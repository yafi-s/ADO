"""Audit sampled physical clearance and independently match ranged front-camera observations."""

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import mmap
from pathlib import Path
import re
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[1]
OFFSETS = ROOT / "results/perception/town03-body-offsets.json"
PARKED = ROOT / "results/perception/town03-static-vehicles.json"
KINDS = {"car", "van", "bus", "truck", "pedestrian", "cyclist", "motorcycle", "bicycle"}
LABELS = {"car": {"car"}, "van": {"car", "truck"}, "bus": {"bus", "truck"},
          "truck": {"truck"}, "pedestrian": {"person"}}
BODY_PAIR_RADIUS = 12.
GAP_LIMITS = (0., .25, .5, 1., 2.)


def read_json(path):
    return json.loads(path.read_bytes())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rotation(pitch, yaw, roll):
    """CARLA's local-to-world rotation in its x-forward, y-right, z-up basis."""
    p, y, r = map(math.radians, (pitch, yaw, roll))
    cp, cy, cr = math.cos(p), math.cos(y), math.cos(r)
    sp, sy, sr = math.sin(p), math.sin(y), math.sin(r)
    return (
        (cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr),
        (cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr),
        (sp, -cp*sr, cp*cr),
    )


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def transform(matrix, xyz):
    return tuple(dot(axis, xyz) for axis in matrix)


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def subtract(a, b):
    return tuple(x-y for x, y in zip(a, b))


def hull(points):
    """Return the CCW convex hull of projected 3D box corners."""
    def half(ordered):
        edge = []
        for x, y in ordered:
            while len(edge) >= 2:
                a, b = edge[-2:]
                if (b[0]-a[0])*(y-a[1]) > (b[1]-a[1])*(x-a[0]) + 1.e-12:
                    break
                edge.pop()
            edge.append((x, y))
        return edge

    ordered = sorted(set(points))
    return half(ordered)[:-1] + half(reversed(ordered))[:-1]


@dataclass
class Body:
    center: tuple
    vertices: tuple
    polygon: list
    bottom: float
    top: float
    radius: float

    def __init__(self, origin, actor_rotation, dimensions, offset, box_rotation):
        self.center = add(origin, transform(actor_rotation, offset))
        axes = tuple(transform(actor_rotation, axis) for axis in zip(*rotation(*box_rotation)))
        length, height, width = dimensions
        self.vertices = tuple(
            tuple(coord + l*x + w*y + h*z for coord, x, y, z in zip(self.center, *axes))
            for l in (-length/2, length/2) for w in (-width/2, width/2) for h in (-height/2, height/2)
        )
        self.polygon = hull((v[0], v[1]) for v in self.vertices)
        self.bottom = min(v[2] for v in self.vertices)
        self.top = max(v[2] for v in self.vertices)
        self.radius = math.sqrt(length**2 + height**2 + width**2) / 2


def point_segment_sq(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    x, y = p[0]-a[0], p[1]-a[1]
    frac = max(0., min(1., (x*dx+y*dy)/(dx*dx+dy*dy)))
    return (x-frac*dx)**2 + (y-frac*dy)**2


def edges(poly):
    return list(zip(poly, poly[1:] + poly[:1]))


def signed_gap(a, b):
    """Euclidean edge separation, or negative minimum planar translation on overlap."""
    a_edges, b_edges = edges(a), edges(b)
    minimum_overlap = math.inf
    for v, w in a_edges + b_edges:
        axis = (v[1]-w[1], w[0]-v[0])
        proj_a, proj_b = [dot(p, axis) for p in a], [dot(p, axis) for p in b]
        overlap = min(max(proj_a)-min(proj_b), max(proj_b)-min(proj_a))
        if overlap < 0.:
            return math.sqrt(min(
                min(point_segment_sq(p, start, end) for p in a for start, end in b_edges),
                min(point_segment_sq(p, start, end) for p in b for start, end in a_edges)
            ))
        minimum_overlap = min(minimum_overlap, overlap / math.hypot(*axis))
    return -minimum_overlap


def box_for(actor, shapes):
    origin = tuple(actor[k] for k in ("x", "y", "z"))
    if "bbox_offset_m" in actor and "bbox_rotation_deg" in actor:
        offset, angles = actor["bbox_offset_m"], actor["bbox_rotation_deg"]
        method = "native_actor_box"
    elif actor.get("role") == "parked-map-vehicle":
        offset, angles = (0., 0., 0.), (0., 0., 0.)
        method = "parked_map_box"
    elif shape := shapes.get(actor.get("blueprint")):
        offset, angles = shape["offset_xyz"], shape["rotation"]
        method = "calibrated_actor_yaw_only"
    else:
        return None, "actor_origin_only"
    heading = rotation(actor.get("pitch", 0.), actor["yaw"], actor.get("roll", 0.))
    return Body(origin, heading, actor["dimensions_m"], offset, angles), method


def stamp(row, trial):
    frame = row["frame"]
    return dict(frame=frame, time_s=row["sim_time"], progress_m=row["progress_m"],
                ego_speed_mps=row["speed_mps"], active_event=row["active_event"],
                inspector="./inspect.html?" + urlencode(dict(run=trial, frame=frame, camera="front")),
                chase_inspector="./inspect.html?" + urlencode(dict(run=trial, frame=frame, camera="chase")))


def actor_info(actor):
    fields = ("actor_id", "type", "role", "blueprint", "name", "x", "y", "z", "yaw", "pitch", "roll",
              "dimensions_m", "bbox_offset_m", "bbox_rotation_deg")
    return {key: actor[key] for key in fields if key in actor}


def finite_record(value):
    return round(value, 4)


class Clearance:
    def __init__(self):
        self.samples = Counter()
        self.limits = {str(v): 0 for v in GAP_LIMITS}
        self.closest = self.center = self.vertical = self.moving = None
        self.overlaps = []

    def observe(self, record, center, box_a, box_b, method):
        self.samples[method] += 1
        if self.center is None or center < self.center[0]:
            self.center = (center, {**record, "actor_origin_distance_m": finite_record(center)})
        if box_a is None or box_b is None:
            return
        vgap = max(box_a.bottom-box_b.top, box_b.bottom-box_a.top)
        # Enclosing spheres provide a lower bound on planar separation.
        lower_bound = math.dist(box_a.center[:2], box_b.center[:2]) - box_a.radius - box_b.radius
        if vgap > 0.:
            self.samples["vertical_disjoint"] += 1
            if self.vertical is not None and lower_bound > self.vertical[0]:
                return
            gap = signed_gap(box_a.polygon, box_b.polygon)
            if self.vertical is None or gap < self.vertical[0]:
                self.vertical = (gap, {**record, "ground_projection_gap_m": finite_record(gap),
                                       "vertical_gap_m": finite_record(vgap)})
            return
        self.samples["vertical_overlap"] += 1
        moving = record.get("actor_moving", False)
        nearest = self.closest[0] if self.closest is not None else math.inf
        moving_limit = self.moving[0] if self.moving else math.inf
        if lower_bound > max(GAP_LIMITS[-1], nearest, moving_limit if moving else 0.):
            return
        gap = signed_gap(box_a.polygon, box_b.polygon)
        details = {**record, "ground_projection_gap_m": finite_record(gap),
                   "vertical_overlap_m": finite_record(min(box_a.top, box_b.top)-max(box_a.bottom, box_b.bottom)),
                   "geometry": method}
        for lim in GAP_LIMITS:
            self.limits[str(lim)] += gap <= lim
        if gap <= 0. and len(self.overlaps) < 10:
            self.overlaps.append(details)
        if self.closest is None or gap < self.closest[0]:
            self.closest = (gap, details)
        if moving and (self.moving is None or gap < self.moving[0]):
            self.moving = (gap, details)

    def report(self):
        return dict(samples=dict(self.samples), at_or_below_m=self.limits,
                    closest_actor_origin=self.center[1] if self.center else None,
                    closest_overlapping_elevation=self.closest[1] if self.closest else None,
                    closest_moving_actor=self.moving[1] if self.moving else None,
                    smallest_xy_with_disjoint_elevation=self.vertical[1] if self.vertical else None,
                    first_projected_overlaps=self.overlaps[:10])


def camera_box(body, origin, basis, camera):
    eye = add(origin, transform(basis, (camera["x"], camera["y"], camera["z"])))
    inv = tuple(zip(*basis))
    fx = .5 / math.tan(math.radians(camera["hfov"]/2))
    fy = fx * camera["width"] / camera["height"]
    points = []
    for vertex in body.vertices:
        x, y, z = transform(inv, subtract(vertex, eye))
        if x <= .25:
            return None
        points.append((.5+fx*y/x, .5-fy*z/x))
    left = max(0., min(x for x, _ in points))
    top = max(0., min(y for _, y in points))
    right = min(1., max(x for x, _ in points))
    bottom = min(1., max(y for _, y in points))
    return (left, top, right, bottom) if right > left and bottom > top else None


def iou(a, b):
    common = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    return common / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - common)


def front_observations(row, targets, fresh):
    tracks = {t["id"]: t for t in row["tracks"] if not t.get("coasting") and "front" in t.get("cameras", ())
              and (t.get("lidar_hits", 0) or t.get("radar_sensors"))}
    matches = []
    for box in fresh:
        if not box.get("range_supported") or not (track := tracks.get(box.get("track_id"))):
            continue
        if box["label"] != track["label"]:
            continue
        candidates = []
        for actor, info, projection in targets:
            if track["label"] not in LABELS.get(actor["type"], ()) or projection is None:
                continue
            overlap = iou(box["box"], projection)
            error = math.dist(info["local_xy_m"], (track["x"], track["y"]))
            if overlap >= .25 and error <= info["center_error_gate_m"]:
                candidates.append((overlap, error, info))
        # Closely aligned bodies can project into the same image rectangle.
        # Require the fresh observation to identify a single eligible actor.
        if len(candidates) != 1:
            continue
        overlap, error, info = candidates[0]
        matches.append((overlap, error, {**info, "box": box["box"], "score": box["score"],
                        "image_id": box.get("image_id"), "track_id": track["id"], "class_label": track["label"],
                        "lidar_hits": track.get("lidar_hits", 0), "radar_sensors": track.get("radar_sensors", []),
                        "range_error_m": finite_record(error), "projected_box_iou": finite_record(overlap)}))
    actors, track_ids, observations = set(), set(), []
    for _, _, hit in sorted(matches, key=lambda h: (-h[0], h[1])):
        if hit["actor_id"] not in actors and hit["track_id"] not in track_ids:
            actors.add(hit["actor_id"])
            track_ids.add(hit["track_id"])
            observations.append(hit)
    return observations


def score(stage, offsets, parked_path, candidate_path):
    capture = read_json(stage / "capture.json")
    raw = stage / "raw"
    readings = raw / "perception.jsonl"
    telemetry_path = raw / "telemetry.json"
    telemetry = read_json(telemetry_path)
    shapes, parked = read_json(offsets), read_json(parked_path)
    if capture["producer"]["map"] != parked["map"]:
        raise ValueError("Parked registry and recording use different maps")
    if len(telemetry) != capture["last_frame"] - capture["first_frame"] + 1:
        raise ValueError("Capture length disagrees with telemetry")
    ego_dimensions = shapes["vehicle.lincoln.mkz"]
    summary = capture["summary"]
    camera = summary["camera_rendering"]["surround"]["camera_mounts"]["front"]
    if (camera["yaw"], camera.get("pitch", 0.), camera.get("roll", 0.)) != (0., 0., 0.):
        raise ValueError("Front matching assumes the recorded level ego-aligned camera")
    actors, pair_groups, pairs = defaultdict(Clearance), defaultdict(Clearance), defaultdict(Clearance)
    groups = defaultdict(Clearance)
    histories, labels = {}, {}
    registered_parked = [(actor, box_for(actor, shapes)[0]) for actor in parked["actors"]]
    braking, acceleration = [], []
    motion = deque(maxlen=6)
    total, fresh_count, optical_frames = 0, 0, Counter()
    with readings.open(encoding="utf-8") as source, candidate_path.open("w", encoding="utf-8", newline="\n") as dest:
        for i, line in enumerate(source):
            row = json.loads(line)
            frame = row["frame"]
            record = stamp(row, capture["trial_id"])
            t = telemetry[i]
            if (frame != t["camera_frame"] or t["frame"] != capture["first_frame"] + i
                    or row["sim_time"] != t["sim_time"] or t["video_index"] != i
                    or not math.isclose(t["video_time"], i / 10.)):
                raise ValueError(f"Journal and native exposure mismatch at {frame}")
            pos = t["ego_pose"]
            record["ego_pose"] = pos
            origin = (pos["x"], pos["y"], pos["z"])
            basis = rotation(pos["pitch"], pos["yaw"], pos["roll"])
            ego = Body(origin, basis, ego_dimensions["dimensions_m"], ego_dimensions["offset_xyz"],
                       ego_dimensions["rotation"])
            motion.append(t)
            if len(motion) == 6:
                interval = row["sim_time"] - motion[0]["sim_time"]
                dv = (row["speed_mps"]-motion[0]["speed_mps"]) / interval
                sample = (dv, {**record, "first_frame": motion[0]["frame"], "interval_s": finite_record(interval),
                               "initial_speed_mps": motion[0]["speed_mps"], "accel_mps2": finite_record(dv),
                               "brake": row["brake"], "throttle": row["throttle"]})
                braking.append(sample)
                acceleration.append(sample)
            fresh = row.get("camera_detections", {}).get("front", []) if row.get("camera_inference_age_s", {}).get("front") == 0. else None
            fresh_count += fresh is not None
            targets, dynamic = [], []
            current = set()
            yaw = math.radians(pos["yaw"])
            for actor in row["observer_map"]["others"]:
                kind, role, identity = actor["type"], actor["role"], actor["actor_id"]
                dist = math.hypot(actor["x"]-origin[0], actor["y"]-origin[1])
                # The query uses 100 m before rounding actor origins to centimetres.
                if dist > 100.02:
                    raise ValueError(f"Recorded actor {identity} is outside the 100 m ego query at frame {frame}")
                if kind not in KINDS:
                    continue
                if identity in current:
                    raise ValueError(f"Recorded actor {identity} appears twice at frame {frame}")
                current.add(identity)
                labels.setdefault(identity, actor_info(actor))
                body, method = box_for(actor, shapes)
                series = histories.setdefault(identity, deque(maxlen=6))
                point = (actor["x"], actor["y"])
                if series and math.dist(point, series[-1]) > 3.:
                    series.clear()
                series.append(point)
                speed = math.dist(series[0], point)/.5 if len(series) == 6 else None
                details = {**record, "actor": actor_info(actor), "actor_origin_speed_mps": finite_record(speed)
                           if speed is not None else None,
                           "actor_moving": speed is not None and speed >= (.5 if kind == "pedestrian" else 1.)}
                key = ("background" if role == "background" else "staged") + "/" + kind
                actors[identity].observe(details, dist, ego, body, method)
                groups[key].observe(details, dist, ego, body, method)
                dynamic.append((actor, body, method, details["actor_origin_speed_mps"]))
                if fresh and body and kind in LABELS and dist <= 100.:
                    dx, dy = body.center[0]-origin[0], body.center[1]-origin[1]
                    x, y = dx*math.cos(yaw)+dy*math.sin(yaw), -dx*math.sin(yaw)+dy*math.cos(yaw)
                    info = dict(actor_id=identity, actor_type=kind, role=role, blueprint=actor["blueprint"],
                                center_range_m=finite_record(math.hypot(x, y)), local_xy_m=(x, y),
                                center_error_gate_m=1. if kind == "pedestrian" else 2.,
                                preceding_half_second_mps=finite_record(speed) if speed is not None else None)
                    if x > 1.3 and abs(y) < x:
                        targets.append((actor, info, camera_box(body, origin, basis, camera)))
            for identity in histories.keys() - current:
                del histories[identity]
            for actor, body in registered_parked:
                dist = math.hypot(actor["x"]-origin[0], actor["y"]-origin[1])
                if dist <= 100.:
                    groups["static_map/" + actor["type"]].observe(
                        {**record, "actor": actor_info(actor)}, dist, ego, body, "parked_map_box")
            dynamic.sort(key=lambda a: a[0]["x"])
            for j, (actor, body, method, speed) in enumerate(dynamic):
                for other, other_body, other_method, other_speed in dynamic[j+1:]:
                    if other["x"]-actor["x"] > BODY_PAIR_RADIUS:
                        break
                    dist = math.hypot(actor["x"]-other["x"], actor["y"]-other["y"])
                    if dist > BODY_PAIR_RADIUS:
                        continue
                    group = "/".join(sorted((actor["type"], other["type"])))
                    pair_method = "+".join(sorted({method, other_method}))
                    details = {**record, "actors": [actor_info(actor), actor_info(other)],
                               "actor_origin_speeds_mps": [speed, other_speed]}
                    pair_groups[group].observe(details, dist, body, other_body, pair_method)
                    pair = tuple(sorted((actor["actor_id"], other["actor_id"])))
                    pairs[pair].observe(details, dist, body, other_body, pair_method)
            if fresh:
                observations = front_observations(row, targets, fresh)
                peds = sum(hit["actor_type"] == "pedestrian" for hit in observations)
                vehicles = len(observations) - peds
                optical_frames["any"] += bool(observations)
                optical_frames["vehicle_and_pedestrian"] += min(vehicles, peds) >= 1
                optical_frames["two_of_each"] += min(vehicles, peds) >= 2
                if peds and vehicles:
                    dest.write(json.dumps({**record, "front_camera_matches": observations}) + "\n")
            total += 1
    if total != len(telemetry):
        raise ValueError("Perception and native image telemetry have different lengths")

    def extrema(samples, reverse):
        # Keep different braking or acceleration episodes rather than adjacent samples of one manoeuvre.
        selected = []
        for _, item in sorted(samples, key=lambda r: r[0], reverse=reverse):
            if all(abs(item["frame"]-previous["frame"]) >= 10 for previous in selected):
                selected.append(item)
            if len(selected) == 6:
                break
        return selected

    keys = ("outcome", "frames", "duration_s", "progress_m", "route_length_m", "contacts",
            "collision_events", "red_light_violations", "yield_violations", "moving_within_1m_frames",
            "min_hazard_gap_m", "turns_completed", "sidewalk_activity", "background_traffic_activity", "events")
    hashes = {p.name: sha256(p) for p in (readings, telemetry_path, offsets, parked_path, stage / "capture.json")}
    for source in (readings.name, telemetry_path.name):
        if hashes[source] != capture["raw_sha256"][source]:
            raise ValueError(f"Audit input changed from the native capture's {source}")
    return dict(schema="ado-city-interaction-audit-v1", capture=capture["capture_id"], trial=capture["trial_id"],
                policy=capture["policy"]["schema"], passed=capture["passed"],
                capture_failure_reasons=capture["failure_reasons"], source_hashes=hashes, frames=total,
                first_frame=capture["first_frame"], last_frame=capture["last_frame"],
                native_grade={key: summary.get(key) for key in keys},
                front_camera=dict(fresh_frames=fresh_count, simultaneous_ranged_matches=dict(optical_frames)),
                by_actor=[{**labels[k], **v.report()} for k, v in sorted(actors.items())],
                ego_clearance={k: v.report() for k, v in sorted(groups.items())},
                participant_pairs_within_12m={k: v.report() for k, v in sorted(pair_groups.items())},
                closest_participant_pairs=[
                    {**row.report(), "pair": key} for key, row in sorted(
                        ((pair, clearance) for pair, clearance in pairs.items() if clearance.closest),
                        key=lambda entry: entry[1].closest[0])[:12]],
                harsh_braking_half_second=extrema(braking, False),
                harsh_acceleration_half_second=extrema(acceleration, True))


def extract_evidence(stage, output, probes):
    capture = read_json(stage / "capture.json")
    requests = defaultdict(set)
    for probe in probes:
        camera, frame_number = probe.split(":")
        frame = int(frame_number)
        if camera not in ("chase", "front", *capture["perimeter_views"]):
            raise ValueError(f"Camera {camera!r} was not recorded")
        if not capture["first_frame"] <= frame <= capture["last_frame"]:
            raise ValueError(f"Frame {frame} was outside the recorded interval")
        requests[camera].add(frame)
    evidence = []
    output.mkdir(parents=True, exist_ok=True)
    for camera, frames in sorted(requests.items()):
        warmup = capture["warmup_frames"] if camera == "chase" else 0
        indices = {frame - capture["first_frame"] + warmup: frame for frame in frames}
        poster_index = capture["poster_frame"] - capture["first_frame"] + warmup
        picks = {}
        source = stage / "raw" / f"{camera}.mjpeg"
        source_sha256 = capture["raw_sha256"][source.name]
        with source.open("rb") as raw, mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ) as data:
            if data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
                raise ValueError(f"Invalid JPEG endpoints in {source}")
            if hashlib.sha256(data).hexdigest() != source_sha256:
                raise ValueError(f"JPEG source changed from the native capture's {source}")
            # The recorder writes consecutive OpenCV JPEGs with no container or padding.
            boundary = re.compile(b"\xff\xd9\xff\xd8")
            first = 0
            for index, mark in enumerate(boundary.finditer(data)):
                end = mark.start() + 2
                if index in indices or index == poster_index:
                    picks[index] = data[first:end]
                first = end
            count = index + 2 if first else 1
            if count - 1 in indices or count - 1 == poster_index:
                picks[count - 1] = data[first:]
            expected = capture.get(f"{camera}_source_frames", capture["summary"]["frames"])
            if count != expected:
                raise ValueError(f"Expected {expected} JPEGs, found {count} in {source}")
        if camera in ("chase", "front"):
            poster = stage / "raw" / ("poster.jpg" if camera == "chase" else "front.jpg")
            if poster.read_bytes() != picks.get(poster_index):
                raise ValueError(f"Camera index does not reproduce the capture's original poster for {camera}")
        for index, frame in sorted(indices.items()):
            name = f"{output.stem}-{camera}-{frame}.jpg"
            image = output / name
            image.write_bytes(picks[index])
            evidence.append(dict(frame=frame, camera=camera, raw_jpeg_index=index, file=name,
                                 raw_stream_sha256=source_sha256, sha256=sha256(image)))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, default=OFFSETS)
    parser.add_argument("--parked", type=Path, default=PARKED)
    parser.add_argument("--image", action="append", default=[], metavar="CAMERA:FRAME")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = score(args.stage, args.offsets, args.parked, args.candidates)
    if args.image:
        folder = args.output.with_suffix("")
        result["native_jpeg_evidence"] = extract_evidence(args.stage, folder, args.image)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
