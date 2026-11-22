"""Audit the Town03 Market drive against independently reconstructed native geometry."""

import argparse
from collections import Counter, defaultdict, deque
import gzip
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon

from tools.score_city_interactions import OFFSETS, PARKED, Body, box_for, rotation, sha256, signed_gap


SECTIONS = (
    ("taper_approach", 358., 375.), ("beside_cones", 375., 398.),
    ("open_curve_entrance", 398., 420.), ("tighter_curve", 420., 456.),
    ("easing_curve", 456., 500.), ("curve_exit_board_approach", 500., 525.),
    ("beside_boards", 525., 547.), ("stopped_lead", 547., 570.),
    ("departing_lead", 570., 640.), ("braking_lead", 950., 1046.),
    ("cyclist_approach", 1175., 1235.),
)
LEADS = {"lead", "braking", "cyclist", "slow-traffic", "in-lane-obstruction"}
VEHICLES = {"car", "van", "bus", "truck", "cyclist", "bicycle", "motorcycle"}
PROPS = {"barrier", "cone"}
VULNERABLE = {"pedestrian", "cyclist", "bicycle", "motorcycle"}
DT = .1
RADII = (35, 50, 65)


def read(path):
    return json.loads(path.read_bytes())


def rnd(number):
    return round(float(number), 3)


def spans(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def load_capture(folder, reference, position_journal, focused=False):
    if (folder / "manifest.json").is_file():
        manifest = read(folder / "manifest.json")
        telemetry = folder / "telemetry.json"
        expected = manifest["sha256"]["telemetry.json"]
    elif (folder / "capture.json").is_file():
        manifest = read(folder / "capture.json")
        telemetry = folder / "raw/telemetry.json"
        expected = manifest["raw_sha256"]["telemetry.json"]
    else:
        summary = read(folder / "summary.json")
        manifest = {"summary": summary, "producer": summary.get("producer")}
        telemetry = folder / "trace.jsonl.gz"
        expected = None
        checkpoint = folder / "checkpoint.npz"
        if checkpoint.is_file():
            manifest["policy"] = {"sha256": sha256(checkpoint)}
        elif focused and (folder.parent / "checkpoint.npz").is_file():
            digest = sha256(folder.parent / "checkpoint.npz")
            if digest != summary.get("model_sha256"):
                raise ValueError("Focused trial checkpoint differs from the retained digest")
            manifest["policy"] = {"sha256": digest}
    digest = sha256(telemetry)
    if expected is not None and digest != expected:
        raise ValueError("Source telemetry digest differs from the sealed capture")
    ref_hash = sha256(reference)
    expected_reference = manifest.get("population_reference_sha256")
    if expected_reference is not None and expected_reference != ref_hash:
        raise ValueError("Population digest differs from the sealed capture")
    if telemetry.suffix == ".gz":
        with gzip.open(telemetry, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
    else:
        rows = read(telemetry)
    source = {"telemetry_sha256": digest, "population_sha256": ref_hash,
              "capture_id": manifest.get("capture_id", folder.name),
              "seal_verified": expected is not None and expected_reference is not None}
    if focused:
        hashes = manifest["summary"].get("artifact_sha256", {})
        if (hashes.get(telemetry.name) != digest or hashes.get(reference.name) != ref_hash):
            raise ValueError("Focused trial trace or footprint journal differs from the retained digest")
        with gzip.open(reference, "rt", encoding="utf-8") as stream:
            for row, sample in zip(rows, (json.loads(line) for line in stream), strict=True):
                if any(row[key] != sample[key] for key in ("frame", "sim_time", "native_s")):
                    raise ValueError("Focused trial pose journal does not match its retained trace")
                x, y, yaw = sample["ego"]
                row["ego_pose"] = dict(x=x, y=y, z=sample["ego_z"], yaw=yaw, pitch=0., roll=0.)
                row["native_ego_polygon_xy"] = [tuple(p) for p in sample["ego_polygon_xy"]]
        source.update(
            verified_focused_trace_and_footprint=True,
            ego_pose_resolution="Native ego polygon in XY. Rounded pose and level pitch/roll for height support.",
        )
    if position_journal is not None:
        if any("ego_pose" in row for row in rows):
            raise ValueError("The primary tape already has full native ego pose")
        with position_journal.open(encoding="utf-8") as stream:
            for row, item in zip(rows, (json.loads(line) for line in stream), strict=True):
                if any(row[key] != item[key] for key in ("frame", "sim_time", "progress_m")):
                    raise ValueError("Rehearsal pose journal does not match the primary tape")
                observed = item["observer_map"]
                x, y, yaw = observed["ego"]
                row["ego_pose"] = dict(x=x, y=y, z=observed["ego_z"], yaw=yaw, pitch=0., roll=0.)
        source.update(position_journal_sha256=sha256(position_journal),
                      ego_pose_resolution="Rounded native position and yaw, with flat pitch and roll for bodies")
    return manifest, rows, source


def stamp(row):
    return dict(frame=row["frame"], time_s=row["sim_time"], station_m=row["progress_m"],
                actual_kmh=rnd(3.6 * row["speed_mps"]), target_kmh=rnd(3.6 * row["speed_request_mps"]))


def lane_permitted(point, side):
    other = point[side]
    if other is None:
        return False
    angle = (other["heading_deg"] - point["heading_deg"] + 180.) % 360. - 180.
    return (point["junction"] is None and other["junction"] is None
            and other["kind"] == "Driving" and point["kind"] == "Driving"
            and point["road"] == other["road"] and point["lane"] * other["lane"] > 0
            and abs(angle) < 4. and min(point["width_m"], other["width_m"]) >= 3.4
            and point[f"{side}_marking"] == "Broken"
            and point["change"] in (side.capitalize(), "Both"))


class Geometry:
    def __init__(self, route):
        self.route = route
        self.sample = route["samples"]
        self.station = np.array([p["station_m"] for p in self.sample])
        self.xyz = np.asarray([p["xyz"] for p in self.sample])
        self.yaw = np.unwrap(np.deg2rad([p["heading_deg"] for p in self.sample]))
        self.legal = {side: np.array([lane_permitted(p, side) for p in self.sample])
                      for side in ("left", "right")}
        self.run_end = {}
        for side in self.legal:
            ends = np.zeros(len(self.sample))
            for begin, end in spans(self.legal[side]):
                ends[begin:end] = self.station[end - 1]
            self.run_end[side] = ends
        offsets = np.r_[1, np.maximum(.001, np.diff(self.station))]
        self.curvature = np.abs(np.diff(self.yaw, prepend=self.yaw[0])) / offsets
        self._corridors = {}

    def sample_index(self, station):
        return int(np.clip(np.searchsorted(self.station, station, side="right") - 1, 0, len(self.station) - 1))

    def describe(self, lo, hi):
        selected = np.flatnonzero((self.station >= lo) & (self.station < hi))
        if not len(selected):
            return {}
        delta = np.rad2deg(self.yaw[selected[-1]] - self.yaw[selected[0]])
        return dict(
            roads=sorted({self.sample[i]["road"] for i in selected}),
            lanes=sorted({self.sample[i]["lane"] for i in selected}),
            heading_change_deg=rnd(delta), curvature_p95_per_m=rnd(np.quantile(self.curvature[selected], .95)),
            curvature_max_per_m=rnd(np.max(self.curvature[selected])),
            legal_lane_samples={side: int(mask[selected].sum()) for side, mask in self.legal.items()},
            samples=len(selected),
        )

    def corridor(self, idx):
        if idx not in self._corridors:
            self._corridors[idx] = self.build_corridor(idx)
        return self._corridors[idx]

    def build_corridor(self, idx):
        center = self.station[idx]
        before = self.sample_index(center - 65.)
        end = min(len(self.sample), self.sample_index(center + 75.) + 2)
        start = self.xyz[before:end-1, :2]
        delta = np.diff(self.xyz[before:end, :2], axis=0)
        length = np.maximum(np.linalg.norm(delta, axis=1), 1.e-8)
        forward = delta / length[:, None]
        arc = self.station[before:end-1]
        # The path occupancy lookahead is deliberately shorter than the adjacent
        # gap lookahead. Rejoins are screened through the next 75 m.
        aim = min(end, self.sample_index(center + 32.) + 2)
        source = LineString(self.xyz[idx:aim, :2]) if aim - idx > 1 else None
        alternatives = {}
        for side in self.legal:
            if not self.legal[side][idx]:
                continue
            near = self.sample[idx]
            points = [p[side]["xyz"][:2] for j, p in enumerate(self.sample[before:end], before)
                      if self.legal[side][j] and (p["road"], p["lane"]) == (near["road"], near["lane"])]
            if len(points) > 1:
                alternatives[side] = LineString(points)
        return dict(start=start, forward=forward, length=length, arc=arc, source=source, alternatives=alternatives)


def project(points, corridor):
    delta = np.asarray(points)[:, None, :] - corridor["start"]
    forward = corridor["forward"]
    advance = np.clip(np.sum(delta * forward, axis=-1), 0., corridor["length"])
    remain = delta - advance[:, :, None] * forward
    i = np.argmin(np.sum(remain * remain, axis=-1), axis=1)
    j = np.arange(len(i))
    return (corridor["arc"][i] + advance[j, i],
            remain[j, i, 1] * forward[i, 0] - remain[j, i, 0] * forward[i, 1])


def planar_clearance(ego, body):
    gap = ego.outline.distance(body.outline)
    if gap == 0. and len(body.polygon) >= 3:
        gap = signed_gap(ego.polygon, body.polygon)
    return gap


def clearance(actor, body, ego, row, motion, method, records):
    if (min(body.top, ego.top) < max(body.bottom, ego.bottom)
            or math.dist(ego.center[:2], body.center[:2]) - ego.radius - body.radius > 4.):
        return
    gap = planar_clearance(ego, body)
    if gap > 4.:
        return
    key = f'{actor["type"]}/{actor["role"]}'
    entry = records.setdefault(key, dict(overlap_samples=0, below_half_m_samples=0, below_1m_samples=0,
                                         minimum_m=math.inf, minimum_moving_m=None, actors={}))
    entry["overlap_samples"] += gap <= 0.
    entry["below_half_m_samples"] += gap < .5
    entry["below_1m_samples"] += gap < 1.
    detail = dict(**stamp(row), actor_id=actor["actor_id"], motion_mps=None if motion is None else rnd(motion),
                  gap_m=rnd(gap), box_method=method)
    if gap <= entry["minimum_m"]:
        entry.update(minimum_m=rnd(gap), nearest=detail)
    if row["speed_mps"] >= 5. / 3.6 and (entry["minimum_moving_m"] is None or gap < entry["minimum_moving_m"]):
        entry.update(minimum_moving_m=rnd(gap), nearest_moving=detail)
    aid = str(actor["actor_id"])
    if gap < entry["actors"].get(aid, {}).get("gap_m", math.inf):
        entry["actors"][aid] = detail


def reference_body(actor, shapes):
    if (not actor.get("blueprint", "").startswith("static.prop.") or "bbox_rotation_deg" not in actor
            or "bbox_offset_m" not in actor):
        return box_for(actor, shapes)
    # CARLA 1.0 prop boxes are already rotated relative to the world. The
    # native scorer calls get_world_vertices with translation only.
    world = tuple(actor[axis] for axis in ("x", "y", "z"))
    body = Body(world, rotation(0., 0., 0.), actor["dimensions_m"], actor["bbox_offset_m"],
                actor["bbox_rotation_deg"])
    return body, "native_world_aligned_prop_box"


def timing_corridor(station, speed, lead):
    """Bound the required route length for a hypothetical comfortable pass."""
    time = np.arange(0., 12.1, .1)
    cap = 40. / 3.6
    speed = min(cap, speed)
    accelerating = np.minimum(time, (cap - speed) / 1.5)
    ego = station + speed * accelerating + .75 * accelerating**2 + cap * (time - accelerating)
    lead_front = lead["front_m"] + lead["speed"] * time
    # Leave 8 m from the front of the lead to the rear of ego before turning back.
    alongside = np.flatnonzero((ego >= station + 24.) & (ego - 2.5 >= lead_front + 8.))
    if not len(alongside):
        return None
    end = float(ego[alongside[0]] + 24.)
    return end if end <= ego[-1] else None


def opportunities(geometry, idx, station, speed, actors, sides, vru):
    output = []
    for lead in actors:
        distance = lead["rear_m"] - station - 2.5
        if (lead["role"] not in LEADS or lead["speed"] is None or not lead["source_path"]
                or not 0. <= distance <= 35. or lead["speed"] >= 40. / 3.6 - 2.):
            continue
        end = timing_corridor(station, speed, lead)
        if end is None:
            output.append((lead, None, "insufficient_speed_advantage", None))
            continue
        available = False
        for side, neighbors in sides.items():
            available = True
            min_front, min_rear, pressure = math.inf, math.inf, []
            failures = []
            if end + 15. > geometry.run_end[side][idx]:
                failures.append("short_legal_return_or_route_branch")
            if vru:
                failures.append("vulnerable_road_user_near_path")
            for other in neighbors:
                if other["rear_m"] > end + 35. or other["front_m"] < station - 60.:
                    continue
                speed_other = other["speed"]
                ahead = other["rear_m"] > station
                gap = (other["rear_m"] - station - 2.5 if ahead else station - other["front_m"] - 2.5)
                if ahead:
                    min_front = min(min_front, gap)
                else:
                    min_rear = min(min_rear, gap)
                actor_failure = None
                if other["kind"] in PROPS and other["front_m"] >= station - 3. and other["rear_m"] < end:
                    actor_failure = "prop_in_alternative"
                elif other["kind"] in VEHICLES and speed_other is None:
                    actor_failure = "adjacent_vehicle_motion_unresolved"
                elif other["kind"] in VEHICLES:
                    threshold = max(15., 2. * speed) if ahead else max(12., 2. * speed_other)
                    closing = speed - speed_other if ahead else speed_other - speed
                    if gap < threshold or (closing > 0. and gap / closing < 5.):
                        actor_failure = "occupied_or_closing_alternative"
                    elif ahead and other["rear_m"] <= end + 15. and speed_other < speed + 1.:
                        actor_failure = "forward_vehicle_in_pass_distance"
                if actor_failure:
                    failures.append(actor_failure)
                    pressure.append(other["id"])
            reason = failures[0] if failures else "potential_gap"
            output.append((lead, side, reason, dict(
                geometric_end_m=rnd(end), legal_run_end_m=rnd(geometry.run_end[side][idx]),
                front_gap_m=rnd(min_front) if math.isfinite(min_front) else None,
                rear_gap_m=rnd(min_rear) if math.isfinite(min_rear) else None,
                reasons=sorted(set(failures)), nearby_target_actor_ids=pressure,
            )))
        if not available:
            output.append((lead, None, "no_permitted_adjacent_lane", None))
    return output


def actual_maneuver(geometry, idx, corridor, ego):
    """Recompute whole-body road occupancy from native boxes and OpenDRIVE waypoints."""
    point = geometry.sample[idx]
    if point["road"] not in (3, 67) or point["junction"] is not None:
        return None
    side = "left" if point["road"] == 67 else "right"
    alternative = point[side]
    if alternative is None or not lane_permitted(point, side):
        return None
    direction = -1. if side == "left" else 1.
    along, laterals = project([ego.center[:2], *ego.polygon], corridor)
    centers, body = direction * laterals[0], direction * laterals[1:]
    edge = point["width_m"] / 2.
    limits = (-edge, edge + alternative["width_m"])
    return dict(
        side=side, road=point["road"], source_lane=point["lane"], alternate_lane=alternative["lane"],
        ego_rear_m=float(along[1:].min()), ego_front_m=float(along[1:].max()),
        center_in_alternate=bool(centers > edge), body_in_alternate=bool(body.min() > edge),
        body_in_source=bool(body.max() <= edge + .05),
        body_within_legal_union=bool(body.min() >= limits[0] - .05 and body.max() <= limits[1] + .05),
    )


def transition_pair(row, geometry, idx, maneuver, actor, body, ego, motion, history, stations, lateral):
    """Measure vehicles in both native lanes while the ego body crosses their divider."""
    if (maneuver is None or maneuver["body_in_source"] or maneuver["body_in_alternate"]
            or motion is None or actor["type"] not in VEHICLES
            or min(body.top, ego.top) < max(body.bottom, ego.bottom)):
        return None
    side = 1. if maneuver["side"] == "right" else -1.
    point = geometry.sample[idx]
    edge = point["width_m"] / 2.
    extent = edge + point[maneuver["side"]]["width_m"]
    side_center = side * float(np.mean(lateral))
    if not -edge <= side_center <= extent:
        return None
    yaw = geometry.yaw[idx]
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    travel = np.array([actor["x"], actor["y"]]) - history[0][1:]
    elapsed = row["sim_time"] - history[0][0]
    moving = motion >= .5
    actor_speed = float(travel @ forward / elapsed) if moving and elapsed > 0. else 0.
    ego_speed = row["speed_mps"] * math.cos(math.radians(row["ego_pose"]["yaw"]) - yaw)
    if moving and actor_speed <= 0.:
        return None
    ego_front, ego_rear = maneuver["ego_front_m"], maneuver["ego_rear_m"]
    actor_rear, actor_front = float(np.min(stations)), float(np.max(stations))
    ahead = (actor_front + actor_rear) >= (ego_front + ego_rear)
    headway = actor_rear - ego_front if ahead else ego_rear - actor_front
    trailing_speed = ego_speed if ahead else actor_speed
    closing = ego_speed - actor_speed if ahead else actor_speed - ego_speed
    gap = planar_clearance(ego, body)
    proximity = abs((actor_front + actor_rear - ego_front - ego_rear) / 2.)
    if proximity > 60.:
        return None
    return dict(
        actor_id=actor["actor_id"], role=actor["role"], kind=actor["type"],
        actor_lane="alternate" if side_center > edge else "source",
        physical_gap_m=rnd(gap), estimated_actor_along_speed_mps=rnd(actor_speed),
        estimated_ego_along_speed_mps=rnd(ego_speed),
        actor_is_ahead=bool(ahead), signed_longitudinal_headway_m=rnd(headway),
        closing_mps=rnd(closing), headway_s=rnd(max(0., headway) / trailing_speed) if trailing_speed > .5 else None,
        closing_ttc_s=rnd(max(0., headway) / closing) if closing > .2 else None,
        moving=moving, below_required_separation=moving and gap < 1., physically_touching=gap <= 0.,
    )


def measure_actors(rows, stream, geometry, ego_blueprint):
    shapes, parked = read(OFFSETS), read(PARKED)["actors"]
    ego_shape = shapes[ego_blueprint]
    width = ego_shape["dimensions_m"][2]
    history = defaultdict(lambda: deque(maxlen=6))
    cache, frames, records, unknown = {}, [], {}, Counter()
    density = {str(r): Counter(scored=0, cars=0, pedestrians=0, both=0, two_each=0) for r in RADII}
    for i, (row, reference) in enumerate(zip(rows, stream, strict=True)):
        pose = row["ego_pose"]
        x, y, z = (pose[k] for k in ("x", "y", "z"))
        if (row["frame"] != reference["frame"] or abs(row["sim_time"] - reference["sim_time"]) > .0001
                or abs(row["progress_m"] - reference.get("progress_m", row["progress_m"])) > .01
                or math.dist((x, y), reference["ego"][:2]) > .02):
            raise ValueError(f"Reference mismatch at {row['frame']}")
        time, station, speed = row["sim_time"], row["progress_m"], row["speed_mps"]
        idx = geometry.sample_index(station)
        corridor = geometry.corridor(idx)
        ego = Body((x, y, z), rotation(pose["pitch"], pose["yaw"], pose["roll"]),
                   ego_shape["dimensions_m"], ego_shape["offset_xyz"], ego_shape["rotation"])
        if "native_ego_polygon_xy" in row:
            ego.polygon = row["native_ego_polygon_xy"]
        ego.outline = Polygon(ego.polygon)
        maneuver = actual_maneuver(geometry, idx, corridor, ego)
        nearby, crossing_pairs, sides = [], [], {side: [] for side in corridor["alternatives"]}
        density_frame = {radius: Counter() for radius in density}
        classifications = defaultdict(list)
        for actor in (*reference["others"], *parked):
            kind, role, aid = actor["type"], actor["role"], actor["actor_id"]
            if kind not in VEHICLES | VULNERABLE | PROPS:
                continue
            ax, ay, az = actor["x"], actor["y"], actor["z"]
            radius = math.hypot(ax-x, ay-y)
            if radius > 80.:
                continue
            mark = (ax, ay, az, actor["yaw"], actor.get("pitch"), actor.get("roll"))
            last_mark, body, method = cache.get(aid, (None, None, None))
            if mark != last_mark:
                body, method = reference_body(actor, shapes)
                if body is not None and len(body.polygon) < 2:
                    body = None
                if body is not None:
                    # CARLA reports zero longitudinal extent for the striped boards.
                    body.outline = (Polygon(body.polygon) if len(body.polygon) >= 3
                                    else LineString(body.polygon))
                cache[aid] = mark, body, method
            if body is None:
                if radius <= 12.:
                    unknown[str(actor.get("blueprint"))] += 1
                continue
            points = history[aid]
            if points and (time - points[-1][0] > .101 or math.dist((ax, ay), points[-1][1:]) > 3.):
                points.clear()
            points.append((time, ax, ay))
            motion = None
            if len(points) == 6:
                motion = math.dist((ax, ay), points[0][1:]) / (time - points[0][0])
            if role == "parked-map-vehicle" or kind in PROPS:
                motion = 0.
            if kind in {"car", "van", "bus", "truck", "pedestrian"} and motion is not None:
                category = "pedestrians" if kind == "pedestrian" else "cars"
                travelled = math.dist((ax, ay), points[0][1:]) if len(points) == 6 else 0.
                if travelled >= (.5 if category == "pedestrians" else 1.) * (time - points[0][0]) > 0.:
                    for distance, counts in density_frame.items():
                        counts[category] += radius <= int(distance)
                        if role == "background":
                            counts["background_" + category] += radius <= int(distance)
            if radius > 76.:
                continue
            clearance(actor, body, ego, row, motion, method, records)
            path = corridor["source"]
            if path is None:
                continue
            stations, lateral = project(body.polygon, corridor)
            path_z = np.interp(float(np.mean(stations)), geometry.station, geometry.xyz[:, 2])
            if min(body.top, path_z+1.75) < max(body.bottom, path_z-.15):
                continue
            pair = transition_pair(row, geometry, idx, maneuver, actor, body, ego, motion, points, stations, lateral)
            if pair is not None:
                crossing_pairs.append(pair)
            ahead = np.max(stations) >= station+2.5
            relation = path.distance(body.outline)
            observation = dict(
                kind=kind, role=role, id=aid,
                rear_m=float(np.min(stations)), front_m=float(np.max(stations)), speed=motion,
                source_path=relation <= width/2. and ahead,
            )
            if ahead and relation <= width/2. + 2.:
                near_road = abs(float(np.mean(lateral))) < 5.
                if near_road and observation["rear_m"] <= station + 33.:
                    # Report physical path overlap distinct from narrowing or side proximity.
                    location = "path" if observation["source_path"] else "beside_path"
                    moving = "unresolved" if motion is None else "moving" if motion >= .5 else "still"
                    bucket = f"{location}/{kind}/{role}/{moving}"
                    classifications[bucket].append(aid)
            for side, alternative in corridor["alternatives"].items():
                if alternative.distance(body.outline) <= width/2.:
                    sides[side].append(observation)
            if role in LEADS:
                # The source path is drawn to 32 m. Extend the screened lead
                # search by the bumper and the positions of its polygon edges.
                observation["source_path"] = (
                    min(lateral) <= width/2. and max(lateral) >= -width/2.
                )
                if maneuver and observation["source_path"] and abs(observation["rear_m"] - station) < 75.:
                    observation["ego_body_gap"] = ego.outline.distance(body.outline)
                nearby.append(observation)
        for radius, counted in density_frame.items():
            if i >= 5:
                tally = density[radius]
                cars, peds = counted["cars"], counted["pedestrians"]
                tally.update(scored=1, cars=int(cars > 0), pedestrians=int(peds > 0), both=int(min(cars, peds) > 0),
                             two_each=int(min(cars, peds) > 1))
        vru = any(bucket.split("/")[1] in VULNERABLE and bucket.startswith("path/")
                  for bucket in classifications)
        wants_pass = [
            dict(actor_id=a["id"], role=a["role"], target_side=side, result=result,
                 lead_speed_mps=rnd(a["speed"]), lead_rear_m=rnd(a["rear_m"]), gap_checks=details)
            for a, side, result, details in opportunities(geometry, idx, station, speed, nearby, sides, vru)
        ]
        frames.append(dict(classifications=classifications, opportunities=wants_pass,
                           actual_maneuver=maneuver, lead_bodies=nearby, crossing_pairs=crossing_pairs,
                           density=density_frame if i >= 5 else {}))
    return frames, records, dict(unknown), density


def departures(rows, physical):
    """Prove passing by pose and signed progress around an actual mapped lateral move."""
    locations = [sample["actual_maneuver"] for sample in physical]
    offsets = [sample is not None and not sample["body_in_source"] for sample in locations]
    moves = []
    for lo, hi in spans(offsets):
        if hi - lo < 3:
            continue
        interval = locations[lo:hi]
        center = next((lo+j for j, p in enumerate(interval) if p["center_in_alternate"]), None)
        if center is None:
            continue
        whole = next((lo+j for j, p in enumerate(interval) if p["body_in_alternate"]), None)
        alongside, ahead = {}, {}
        for i in range(center, min(hi+1, len(rows))):
            alignment = locations[i]
            if alignment is None:
                continue
            for a in physical[i]["lead_bodies"]:
                if not a["source_path"] or "ego_body_gap" not in a:
                    continue
                identity = str(a["id"])
                detail = dict(**stamp(rows[i]), actor_id=a["id"], role=a["role"],
                              body_clearance_m=rnd(a["ego_body_gap"]))
                if (alignment["center_in_alternate"] and alignment["ego_rear_m"] <= a["front_m"]
                        and alignment["ego_front_m"] >= a["rear_m"]):
                    group = alongside.setdefault(identity, {"first": detail, "last": detail, "nearest": detail})
                    group["last"] = detail
                    if detail["body_clearance_m"] < group["nearest"]["body_clearance_m"]:
                        group["nearest"] = detail
                if identity in alongside and alignment["ego_rear_m"] - a["front_m"] >= 1.:
                    ahead.setdefault(identity, dict(**detail, longitudinal_gap_m=rnd(
                        alignment["ego_rear_m"] - a["front_m"])))
        first = locations[lo]
        illegal = [stamp(rows[i]) for i in range(lo, hi) if not locations[i]["body_within_legal_union"]]
        rejoin = hi < len(rows) and locations[hi] and locations[hi]["body_in_source"]
        first_unknown = next((stamp(rows[i]) for i in range(lo, hi)
                              if rows[i].get("passing_footprint_in_legal_lanes") is False), None)
        moves.append(dict(
            road=first["road"], source_lane=first["source_lane"], alternate_lane=first["alternate_lane"],
            side=first["side"], footprint_departs=stamp(rows[lo]), center_crosses=stamp(rows[center]),
            whole_body_in_alternate=stamp(rows[whole]) if whole is not None else None,
            last_alternate_sample=stamp(rows[hi-1]), restored_source=stamp(rows[hi]) if rejoin else None,
            shoulder_or_departure_samples=len(illegal), first_outside_legal_union=next(iter(illegal), None),
            first_native_legality_failure=first_unknown, alongside=alongside, ahead_with_sampled_clearance=ahead,
            moving_vehicle_clearance=moving_separation(rows, physical, lo, hi),
        ))
    return moves


def moving_separation(rows, physical, begin=0, end=None):
    participants, closest_stopped, nearest_by_phase, near, frame_set = {}, {}, {}, [], set()
    for i in range(begin, len(rows) if end is None else end):
        for pair in physical[i]["crossing_pairs"]:
            detail = {**stamp(rows[i]), **pair, "reported_maneuver_phase": rows[i].get("tactical_pass", {}).get("maneuver_phase")}
            key = str(pair["actor_id"])
            if not detail["moving"]:
                prior = closest_stopped.get(key)
                if prior is None or detail["physical_gap_m"] < prior["physical_gap_m"]:
                    closest_stopped[key] = detail
                continue
            frame_set.add(i)
            phase = detail["reported_maneuver_phase"] or "phase_unavailable"
            phase_min = nearest_by_phase.get(phase)
            if phase_min is None or detail["physical_gap_m"] < phase_min["physical_gap_m"]:
                nearest_by_phase[phase] = detail
            data = participants.setdefault(key, dict(samples=0, minimum_gap=detail, shortest_headway=None,
                                                    shortest_closing_ttc=None, lowest_longitudinal_gap=detail))
            data["samples"] += 1
            if detail["physical_gap_m"] < data["minimum_gap"]["physical_gap_m"]:
                data["minimum_gap"] = detail
            for name, field in (("shortest_headway", "headway_s"), ("shortest_closing_ttc", "closing_ttc_s")):
                prior = data[name]
                if detail[field] is not None and (prior is None or detail[field] < prior[field]):
                    data[name] = detail
            if detail["signed_longitudinal_headway_m"] < data["lowest_longitudinal_gap"]["signed_longitudinal_headway_m"]:
                data["lowest_longitudinal_gap"] = detail
            if detail["below_required_separation"]:
                near.append(detail)
    return dict(
        minimum_required_straddle_separation_m=1.,
        scored_frames_with_moving_counterpart=len(frame_set),
        moving_participants=participants, closest_stopped_vehicles=closest_stopped,
        minimum_moving_gap_by_phase=nearest_by_phase,
        sub_1m_samples=len(near), overlap_samples=sum(detail["physically_touching"] for detail in near),
        below_1m=near, sampled_pass=not near if frame_set else None,
    )


def tactical_phases(rows):
    """Keep brief invitations and withdrawals beside the physical pose score."""
    phase_at = [r.get("tactical_pass", {}).get("maneuver_phase") for r in rows]
    counts = Counter(p for p in phase_at if p is not None)
    periods = []
    for phase in counts:
        if phase == "keep":
            continue
        for lo, hi in spans(np.asarray([p == phase for p in phase_at])):
            first = rows[lo]["tactical_pass"]
            periods.append(dict(
                phase=phase, first=stamp(rows[lo]), last=stamp(rows[hi-1]), seconds=rnd((hi-lo)*DT),
                road_source_lane=first.get("source_lane"), road_target_lane=first.get("target_lane"),
                lead_kind=first.get("lead_kind"), source_gap_m=first.get("source_gap_m"),
                target_front_m=first.get("target_front_m"), target_rear_m=first.get("target_rear_m"),
                target_gap_accepted=first.get("target_gap_accepted"),
                return_gap_accepted=first.get("return_gap_accepted"),
            ))
    return dict(frame_counts=dict(counts), non_keep_periods=sorted(periods, key=lambda p: p["first"]["frame"]))


def inspect_range(rows, geometry, ranges, inventory, wanted):
    """Read sealed sweeps only at registered exposures around a possible pass."""
    hashes = {entry["frame"]: entry["sha256"] for entry in read(inventory)["files"]}
    chosen = {r["frame"]: r for r in rows if r["frame"] in wanted}
    if set(chosen) != set(wanted):
        raise ValueError("Requested range exposures are missing")
    samples = {}
    for frame, row in chosen.items():
        path = ranges / f"{frame}.npz"
        digest = sha256(path)
        if digest != hashes.get(frame):
            raise ValueError(f"Unmatched raw range exposure at {frame}")
        pose = row["ego_pose"]
        station = row["progress_m"]
        idx = geometry.sample_index(station)
        basis = np.asarray(rotation(*(pose[k] for k in ("pitch", "yaw", "roll"))))
        mount = np.array([pose[k] for k in ("x", "y", "z")]) + 2.3 * basis[:, 2]
        with np.load(path, allow_pickle=False) as sweep:
            np.testing.assert_allclose(sweep["ego_xy_yaw"], [pose[k] for k in ("x", "y", "yaw")], atol=.015, rtol=0.)
            points = sweep["lidar"][:, :3] @ basis.T + mount
        arc, lateral = project(points[:, :2], geometry.corridor(idx))
        longitudinal = arc - station
        height = points[:, 2] - np.interp(arc, geometry.station, geometry.xyz[:, 2])
        labeled = {}
        for side in ("left", "right"):
            if not geometry.legal[side][idx]:
                continue
            wp = geometry.sample[idx]
            other = wp[side]
            width = math.dist(other["xyz"][:2], wp["xyz"][:2])
            lane = abs(lateral + width if side == "left" else lateral - width) < 1.25
            roadway = lane & (abs(height) < .4)
            obstruction = lane & (height > .44) & (height < 3.7)
            labeled[side] = {}
            for direction, sign in (("forward", 1.), ("rear", -1.)):
                distance = sign * longitudinal
                road = distance[roadway & (distance > 4.) & (distance < 60.)]
                hits = distance[obstruction & (distance > 3.) & (distance < 58.)]
                labeled[side][direction] = dict(
                    pavement_return_count_4_to_24m=int(np.sum(road < 24.)),
                    pavement_observed_meter_bins=sorted(map(int, np.unique(np.floor(road)))),
                    near_above_road_surface_m=rnd(np.min(hits)) if len(hits) else None,
                )
        evidence = {key: row.get("progress_evidence", {}).get(key)
                    for key in ("binding", "raw_lidar_gap_m", "raw_radar_gap_m", "sight_m", "corridor_m")}
        samples[str(frame)] = dict(**stamp(row), sha256=digest, existing_policy_evidence=evidence,
                                   adjacent_lane_lidar=labeled)
    return samples


def audit(folder, reference, route_path, ranges=None, inventory=None, range_frames=(), position_journal=None,
          focused=False):
    manifest, rows, source = load_capture(folder, reference, position_journal, focused)
    summary = manifest["summary"]
    route = read(route_path)
    if route["route_id"] != summary["route_id"] or abs(route["route_length_m"] - summary["route_length_m"]) > .03:
        raise ValueError("OpenDRIVE route differs from the capture")
    geometry = Geometry(route)
    time = np.asarray([r["sim_time"] for r in rows])
    frames = np.asarray([r["frame"] for r in rows])
    if (not len(rows) or len(rows) != summary["frames"] or not np.allclose(np.diff(time), DT, atol=.0001)
            or not np.all(np.diff(frames) == 1)
            or int(frames[0]) != manifest.get("first_frame", frames[0])
            or int(frames[-1]) != manifest.get("last_frame", frames[-1])):
        raise ValueError("Inconsistent 10 Hz exposure clock")
    if any(r.get("camera_frame", r["frame"]) != r["frame"] or r.get("video_index", i) != i
           for i, r in enumerate(rows)):
        raise ValueError("Camera exposures and telemetry clock disagree")
    station = np.asarray([r["progress_m"] for r in rows])
    posted = np.asarray([r["posted_limit_kmh"] for r in rows])
    speed = np.asarray([r["speed_mps"] for r in rows]) * 3.6
    target = np.asarray([r["speed_request_mps"] for r in rows]) * 3.6
    yaw = np.unwrap(np.deg2rad([r["ego_pose"]["yaw"] for r in rows]))
    accel, lateral = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    if len(rows) > 5:
        accel[5:] = (speed[5:] - speed[:-5]) / (5 * DT * 3.6)
        means = np.convolve(speed / 3.6, [.1, .2, .2, .2, .2, .1], mode="valid")
        lateral[5:] = means * np.abs(yaw[5:] - yaw[:-5]) / (5 * DT)
    with gzip.open(reference, "rt", encoding="utf-8") as stream:
        physical, clearance_records, unresolved, density = measure_actors(
            rows, (json.loads(line) for line in stream), geometry, summary["ego_blueprint"])

    def stats(mask):
        samples = np.flatnonzero(mask)
        if not len(samples):
            return dict(frames=0)
        motion_windows = mask & np.isfinite(accel)
        steady = spans(mask & (speed >= 35.))
        cruise = [int(a) for a, b in steady if (b-a)*DT >= 1.]
        buckets, examples = Counter(), {}
        for i in samples:
            for bucket, ids in physical[i]["classifications"].items():
                buckets[bucket] += 1
                if bucket not in examples:
                    examples[bucket] = dict(**stamp(rows[i]), actor_ids=ids)
        duration = lambda eligible: rnd(np.sum(mask & eligible) * DT)
        population = {}
        for radius in RADII:
            local = [physical[i]["density"][str(radius)] for i in samples if physical[i]["density"]]
            population[str(radius)] = dict(
                scored=len(local), moving_car_frames=sum(row["cars"] > 0 for row in local),
                moving_pedestrian_frames=sum(row["pedestrians"] > 0 for row in local),
                both_frames=sum(min(row["cars"], row["pedestrians"]) > 0 for row in local),
                two_of_each_frames=sum(min(row["cars"], row["pedestrians"]) > 1 for row in local),
                background_both_frames=sum(min(row["background_cars"], row["background_pedestrians"]) > 0
                                           for row in local),
            )
        high_pace = {}
        for level in (45, 50, 55, 60):
            qualified = spans(mask & (speed >= level))
            sustained = next((i for i, j in qualified if j-i >= 10), None)
            high_pace[str(level)] = dict(
                actual_seconds=duration(speed >= level), requested_seconds=duration(target >= level),
                first_sustained=stamp(rows[sustained]) if sustained is not None else None,
                time_from_section_entry_to_sustained_s=rnd(time[sustained]-time[samples[0]])
                if sustained is not None else None,
            )
        report = dict(
            frames=int(len(samples)), seconds=duration(mask), entry=stamp(rows[samples[0]]), exit=stamp(rows[samples[-1]]),
            kmh_p10_p50_p95_max=[rnd(a) for a in np.r_[np.quantile(speed[mask], [.1, .5, .95]), speed[mask].max()]],
            request_kmh_p10_p50_p95_max=[rnd(a) for a in np.r_[np.quantile(target[mask], [.1, .5, .95]), target[mask].max()]],
            seconds_under_10=duration(speed < 10.), seconds_under_15=duration(speed < 15.),
            seconds_at_or_above_35=duration(speed >= 35.), seconds_at_or_above_37=duration(speed >= 37.),
            longest_continuous_35_kmh_s=rnd(max((b - a for a, b in steady), default=0) * DT),
            sustained_35_kmh_start=stamp(rows[cruise[0]]) if cruise else None,
            seconds_from_section_entry_to_sustained_35=rnd(time[cruise[0]] - time[samples[0]]) if cruise else None,
            overspeed_by_more_than_1_kmh_seconds=duration(speed > posted + 1.),
            over_40_kmh_seconds=duration(speed > 40.),
            actual_over_posted_seconds=duration(speed > posted),
            actual_over_posted_percent=rnd(100. * np.sum(mask & (speed > posted)) / len(samples)),
            maximum_overspeed_kmh=rnd(max(0., (speed[mask] - posted[mask]).max())),
            maximum_request_above_posted_kmh=rnd(max(0., (target[mask] - posted[mask]).max())),
            higher_thresholds=high_pace,
            binding_counts=dict(Counter(rows[i].get("progress_evidence", {}).get("binding") for i in samples
                                        if rows[i].get("progress_evidence", {}).get("binding"))),
            observed_path_actor_seconds={bucket: rnd(count * DT) for bucket, count in buckets.items()},
            observed_path_actor_first_example=examples, moving_population_exposure=population,
        )
        faster = spans(mask & (speed > 40.))
        if len(faster):
            a, b = max(faster, key=lambda limits: limits[1] - limits[0])
            after = rows[b] if b < len(rows) else None
            end_actor_groups = dict(physical[b]["classifications"]) if after else {}
            evidence = after.get("progress_evidence", {}) if after else {}
            report["longest_continuous_above_40"] = dict(
                first=stamp(rows[a]), last=stamp(rows[b-1]), seconds=rnd((b-a)*DT),
                next_exposure=stamp(after) if after else None,
                next_binding=evidence.get("binding"),
                next_maneuver_binding=evidence.get("maneuver_binding"),
                next_speed_regime=evidence.get("speed_regime"),
                next_phase=after["phase"] if after else None,
                next_observed_actor_groups=end_actor_groups,
            )
        regimes = Counter(rows[i].get("progress_evidence", {}).get("speed_regime") for i in samples)
        regimes.pop(None, None)
        if regimes:
            report["speed_regime_frame_counts"] = dict(regimes)
        if motion_windows.any():
            brake_spot = int(np.nanargmin(np.where(motion_windows, accel, np.nan)))
            lateral_spot = int(np.nanargmax(np.where(motion_windows, lateral, np.nan)))
            report.update(
                acceleration_peak_mps2=rnd(accel[motion_windows].max()),
                braking_peak_mps2=rnd(-accel[motion_windows].min()),
                braking_peak_location=stamp(rows[brake_spot]),
                braking_harder_than_3_seconds=duration(accel < -3.),
                lateral_p95_mps2=rnd(np.quantile(lateral[motion_windows], .95)),
                lateral_peak_mps2=rnd(lateral[motion_windows].max()),
                lateral_peak_location=stamp(rows[lateral_spot]),
            )
        return report

    groups = defaultdict(list)
    for i, sample in enumerate(physical):
        for entry in sample["opportunities"]:
            key = (entry["role"], entry["target_side"], entry["result"])
            groups[key].append((i, entry))
    checks = []
    for (role, side, outcome), occurrences in groups.items():
        hits = np.zeros(len(rows), dtype=bool)
        hits[[i for i, _ in occurrences]] = True
        runs = spans(hits)
        considered = [(a, b) for a, b in runs if (b-a)*DT >= 1.]
        checks.append(dict(role=role, side=side, outcome=outcome, frames=len(occurrences),
                           periods=[dict(first=stamp(rows[int(a)]), last=stamp(rows[int(b)-1]),
                                         seconds=rnd((b-a)*DT)) for a, b in considered],
                           first={**stamp(rows[occurrences[0][0]]), **occurrences[0][1]},
                           mid={**stamp(rows[occurrences[len(occurrences)//2][0]]), **occurrences[len(occurrences)//2][1]},
                           last={**stamp(rows[occurrences[-1][0]]), **occurrences[-1][1]}))

    sections = {}
    for name, lo, hi in SECTIONS:
        mask = (station >= lo) & (station < hi)
        if hi <= 570.:
            mask &= posted == 90.
        sections[name] = dict(bounds_m=[lo, hi], geometry=geometry.describe(lo, hi), **stats(mask))
    keys = ("frames", "duration_s", "progress_m", "route_length_m", "outcome", "collision_events",
            "yield_violations", "red_light_violations", "turns_completed")
    grade = {k: summary.get(k) for k in keys}
    grade["events"] = summary["events"]
    ambient = summary.get("background_traffic_activity", {})
    grade["participant_contacts"] = ambient.get("participant_contacts")
    grade["ambient_vehicles_spawned"] = ambient.get("spawned")
    grade["ambient_vehicles_moving_near_ego"] = ambient.get("moving_near_ego")
    result = dict(
        schema="ado-city-maneuver-audit-v1",
        source=dict(**source, route_id=route["route_id"], map_sha256=route["opendrive_sha256"],
                    route_sha256=sha256(route_path), producer=manifest.get("producer"),
                    quality=manifest.get("quality_requested"), policy_sha256=manifest.get("policy", {}).get("sha256"),
                    first_frame=rows[0]["frame"], last_frame=rows[-1]["frame"],
                    seed=summary["seed"], crossing_delay_s=summary.get("crossing_delay_s")),
        grade=grade, whole=stats(np.ones(len(rows), dtype=bool)), posted_90=stats(posted == 90.),
        by_posted_kmh={str(int(value)): stats(posted == value) for value in np.unique(posted)},
        sections=sections, independent_clearance=clearance_records, unresolved_close_actor_samples=unresolved,
        moving_population_exposure=density, passing_screen=checks, actual_lateral_departures=departures(rows, physical),
        straddling_moving_separation=moving_separation(rows, physical),
    )
    phases = tactical_phases(rows)
    if phases["frame_counts"]:
        result["tactical_decisions"] = phases
    if ranges is not None and inventory is not None:
        result["registered_range_exposures"] = inspect_range(rows, geometry, ranges, inventory, range_frames)
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("drive", type=Path)
    cli.add_argument("reference", type=Path)
    cli.add_argument("route", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--ranges", type=Path)
    cli.add_argument("--range-inventory", type=Path)
    cli.add_argument("--range-frames", type=int, nargs="+", default=[])
    cli.add_argument("--position-journal", type=Path, help="Aligned rounded native poses for a rehearsal trace")
    cli.add_argument("--focused", action="store_true", help="Use the exact ego polygon recorded by a focused trial")
    args = cli.parse_args()
    if (args.ranges is None) != (args.range_inventory is None) or (args.ranges is None and args.range_frames):
        cli.error("Supply the range directory and sealed inventory together")
    result = audit(args.drive, args.reference, args.route, args.ranges, args.range_inventory,
                   args.range_frames, args.position_journal, args.focused)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
