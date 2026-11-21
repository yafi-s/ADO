"""Causal tracks from calibrated RGB views, roof lidar and front radar.

Positions describe estimated object centres in metres. A fresh observation needs
independent depth support. The legacy rig coasts for half a second. Surround mode
holds established tracks for at most 0.8 s without new measurements, and one-off
observations for at most 0.3 s. Only observed travel is extrapolated.

Heading starts parallel to the ego vehicle, or across it for a broadside view, and
afterwards follows only sustained travel. A single view shows one or two faces of
an object, so its apparent centre shifts as the viewpoint changes; that shift is
not motion and must not turn the object. Dimensions settle per track for the same reason.
"""

from dataclasses import dataclass
import math

import numpy as np

from .robotaxi import MOUNTS
from .camera_geometry import CAMERAS
from .radar_geometry import ego_returns


# Length, width, height in metres. Single-view returns sample visible surfaces.
SIZE = {"car": (4.4, 1.85, 1.52), "truck": (7.0, 2.5, 3.2),
        "barrier": (1.19, .40, 1.09), "bus": (10.5, 2.55, 3.2), "person": (.52, .52, 1.70),
        "bicycle": (1.7, .60, 1.45), "motorcycle": (2.1, .78, 1.45)}
COAST_S = .50
SIZE_SETTLE_S = .6      # a track's dimensions approach each new view over this long
COURSE_WINDOW_S = 1.    # travel is judged over this much recent history
COURSE_MIN_S = .45
COURSE_MIN_M = {"person": .6}   # net travel needed before it sets the heading; 1 m otherwise
COURSE_STRAIGHTNESS = .8        # net travel over path length; jitter scores low
REPEAT_IOU = .5         # two boxes of one family overlapping this much are one object


def _family(label):
    """Classes the detector confuses with one another. A van reads as car, bus or bicycle."""
    return label if label in ("person", "barrier") else "wheeled"


def _distinct(detections):
    """Keep the surer box where another class of the same family repeats it.

    The detector suppresses repeats within a class only. Left alone, two boxes on one
    object share every lidar return, and shared returns support neither.
    """
    kept = []
    for detection in sorted(detections, key=lambda d: -d["score"]):
        left, top, right, bottom = detection["box"]
        for other in kept:
            if other["label"] == detection["label"] or _family(other["label"]) != _family(detection["label"]):
                continue
            a, b, c, d = other["box"]
            overlap = max(0., min(right, c) - max(left, a)) * max(0., min(bottom, d) - max(top, b))
            union = (right - left) * (bottom - top) + (c - a) * (d - b) - overlap
            if union > 0 and overlap / union >= REPEAT_IOU:
                break
        else:
            kept.append(detection)
    return kept


def _project(lidar, camera):
    points = np.asarray(lidar).reshape(-1, 4)[:, :3]
    ids = np.flatnonzero(np.isfinite(points).all(axis=1))
    points = points[ids] + MOUNTS["lidar"][:3]
    dx, dy = points[:, 0] - camera.x, points[:, 1] - camera.y
    yaw = math.radians(camera.yaw)
    depth = dx * math.cos(yaw) + dy * math.sin(yaw)
    lateral = dy * math.cos(yaw) - dx * math.sin(yaw)
    positive = (depth > .15) & (points[:, 2] > .22)
    points, ids, depth = points[positive], ids[positive], depth[positive]
    u = .5 + camera.fx * lateral[positive] / depth
    v = .5 - camera.fy * (points[:, 2] - camera.z) / depth
    visible = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    return points[visible], ids[visible], depth[visible], u[visible], v[visible]


def _object_surface(depths, indices, label, box, camera):
    if len(indices) < 3:
        return None
    ordered = indices[np.argsort(depths[indices])]
    clusters = np.split(ordered, np.flatnonzero(np.diff(depths[ordered]) > .85) + 1)
    groups = []
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        depth = float(np.median(depths[cluster]))
        # A background surface behind a road user can land in the same pixels.
        # Reject depths incompatible with the box's apparent height or width.
        apparent_h = (box[3] - box[1]) * depth / camera.fy
        apparent_w = (box[2] - box[0]) * depth / camera.fx
        length, width, height = SIZE[label]
        if apparent_h > 1.7 * height or apparent_w > 1.65 * max(width, length):
            continue
        if box[1] > .01 and box[3] < .99 and apparent_h < .35 * height:
            continue
        groups.append(cluster)
    if not groups:
        return None
    # Prefer the first supported surface, suppress competing reliable depths.
    first = groups[0]
    if any(len(other) >= max(3, .55 * len(first)) for other in groups[1:]):
        return None
    return first


def _measure(lidar, detections, view="front", *, cycle_context=None):
    if not detections:
        return []
    camera = CAMERAS[view]
    points, ids, depths, u, v = _project(lidar, camera)
    selections = []
    owners = np.zeros(len(points), dtype=int)
    for detection in _distinct([d for d in detections if d["label"] in SIZE]):
        box = np.clip(detection["box"], 0., 1.)
        left, top, right, bottom = box
        mask = (u >= left) & (u <= right) & (v >= top) & (v <= bottom)
        owners[mask] += 1
        selections.append((detection, box, mask))
    result = []
    for detection, box, mask in selections:
        label = detection["label"]
        if detection.get("barrier_subtype") == "chain":
            groups = [np.flatnonzero(np.isin(ids, group) & (owners == 1)) for group in detection["chain_return_ids"]]
            if len(groups) != 3 or any(len(g) < n for g, n in zip(groups, (2, 1, 2))):
                continue
            supported = np.concatenate(groups)
        else:
            supported = _object_surface(depths, np.flatnonzero(mask & (owners == 1)), label, box, camera)
        if supported is None:
            continue
        surface = points[supported]
        if cycle_context and label == "bicycle" and surface[:, 2].max() < .60:
            # On sloping streets a single lidar ring can draw a horizontal strip
            # of asphalt through a bicycle box. Require another elevation beam
            # before that low strip alone can establish a standalone bicycle.
            rays = surface - MOUNTS["lidar"][:3]
            elevation = np.arctan2(rays[:, 2], np.hypot(rays[:, 0], rays[:, 1]))
            if np.ptp(elevation) < math.radians(.6):
                continue
        item = _fit_surface(surface, label, detection["score"], set(ids[supported].tolist()), [view],
                            box=detection["box"])
        if "barrier_subtype" in detection:
            item["barrier_subtype"] = detection["barrier_subtype"]
            item["subtype_confidence"] = detection["subtype_confidence"]
            if detection["barrier_subtype"] == "chain":
                ends = np.asarray(detection["chain_ends_xy"])
                axis = ends[1]-ends[0]
                center = ends.mean(axis=0)
                item.update(chain_center=center, center=np.r_[center,.35], length=np.linalg.norm(axis)+.20,
                            width=.24, height=.70, angle=math.degrees(math.atan2(axis[1], axis[0])),
                            chain_support=dict(posts=[len(groups[0]),len(groups[2])],span_hits=len(groups[1])))
        if "image_id" in detection:
            item["image_ids"][view] = detection["image_id"]
        result.append(item)
    if cycle_context and any(d["label"] == "bicycle" for d, _, _ in selections):
        from .cycle_association import pairs, pair_surface
        cycles, pose, now = cycle_context
        masks = [(d, mask) for d, _, mask in selections]
        for person, bike in pairs([d for d, _ in masks]):
            joint = pair_surface(points, depths, u, v, masks, person, bike, camera)
            if joint is None:
                continue
            info = cycles.observe(view, person, bike, joint, points, depths, camera, pose, now)
            if info is None:
                continue
            group, _, _ = joint
            # Separate well-supported depths could be a walker in front of a parked
            # bike. Do not replace either object with an assumed joint surface.
            members = [m for m in result if m["boxes"][view] in (person["box"], bike["box"])]
            if any(np.linalg.norm(m["center"][:2]-joint[2]) > 1.6 for m in members):
                continue
            result = [m for m in result if all(m is not member for member in members)]
            item = _fit_surface(points[group], "bicycle", min(person["score"], bike["score"]), set(ids[group].tolist()),
                                [view], box=person["box"])
            # Keep the rider's association family when the wheels disappear in a
            # later view. Radar and the reported footprint use the mounted body.
            item["label"] = "person"
            info["estimated_size_m"] = [item[k] for k in ("length", "width", "height")]
            item["image_ids"][view] = person["image_id"]
            item["cycle_association"] = info
            result.append(item)
    return result


def _fit_surface(surface, label, score, hit_ids, cameras, *, box=None):
    # Fit along the sightline so an ego yaw change cannot rotate a hidden
    # thickness correction in world space while the object stays still.
    mean_xy = np.median(surface[:, :2], axis=0)
    bearing = math.atan2(mean_xy[1], mean_xy[0])
    sin, cos = math.sin(bearing), math.cos(bearing)
    axes = np.array([[cos, -sin], [sin, cos]])
    aligned = np.column_stack((surface[:, :2] @ axes, surface[:, 2]))
    lo, hi = np.quantile(aligned, [.05, .95], axis=0)
    dx, dy, dz = hi - lo
    length, width, height = SIZE.get(label, (max(.35, dx, dy), max(.28, min(dx, dy)), max(.35, hi[2])))
    sideways = dy > width * 1.45 and dy > dx
    along, across = (dy, dx) if sideways else (dx, dy)
    length = max(min(along, length * 1.55), length * (.78 if along > .55 * length else 1))
    width = max(min(across, width * 1.45), width * .85)
    height = max(min(hi[2], height * 1.35), height * .85, min(dz, height * 1.35))
    extent_x = width if sideways else length
    # Missing depth extends behind the visible surface.
    center = (lo + hi) / 2
    center[0] += max(0., extent_x - dx) / 2
    center[:2] = axes @ center[:2]
    center[2] = max(height / 2, hi[2] - height / 2)
    return dict(label=label, center=center, length=length, width=width, height=height,
                angle=math.degrees(bearing) + (90. if sideways else 0.),
                confidence=float(score), radial_mps=None, cameras=cameras,
                hit_ids=hit_ids, boxes={cameras[0]: box} if cameras else {}, image_ids={},
                bearing=bearing, near=float(lo[0]), lateral=float((lo[1] + hi[1]) / 2),
                seen=float(dx), top=float(hi[2]), sideways=bool(sideways))


def _centre(item, length, width):
    """Ego-frame centre of an object whose near surface was seen, given its footprint."""
    if "chain_center" in item:
        return item["chain_center"]
    extent = width if item["sideways"] else length
    along = item["near"] + max(item["seen"], extent) / 2
    sin, cos = math.sin(item["bearing"]), math.cos(item["bearing"])
    return np.array([cos * along - sin * item["lateral"], sin * along + cos * item["lateral"]])


def _across_views(lidar, by_view, cycle_context=None):
    """Deduplicate overlap only when two independent camera boxes own the same laser surface."""
    measured = [item for view, detections in by_view.items() for item in _measure(lidar, detections, view, cycle_context=cycle_context)]
    kept = []
    for item in sorted(measured, key=lambda m: ("cycle_association" not in m, -len(m["hit_ids"]), -m["confidence"])):
        for other in kept:
            same_kind = (_family(other["label"]) == _family(item["label"]) or
                         ("cycle_association" in other and item["label"] == "bicycle"))
            if (not same_kind or
                    not set(other["cameras"]).isdisjoint(item["cameras"])):
                continue
            common = len(other["hit_ids"] & item["hit_ids"])
            if common >= max(3, .5 * min(len(other["hit_ids"]), len(item["hit_ids"]))):
                other["cameras"] += item["cameras"]
                other["hit_ids"].update(item["hit_ids"])
                other["boxes"].update(item["boxes"])
                other["image_ids"].update(item["image_ids"])
                break
        else:
            kept.append(item)
    for item in kept:
        item["weak_edge"] = (len(item["boxes"]) == 1 and
                             weak_edge_box(next(iter(item["boxes"].values())), item["confidence"]))
    return kept


def weak_edge_box(box, score):
    """A thin low-confidence sliver at the horizontal image boundary cannot establish class."""
    return score < .55 and (box[0] <= .005 or box[2] >= .995) and box[2] - box[0] < .075


def _course(trail, label):
    """World heading in degrees from sustained travel, or None while travel is unproven."""
    (start_time, start), (end_time, end) = trail[0], trail[-1]
    if end_time - start_time < COURSE_MIN_S:
        return None
    net = end - start
    distance = float(np.linalg.norm(net))
    path = sum(float(np.linalg.norm(b - a)) for (_, a), (_, b) in zip(trail, trail[1:]))
    if distance < COURSE_MIN_M.get(label, 1.) or distance < COURSE_STRAIGHTNESS * path:
        return None
    return math.degrees(math.atan2(net[1], net[0]))


def _match_radar(measurements, radar, name="radar"):
    if not len(radar) or not measurements:
        return
    samples = ego_returns(radar, name)
    xyz, vel = samples[:, :3], samples[:, 3]
    valid = xyz[:, 2] > .18
    masks = []
    for item in measurements:
        limits = np.array([item["length"], item["width"], item["height"]])
        angle = math.radians(item["angle"])
        c, s = math.cos(angle), math.sin(angle)
        delta = xyz - item["center"]
        delta[:, :2] = delta[:, :2] @ np.array([[c, -s], [s, c]])
        masks.append(valid & (np.abs(delta) <= limits / 2 + [.35, .25, .15]).all(axis=1))
    unique = np.sum(masks, axis=0) == 1
    for item, mask in zip(measurements, masks):
        speeds = vel[mask & unique]
        if len(speeds):
            sources = item.setdefault("doppler", {})
            sources[name] = dict(hits=len(speeds), radial_mps=float(np.median(speeds)))
            # The line of sight depends on the mount. Keep one supported value for
            # the legacy scalar, and retain each sensor's own measurement separately.
            item["radial_mps"] = max(sources.values(), key=lambda s: s["hits"])["radial_mps"]


@dataclass
class _Track:
    position: np.ndarray
    velocity: np.ndarray
    last_seen: float
    measurement: dict
    heading_world: float
    size: np.ndarray        # settled length, width, height
    trail: list             # recent (time, world position) observations
    votes: dict             # confidence each class has gathered; the leader names the track
    observations: int = 1
    travelling: bool = False    # sustained travel was seen at the last observation
    last_classified: float | None = None
    last_cameras: tuple = ()
    last_confirmed: float = 0.
    image_keys: frozenset = frozenset()
    hidden: bool = False
    cycle_evidence: tuple | None = None
    barrier_read: tuple | None = None

    @property
    def label(self):
        return max(self.votes, key=self.votes.get)


class Tracker:
    def __init__(self, *, surround=False):
        self._tracks = {}
        self._next_id = 1
        self.surround = surround
        if surround:
            from .cycle_association import CycleAssociator
            from .lidar_motion import LidarMotion
            self.cycles = CycleAssociator()
            self.motion = LidarMotion()

    def _range_measurements(self, clusters, optical, origin, rotation, now):
        """Follow uniquely associated isolated surfaces; only proven movers can start unlabelled tracks."""
        occupied = {idx for item in optical for idx in item["hit_ids"]}
        result = []
        for cluster in clusters:
            ids = set(cluster.ids.tolist())
            if occupied.intersection(ids):
                continue
            matches = []
            for key, track in self._tracks.items():
                # A single post cannot refresh the depth or pose of a whole suspended chain.
                if track.barrier_read and track.barrier_read[0] == "chain":
                    continue
                if track.hidden or now - track.last_seen > self._lifetime(track) or now - track.last_confirmed > 1.2:
                    continue
                dt = now - track.last_seen
                prediction = track.position + (dt * track.velocity if track.travelling else 0.)
                # The measured centroid is usually on the object's visible face.
                if np.linalg.norm(cluster.center - prediction) > .9 + track.size[0] * .65:
                    continue
                if cluster.scale > np.hypot(*track.size[:2]) * 1.35 or cluster.surface[:, 2].max() > track.size[2]*1.55:
                    continue
                fitted = _fit_surface(cluster.surface, track.label, .65, ids, [])
                place = origin + rotation @ _centre(fitted, *track.size[:2])
                limit = .45 if track.label == "person" else .85
                if np.linalg.norm(place - prediction) <= limit + dt * 2.5:
                    fitted.update(required_track=key, moving=cluster.moving)
                    matches.append(fitted)
            if len(matches) == 1:
                result.append(matches[0])
            elif not matches and cluster.moving:
                item = _fit_surface(cluster.surface, "unknown", .65, ids, [])
                item["moving"] = True
                result.append(item)
        # An ambiguous split into two surfaces cannot maintain the same ID.
        return [m for m in result if "required_track" not in m or
                sum(n.get("required_track") == m["required_track"] for n in result) == 1]

    def _lifetime(self, track):
        if not self.surround:
            return COAST_S
        return .8 if track.observations >= 3 else .3

    def update(self, lidar, radar, detections, ego_pose, sim_time, *, corner_radars=None):
        """Update once per chronological frame. Pass empty detections on skipped RGB inference."""
        now = float(sim_time)
        origin = np.asarray(ego_pose[:2], dtype=float)
        yaw = float(ego_pose[2])
        cos, sin = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        rotation = np.array([[cos, -sin], [sin, cos]])
        measurements = (_across_views(lidar, detections, (self.cycles, ego_pose, now)) if self.surround
                        else _measure(lidar, detections))
        self._tracks = {k: t for k, t in self._tracks.items()
                        if 0 <= now - t.last_seen <= (2. if self.surround and t.image_keys else self._lifetime(t)) + 1e-8}
        if self.surround:
            clusters = self.motion.read(lidar, origin, rotation, now)
            measurements.extend(self._range_measurements(clusters, measurements, origin, rotation, now))
            raw = np.asarray(lidar).reshape(-1, 4)
            for item in measurements:
                hits = sorted(item["hit_ids"])
                sample = raw[hits[::max(1, math.ceil(len(hits) / 24))]]
                item["world_surface"] = np.column_stack((sample[:, :2] @ rotation.T + origin, sample[:, 2]))
        _match_radar(measurements, radar)
        if corner_radars:
            for name, raw in corner_radars.items():
                _match_radar(measurements, raw, name)
        positions = [origin + rotation @ item["center"][:2] for item in measurements]
        costs = []
        for key, track in self._tracks.items():
            dt = now - track.last_seen
            dormant = track.hidden or dt > self._lifetime(track) + 1e-8
            prediction = track.position + dt * track.velocity
            if self.surround and track.observations >= 3 and not track.travelling:
                prediction = track.position
            for idx, (item, pos) in enumerate(zip(measurements, positions)):
                # A re-read of the class is not a new object.
                if track.barrier_read:
                    prior_subtype = track.barrier_read[0]
                    next_subtype = item.get("barrier_subtype")
                    if (next_subtype and next_subtype != prior_subtype) or (prior_subtype == "chain" and not next_subtype):
                        continue
                if item.get("required_track", key) != key:
                    continue
                if item.get("weak_edge") and (dormant or track.label != item["label"]):
                    continue
                if dormant and track.image_keys.isdisjoint(item["image_ids"].items()):
                    continue
                if "unknown" not in (item["label"], track.label) and _family(item["label"]) != _family(track.label):
                    continue
                distance = float(np.linalg.norm(pos - prediction))
                max_speed = 4. if item["label"] == "person" else 20.
                gate = .85 + dt * max_speed
                if self.surround and track.observations >= 3:
                    # Large blind gates can snap a recently occluded object to its neighbour.
                    # Keep a short allowance for a new face becoming visible while passing.
                    gate = min(gate, (.7 if item["label"] == "person" else 1.5) + 3. * dt)
                if dormant:
                    gate = min(gate, .6 + dt * (2.4 if item["label"] == "person" else 5.))
                if distance <= gate:
                    costs.append((not item["cameras"], distance, key, idx))
        used, updated = set(), set()
        for _, _, key, idx in sorted(costs):
            if idx in used or key in updated:
                continue
            track = self._tracks[key]
            item = measurements[idx]
            dt = now - track.last_seen
            # Each view measures different faces. Let the dimensions settle, then place the
            # centre with them, so a changing view does not read as the object moving.
            seen = np.array([item["length"], item["width"], item["height"]])
            track.size += min(1., dt / SIZE_SETTLE_S) * (seen - track.size)
            pos = origin + rotation @ _centre(item, track.size[0], track.size[1])
            if dt > .001:
                weight = 1. if track.observations == 1 else min(1., dt / .4)
                track.velocity += weight * ((pos - track.position) / dt - track.velocity)
                track.observations += 1
            track.trail = [entry for entry in track.trail if now - entry[0] <= COURSE_WINDOW_S] + [(now, pos)]
            if item["cameras"]:
                track.votes.pop("unknown", None)
                track.votes[item["label"]] = track.votes.get(item["label"], 0.) + item["confidence"]
                track.last_classified, track.last_cameras = now, tuple(item["cameras"])
                track.image_keys = frozenset(item["image_ids"].items())
            track.hidden = False
            if item["cameras"] or item.get("moving"):
                track.last_confirmed = now
            course = _course(track.trail, track.label)
            track.travelling = course is not None
            if course is not None:
                track.heading_world = course
            track.position, track.last_seen, track.measurement = pos, now, item
            used.add(idx)
            updated.add(key)
        for idx, (item, pos) in enumerate(zip(measurements, positions)):
            if idx not in used and "required_track" not in item:
                if item.get("weak_edge"):
                    continue
                size = np.array([item["length"], item["width"], item["height"]])
                # Road users mostly travel along or across the ego vehicle's own heading.
                facing = yaw + (90. if item["sideways"] else 0.)
                if self.surround:
                    # Broadside at a lateral camera generally means longitudinal on the road.
                    facing = yaw + (90. if abs(math.sin(math.radians(item["angle"]))) > .7071 else 0.)
                self._tracks[self._next_id] = _Track(pos, np.zeros(2), now, item, facing, size, [(now, pos)],
                                                     {item["label"]: item["confidence"]},
                                                     last_classified=now if item["cameras"] else None,
                                                     last_cameras=tuple(item["cameras"]), last_confirmed=now,
                                                     image_keys=frozenset(item["image_ids"].items()))
                self._next_id += 1
        if self.surround:
            for track in self._tracks.values():
                item = track.measurement
                if track.last_seen == now and item.get("barrier_subtype"):
                    track.barrier_read = (item["barrier_subtype"], item["subtype_confidence"], now)
                    if item["barrier_subtype"] == "chain":
                        track.heading_world = yaw + item["angle"]
            for key, track in list(self._tracks.items()):
                item = track.measurement
                info = item.get("cycle_association")
                if track.last_seen != now or not info:
                    continue
                track.cycle_evidence = now, info
                # A previously separate bike box can still be coasting after both
                # image identities were measured as one body. Retire only that
                # exact optical ID near the newly confirmed joint measurement.
                for other_key, other in list(self._tracks.items()):
                    if (other_key != key and other.label == "bicycle" and
                        (info["camera"],info["bicycle_image_id"]) in other.image_keys and
                        np.linalg.norm(other.position-track.position)<1.5):
                        del self._tracks[other_key]
        if self.surround:
            # A positive sweep through the predicted surface contradicts a coast.
            # Missing returns and foreground occlusion cannot erase an object.
            current = self.motion.history[-1]
            for key, track in list(self._tracks.items()):
                age = now - track.last_seen
                if age <= 0:
                    continue
                world = track.measurement["world_surface"].copy()
                if track.travelling:
                    world[:, :2] += age * track.velocity
                if current.was_clear(world, fraction=.7):
                    if track.image_keys:
                        track.hidden = True
                    else:
                        del self._tracks[key]
        result = []
        for key, track in self._tracks.items():
            age = now - track.last_seen
            if track.hidden or age > self._lifetime(track) + 1e-8:
                continue
            item = track.measurement
            # Only an object seen travelling is carried forward through a coast. The small
            # velocity a stopped object picks up from a changing view would slide it away.
            carried = age * track.velocity if track.travelling else 0.
            x, y = rotation.T @ (track.position + carried - origin)
            radial = item["radial_mps"] if not age else None
            sources = (["rgb", "lidar"] if item["cameras"] else ["lidar"]) if not age else []
            if radial is not None:
                sources.append("radar")
            length, width, height = (float(value) for value in track.size)
            result.append(dict(id=key, label=track.label, x=float(x), y=float(y),
                               z=max(height / 2, item["top"] - height / 2), length=length,
                               width=width, height=height,
                               heading=(track.heading_world - yaw + 180) % 360 - 180,
                               speed_mps=float(np.linalg.norm(track.velocity)) if track.observations > 1 else None,
                               confidence=item["confidence"] * max(0., 1. - age / self._lifetime(track)),
                               radial_mps=radial, age_s=age, coasting=bool(age), sources=sources,
                               estimated_fields=["length", "width", "height", "heading"]))
            if self.surround:
                hits = sorted(item["hit_ids"]) if not age else []
                # Publish a bounded example of the real supporting surface in ego coordinates.
                sample = hits[::max(1, math.ceil(len(hits) / 12))]
                xyz = np.asarray(lidar).reshape(-1, 4)[sample, :3] + MOUNTS["lidar"][:3]
                result[-1].update(cameras=[] if age else item["cameras"], last_cameras=list(track.last_cameras),
                                  rgb_boxes={} if age else item["boxes"],
                                  radar_sensors=[] if age else list(item.get("doppler", {})),
                                  doppler={} if age else item.get("doppler", {}),
                                  class_age_s=None if track.last_classified is None else now-track.last_classified,
                                  lidar_hits=len(hits), surface_xyz=xyz.astype(float).round(3).tolist())
                if track.label == "barrier" and track.barrier_read:
                    subtype, confidence, stamp = track.barrier_read
                    result[-1].update(barrier_subtype=subtype, subtype_confidence=confidence,
                                      subtype_age_s=round(now-stamp, 4))
                    if not age and "chain_support" in item:
                        result[-1]["chain_support"] = item["chain_support"]
                if track.cycle_evidence:
                    stamp, info = track.cycle_evidence
                    elapsed = now-stamp
                    # Fill the gap between scheduled optical exposures only.
                    if elapsed <= .21 and (not elapsed or info["camera"] not in detections):
                        cycle = dict(info, age_s=elapsed)
                        if elapsed:
                            cycle.update(surface_hits=0, bicycle_exclusive_hits=0, bicycle_surface_xyz=[])
                        length, width, height = map(float, info["estimated_size_m"])
                        result[-1].update(label="bicycle", cycle_association=cycle,
                                          length=length, width=width, height=height,
                                          z=max(height / 2, item["top"] - height / 2))
        return result
