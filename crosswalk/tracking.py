"""Causal viewer tracks from front RGB, roof lidar and front radar.

Positions describe estimated object centres in metres. Dimensions and heading are
estimates. A fresh detection needs independent depth support. A coast extrapolates
a previously measured world position for at most half a second.
"""

from dataclasses import dataclass
import math

import numpy as np

from .robotaxi import MOUNTS


# Length, width, height in metres. Single-view returns sample visible surfaces.
SIZE = {"car": (4.4, 1.85, 1.52), "truck": (7.0, 2.5, 3.2),
        "bus": (10.5, 2.55, 3.2), "person": (.52, .52, 1.70),
        "bicycle": (1.7, .60, 1.45), "motorcycle": (2.1, .78, 1.45)}
CAMERA_X, _, CAMERA_Z, _ = MOUNTS["front"]
VERTICAL_FOCAL = 320 / 360  # 90-degree horizontal FOV, 640 x 360, square pixels
COAST_S = .50


def _project(lidar):
    points = np.asarray(lidar).reshape(-1, 4)[:, :3]
    points = points[np.isfinite(points).all(axis=1)] + MOUNTS["lidar"][:3]
    points = points[(points[:, 0] > CAMERA_X + .15) & (points[:, 2] > .22)]
    depth = points[:, 0] - CAMERA_X
    u = .5 + .5 * points[:, 1] / depth
    v = .5 - VERTICAL_FOCAL * (points[:, 2] - CAMERA_Z) / depth
    visible = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    return points[visible], u[visible], v[visible]


def _object_surface(points, label, box):
    if len(points) < 3:
        return None
    ordered = points[np.argsort(points[:, 0])]
    clusters = np.split(ordered, np.flatnonzero(np.diff(ordered[:, 0]) > .85) + 1)
    groups = []
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        depth = float(np.median(cluster[:, 0])) - CAMERA_X
        # A background surface behind a road user can land in the same pixels.
        # Reject depths incompatible with the box's apparent height or width.
        apparent_h = (box[3] - box[1]) * depth / VERTICAL_FOCAL
        apparent_w = (box[2] - box[0]) * depth * 2.
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


def _measure(lidar, detections):
    points, u, v = _project(lidar)
    selections = []
    owners = np.zeros(len(points), dtype=int)
    for detection in detections:
        if detection["label"] not in SIZE:
            continue
        box = np.clip(detection["box"], 0., 1.)
        left, top, right, bottom = box
        mask = (u >= left) & (u <= right) & (v >= top) & (v <= bottom)
        owners[mask] += 1
        selections.append((detection, box, mask))
    result = []
    for detection, box, mask in selections:
        label = detection["label"]
        surface = _object_surface(points[mask & (owners == 1)], label, box)
        if surface is None:
            continue
        # Fit along the sightline so an ego yaw change cannot rotate a hidden
        # thickness correction in world space while the object stays still.
        mean_xy = np.median(surface[:, :2], axis=0)
        bearing = math.atan2(mean_xy[1], mean_xy[0])
        sin, cos = math.sin(bearing), math.cos(bearing)
        axes = np.array([[cos, -sin], [sin, cos]])
        aligned = np.column_stack((surface[:, :2] @ axes, surface[:, 2]))
        lo, hi = np.quantile(aligned, [.05, .95], axis=0)
        dx, dy, dz = hi - lo
        length, width, height = SIZE[label]
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
        result.append(dict(label=label, center=center, length=length, width=width,
                           height=height, angle=math.degrees(bearing) + (90. if sideways else 0.),
                           confidence=float(detection["score"]), radial_mps=None))
    return result


def _match_radar(measurements, radar):
    vel, az, alt, depth = np.asarray(radar).reshape(-1, 4).T
    if not len(vel) or not measurements:
        return
    flat = depth * np.cos(alt)
    xyz = np.column_stack((flat * np.cos(az), flat * np.sin(az),
                           depth * np.sin(alt))) + MOUNTS["radar"][:3]
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(vel) & (depth > 0) & (xyz[:, 2] > .18)
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
            item["radial_mps"] = float(np.median(speeds))  # CARLA signed Doppler, toward sensor positive


@dataclass
class _Track:
    position: np.ndarray
    velocity: np.ndarray
    last_seen: float
    measurement: dict
    heading_world: float
    observations: int = 1


class Tracker:
    def __init__(self):
        self._tracks = {}
        self._next_id = 1

    def update(self, lidar, radar, detections, ego_pose, sim_time):
        """Update once per chronological frame. Pass empty detections on skipped RGB inference."""
        now = float(sim_time)
        origin = np.asarray(ego_pose[:2], dtype=float)
        yaw = float(ego_pose[2])
        cos, sin = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        rotation = np.array([[cos, -sin], [sin, cos]])
        measurements = _measure(lidar, detections)
        _match_radar(measurements, radar)
        positions = [origin + rotation @ item["center"][:2] for item in measurements]
        self._tracks = {k: t for k, t in self._tracks.items() if 0 <= now - t.last_seen <= COAST_S}
        costs = []
        for key, track in self._tracks.items():
            dt = now - track.last_seen
            prediction = track.position + dt * track.velocity
            for idx, (item, pos) in enumerate(zip(measurements, positions)):
                if item["label"] != track.measurement["label"]:
                    continue
                distance = float(np.linalg.norm(pos - prediction))
                max_speed = 4. if item["label"] == "person" else 20.
                if distance <= .85 + dt * max_speed:
                    costs.append((distance, key, idx))
        used, updated = set(), set()
        for _, key, idx in sorted(costs):
            if idx in used or key in updated:
                continue
            track = self._tracks[key]
            item, pos = measurements[idx], positions[idx]
            dt = now - track.last_seen
            if dt > .001:
                weight = 1. if track.observations == 1 else min(1., dt / .4)
                track.velocity += weight * ((pos - track.position) / dt - track.velocity)
                track.observations += 1
            if np.linalg.norm(track.velocity) > .9:
                track.heading_world = math.degrees(math.atan2(*track.velocity[::-1]))
            track.position, track.last_seen, track.measurement = pos, now, item
            used.add(idx)
            updated.add(key)
        for idx, (item, pos) in enumerate(zip(measurements, positions)):
            if idx not in used:
                self._tracks[self._next_id] = _Track(pos, np.zeros(2), now, item, yaw + item["angle"])
                self._next_id += 1
        result = []
        for key, track in self._tracks.items():
            age = now - track.last_seen
            item = track.measurement
            x, y = rotation.T @ (track.position + age * track.velocity - origin)
            radial = item["radial_mps"] if not age else None
            sources = ["rgb", "lidar"] if not age else []
            if radial is not None:
                sources.append("radar")
            result.append(dict(id=key, label=item["label"], x=float(x), y=float(y),
                               z=float(item["center"][2]), length=float(item["length"]),
                               width=float(item["width"]), height=float(item["height"]),
                               heading=(track.heading_world - yaw + 180) % 360 - 180,
                               speed_mps=float(np.linalg.norm(track.velocity)) if track.observations > 1 else None,
                               confidence=item["confidence"] * max(0., 1. - age / COAST_S),
                               radial_mps=radial, age_s=age, coasting=bool(age), sources=sources,
                               estimated_fields=["length", "width", "height", "heading"]))
        return result
