"""Positive motion evidence from successive ordinary lidar sweeps and measured ego odometry."""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


CELL = .25
RADIUS = 36.
BEAMS = 32
AZIMUTHS = 720


def _ray_cells(xyz):
    horizontal = np.hypot(xyz[:, 0], xyz[:, 1])
    elevation = np.degrees(np.arctan2(xyz[:, 2], horizontal))
    azimuth = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    beam = np.rint((elevation + 28.) * (BEAMS-1) / 38.).astype(int)
    az = np.floor((azimuth + 180.) * AZIMUTHS / 360.).astype(int) % AZIMUTHS
    return beam, az, np.hypot(horizontal, xyz[:, 2])


@dataclass
class Cluster:
    ids: np.ndarray
    surface: np.ndarray       # ego XYZ including mount height
    world: np.ndarray         # odometry-compensated points, Z relative to the roof lidar
    center: np.ndarray
    scale: float
    moving: bool = False


class Sweep:
    def __init__(self, raw, origin, rotation, now):
        self.now, self.origin, self.rotation = now, origin, rotation
        points = np.asarray(raw).reshape(-1, 4)[:, :3]
        good = np.flatnonzero(np.isfinite(points).all(axis=1))
        xyz = points[good]
        beam, az, depth = _ray_cells(xyz)
        ok = (beam >= 0) & (beam < BEAMS)
        self.range = np.full((BEAMS, AZIMUTHS), np.inf)
        np.minimum.at(self.range, (beam[ok], az[ok]), depth[ok])
        # A connected surface can maintain a prior RGB observation. Unclassified
        # rails, signs and parked vehicles cannot initiate tracks from shape alone.
        road_height = xyz[:, 2] + 2.3
        good = good[(road_height > .23) & (road_height < 3.6) & (abs(xyz[:, :2]) < RADIUS).all(axis=1)
                    & ((abs(xyz[:, 0]) > 2.75) | (abs(xyz[:, 1]) > 1.18))]
        self.clusters = []
        if len(good) < 6:
            return
        xy = np.floor((points[good, :2] + RADIUS) / CELL).astype(int)
        n = int(2 * RADIUS / CELL)
        # A float32 return just inside +RADIUS can round to the upper edge
        # when translated to grid coordinates. It belongs to the last cell.
        np.minimum(xy, n - 1, out=xy)
        mask = np.zeros((n, n), np.uint8)
        mask[xy[:, 0], xy[:, 1]] = 1
        _, labels = cv2.connectedComponents(cv2.dilate(mask, np.ones((3, 3), np.uint8)), connectivity=8)
        tags = labels[xy[:, 0], xy[:, 1]]
        order = tags.argsort()
        for indices in np.split(good[order], np.flatnonzero(np.diff(tags[order])) + 1):
            if len(indices) < 6:
                continue
            local = points[indices]
            span = np.ptp(local, axis=0)
            scale = float(np.linalg.norm(span[:2]))
            if not (.12 <= scale <= 8. and .35 <= span[2] <= 3.4):
                continue
            world = np.column_stack((local[:, :2] @ rotation.T + origin, local[:, 2]))
            center = (world[:, :2].min(axis=0) + world[:, :2].max(axis=0)) * .5
            self.clusters.append(Cluster(indices, local + [0., 0., 2.3], world, center, scale))

    def was_clear(self, world, fraction=0.):
        """Do at least three sampled rays pass beyond the proposed surface in this scan?"""
        local = np.column_stack(((world[:, :2] - self.origin) @ self.rotation, world[:, 2]))
        beam, az, depth = _ray_cells(local)
        ok = (beam >= 0) & (beam < BEAMS)
        beam, az, depth = beam[ok], az[ok], depth[ok]
        far = self.range[beam, az]
        free = np.isfinite(far) & (far > depth + 1.)
        return (free.sum() >= fraction * len(world)
                and len(np.unique(beam[free] * AZIMUTHS + az[free])) >= 3)


class LidarMotion:
    def __init__(self):
        self.history = deque(maxlen=6)

    def read(self, lidar, origin, rotation, now):
        current = Sweep(lidar, origin, rotation, now)
        prior = next((s for s in reversed(self.history) if .39 <= now - s.now <= .61), None)
        self.history.append(current)
        if prior is None or not prior.clusters:
            return current.clusters
        centers = np.array([c.center for c in prior.clusters])
        for cluster in current.clusters:
            distance = np.linalg.norm(centers - cluster.center, axis=1)
            nearest = int(np.argmin(distance))
            old = prior.clusters[nearest]
            delta = distance[nearest]
            if not (.45 < delta < 12. * (now - prior.now) and .55 < cluster.scale / old.scale < 1.8):
                continue
            # Both newly occupied and vacated space need positive measured rays. A
            # change in a visible face, a dropped return or an occluder is insufficient.
            cluster.moving = prior.was_clear(cluster.world) and current.was_clear(old.world)
        return current.clusters
