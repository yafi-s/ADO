"""Research observer for repeated pedestrian bearings with sparse lidar and dense, agreeing Doppler returns."""
import math

import numpy as np

from .camera_geometry import CAMERAS
from .radar_geometry import ego_returns
from .tracking import _project


def _inside(u, v, box):
    left, top, right, bottom = box
    return (u >= left) & (u <= right) & (v >= top) & (v <= bottom)


class SparsePedestrianStudy:
    """Uses fresh front optics and independent range sensors. Its output is an offline study."""
    def __init__(self):
        self.previous = {}

    def observe(self, boxes, scan, radar, pose, now):
        self.previous = {key: old for key, old in self.previous.items() if 0 < now-old[0] <= .500001}
        candidates = [b for b in boxes if b.get("label") == "person" and b.get("image_id") is not None
                      and b.get("score", 0) >= .60 and b.get("range_supported") is False]
        if not candidates:
            return []
        camera = CAMERAS["front"]
        points, ids, depth, u, v = _project(scan, camera)
        doppler = ego_returns(radar, "radar")
        doppler = doppler[doppler[:, 0] - camera.x > 1.]
        rd = doppler[:, 0] - camera.x
        ru = .5 + camera.fx * (doppler[:, 1] - camera.y) / rd
        rv = .5 - camera.fy * (doppler[:, 2] - camera.z) / rd
        laser_owners = np.zeros(len(u), dtype=int)
        radar_owners = np.zeros(len(ru), dtype=int)
        for b in boxes:
            laser_owners += _inside(u, v, b["box"])
            radar_owners += _inside(ru, rv, b["box"])
        angle = math.radians(pose[2])
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        readings = []
        for b in candidates:
            box = b["box"]
            laser = np.flatnonzero(_inside(u, v, box) & (laser_owners == 1) & (points[:, 2] > .35)
                                   & (points[:, 2] < 2.4) & (depth >= 15) & (depth <= 60))
            radar_ids = np.flatnonzero(_inside(ru, rv, box) & (radar_owners == 1))
            supported = []
            for j in laser:
                width = (box[2]-box[0]) * depth[j] / camera.fx
                height = (box[3]-box[1]) * depth[j] / camera.fy
                if not (1.05 <= height <= 2.18 and .18 <= width <= .94):
                    continue
                agrees = (abs(rd[radar_ids]-depth[j]) <= .29) & (
                    np.linalg.norm(doppler[radar_ids, :2]-points[j, :2], axis=1) < .5)
                corroborating = radar_ids[agrees]
                if len(corroborating) >= 3 and np.ptp(doppler[corroborating, 3]) <= .65:
                    supported.append((j, corroborating))
            if not 1 <= len(supported) <= 2:
                continue
            indices = [j for j, _ in supported]
            if np.ptp(depth[indices]) > .5:
                continue
            # At this range the laser usually samples the near face. Half the nominal
            # pedestrian width behind it approximates the body center (the evaluator uses a 1.5 m gate).
            center = np.median(points[indices, :2], axis=0)
            center += .18 * center / np.linalg.norm(center)
            world = pose[:2] + rotation @ center
            key = b["image_id"]
            old = self.previous.get(key)
            self.previous[key] = now, world
            if old is None or np.linalg.norm(world-old[1]) > .50 + 1.8*(now-old[0]):
                continue
            rays = np.unique(np.concatenate([hits for _, hits in supported]))
            readings.append(dict(image_id=key, box=box, confidence=b["score"], xy=center.tolist(),
                                 world_xy=world.tolist(), range_m=float(np.linalg.norm(center)),
                                 prior_age_s=round(now-old[0],3), lidar_indices=ids[indices].tolist(),
                                 radar_hits=len(rays), radial_mps=float(np.median(doppler[rays, 3]))))
        return readings
