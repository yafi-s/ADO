"""Vehicle speed and map-relative navigation inputs for the learned policy."""
import math

import numpy as np


def vehicle_speed(ego):
    v = ego.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


class Navigation:
    """Sample map points at fixed arc lengths in the ego frame, in metres."""
    def __init__(self, route):
        self.points = np.array([(p.x, p.y) for p in route], dtype=float)
        self.delta = np.diff(self.points, axis=0)
        lengths = np.linalg.norm(self.delta, axis=1)
        self.lengths = np.maximum(lengths, 1e-7)
        self.arc = np.r_[0., np.cumsum(lengths)]

    def read(self, ego, lookaheads=(3., 7., 13.)):
        t = ego.get_transform()
        pos = np.array((t.location.x, t.location.y))
        yaw = math.radians(t.rotation.yaw)
        offset = pos - self.points[:-1]
        fraction = np.clip((offset * self.delta).sum(axis=1) / self.lengths**2, 0, 1)
        residual = offset - self.delta * fraction[:, None]
        idx = int(np.argmin((residual**2).sum(axis=1)))
        progress = self.arc[idx] + fraction[idx] * self.lengths[idx]
        self.progress = float(progress)
        distances = progress + np.asarray(lookaheads)
        # Extend the last route tangent beyond the measured finish so navigation
        # inputs don't collapse in the final metres of an otherwise straight lane.
        extra = np.maximum(0., distances - self.arc[-1])
        goals = np.column_stack([np.interp(distances, self.arc, p) for p in self.points.T])
        goals += extra[:, None] * self.delta[-1] / self.lengths[-1]
        dx, dy = (goals - pos).T
        c, s = math.cos(yaw), math.sin(yaw)
        return np.column_stack([dx*c + dy*s, -dx*s + dy*c]).ravel()
