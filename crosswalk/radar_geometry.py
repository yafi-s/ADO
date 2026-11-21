"""CARLA Doppler radar geometry in vehicle-local metres and signed metres per second."""

import math

import numpy as np

from .robotaxi import MOUNTS


RADARS = {
    "radar": MOUNTS["radar"],
    "radar_front_left": (1.85, -.85, .82, -45.),
    "radar_front_right": (1.85, .85, .82, 45.),
    "radar_rear_left": (-2.05, -.85, .82, -135.),
    "radar_rear_right": (-2.05, .85, .82, 135.),
}
CORNER_RADARS = tuple(name for name in RADARS if name != "radar")


def ego_returns(raw, name):
    """Convert velocity/azimuth/altitude/range into ego XYZ and unchanged line-of-sight Doppler."""
    vel, az, alt, depth = np.asarray(raw).reshape(-1, 4).T
    x, y, z, yaw = RADARS[name]
    bearing = az + math.radians(yaw)
    flat = depth * np.cos(alt)
    values = np.column_stack((x + flat * np.cos(bearing), y + flat * np.sin(bearing),
                              z + depth * np.sin(alt), vel))
    return values[np.isfinite(values).all(axis=1) & (depth > 0)]
