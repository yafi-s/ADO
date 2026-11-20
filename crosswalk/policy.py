"""Sensor features and two longitudinal policies for short, near-level urban routes.

Geometry uses CARLA's vehicle coordinate convention (x forward, y right, z up).
Only the training harness is allowed to supply an actor-derived expert label.
"""

from pathlib import Path

import numpy as np


CRUISE_SPEED = 6.0
MAX_RANGE = 60.0
BUMPER_X = 2.25
LIDAR_Z = 2.3
RADAR_X = 2.0
RADAR_Z = 0.9
CORRIDOR_HALF_WIDTH = 1.35
MODEL_VERSION = 2
FEATURE_NAMES = (
    "corridor_gap_m", "lidar_gap_m", "radar_gap_m", "left_lidar_gap_m",
    "right_lidar_gap_m", "radar_velocity_mps", "lidar_hits_log", "radar_hits_log",
    "ego_speed_mps",
) + tuple(
    f"rgb_{row}_{column}_{measure}"
    for row in ("horizon", "near")
    for column in ("left", "center", "right")
    for measure in ("r", "g", "b", "gray_std")
)


def _returns(data):
    data = np.asarray(data, dtype=np.float32)
    if not data.size:
        return np.empty((0, 4), dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("expected an N x 4 sensor return array")
    return data[np.isfinite(data).all(axis=1)]


def _gap(distances):
    return float(np.clip(distances.min(initial=MAX_RANGE), 0.0, MAX_RANGE))


def _camera_features(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 8:
        raise ValueError("expected an RGB image at least 8 x 8")
    h, w = image.shape[:2]
    # Restrict the field to the approach lane. Decimation bounds per-frame cost.
    road = image[h * 3 // 8:h * 7 // 8:max(1, h // 80),
                 w // 4:w * 3 // 4:max(1, w // 120)].astype(np.float32) / 255.0
    values = []
    for row in np.array_split(road, 2, axis=0):
        for patch in np.array_split(row, 3, axis=1):
            values.extend(patch.mean(axis=(0, 1)))
            values.append(patch.mean(axis=2).std())
    return values


def features(rgb, lidar, radar, speed):
    """Encode a synchronized RGB image, lidar sweep, Doppler scan and ego speed.

    Lidar is mounted at (0, 0, 2.3) and radar at (2, 0, 0.9), both level.
    CARLA's raw radar columns are radial velocity, azimuth, altitude, depth.
    Empty scans yield the 60 m sentinel. Missing frames are handled upstream.
    """
    lidar = _returns(lidar)
    x, y, z = lidar[:, :3].T
    height = z + LIDAR_Z
    above_road = (x > BUMPER_X) & (height > 0.24) & (height < 2.6)
    center = above_road & (np.abs(y) < CORRIDOR_HALF_WIDTH)
    left = above_road & (y <= -CORRIDOR_HALF_WIDTH) & (y > -3.0)
    right = above_road & (y >= CORRIDOR_HALF_WIDTH) & (y < 3.0)
    gap_lidar = _gap(x[center] - BUMPER_X)

    vel, az, alt, depth = _returns(radar).T
    horizontal = depth * np.cos(alt)
    rx = horizontal * np.cos(az) + RADAR_X - BUMPER_X
    ry = horizontal * np.sin(az)
    rz = depth * np.sin(alt) + RADAR_Z
    hits = (rx >= 0) & (rx < MAX_RANGE) & (np.abs(ry) < CORRIDOR_HALF_WIDTH)
    hits &= (rz > 0.24) & (rz < 2.6)
    gap_radar = _gap(rx[hits])
    radial_velocity = float(vel[hits][rx[hits].argmin()]) if hits.any() else 0.0

    geometry = [
        min(gap_lidar, gap_radar), gap_lidar, gap_radar, _gap(x[left] - BUMPER_X),
        _gap(x[right] - BUMPER_X), radial_velocity, np.log1p(center.sum()),
        np.log1p(hits.sum()), speed,
    ]
    return np.asarray(geometry + _camera_features(rgb), dtype=np.float32)


def expert_speed(gap):
    """Preview braking with 3 m of standstill clearance and 1.65 m/s² deceleration."""
    return min(CRUISE_SPEED, float(np.sqrt(2 * 1.65 * max(0.0, float(gap) - 3.0))))


def rule_speed(x):
    """Apply the explicit braking law to measured in-lane clearance."""
    return expert_speed(x[0])


def input_mask(inputs):
    """Columns available to a model. Both configurations receive the ego speed."""
    if inputs == "fusion":
        return np.ones(len(FEATURE_NAMES), dtype=bool)
    if inputs == "camera":
        return np.asarray([name == "ego_speed_mps" or name.startswith("rgb_")
                           for name in FEATURE_NAMES])
    raise ValueError(f"unknown model inputs: {inputs}")


class Model:
    """Small boosted CART ensemble. All split thresholds and leaf speeds are fitted."""

    def __init__(self, bias, splits, thresholds, values, inputs="fusion"):
        self.bias = float(bias)
        self.inputs = inputs
        self.mask = input_mask(inputs)
        self.splits = np.asarray(splits, dtype=np.int32)
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        if (
            self.splits.ndim != 2 or self.splits.shape != self.thresholds.shape
            or self.splits.shape != self.values.shape or not self.splits.size
            or not (-1 <= self.splits).all() or not (self.splits < len(FEATURE_NAMES)).all()
            or not np.isfinite(self.values).all() or not np.isfinite(self.thresholds).all()
            or not np.isfinite(self.bias)
        ):
            raise ValueError("invalid policy arrays")
        if not self.mask[self.splits[self.splits >= 0]].all():
            raise ValueError("model splits use unavailable sensor inputs")
        # All actual children must be inside the fixed heap of each tree.
        occupied = np.where(self.splits >= 0)[1]
        if (2 * occupied + 2 >= self.splits.shape[1]).any():
            raise ValueError("invalid tree depth")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            version = int(data["version"])
            if (version not in (1, MODEL_VERSION)
                    or tuple(data["feature_names"].tolist()) != FEATURE_NAMES):
                raise ValueError("policy is incompatible with this feature extractor")
            inputs = str(data["inputs"].item()) if version >= 2 else "fusion"
            if version >= 2 and not np.array_equal(data["input_mask"], input_mask(inputs)):
                raise ValueError("invalid model input mask")
            return cls(data["bias"], data["splits"], data["thresholds"], data["values"], inputs)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            np.savez_compressed(
                output, version=MODEL_VERSION, feature_names=FEATURE_NAMES, bias=self.bias,
                inputs=self.inputs, input_mask=self.mask,
                splits=self.splits, thresholds=self.thresholds, values=self.values,
            )

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x[self.mask]).all():
            raise ValueError("invalid policy input")
        trees = np.arange(len(self.splits))
        nodes = np.zeros(len(trees), dtype=np.int32)
        while True:
            columns = self.splits[trees, nodes]
            active = columns >= 0
            if not active.any():
                break
            right = x[columns[active]] > self.thresholds[trees[active], nodes[active]]
            nodes[active] = 2 * nodes[active] + 1 + right
        return float(np.clip(self.bias + self.values[trees, nodes].sum(), 0, CRUISE_SPEED))
