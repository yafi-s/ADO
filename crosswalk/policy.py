"""Sensor and navigation features for a learned CARLA actuator policy."""

from pathlib import Path

import numpy as np

from .vision import VISION_NAMES, summarize


ACTION_NAMES = ("steer", "throttle", "brake")
MAX_RANGE = 60.0
BUMPER_X = 2.25
LIDAR_Z = 2.3
RADAR_X = 2.0
RADAR_Z = 0.9
CORRIDOR_HALF_WIDTH = 1.35
MODEL_VERSION = 4
NAV_NAMES = tuple(f"route_{distance}m_{axis}_m" for distance in (3, 7, 13) for axis in ("x", "y"))
FEATURE_NAMES = (
    "corridor_gap_m", "lidar_gap_m", "radar_gap_m", "left_lidar_gap_m",
    "right_lidar_gap_m", "radar_velocity_mps", "lidar_hits_log", "radar_hits_log",
    "ego_speed_mps",
) + tuple(
    f"rgb_{row}_{column}_{measure}"
    for row in ("horizon", "near")
    for column in ("left", "center", "right")
    for measure in ("r", "g", "b", "gray_std")
) + VISION_NAMES + NAV_NAMES


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


def features(rgb, lidar, radar, speed, detections=(), navigation=None):
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
    nav = np.asarray(navigation, dtype=np.float32)
    if nav.shape != (6,) or not np.isfinite(nav).all():
        raise ValueError("expected three ego-local route points [x3,y3,x7,y7,x13,y13]")
    return np.asarray(geometry + _camera_features(rgb) + summarize(detections) + nav.tolist(),
                      dtype=np.float32)


def input_mask(inputs):
    if inputs not in ("fusion", "camera"):
        raise ValueError(f"unknown model inputs: {inputs}")
    return np.asarray([inputs == "fusion" or name.startswith(("rgb_", "route_"))
                       or name == "ego_speed_mps" for name in FEATURE_NAMES])


def head_columns(inputs):
    nav = np.asarray([name in NAV_NAMES for name in FEATURE_NAMES])
    speed = np.asarray([name == "ego_speed_mps" for name in FEATURE_NAMES])
    return np.flatnonzero(nav | speed), np.flatnonzero(input_mask(inputs) & ~nav)


def linear(x, w):
    # Direct einsum runs these small contractions on one CPU thread.
    return np.einsum("bi,ij->bj", x, w, optimize=False)


def forward(x, weights):
    w1, b1, w2, b2 = weights
    hidden = np.tanh(linear(x, w1) + b1)
    return hidden, linear(hidden, w2) + b2


class Model:
    """Independent neural steering and longitudinal heads trained on demonstrated controls."""

    def __init__(self, inputs, mean, scale, heads):
        self.inputs = inputs
        self.mask = input_mask(inputs)
        self.columns = head_columns(inputs)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.heads = heads
        if (self.mean.shape != self.mask.shape or self.scale.shape != self.mask.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or not (self.scale > 0).all()):
            raise ValueError("invalid model normalization")
        if len(heads) != 2:
            raise ValueError("expected two control heads")
        for columns, (w1, b1, w2, b2) in zip(self.columns, heads):
            if (b1.ndim != 1 or not b1.size or w1.shape != (len(columns), b1.size)
                    or w2.shape != (b1.size, 1) or b2.shape != (1,)
                    or not all(np.isfinite(p).all() for p in (w1, b1, w2, b2))):
                raise ValueError("invalid neural policy parameters")

    def predict_batch(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or not np.isfinite(X[:, self.mask]).all():
            raise ValueError("invalid neural policy inputs")
        outputs = []
        for columns, weights in zip(self.columns, self.heads):
            z = np.clip((X[:, columns] - self.mean[columns]) / self.scale[columns], -8, 8)
            _, output = forward(z, weights)
            outputs.append(np.clip(output[:, 0], -1, 1))
        steer, drive = outputs
        return np.column_stack((steer, np.maximum(drive, 0), np.maximum(-drive, 0)))

    def predict(self, x):
        steer, throttle, brake = self.predict_batch(np.asarray(x)[None, :])[0]
        return float(steer), float(throttle), float(brake)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {f"{head}_{i}": param for head, weights in zip(("steer", "drive"), self.heads)
                  for i, param in enumerate(weights)}
        with path.open("wb") as output:
            np.savez_compressed(output, version=MODEL_VERSION, feature_names=FEATURE_NAMES,
                                action_names=ACTION_NAMES, inputs=self.inputs, input_mask=self.mask,
                                mean=self.mean, scale=self.scale, **params)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if (int(data["version"]) != MODEL_VERSION
                    or tuple(data["feature_names"].tolist()) != FEATURE_NAMES
                    or tuple(data["action_names"].tolist()) != ACTION_NAMES):
                raise ValueError("incompatible policy; train an actuator model from demonstrations")
            inputs = str(data["inputs"].item())
            if not np.array_equal(data["input_mask"], input_mask(inputs)):
                raise ValueError("invalid policy input mask")
            heads = [[data[f"{head}_{i}"] for i in range(4)] for head in ("steer", "drive")]
            return cls(inputs, data["mean"], data["scale"], heads)
