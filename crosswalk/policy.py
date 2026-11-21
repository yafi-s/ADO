"""Sensor and navigation features for a learned CARLA actuator policy."""

from pathlib import Path

import numpy as np

from .signal_vision import SIGNAL_NAMES, SIDE_NAMES, signal_candidates, side_features
from .vision import VISION_NAMES, summarize


ACTION_NAMES = ("steer", "throttle", "brake")
MAX_RANGE = 60.0
BUMPER_X = 2.25
LIDAR_Z = 2.3
RADAR_X = 2.0
RADAR_Z = 0.9
CORRIDOR_HALF_WIDTH = 1.35
MODEL_VERSION = 6
NAV_NAMES = tuple(f"route_{distance}m_{axis}_m" for distance in (3, 7, 13) for axis in ("x", "y"))
LEGACY_FEATURE_NAMES = (
    "corridor_gap_m", "lidar_gap_m", "radar_gap_m", "left_lidar_gap_m",
    "right_lidar_gap_m", "radar_velocity_mps", "lidar_hits_log", "radar_hits_log",
    "ego_speed_mps",
) + tuple(
    f"rgb_{row}_{column}_{measure}"
    for row in ("horizon", "near")
    for column in ("left", "center", "right")
    for measure in ("r", "g", "b", "gray_std")
) + VISION_NAMES + NAV_NAMES

PATH_OBJECT_NAMES = tuple(f"rgb_path_{group}_{measure}" for group in ("person", "vehicle")
                          for measure in ("height", "offset_in_heights", "confidence"))
RANGE_NAMES = ("lidar_inverse_gap", "radar_inverse_gap", "ego_speed_over_gap")
GRID_FORWARD_EDGES = (0, 4, 8, 14, 22, 36, 60)
GRID_LATERAL_EDGES = (-9, -3.6, -1.35, 1.35, 3.6, 9)
GRID_LANES = ("far_left", "left", "center", "right", "far_right")
GRID_MEASURES = ("lidar_density", "radar_density", "radar_radial_velocity")
GRID_NAMES = tuple(
    f"bev_{measure}_{low}_{high}m_{lane}"
    for measure in GRID_MEASURES
    for low, high in zip(GRID_FORWARD_EDGES[:-1], GRID_FORWARD_EDGES[1:])
    for lane in GRID_LANES
)
FEATURE_NAMES = (LEGACY_FEATURE_NAMES + SIGNAL_NAMES + SIDE_NAMES + PATH_OBJECT_NAMES
                 + RANGE_NAMES + GRID_NAMES)
RGB_COLUMNS = np.asarray([name.startswith("rgb_") for name in FEATURE_NAMES])
RANGE_FEATURE_NAMES = tuple(name for name in FEATURE_NAMES if not name.startswith("rgb_"))


def _path_objects(detections):
    # A large oncoming car can dominate a center crop. Measure lateral offset
    # relative to apparent height, so that a parallel lane stays off axis at range.
    best = {group: (0., 0., 0., 0.) for group in ("person", "vehicle")}
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        height = max(float(y2 - y1), 1.e-4)
        offset = abs(float(x1 + x2 - 1.)) / height
        group = "person" if item["label"] == "person" else "vehicle"
        priority = height * item["score"] / (1. + offset**2)
        best[group] = max(best[group], (priority, height, min(offset, 12.), float(item["score"])))
    return [value for candidate in best.values() for value in candidate[1:]]



def _returns(data):
    data = np.asarray(data, dtype=np.float32)
    if not data.size:
        return np.empty((0, 4), dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("expected an N x 4 sensor return array")
    return data[np.isfinite(data).all(axis=1)]


def _gap(distances):
    return float(np.clip(distances.min(initial=MAX_RANGE), 0.0, MAX_RANGE))


def _grid_cells(forward, right, height):
    """Bin obstacle-height sensor returns, in meters from the front bumper."""
    keep = ((height > .24) & (height < 2.6)
            & (forward >= GRID_FORWARD_EDGES[0]) & (forward < GRID_FORWARD_EDGES[-1])
            & (right >= GRID_LATERAL_EDGES[0]) & (right < GRID_LATERAL_EDGES[-1]))
    rows = np.searchsorted(GRID_FORWARD_EDGES[1:], forward[keep], side="right")
    cols = np.searchsorted(GRID_LATERAL_EDGES[1:], right[keep], side="right")
    return rows * len(GRID_LANES) + cols, keep


def _spatial_grid(lidar_x, lidar_y, lidar_z, radar_x, radar_y, radar_z, velocity):
    """Lidar density, radar density and signed mean Doppler on the same 6 x 5 grid."""
    num_cells = (len(GRID_FORWARD_EDGES) - 1) * len(GRID_LANES)
    lidar_cell, _ = _grid_cells(lidar_x, lidar_y, lidar_z)
    radar_cell, radar_keep = _grid_cells(radar_x, radar_y, radar_z)
    lidar_counts = np.bincount(lidar_cell, minlength=num_cells)
    radar_counts = np.bincount(radar_cell, minlength=num_cells)
    # Compress dense near-field scans, keeping weak single-hit evidence.
    counts = np.concatenate((lidar_counts, radar_counts))
    log_density = np.log1p(np.minimum(counts, 64)) / np.log(65.)
    # Keep the native signed radial velocity. An occupied stationary cell
    # and an unobserved cell have different radar density measurements.
    scaled_velocity = np.clip(velocity[radar_keep] / 15., -1., 1.)
    sums = np.bincount(radar_cell, weights=scaled_velocity, minlength=num_cells)
    mean_velocity = sums / np.maximum(radar_counts, 1)
    return np.concatenate((log_density, mean_velocity)).tolist()


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


def range_features(lidar, radar, speed, navigation):
    """Pack lidar, radar, speed and navigation in RANGE_FEATURE_NAMES order.

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
    proximity = [1./(1.+gap_lidar), 1./(1.+gap_radar), speed/(1.+min(gap_lidar, gap_radar))]
    grid = _spatial_grid(x - BUMPER_X, y, height, rx, ry, rz, vel)
    return np.asarray(geometry + nav.tolist() + proximity + grid, dtype=np.float32)


def features(rgb, lidar, radar, speed, detections=(), navigation=None, *, side_views):
    """Combine synchronized range measurements and the three physical RGB camera views."""
    if len(side_views) != 2:
        raise ValueError("expected synchronized left and right RGB camera images")
    side = [value for image in side_views for value in side_features(image)]
    full = np.empty(len(FEATURE_NAMES), dtype=np.float32)
    full[~RGB_COLUMNS] = range_features(lidar, radar, speed, navigation)
    full[RGB_COLUMNS] = (_camera_features(rgb) + summarize(detections) + signal_candidates(rgb)
                         + side + _path_objects(detections))
    return full


def expand_range(packed):
    """Align measured rows with shared head indices; missing camera columns stay unavailable."""
    packed = np.asarray(packed, dtype=np.float32)
    if packed.ndim != 2 or packed.shape[1] != len(RANGE_FEATURE_NAMES):
        raise ValueError("expected packed range rows in RANGE_FEATURE_NAMES order")
    full = np.full((len(packed), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    full[:, ~RGB_COLUMNS] = packed
    return full


def input_mask(inputs):
    if inputs == "range":
        return ~RGB_COLUMNS
    if inputs not in ("fusion", "camera"):
        raise ValueError(f"unknown model inputs: {inputs}")
    return np.asarray([inputs == "fusion" or name.startswith(("rgb_", "route_"))
                       or name == "ego_speed_mps" for name in FEATURE_NAMES])


def head_columns(inputs):
    nav = np.asarray([name in NAV_NAMES for name in FEATURE_NAMES])
    speed = np.asarray([name == "ego_speed_mps" for name in FEATURE_NAMES])
    return np.flatnonzero(nav | speed), np.flatnonzero(input_mask(inputs))


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
        if (self.mean.shape != (2, len(self.mask)) or self.scale.shape != self.mean.shape
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
        for head, (columns, weights) in enumerate(zip(self.columns, self.heads)):
            z = np.clip((X[:, columns] - self.mean[head, columns]) / self.scale[head, columns], -8, 8)
            _, output = forward(z, weights)
            outputs.append(np.clip(output[:, 0], -1, 1))
        steer, drive = outputs
        return np.column_stack((steer, np.maximum(drive, 0), np.maximum(-drive, 0)))

    def predict(self, x):
        steer, throttle, brake = self.predict_batch(np.asarray(x)[None, :])[0]
        return float(steer), float(throttle), float(brake)

    def predict_range(self, packed):
        """Apply a range-only diagnostic model to one physical lidar/radar observation."""
        if self.inputs != "range":
            raise ValueError("packed range inference requires a range-only model")
        return self.predict(expand_range(np.asarray(packed)[None, :])[0])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {f"{head}_{i}": param for head, weights in zip(("steer", "drive"), self.heads)
                  for i, param in enumerate(weights)}
        with path.open("wb") as output:
            packed = {"packed_feature_names": RANGE_FEATURE_NAMES} if self.inputs == "range" else {}
            np.savez_compressed(output, version=MODEL_VERSION, feature_names=FEATURE_NAMES,
                                action_names=ACTION_NAMES, inputs=self.inputs, input_mask=self.mask,
                                mean=self.mean, scale=self.scale, **packed, **params)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if (int(data["version"]) != MODEL_VERSION
                    or tuple(data["feature_names"].tolist()) != FEATURE_NAMES
                    or tuple(data["action_names"].tolist()) != ACTION_NAMES):
                raise ValueError("incompatible policy; train an actuator model from demonstrations")
            inputs = str(data["inputs"].item())
            if inputs == "range" and ("packed_feature_names" not in data
                                      or tuple(data["packed_feature_names"]) != RANGE_FEATURE_NAMES):
                raise ValueError("invalid packed range model schema")
            if not np.array_equal(data["input_mask"], input_mask(inputs)):
                raise ValueError("invalid policy input mask")
            heads = [[data[f"{head}_{i}"] for i in range(4)] for head in ("steer", "drive")]
            return cls(inputs, data["mean"], data["scale"], heads)
