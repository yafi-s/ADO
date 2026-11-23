"""Cost-fitted stationary vehicle intent; invoked only for a sensor-confirmed source car."""
import hashlib
import io
from pathlib import Path

import numpy as np


FEATURES = (
    "source_gap_m", "ego_speed_mps", "source_speed_mps", "previous_source_speed_mps",
    "source_acceleration_mps2", "observed_stopped_wait_s", "source_observed_s",
    "projected_lane_gain_m",
)
MODEL = Path(__file__).resolve().parents[1] / "models/city-maneuvers/stationary-intent.npz"
VERSION = "ado-stationary-vehicle-intent-v1"


class StationaryIntent:
    def __init__(self, path=MODEL):
        self.checkpoint_path = Path(path).resolve()
        packed = self.checkpoint_path.read_bytes()
        self.sha256 = hashlib.sha256(packed).hexdigest()
        with np.load(io.BytesIO(packed), allow_pickle=False) as d:
            if str(d["version"]) != VERSION or tuple(d["features"]) != FEATURES:
                raise ValueError("Incompatible stationary vehicle intent head")
            self.feature, self.threshold, self.left, self.right = (
                d[k] for k in ("feature", "threshold", "left", "right"))
            self.pass_gain = d["pass_gain_m"]
            self.follow_fraction = d["follow_fraction"]
        n = len(self.feature)
        if not n or any(len(a) != n for a in (self.threshold, self.left, self.right,
                                               self.pass_gain, self.follow_fraction)):
            raise ValueError("Invalid stationary intent tree")
        for i, j in enumerate(self.feature):
            if j != -1 and not (0 <= j < len(FEATURES) and i < self.left[i] < n and i < self.right[i] < n):
                raise ValueError("Invalid stationary intent branch")
        if not np.isfinite(np.r_[self.threshold,self.pass_gain,self.follow_fraction]).all():
            raise ValueError("Nonfinite stationary intent estimates")

    def details(self, features):
        """Return the action, calibrated leaf fraction, relative utility in equivalent metres, and leaf."""
        row = np.asarray([features[name] for name in FEATURES] if hasattr(features, "keys") else features, dtype=float)
        if row.shape != (len(FEATURES),) or not np.isfinite(row).all():
            raise ValueError("Expected eight finite causal stationary intent features")
        node = 0
        while self.feature[node] >= 0:
            node = self.left[node] if row[self.feature[node]] <= self.threshold[node] else self.right[node]
        decision = "follow" if self.pass_gain[node] <= 0. else "defer_to_existing_choice"
        return decision, float(self.follow_fraction[node]), float(self.pass_gain[node]), int(node)

    def predict(self, features):
        decision, follow_fraction, _, _ = self.details(features)
        return decision, follow_fraction
