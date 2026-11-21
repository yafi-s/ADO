"""Town01 interaction residual for the frozen v0.2 driver. One instance per episode."""
from pathlib import Path
import hashlib
import io
import json
import zipfile

import numpy as np

from .behavior_v2 import BehaviorModel, FEATURE_NAMES, HORIZON, Observer, _feedback_features

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/behavior-v2/candidate.npz"
MODEL = ROOT / "models/town01-yield-v04/calibration.json"
LESSONS = ROOT / "data/behavior-v2/lessons.zip"
COL = {n: i for i, n in enumerate(FEATURE_NAMES)}
VERSION = "town01-yield-v04"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibrate(lessons=LESSONS):
    """Estimate stationary stand-off and approach deceleration in disjoint teacher episodes."""
    samples = {"junction": [], "crossing": []}
    braking = []
    with zipfile.ZipFile(lessons) as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith(("left-", "pedestrian-")):
                continue
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as lesson:
                if tuple(lesson["feature_names"]) != FEATURE_NAMES:
                    raise ValueError("Unexpected training columns")
                x, gap = lesson["X"], lesson["expert_gap_m"]
                speed = x[:, COL["ego_speed_mps"]]
                stopped = (speed < .25) & (gap < 8)
                if not stopped.any():
                    continue
                is_left = name.startswith("left-")
                observed = x[:, COL["junction_gap" if is_left else "path_body_lidar"]]
                measured = observed[stopped & (observed < HORIZON)]
                if not len(measured):
                    continue
                standoff = float(np.median(measured))
                family = "junction" if is_left else "crossing"
                samples[family].append((name, standoff))
                # Pick the observed braking envelope upstream of the final stop.
                approach = (gap < 16) & (observed > standoff + 1.) & (observed < 18.) & (speed > 1.5) & (lesson["y"][:, 2] > .2)
                if approach.any():
                    braking.append(float(np.median(speed[approach] ** 2 / (2 * (observed[approach] - standoff)))))
    if any(len(v) < 2 for v in samples.values()) or not braking:
        raise ValueError("Insufficient native demonstration coverage")
    return dict(version=VERSION, baseline_sha256=digest(BASE), lessons_sha256=digest(lessons),
                training_episodes=[n for values in samples.values() for n, _ in values],
                junction_standoff_m=float(np.median([s for _, s in samples["junction"]])),
                crossing_standoff_m=float(np.median([s for _, s in samples["crossing"]])),
                deceleration_mps2=float(np.quantile(braking, .25)))


class InteractionModel:
    inputs = "fusion"
    version = VERSION

    def __init__(self, base, config):
        self.base = base
        self.config = config
        self.lamp_expert = base.lamp_expert
        self.observer = None

    @classmethod
    def load(cls, path=MODEL):
        config = json.loads(Path(path).read_text())
        if config["version"] != cls.version or config["baseline_sha256"] != digest(BASE):
            raise ValueError("Incompatible Town01 interaction model")
        for k in ("junction_standoff_m", "crossing_standoff_m", "deceleration_mps2"):
            if not 0 < float(config[k]) < 12:
                raise ValueError(f"Invalid calibration: {k}")
        return cls(BehaviorModel.load(BASE), config)

    def new_observer(self):
        self.observer = Observer()
        self.observer.last_info = {}
        self.opposing_memory = 0
        self.previous_body = HORIZON
        self.missed_body = 0
        return self.observer

    def edge_distance(self, x, request):
        return float(x[COL["right_edge_lidar"]])

    def predict(self, x):
        if self.observer is None:
            raise RuntimeError("Start a new episode before predicting")
        def value(name):
            return float(x[COL[name]])
        speed = value("ego_speed_mps")
        action = self.base.predict(x)
        request = self.base._plan(x)
        side = self.edge_distance(x, request)
        gap = value("junction_gap")
        # Curvature is signed in ego coordinates (+right). This is static navigation.
        left = gap < 25 and sum(value(k) for k in ("bend_7_13", "bend_13_25", "bend_25_40")) < -.45
        opposing = value("oncoming_gap")
        self.opposing_memory = 15 if left and opposing < 35.9 else max(0, self.opposing_memory - 1)
        cap, reason = request, "base"
        decel = self.config["deceleration_mps2"]
        def stop_speed(measured, margin):
            return float(np.sqrt(2 * decel * max(0., measured - margin - .25 * speed)))
        if left and self.opposing_memory:
            cap = min(cap, stop_speed(gap, self.config["junction_standoff_m"]))
            reason = "oncoming"
        elif gap > 24:
            # A curbside return warrants slowing until its motion is resolved. Preserve
            # a moving floor for a stationary roadside object outside the vehicle body.
            body = min(value("path_body_lidar"), value("path_body_radar"))
            if body < 14:
                self.previous_body = body
                self.missed_body = 0
            else:
                self.missed_body += 1
                if self.missed_body <= 5:
                    self.previous_body = max(0., self.previous_body - .1 * speed)
                    body = min(body, self.previous_body)
            margin = self.config["crossing_standoff_m"]
            side_cap = max(2.8, stop_speed(side, margin + 1.))
            body_cap = stop_speed(body, margin)
            cap = min(cap, side_cap, body_cap)
            if cap < request:
                reason = "path occupancy" if body_cap < side_cap else "path edge"
        self.observer.last_info = dict(base_target_mps=round(request, 3), target_mps=round(cap, 3),
            cue=reason, junction_m=round(gap, 2), opposing_m=round(opposing, 2),
            body_m=round(value("path_body_lidar"), 2), right_m=round(value("right_edge_lidar"), 2))
        if cap >= request:
            return action
        pedal = float(np.clip(_feedback_features(cap, speed) @ self.base.feedback, -1., 1.))
        return action[0], max(0., pedal), max(0., -pedal)


if __name__ == "__main__":
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    result = calibrate()
    MODEL.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
