"""Fit a cruise-speed quantile and speed-dependent inverse pedal response from city telemetry."""
import argparse
import gzip
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from . import behavior_v2 as v2
from .city_cruise import CRUISE_MAX, CRUISE_NAMES, FOREST_FIELDS, UrbanCruiseModel, cruise_inputs
from .city_policy import CityModel, FEATURE_NAMES, INDEX, city_record, clearance


VERSION = "ado-city-pace-v1"
SPEED_KNOTS = np.array([0., 2., 4., 6., 8., 10., 12.])


class CityPaceModel(UrbanCruiseModel):
    uses_signed_speed_preview = True

    def __init__(self, base, forest, actuator, side_speed):
        super().__init__(base, forest)
        self.actuator = np.asarray(actuator)
        self.tactical_speed_limit = None
        self.signed_speed_limit = None
        self.side_speed = float(side_speed)
        if not np.isfinite(self.side_speed) or not 1. <= self.side_speed <= 6.:
            raise ValueError("Invalid demonstrated curbside speed")
        self.standoff = float(self.distances[self.requests <= .35][-1])
        if (self.actuator.shape != (len(SPEED_KNOTS), 3) or not np.isfinite(self.actuator).all()
                or np.any(self.actuator < 0.) or np.any(self.actuator > 1.)):
            raise ValueError("Invalid fitted city pedal calibration")

    def new_observer(self):
        self.tactical_speed_limit = None
        self.signed_speed_limit = None
        return super().new_observer()

    def _edge_speed(self):
        return self.side_speed

    def _plan(self, x):
        baseline = CityModel._plan(self, x)
        starts, splits, thresholds, left, right, values = self.cruise_forest
        features = cruise_inputs(x)
        nodes = starts.copy()
        active = left[nodes] >= 0
        while active.any():
            n = nodes[active]
            nodes[active] = np.where(features[splits[n]] <= thresholds[n], left[n], right[n])
            active = left[nodes] >= 0
        increment = float(values[nodes].mean())
        confidence = np.clip((baseline - 4.5) / (self.requests[-1] - 4.5), 0., 1.)
        requested = baseline + confidence * increment
        distance = min(clearance(x, self.radar_weight), x[INDEX["path_body_lidar_recent"]])
        # 36 m denotes no corridor return. Finite returns keep the original stopping margin.
        if distance < 36.:
            requested = min(requested, np.sqrt(baseline**2 + 3.6 * max(0., distance - 12.)))
        junction = np.sqrt(baseline**2 + 3.2 * max(0., x[INDEX["junction_gap"]] - 4.))
        posted = self.posted_limit_kmh
        if not np.isfinite(posted) or posted <= 0.:
            posted = 30.
        # A close lateral return can enter the corridor even before a precise
        # body range is available. Use a moving curbside approach set by the
        # demonstration; a stationary object outside the lane permits passage.
        side = min(x[INDEX["right_edge_lidar"]], x[INDEX["left_edge_lidar"]],
                   x[INDEX["right_edge_lidar_recent"]])
        if side < 36.:
            requested = min(requested, np.sqrt(self._edge_speed()**2 + 3.6 * max(0., side - self.standoff)))
        requested = min(requested, junction, posted / 3.6, CRUISE_MAX)
        if self.signed_speed_limit is not None:
            requested = min(requested, self.signed_speed_limit)
        if self.tactical_speed_limit is not None:
            requested = min(requested, self.tactical_speed_limit)
        # A briefly clear frame must not cause an immediate surge after a slowdown.
        if self.last_speed_request is not None:
            requested = min(requested, self.last_speed_request + .185)
        return float(requested)

    def predict(self, x):
        request = self._plan(x)
        self.last_speed_request = request
        nav = np.asarray(x)[v2.STEER_IDX]
        steer = float(np.clip(v2._steering_basis(nav) @ self.steering, -1., 1.))
        speed = nav[0]
        original = float(v2._feedback_features(request, speed) @ self.feedback)
        # Smooth ordinary speed corrections. A strong demand for deceleration still
        # permits the fitted legacy response when a hazard closes the stopping gap.
        error = request - speed
        accel = np.clip(1.4 * error, -2.35, 1.85)
        hold, accelerate, brake = (np.interp(speed, SPEED_KNOTS, c) for c in self.actuator.T)
        distance = clearance(x, self.radar_weight)
        if distance < 36. and error < 0.:
            # Reserve the demonstrated stand-off in the measured corridor. This
            # bound retains braking as speed falls and the residual gap shortens.
            required = (speed**2 - request**2) / (2. * max(.35, distance - self.standoff))
            accel = min(accel, -required)
        pedal = hold * np.exp(min(0., accel) / .35) + accelerate * max(0., accel) + brake * min(0., accel)
        # Return to the learned standstill/creep actuator near rest. Its training
        # includes brake hold at zero where moving inverse dynamics are undefined.
        near_rest = np.clip((1.4 - max(request, speed)) / 1., 0., 1.)
        urgency = np.clip((-error - 3.) / 2., 0., 1.)
        legacy = max(near_rest, urgency)
        ordinary = (1.-legacy)*pedal + legacy*original
        if accel < -2.35:
            ordinary = min(ordinary, pedal)
        pedal = float(np.clip(ordinary, -1., 1.))
        return steer, max(0., pedal), max(0., -pedal)

    def save(self, path):
        np.savez_compressed(path, version=VERSION, feature_names=FEATURE_NAMES,
                            cruise_features=CRUISE_NAMES, base=np.frombuffer(self.base_bytes, np.uint8),
                            actuator=self.actuator, side_speed=self.side_speed, **dict(zip(FOREST_FIELDS, self.cruise_forest)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if (str(d["version"].item()) != VERSION or tuple(d["feature_names"]) != FEATURE_NAMES
                    or tuple(d["cruise_features"]) != CRUISE_NAMES):
                raise ValueError("Incompatible city pace checkpoint")
            return cls(d["base"].tobytes(), tuple(d[key] for key in FOREST_FIELDS), d["actuator"], d["side_speed"].item())


def fit(lesson, base):
    from scipy.optimize import least_squares
    from sklearn.ensemble import ExtraTreesRegressor
    sample = city_record(lesson)
    x = cruise_inputs(sample["X"])
    goal = np.maximum(0., sample["goal"] - 5.2)
    estimator = ExtraTreesRegressor(n_estimators=48, max_depth=11, min_samples_leaf=5,
                                   random_state=2501, n_jobs=1)
    estimator.fit(x, goal)
    # Demonstrations anticipate scripted encounters before a sensor can see them.
    # Estimate available cruise speed from the upper tail of compatible requests.
    # Obstacle, turn, yield and signal constraints are applied independently.
    for tree in estimator.estimators_:
        leaves = tree.apply(x)
        for leaf in np.unique(leaves):
            tree.tree_.value[leaf, 0, 0] = np.quantile(goal[leaves == leaf], .85)
    with gzip.open(lesson.with_name("trace.jsonl.gz"), "rt") as stream:
        rows = [json.loads(line) for line in stream][:len(x)]
    speed = np.array([r["speed_mps"] for r in rows])
    dt = np.diff([r["sim_time"] for r in rows])
    np.testing.assert_allclose(dt, .1)
    accel = np.diff(speed) / dt
    # The control recorded in frame t is applied before observing frame t+1.
    pedal = np.array([r["throttle"] - r["brake"] for r in rows[:-1]])
    moving = (speed[:-1] > .45) & (np.abs(accel) < 7.)
    acceleration = accel[moving]
    design = np.stack((np.exp(np.minimum(0., acceleration) / .35), np.maximum(0., acceleration),
                       np.minimum(0., acceleration)), axis=1)
    calibrations = []
    for knot in SPEED_KNOTS:
        weight = np.exp(-.5 * ((speed[:-1][moving] - knot)/1.4)**2)
        # Bound coefficients and reduce the influence of gear changes and slopes.
        result = least_squares(lambda coef: np.sqrt(weight) * (design @ coef - pedal[moving]),
                               x0=[.12, .2, .13], bounds=([0., .03, .04], [.35, .4, .25]),
                               loss="soft_l1", f_scale=.035)
        calibrations.append(result.x)
    raw = sample["X"]
    sides = np.minimum(raw[:, INDEX["left_edge_lidar"]], raw[:, INDEX["right_edge_lidar"]])
    body = np.minimum(raw[:, INDEX["path_body_lidar"]], raw[:, INDEX["path_body_radar"]])
    # Conservative curb approach in motion, measured before the object
    # occupies the lane. No event name or position enters the estimator.
    curb_rows = (sides < 10.) & (body > 15.) & (raw[:, INDEX["ego_speed_mps"]] > .5)
    side_speed = float(np.quantile(raw[curb_rows, INDEX["ego_speed_mps"]], .15))
    return CityPaceModel(base.read_bytes(), v2._forest_arrays(estimator), np.array(calibrations),
                         side_speed), len(x), int(moving.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("models/city-v1/candidate-73-74.npz"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import scipy
    import sklearn
    model, n, moving = fit(args.lesson, args.base)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "candidate.npz"
    model.save(checkpoint)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(schema=VERSION, model_sha256=digest(checkpoint), instructor_rows=n,
                  moving_transition_rows=moving, actuator_columns=["hold", "accelerate", "brake"],
                  speed_knots_mps=SPEED_KNOTS.tolist(), actuator=model.actuator.tolist(),
                  curb_approach_mps=model.side_speed, learned_standoff_m=model.standoff,
                  dependencies=dict(python=platform.python_version(), numpy=np.__version__,
                                    scipy=scipy.__version__, sklearn=sklearn.__version__),
                  sources={p.name: digest(p) for p in (Path(__file__), args.base, args.lesson,
                          args.lesson.with_name("summary.json"), args.lesson.with_name("trace.jsonl.gz"))})
    (args.output / "fit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
