"""Fit and run a normal-speed Town03 cruise head on measured navigation and sensors."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from . import behavior_v2 as v2
from .city_policy import CityModel, FEATURE_NAMES, INDEX, city_record, clearance


VERSION = "ado-city-cruise-v1"
CRUISE_NAMES = (
    "junction_gap", "bend_7_13", "bend_13_25", "bend_25_40",
    "path_body_lidar", "path_body_radar", "path_body_lidar_recent", "path_body_radar_recent",
    "left_edge_lidar", "right_edge_lidar", "right_edge_lidar_recent",
    "oncoming_gap_recent", "rgb_person_height_recent", "rgb_vehicle_height",
    "mapped_lamp_gap_m",
)
COLUMNS = np.array([INDEX[name] for name in CRUISE_NAMES])
FOREST_FIELDS = ("starts", "splits", "thresholds", "left", "right", "values")
CRUISE_MAX = 11.0  # 39.6 km/h, below the minimum posted speed on the Town03 market route.


def cruise_inputs(x):
    values = np.asarray(x)[..., COLUMNS].copy()
    values[..., 1:4] = np.abs(values[..., 1:4])
    return values


class UrbanCruiseModel(CityModel):
    """Keep the low-speed fitted policy, extend its unconstrained cruise range."""

    uses_posted_limit = True

    def __init__(self, base_bytes, forest):
        base = CityModel.load(io.BytesIO(base_bytes))
        super().__init__(base.distances, base.requests, base.radar_weight, base.forest,
                         base.steering, base.feedback, base.sensor_lamp_classifier)
        self.base_bytes = base_bytes
        self.posted_limit_kmh = CRUISE_MAX * 3.6
        self.cruise_forest = forest
        # Validate ordered branches, targets, fields and all node values on load.
        v2.BehaviorModel(*forest, self.steering, self.feedback, None)
        if (np.any(forest[1][forest[3] >= 0] >= len(COLUMNS))
                or np.min(forest[5]) < 0 or np.max(forest[5]) > CRUISE_MAX - 5.2 + 1e-5):
            raise ValueError("Invalid urban cruise estimator")

    def _plan(self, x):
        baseline = super()._plan(x)
        starts, splits, thresholds, left, right, values = self.cruise_forest
        features = cruise_inputs(x)
        nodes = starts.copy()
        active = left[nodes] >= 0
        while active.any():
            n = nodes[active]
            nodes[active] = np.where(features[splits[n]] <= thresholds[n], left[n], right[n])
            active = left[nodes] >= 0
        # A low request from the fitted obstacle, signal or yield planner has
        # precedence. Use its measured free-road plateau to avoid a speed jump.
        confidence = np.clip((baseline-4.5)/(self.requests[-1]-4.5), 0., 1.)
        requested = baseline + confidence * values[nodes].mean()
        # Preserve stopping room at the wider speed range. An occupied corridor
        # and an approaching junction each impose a continuous deceleration bound.
        distance = min(clearance(x, self.radar_weight), x[INDEX["path_body_lidar_recent"]])
        obstacle_limit = np.sqrt(baseline**2 + 3.6*max(0., distance-12.))
        junction_limit = np.sqrt(baseline**2 + 3.2*max(0., x[INDEX["junction_gap"]]-4.))
        posted = self.posted_limit_kmh
        if not np.isfinite(posted) or posted <= 0.:
            posted = 30.
        return float(min(requested, obstacle_limit, junction_limit, posted/3.6, CRUISE_MAX))

    def save(self, path):
        np.savez_compressed(path, version=VERSION, feature_names=FEATURE_NAMES,
                            cruise_features=CRUISE_NAMES, base=np.frombuffer(self.base_bytes, np.uint8),
                            **dict(zip(FOREST_FIELDS, self.cruise_forest)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if (str(d["version"].item()) != VERSION or tuple(d["feature_names"]) != FEATURE_NAMES
                    or tuple(d["cruise_features"]) != CRUISE_NAMES):
                raise ValueError("Incompatible urban cruise checkpoint")
            return cls(d["base"].tobytes(), tuple(d[key] for key in FOREST_FIELDS))


def train(episodes, base):
    from sklearn.ensemble import ExtraTreesRegressor
    X = np.concatenate([cruise_inputs(e["X"]) for e in episodes])
    y = np.concatenate([np.maximum(0., e["goal"]-5.2) for e in episodes])
    forest = ExtraTreesRegressor(n_estimators=48, max_depth=11, min_samples_leaf=3,
                                 random_state=2501, n_jobs=1)
    forest.fit(X, y)
    return UrbanCruiseModel(base, v2._forest_arrays(forest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("models/city-v1/candidate-73-74.npz"))
    parser.add_argument("--lessons", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import platform
    import sklearn
    episodes = [city_record(p) for p in args.lessons]
    base = args.base.read_bytes()
    model = train(episodes, base)
    args.output.mkdir(exist_ok=True, parents=True)
    checkpoint = args.output/"candidate.npz"
    model.save(checkpoint)
    recovered = UrbanCruiseModel.load(checkpoint)
    for x in episodes[0]["X"][:25]:
        np.testing.assert_allclose(model.predict(x), recovered.predict(x), atol=1e-7)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    sources = [Path(__file__), args.base]
    report = dict(schema=VERSION, model_sha256=digest(checkpoint), dependencies={
        "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        training=[], features=list(CRUISE_NAMES), sources={})
    for path, item in zip(args.lessons, episodes):
        sources.extend((path, path.with_name("trace.jsonl.gz"), path.with_name("summary.json")))
        report["training"].append(dict(seed=item["seed"], episode=item["name"], rows=len(item["X"]),
            excluded_collision_tail=item["trim"], requests_above_35_kmh=int((item["goal"]>35/3.6).sum())))
    report["sources"] = {str(p): digest(p) for p in sources}
    (args.output/"fit.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
