"""A measured sightline refinement of the frozen City Pace driver.

It relaxes only the close lateral-object speed cap when scans repeatedly resolve
separate narrow, static returns outside the swept body and pavement beyond them.
"""
import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from . import behavior_v2 as v2
from .city_pace import CityPaceModel, FEATURE_NAMES as BASE_NAMES, FOREST_FIELDS, CRUISE_NAMES
from .city_policy import INDEX
from .policy import LIDAR_Z, _returns


VERSION = "ado-city-clearance-v1"
PROFILE = ("open_edge_margin_m", "open_edge_sight_m")
FEATURE_NAMES = (*BASE_NAMES[:-2], *PROFILE, *BASE_NAMES[-2:])
BODY = 1.08


def _edge_profile(lidar, path):
    """Resolve compact surfaces, their clearance, and visible ground through the gaps."""
    cloud = _returns(lidar)
    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2] + LIDAR_Z
    plane = v2._ground_plane(x, y, z)
    h = z - (plane[0] * x + plane[1] * y + plane[2])
    nearby = (x > .5) & (x < 24.) & (abs(y) < 8.)
    station, lateral = v2._project(x[nearby], y[nearby], path)
    height = h[nearby]
    surface = (height > .28) & (height < 3.)
    cross = abs(lateral)
    flank = surface & (station < 17.) & (cross < 3.6)
    if np.any(flank & (cross < BODY)):
        return []
    cells = []
    for sign in (-1, 1):
        side = lateral * sign
        hits = np.flatnonzero(flank & (side >= BODY))
        if not len(hits):
            continue
        # Validate visible pavement beyond this edge across several longitudinal
        # metres. Two isolated beams behind the object cannot establish sight.
        pavement = (side > 3.6) & (side < 7.) & (abs(height) < .32) & (station > 0.) & (station < 15.)
        counts = np.bincount((station[pavement] // 2.).astype(int), minlength=8)
        # A pair of beams in each consecutive two-metre cell measures a continuous
        # view. Isolated long-range hits do not extend the stopping budget.
        gaps = np.flatnonzero(counts < 2)
        sight = 2. * (int(gaps[0]) if len(gaps) else len(counts))
        if sight < 8.:
            return []
        hits = hits[np.argsort(station[hits])]
        clusters = np.split(hits, np.flatnonzero(np.diff(station[hits]) > .65) + 1)
        for cluster in clusters:
            s = station[cluster]
            if (np.ptp(s) > .95 or np.ptp(side[cluster]) > .8 or np.max(height[cluster]) > 1.45):
                return []
            if len(s) >= 4:
                cells.append((float(np.mean(s)), float(np.mean(lateral[cluster])),
                              float(np.min(cross[cluster]) - BODY), min(15., sight)))
    return cells


class EdgeObserver(v2.Observer):
    def __init__(self):
        super().__init__()
        self.sweeps = deque(maxlen=5)
        self.travel = 0.
        self.streak = 0

    def observe(self, obs, detections, local_route, speed, *, route_preview=None, stopline_gap=None):
        x = super().observe(obs, detections, local_route, speed,
                            route_preview=route_preview, stopline_gap=stopline_gap)
        cells = _edge_profile(obs["lidar"], v2._checked_route(local_route, route_preview))
        # Station falls by ego travel for a stationary return. Track relative lateral
        # displacement too; an object crossing into the lane cancels the release.
        self.travel += speed * .1
        def compatible(cell, sweep):
            travel, prior = sweep
            return any(abs(cell[0] - (p[0] - (self.travel - travel))) < .36
                       and abs(cell[1] - p[1]) < .16 for p in prior)
        matched = [c for c in cells if len(self.sweeps) == self.sweeps.maxlen
                   and compatible(c, self.sweeps[-1]) and compatible(c, self.sweeps[0])]
        if len(matched) >= 2:
            self.streak += 1
        elif not matched or not self.streak:
            self.streak = 0
        self.sweeps.append((self.travel, cells))
        person_near = any(d["label"] == "person" and d["box"][3] - d["box"][1] > .055 for d in detections)
        margin, sight = -1., 0.
        near_unconfirmed = any(c not in matched and c[0] < 8. for c in cells)
        if (self.streak >= 3 and not near_unconfirmed and not person_near
                and min(x[INDEX[n]] for n in ("path_body_lidar", "path_body_radar",
                                               "path_body_lidar_recent", "path_body_radar_recent")) > 17.
                and min(c[0] for c in matched) < 10.):
            margin = min(c[2] for c in cells)
            sight = min(c[3] for c in cells)
        return np.r_[x, np.float32(margin), np.float32(sight)]


class ClearanceModel(CityPaceModel):
    def __init__(self, base, forest, actuator, side_speed, stopping):
        super().__init__(base, forest, actuator, side_speed)
        self.stopping = np.asarray(stopping, dtype=float)
        if (self.stopping.shape != (4,) or not np.isfinite(self.stopping).all()
                or self.stopping[0] <= 0. or np.any(self.stopping[1:] < 0.) or self.stopping[3] <= 0.):
            raise ValueError("Invalid measured stopping response")
        self.open_edge_margin, self.edge_sight = -1., 0.

    def new_observer(self):
        super().new_observer()
        return EdgeObserver()

    def predict(self, x):
        x = np.asarray(x)
        if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
            raise ValueError("Invalid city clearance measurements")
        self.open_edge_margin, self.edge_sight = x[-4:-2]
        return super().predict(np.concatenate((x[:-4], x[-2:])))

    def _edge_speed(self):
        # Reserve the fitted stand-off in the visible corridor. Discount sight on
        # a tight pass in proportion to clearance relative to half the body half-width.
        # An empty or unknown profile yields the frozen conservative speed.
        fraction = np.clip(self.open_edge_margin / (.5 * BODY), 0., 1.)
        quadratic, linear, slack, max_tested_speed = self.stopping
        budget = max(0., (self.edge_sight - self.standoff) * fraction - slack)
        # Invert the measured reaction-and-braking distance. Limit extrapolation
        # to the highest actual speed used in the development trials.
        speed = 2.*budget / max(1.e-9, linear + np.sqrt(linear**2 + 4.*quadratic*budget))
        return max(self.side_speed, min(speed, max_tested_speed))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if (str(d["version"].item()) != VERSION or tuple(d["feature_names"]) != FEATURE_NAMES
                    or tuple(d["cruise_features"]) != CRUISE_NAMES):
                raise ValueError("Incompatible city clearance checkpoint")
            return cls(d["base"].tobytes(), tuple(d[key] for key in FOREST_FIELDS), d["actuator"], d["side_speed"].item(), d["stopping"])

    def save(self, path):
        np.savez_compressed(path, version=VERSION, feature_names=FEATURE_NAMES, cruise_features=CRUISE_NAMES,
                            base=np.frombuffer(self.base_bytes, np.uint8), actuator=self.actuator,
                            side_speed=self.side_speed, stopping=self.stopping, **dict(zip(FOREST_FIELDS, self.cruise_forest)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("models/city-pace/candidate.npz"))
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    base = CityPaceModel.load(a.base)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    trials = json.loads(a.calibration.read_bytes())["trials"]
    train = [r for r in trials if r["split"] == "fit"]
    speed = np.array([r["cue_speed_mps"] for r in train])
    observed = np.array([r["stopping_distance_m"] for r in train])
    X = np.column_stack([speed**2, speed])
    quadratic, linear = np.linalg.lstsq(X, observed, rcond=None)[0]
    slack = max(0., float(max(observed - X @ [quadratic, linear])))
    coefficients = [float(quadratic), float(linear), slack, float(max(speed))]
    ClearanceModel(base.base_bytes, base.cruise_forest, base.actuator, base.side_speed,
                   coefficients).save(a.output)
    scores = []
    for row in trials:
        v = row["cue_speed_mps"]
        physical = row["stopping_distance_m"]
        prediction = quadratic*v*v + linear*v + slack
        scores.append(dict(split=row["split"], cue_speed_mps=v, observed_m=physical,
                           predicted_m=prediction, signed_error_m=prediction-physical))
    digest = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    report = dict(schema=VERSION, stopping_coefficients=coefficients, trials=scores,
                  base_sha256=digest(a.base), calibration_sha256=digest(a.calibration),
                  checkpoint_sha256=digest(a.output), source_sha256=digest(Path(__file__)))
    a.output.with_name("fit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
