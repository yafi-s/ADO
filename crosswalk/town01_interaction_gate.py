"""Select the off-path curb response using independent front-camera and deceleration cues."""
import io
import itertools
import json
import zipfile

import numpy as np

from .behavior_v2 import BehaviorModel, FEATURE_NAMES, HORIZON
from .town01_interaction import BASE, COL, LESSONS, ROOT, InteractionModel, calibrate

VERSION = "town01-yield-gated-v04"
MODEL = ROOT/"models/town01-yield-gated-v04/calibration.json"


def risk_signal(x, plan, height, error):
    person = ((x[..., COL["rgb_person_height"]] >= height)
              & (np.abs(x[..., COL["rgb_person_center_offset"]]) < .55))
    slowing = x[..., COL["ego_speed_mps"]] - plan > error
    return person | slowing


def training_records():
    base = BehaviorModel.load(BASE)
    with zipfile.ZipFile(LESSONS) as z:
        for name in sorted(z.namelist()):
            with np.load(io.BytesIO(z.read(name))) as d:
                if tuple(d["feature_names"]) != FEATURE_NAMES:
                    raise ValueError("Incompatible native training observations")
                x, g = d["X"], d["expert_gap_m"]
                if not (name.startswith(("pedestrian-", "occluded-", "clear-", "turn-"))):
                    continue
                # Project upcoming ground-truth pedestrian occupancy back to the
                # current bumper; use this only to score gate candidates during fitting.
                progress = d["progress_m"]
                nearby = np.full(len(g), 60.)
                if not name.startswith(("clear-", "turn-")):
                    for offset in range(11):
                        j = np.minimum(np.arange(len(g)) + offset, len(g)-1)
                        nearby = np.minimum(nearby, g[j]+progress[j]-progress)
                yield name, x, np.array([base._plan(v) for v in x]), nearby < 13.


def gate_score(records, height, error, hold):
    misses = false = positives = negatives = 0
    for _, x, plan, label in records:
        trigger = risk_signal(x, plan, height, error)
        # Each trigger remains available for hold frames, including its arrival.
        risk = np.convolve(trigger.astype(int), np.ones(hold, dtype=int))[:len(trigger)] > 0
        edge = (x[:, COL["right_edge_lidar"]] < 14.) & (x[:, COL["junction_gap"]] > 24)
        moving = x[:, COL["ego_speed_mps"]] > 1.
        relevant = edge & moving
        positives += int(np.count_nonzero(relevant & label))
        negatives += int(np.count_nonzero(relevant & ~label))
        misses += int(np.count_nonzero(relevant & label & ~risk))
        false += int(np.count_nonzero(relevant & ~label & risk))
    return dict(misses=misses, positives=positives, false=false, negatives=negatives,
                loss=2*misses/max(1,positives) + false/max(1,negatives))


def fit():
    records = list(training_records())
    train = [r for r in records if not r[0].endswith("-48.npz")]
    dev = [r for r in records if r[0].endswith("-48.npz")]
    candidates = itertools.product((.04,.08,.12,.18), (.2,.5,.9,1.4), (4,8,12))
    scored = [(gate_score(train, *parameters), parameters) for parameters in candidates]
    score, parameters = min(scored, key=lambda pair: (pair[0]["loss"], pair[1]))
    height, error, hold = parameters
    config = dict(calibrate(), version=VERSION, gate_person_height=height, gate_speed_error=error,
                  gate_hold_frames=hold, gate_train=[r[0] for r in train], gate_dev=[r[0] for r in dev],
                  gate_train_score=score, gate_dev_score=gate_score(dev,*parameters))
    return config


class GatedInteractionModel(InteractionModel):
    version = VERSION

    @classmethod
    def load(cls, path=MODEL):
        model = super().load(path)
        cfg = model.config
        if not (0 < cfg["gate_person_height"] <= 1 and 0 < cfg["gate_speed_error"] <= 5
                and isinstance(cfg["gate_hold_frames"], int) and 1 <= cfg["gate_hold_frames"] <= 50):
            raise ValueError("Invalid curb-risk calibration")
        return model

    def new_observer(self):
        self.risk_memory = 0
        return super().new_observer()

    def edge_distance(self, x, request):
        if risk_signal(x,request,self.config["gate_person_height"],self.config["gate_speed_error"]):
            self.risk_memory = self.config["gate_hold_frames"]
        enabled = self.risk_memory > 0
        self.risk_memory = max(0, self.risk_memory - 1)
        return super().edge_distance(x, request) if enabled else HORIZON


if __name__ == "__main__":
    MODEL.parent.mkdir(parents=True,exist_ok=True)
    config = fit()
    MODEL.write_text(json.dumps(config,indent=2)+"\n")
    print(json.dumps(config,indent=2))
