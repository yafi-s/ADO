"""Fit a four-action maneuver tree to simulated stay/merge counterfactuals.

Each row is a separate traffic realization. No CARLA evaluation fixture,
actor identities or scores enter the fit. Inference only needs NumPy.
"""
import argparse
import json
from pathlib import Path

import numpy as np


MODEL = Path(__file__).resolve().parents[1] / "models/route-maneuvers/choice.json"
FEATURES = ("source_gap_m", "source_speed_mps", "target_front_m", "target_rear_m",
            "target_front_closing_mps", "target_rear_closing_mps", "ego_speed_mps")
ACTIONS = ("keep", "follow", "yield", "pass")


def traffic(seed, n):
    rng = np.random.default_rng(seed)
    lead = rng.uniform(4., 51., n)
    lead[rng.random(n) < .18] = 60.
    lv = rng.uniform(0., 13., n)
    lv[rng.random(n) < .24] = 0.
    front, rear = rng.uniform(1., 57., (2, n))
    front[rng.random(n) < .48] = 60.
    rear[rng.random(n) < .48] = 60.
    ev = rng.uniform(0., 11., n)
    closing_front = np.where(front < 60., ev-rng.uniform(0., 14., n), 0.)
    closing_rear = np.where(rear < 60., rng.uniform(0., 17., n)-ev, 0.)
    # Compare distance after a lane change to comfortable progress while following.
    horizon, cruise = 6., 8.
    follow_travel = np.minimum(cruise*horizon, np.maximum(0., lead-5. + lv*(horizon-1.2)))
    worthwhile = (cruise*horizon-follow_travel > 7.) & (lv < 4.3)
    front_at_merge = np.where(front < 60., front-3.5*np.maximum(closing_front, 0.), 60.)
    rear_at_merge = np.where(rear < 60., rear-4.*np.maximum(closing_rear, 0.), 60.)
    safe = (front_at_merge > 14.) & (rear_at_merge > 12.)
    labels = np.where(lead >= 59., 0, np.where(~worthwhile, 1, np.where(safe, 3, 2)))
    return np.column_stack((lead, lv, front, rear, closing_front, closing_rear, ev)), labels


def fit(x, y):
    # Bounded CART using Gini gain over per-node quantile split candidates.
    tree = []
    def node(indices, depth):
        idx = len(tree)
        labels = y[indices]
        counts = np.bincount(labels, minlength=4)
        tree.append({"prob": (counts / len(labels)).round(6).tolist()})
        if depth >= 11 or np.count_nonzero(counts) == 1 or len(indices) < 50:
            return idx
        best = None
        for j in range(x.shape[1]):
            order = np.argsort(x[indices, j], kind="stable")
            values = x[indices[order], j]
            cuts = np.flatnonzero(np.diff(values) > 1e-5) + 1
            cuts = cuts[(cuts >= 18) & (len(indices)-cuts >= 18)]
            if not len(cuts):
                continue
            sizes = cuts[:, None]
            steps = np.stack([np.cumsum(labels[order] == k) for k in range(4)], axis=1)[cuts-1]
            scores = ((steps**2/sizes) + ((counts-steps)**2/(len(indices)-sizes))).sum(axis=1)
            top = int(np.argmax(scores))
            cut = cuts[top]
            if best is None or scores[top] > best[0]:
                best = (scores[top], j, float((values[cut-1]+values[cut])/2), order, cut)
        if best is not None:
            _, feature, threshold, order, cut = best
            tree[idx].update(feature=feature, threshold=threshold,
                             left=node(indices[order[:cut]], depth+1),
                             right=node(indices[order[cut:]], depth+1))
        return idx
    node(np.arange(len(y)), 0)
    return tree


class ManeuverChoice:
    def __init__(self, path=MODEL, *, tree=None):
        if tree is None:
            data = json.loads(path.read_bytes())
            if data["format"] != "ado-route-maneuvers-1" or data["features"] != list(FEATURES) or data["actions"] != list(ACTIONS):
                raise ValueError("Incompatible route maneuver checkpoint")
            tree = data["tree"]
        self.tree = tree

    def predict(self, x):
        n = self.tree[0]
        while "feature" in n:
            n = self.tree[n["left"] if x[n["feature"]] <= n["threshold"] else n["right"]]
        return ACTIONS[int(np.argmax(n["prob"]))], n["prob"]


def train(path):
    seed, heldout_seed = 71105, 81919
    x, y = traffic(seed, 56000)
    tree = fit(x, y)
    model = ManeuverChoice(tree=tree)
    test, truth = traffic(heldout_seed, 16000)
    pred = [ACTIONS.index(model.predict(row)[0]) for row in test]
    confusion = np.zeros((4,4), dtype=int)
    np.add.at(confusion, (truth, pred), 1)
    result = dict(format="ado-route-maneuvers-1", features=FEATURES, actions=ACTIONS, tree=tree,
                  training={"source":"synthetic independent traffic rollouts", "seed":seed, "n":len(x)},
                  holdout={"source":"synthetic independent traffic rollouts", "seed":heldout_seed,
                           "n":len(test), "confusion_truth_by_prediction":confusion.tolist(),
                           "accuracy":float(np.mean(truth==pred))})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, separators=(",", ":"))+"\n", encoding="utf8")
    print(result["holdout"])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=MODEL)
    train(cli.parse_args().output)
