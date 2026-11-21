"""Fit the tactical pass classifier to synthetic longitudinal traffic counterfactuals.

The teacher compares projected two-lane traffic gaps while the ego would
merge. Actor kinematics exist only in this generator. Deployment takes the
four measured LiDAR features in overtake.py and loads the fitted tree.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .overtake import MODEL, FEATURES, ACTIONS


def sample(rng, count):
    # Independent lane occupancy, including no source obstruction or empty neighbors.
    lead = rng.uniform(4, 58, count)
    lead_speed = rng.uniform(0, 16, count)
    has_lead = rng.random(count) >= .27
    lead[~has_lead] = 60.
    front = rng.uniform(0., 59., count)
    rear = rng.uniform(0., 59., count)
    front[rng.random(count) < .43] = 60.
    rear[rng.random(count) < .43] = 60.
    # Counterfactual teacher: project the distance a lane follower can make
    # with a comfortable following headway, against the progress and effort of
    # a legal pass. Neighbor speed is uncertain from a single occupied slice.
    cruise = 9.5
    horizon = 6.4
    ideal_travel = cruise * horizon
    following_room = lead + horizon * lead_speed - (3.4 + 1.1 * lead_speed)
    kept_travel = np.where(has_lead, np.clip(following_room,0.,ideal_travel), ideal_travel)
    pass_gain = ideal_travel - kept_travel - 5.  # maneuver effort, metres-equivalent

    # Evaluate overlapping vehicle envelopes once lateral ingress starts.
    # Each rollout draws an independent adjacent traffic realization. A
    # virtual near collision outweighs a small gain in destination progress.
    risk = np.zeros(count)
    for _ in range(15):
        front_speed = rng.uniform(2.,14.,count)
        rear_speed = rng.uniform(7.,18.,count)
        clearance = np.full(count,60.)
        for time in (1.,2.,3.,4.):
            position = cruise * time
            ahead = np.where(front < 60., front + time*front_speed-position, 60.)
            behind = np.where(rear < 60., rear + position-time*rear_speed, 60.)
            clearance = np.minimum(clearance,np.minimum(ahead,behind))
        risk += np.clip((6.-clearance)/3.,0.,1.)
    pass_utility = pass_gain - 400.*risk/15.
    action = np.where(pass_gain <= 0., 0, np.where(pass_utility > 0., 2, 1))
    observed = np.column_stack((lead, lead_speed, front, rear))
    noise = rng.normal(0, (.45, .3, .35, .35), observed.shape)
    return np.clip(observed + noise, 0., 60.), action


def train(path):
    rng = np.random.default_rng(3029)
    x, y = sample(rng, 36000)
    tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=20, random_state=3029).fit(x, y)
    test, labels = sample(np.random.default_rng(4113), 9000)
    measured = tree.predict(test)
    fitted = tree.tree_
    doc = dict(format='ado-tactical-pass-v1', features=FEATURES, actions=ACTIONS,
               training='synthetic counterfactual stay/merge rollouts with privileged longitudinal kinematics, progress, maneuver effort, uncertain adjacent speeds and near-collision costs',
               training_examples=len(x), synthetic_holdout_count=len(test),
               synthetic_holdout_confusion=[[int(np.sum((labels==i) & (measured==j))) for j in range(3)]
                                            for i in range(3)],
               left=fitted.children_left.tolist(), right=fitted.children_right.tolist(),
               feature=fitted.feature.tolist(), threshold=np.round(fitted.threshold, 6).tolist(),
               probability=np.round(fitted.value[:,0,:]/fitted.value[:,0,:].sum(axis=1,keepdims=True),5).tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(',', ':'))+'\n', encoding='utf8')
    print(f'{len(x)} synthetic training cases, {100*np.mean(measured==labels):.2f}% synthetic held-out accuracy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=MODEL)
    train(parser.parse_args().output)
