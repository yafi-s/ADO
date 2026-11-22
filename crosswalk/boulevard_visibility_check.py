"""Stress the long-range speed extension with truncated recorded native sweeps."""
import argparse
import copy
import gzip
import json
from pathlib import Path

import numpy as np

from .behavior_v2 import _checked_route
from .city_drive import load_driver
from .city_open_road import long_sight
from .experiment import fingerprint, write_json


def check(trial, sweeps, checkpoint):
    policy, names, _ = load_driver(checkpoint)
    policy.new_observer()
    with np.load(trial/"observations.npz", allow_pickle=False) as saved:
        if tuple(saved["feature_names"]) != tuple(names):
            raise ValueError("Input feature schemas differ")
        features, frames = saved["features"], saved["frame"]
    with gzip.open(trial/"trace.jsonl.gz","rt") as stream:
        rows = [json.loads(line) for line in stream]
    samples = []
    for x, frame, row in zip(features,frames,rows,strict=True):
        assert frame == row["frame"]
        policy.posted_limit_kmh = row["posted_limit_kmh"]
        policy.signed_speed_limit = row.get("signed_approach_cap_mps")
        policy.tactical_speed_limit = row.get("tactical_pass",{}).get("tactical_speed_limit_mps")
        tape = sweeps/f"{frame}.npz"
        branch = copy.deepcopy(policy) if tape.is_file() else None
        policy.predict(x)
        normal = policy.last_speed_request
        if branch is None or normal <= 40/3.6:
            continue
        with np.load(tape,allow_pickle=False) as raw:
            lidar, radar = raw["lidar"], raw["radar"]
            cutoff = 22.
            laser = lidar[np.linalg.norm(lidar[:,:3],axis=1) <= cutoff]
            doppler = radar[radar[:,3] <= cutoff]
            path = np.vstack((_checked_route(raw["local_route"],raw["preview"]),raw["extended"]))
            horizon,body = long_sight(laser,doppler,path,raw["speed_mps"].item())
        restricted = x.copy()
        restricted[names.index("long_sight_m")] = min(x[names.index("long_sight_m")],horizon)
        restricted[names.index("long_body_gap_m")] = min(x[names.index("long_body_gap_m")],body)
        steer,throttle,brake = branch.predict(restricted)
        samples.append(dict(frame=int(frame),station_m=row["progress_m"],actual_recorded_kmh=3.6*row["speed_mps"],
                            original_recorded_request_kmh=3.6*row["speed_request_mps"],
                            full_replay_request_kmh=3.6*normal, shortened_request_kmh=3.6*branch.last_speed_request,
                            original_sight_m=float(x[names.index("long_sight_m")]),
                            shortened_sight_m=float(restricted[names.index("long_sight_m")]),
                            shortened_regime=branch.progress_evidence["speed_regime"], brake=brake,
                            retained_lidar_points=len(laser),retained_radar_points=len(doppler)))
    if not samples:
        raise RuntimeError("No higher-speed proposals on sampled native frames")
    return dict(native_trial=trial.name, recorded_feature_frames=len(rows), sampled_high_proposals=len(samples),
                native_sensor_cutoff_m=22., checkpoint_sha256=fingerprint(checkpoint),
                scope="One-frame policy counterfactual on one-Hz raw samples. Long-range lidar and radar cropped; other features and pre-tick policy history held at recorded conditions. Vehicle motion is not re-simulated.",
                maximum_restricted_request_kmh=max(s["shortened_request_kmh"] for s in samples),
                maximum_input_replay_error_kmh=max(abs(s["full_replay_request_kmh"]-s["original_recorded_request_kmh"]) for s in samples),
                all_withdrew_high_regime=all(s["shortened_regime"]=="urban" and s["shortened_request_kmh"]<=40.
                                              for s in samples), samples=samples)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("trial","sweeps","checkpoint","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    write_json(a.output,check(a.trial,a.sweeps,a.checkpoint))


if __name__=="__main__":
    main()
