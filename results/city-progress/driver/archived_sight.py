"""Measure candidate visibility evidence on the retained v0.5 city range tape, without issuing controls."""
import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

from crosswalk.city_progress import FlankMotion, ProgressModel, _profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--sweeps", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/city-progress/candidate.npz"))
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    replay = a.session / "raw" / "replay.json"
    session = json.loads((a.session / "capture.json").read_bytes())
    source_digest = hashlib.sha256(replay.read_bytes()).hexdigest()
    rows = json.loads(replay.read_bytes())
    model = ProgressModel.load(a.model)
    if len(rows) != session["summary"]["frames"]:
        raise ValueError("Capture length mismatch")
    samples, hist = [], deque(maxlen=3)
    first = int(rows[0]["frame"])
    motion, travel = FlankMotion(), 0.
    for i,r in enumerate(rows):
        frame = int(r["frame"])
        if frame != first+i:
            raise ValueError("Non-contiguous range tape")
        pose = r["ego_pose"]
        travel += r["speed_mps"] * .1
        xy = np.array([pose["x"], pose["y"]])
        yaw = np.deg2rad(pose["yaw"])
        path = (np.array(r["plan_xy"])-xy) @ np.array([[np.cos(yaw), -np.sin(yaw)],[np.sin(yaw), np.cos(yaw)]])
        with np.load(a.sweeps/f"{frame}.npz", allow_pickle=False) as scan:
            if not np.allclose(scan["ego_xy_yaw"], [*xy, pose["yaw"]], atol=.002):
                raise ValueError(f"Range and camera poses disagree at {frame}")
            corridor, crossing, corner, margin = _profile(scan["lidar"], path, r.get("tactical_pass",{}),
                                                        motion=motion, frame=frame, travel=travel)
            radar_points = len(scan["radar"])
        hist.append((corridor, crossing, corner, margin))
        core, cross, end, separation = np.min(hist, axis=0) if len(hist)==3 else [0.]*4
        sight = min(core, max(cross,end*min(1.,separation/1.08)))
        samples.append([r["progress_m"],core,cross,end,separation,sight,
                        model._stoppable_speed(sight-model.standoff), radar_points,r["speed_request_mps"]])
    x=np.asarray(samples)
    turns = session["summary"]["junction_traversals"]
    windows = [(f"junction_{t['junction_id']}_{t['direction']}", t["entry_m"], t["exit_m"]) for t in turns]
    windows.extend([("board_approach",525.,550.),("unobstructed_boulevard",420.,500.),
                    ("bus_approach",835.,858.),("bus_alongside",858.,872.),
                    ("bus_rear_half",858.,865.),("bus_blind_nose",865.,872.),("bus_recovery",872.,882.),
                    ("occluded_pedestrian",130.,152.),("post_lead",645.,735.)])
    def summarize(x):
        return dict(frames=len(x), median_path_sight_m=float(np.median(x[:,1])),
            median_conflict_sight_m=float(np.median(x[:,2])), confirmed_flank_frames=int(sum(x[:,3]>0)),
            nonempty_radar_frames=int(sum(x[:,7]>0)), median_measured_sight_limit_kmh=float(np.median(x[:,6])*3.6),
            median_recorded_v05_request_kmh=float(np.median(x[:,8])*3.6),
            potential_extra_speed_frames=int(sum(x[:,6]>x[:,8]+.25)))
    output=dict(schema="city-progress-archived-visibility-v1", frames=len(x), session=a.session.name,
        replay_sha256=source_digest, checkpoint_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
        interpretation="Measured geometric sight ceiling only, before signals, opposing traffic, vulnerable users, current/recent body, posted/tactical limits and recovery. This is not a prediction or a closed-loop result.",
        complete=summarize(x), windows=[dict(name=name,station_m=[lo,hi],**summarize(x[(x[:,0]>=lo)&(x[:,0]<hi)]))
                                       for name,lo,hi in windows])
    a.out.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
