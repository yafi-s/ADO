"""Swap genuine synchronized CARLA camera triplets in the frozen E2E predictor."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .portable import TrajectoryPolicy
from .simulator_transfer import track_path


def read_trial(folder):
    rows = json.loads((folder / "trace.json").read_bytes())
    replay = json.loads((folder / "replay/replay.json").read_bytes())
    if [r["frame"] for r in rows] != [r["frame"] for r in replay]:
        raise ValueError("Video telemetry is not aligned to the decision trace")
    views = []
    for name in ("front", "left", "right"):
        capture = cv2.VideoCapture(str(folder / "replay" / (name + ".mp4")))
        try:
            if not capture.isOpened() or capture.get(cv2.CAP_PROP_FPS) != 10.:
                raise ValueError("Expected a 10 Hz CARLA video")
            images = []
            for _ in rows:
                ok, bgr = capture.read()
                if not ok:
                    raise ValueError("Missing camera frame")
                small = cv2.resize(bgr, (96, 64), interpolation=cv2.INTER_AREA)
                images.append(small[:, :, ::-1])
            if capture.read()[0]:
                raise ValueError("Extra camera frame")
        finally:
            capture.release()
        views.append(images)
    return rows, np.stack(views, axis=1)


def compare(root, checkpoint):
    report = json.loads((root / "report.json").read_bytes())
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if sha != report["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from original transfer run")
    model = TrajectoryPolicy.load(checkpoint)
    clear_rows, clear = read_trial(root / "clear-seed41-start3")
    results = {}
    for case in ("lead", "pedestrian"):
        rows, rgb = read_trial(root / (case + "-seed41-start3"))
        if [r["sim_time"] for r in rows] != [r["sim_time"] for r in clear_rows]:
            raise ValueError("Mismatched clear-road and hazard simulation times")
        effects, speed_effects, encode_errors = [], [], []
        pose_errors = []
        for i, row in enumerate(rows):
            if not row["applied_for_next_tick"]:
                continue
            hist, intent = row["history_input"], row["intent"]
            observed = model.predict(rgb[i], hist, intent)
            swapped = model.predict(clear[i], hist, intent)
            effects.append(np.linalg.norm(observed - swapped, axis=1))
            speed = row["speed_mps"]
            target = track_path(observed, speed)["target_speed_mps"]
            speed_effects.append(target - track_path(swapped, speed)["target_speed_mps"])
            encode_errors.append(abs(target - row["target_speed_mps"]))
            pose_errors.append(np.linalg.norm(np.subtract(row["rear_axle_world_xy_m"],
                                                          clear_rows[i]["rear_axle_world_xy_m"])))
        results[case] = dict(applied_decisions=len(effects),
                             mean_waypoint_change_m=float(np.mean(effects)),
                             mean_abs_target_speed_change_mps=float(np.mean(np.abs(speed_effects))),
                             max_abs_target_speed_change_mps=float(np.max(np.abs(speed_effects))),
                             mean_hazard_minus_clear_target_mps=float(np.mean(speed_effects)),
                             mean_abs_decoded_vs_original_target_mps=float(np.mean(encode_errors)),
                             maximum_ego_position_difference_m=float(np.max(pose_errors)))
    return dict(checkpoint_sha256=sha, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                producer=report["producer"], seed=41, rolling_handoff_mps=3.,
                terms="Waymo Dataset License Agreement for Non-Commercial Use; see WAYMO_DATASET_AGREEMENT.txt",
                results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = compare(args.archive, args.checkpoint)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
