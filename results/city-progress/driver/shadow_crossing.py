"""Score the predictive crossing guard on the dense native City Life sensor tape without driving."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from crosswalk import city_crossing
from crosswalk.city_crossing import CrossingObserver
from crosswalk.experiment import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--sweeps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay = args.session / "raw" / "replay.json"
    rows = json.loads(replay.read_bytes())
    guard = CrossingObserver()
    measurements = hashlib.sha256()
    segments = []
    segment = None
    first = rows[0]["frame"]
    for i, row in enumerate(rows):
        frame = row["frame"]
        if frame != first + i:
            raise ValueError("Non-contiguous native frame sequence")
        e = row["ego_pose"]
        pose = np.array([e["x"], e["y"], e["yaw"]])
        yaw = np.deg2rad(pose[2])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        path = (np.asarray(row["plan_xy"]) - pose[:2]) @ rotation
        with np.load(args.sweeps / f"{frame}.npz", allow_pickle=False) as data:
            lidar = data["lidar"]
            if not np.allclose(data["ego_xy_yaw"], pose, atol=.002):
                raise ValueError(f"Pose mismatch at frame {frame}")
            measurements.update(np.int64(frame).tobytes())
            measurements.update(lidar.tobytes())
            values = guard.observe(lidar, pose, frame, row["speed_mps"], path)
        # Scorer annotations are consulted only after the measured control evidence was produced.
        if values[0] < 60.:
            if segment is None:
                segment = dict(start_frame=frame, start_station_m=row["progress_m"],
                               start_phase=row["phase"], minimum_gap_m=60., direct_measurement_frames=0)
            segment["minimum_gap_m"] = min(segment["minimum_gap_m"], float(values[0]))
            segment["direct_measurement_frames"] += int(values[3] < .001)
            segment.update(last_frame=frame, last_station_m=row["progress_m"], last_phase=row["phase"])
        elif segment is not None:
            segments.append(segment)
            segment = None
    if segment is not None:
        segments.append(segment)
    output = dict(schema="city-progress-crossing-shadow-v1", frames=len(rows),
                  source_replay=fingerprint(replay), source_lidar_frame_sequence=measurements.hexdigest(),
                  guard_source=fingerprint(Path(city_crossing.__file__)),
                  flagged_frames=sum(s["last_frame"]-s["start_frame"]+1 for s in segments),
                  interpretation="Open-loop checks on the tagged driver's measured route and odometry. "
                                 "The ego continues moving on its recorded schedule. A changed trajectory may encounter different traffic.",
                  windows=segments)
    args.out.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
