"""Measure grade geometry over an archived City Life route without issuing controls."""

import argparse
from collections import Counter, deque
import hashlib
from itertools import groupby
import json
from pathlib import Path

import numpy as np

from crosswalk import behavior_v2, city_grade
from crosswalk.policy import _returns


BODY_CUES = [behavior_v2.RANGE_CUES.index(name) for name in ("path_body_lidar", "path_body_radar")]
CONTROL_STATIONS = {
    "occluded_pedestrian_approach": (130., 152.),
    "boards_and_stopped_lead": (525., 550.),
    "departing_lead": (600., 645.),
    "bus_approach": (835., 858.),
    "bus_alongside": (858., 872.),
    "bus_blind_nose": (865., 872.),
    "bus_recovery": (872., 882.),
}
CONTROL_EVENTS = {"pedestrian", "occluded", "pedestrian-delayed", "barrier", "lead", "braking", "bus"}


def estimate(rows, sweeps):
    body_history, grade_history = deque(maxlen=9), deque(maxlen=9)
    previous = None
    for row in rows:
        frame = int(row["frame"])
        if previous is not None and frame != previous + 1:
            raise ValueError(f"Noncontiguous archive at frame {frame}")
        previous = frame
        pose = row["ego_pose"]
        xy = np.array([pose["x"], pose["y"]])
        yaw = np.deg2rad(pose["yaw"])
        c, s = np.cos(yaw), np.sin(yaw)
        plan = (np.asarray(row["plan_xy"]) - xy) @ np.array([[c, -s], [s, c]])
        path = behavior_v2._checked_route(plan[1:4], plan[4:])
        with np.load(sweeps / f"{frame}.npz", allow_pickle=False) as scan:
            if not np.allclose(scan["ego_xy_yaw"], [*xy, pose["yaw"]], atol=.002, rtol=0.):
                raise ValueError(f"Range and route poses disagree at frame {frame}")
            lidar, radar = scan["lidar"], scan["radar"]
            speed = float(row["speed_mps"])
            cues = behavior_v2._range_cues(lidar, radar, speed, path)
            body = np.asarray([cues[i] for i in BODY_CUES], dtype=np.float32)
            distance = float(min(body))
            eligible = distance < 32. and bool(len(_returns(radar)))
            extent = city_grade.supported_grade(lidar, radar, path, speed) if eligible else 0.

        body_history.append(body)
        recent_body = np.mean(body_history, axis=0)
        raw = float(min(*body, *recent_body))
        grade_history.append((extent, distance))
        confirmed = 0.
        if len(grade_history) == 9 and all(proof > 0. or gap >= 24. for proof, gap in grade_history):
            confirmed = float(np.float32(min(proof for proof, _ in list(grade_history)[-3:])))
        yield dict(frame=frame, time_s=row["sim_time"], station_m=row["progress_m"], eligible=eligible,
                   current_body_lidar_m=float(body[0]), current_body_radar_m=float(body[1]),
                   recent_body_lidar_m=float(recent_body[0]), recent_body_radar_m=float(recent_body[1]),
                   raw_body_m=raw, grade_extent_m=float(np.float32(extent)), confirmed_m=confirmed)


def affected(sample):
    return 0. < sample["confirmed_m"] <= 36. and sample["raw_body_m"] < 24.


def summarize(samples):
    eligible = sum(r["eligible"] for r in samples)
    proven = sum(r["grade_extent_m"] > 0. for r in samples)
    return dict(frames=len(samples), geometry_fit_attempts=eligible, fit_gate_closed=len(samples) - eligible,
                fit_positive=proven, fit_unknown_after_attempt=eligible - proven,
                confirmed=sum(r["confirmed_m"] > 0. for r in samples),
                raw_body_below_24_m=sum(r["raw_body_m"] < 24. for r in samples),
                geometry_only_candidates=sum(map(affected, samples)))


def report_windows(samples, windows):
    return {name: {"station_m_inclusive_exclusive": [lo, hi],
                   **summarize([r for r in samples if lo <= r["station_m"] < hi])}
            for name, (lo, hi) in windows.items()}


def attach_context(samples, rows):
    for sample, row in zip(samples, rows, strict=True):
        sample.update(active_event=row["active_event"], phase=row["phase"],
                      recorded_request_kmh=row["speed_request_mps"] * 3.6,
                      measured_speed_kmh=row["speed_mps"] * 3.6)


def contiguous_windows(samples, predicate, *, include_samples=False):
    for selected, group in groupby(samples, key=predicate):
        if selected:
            items = list(group)
            phases = Counter((r["active_event"], r["phase"]) for r in items)
            window = dict(start_frame=items[0]["frame"], end_frame=items[-1]["frame"],
                          time_s=[items[0]["time_s"], items[-1]["time_s"]],
                          station_m=[items[0]["station_m"], items[-1]["station_m"]],
                          **summarize(items),
                          native_context=[dict(active_event=event, phase=phase, frames=count)
                                          for (event, phase), count in phases.items()])
            if include_samples:
                window["samples"] = items
            yield window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--sweeps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay = (args.session / "raw" / "replay.json").read_bytes()
    rows = json.loads(replay)
    if {str(row["frame"]) for row in rows} != {scan.stem for scan in args.sweeps.glob("*.npz")}:
        raise ValueError("Replay and retained range frames differ")
    samples = list(estimate(rows, args.sweeps))
    attach_context(samples, rows)
    output = dict(
        schema="city-progress-city-life-grade-shadow-v1", session=args.session.name,
        sweeps=args.sweeps.name, replay_sha256=hashlib.sha256(replay).hexdigest(),
        detector_source_sha256=hashlib.sha256(Path(city_grade.__file__).read_bytes()).hexdigest(),
        legacy_cues_source_sha256=hashlib.sha256(Path(behavior_v2.__file__).read_bytes()).hexdigest(),
        method="Use the seven archived route waypoints in the measured ego frame with an exact (0,0) origin. "
               "Use native vehicle speed and matching range frames. Float32 body cues and nine-scan means follow "
               "behavior_v2.Observer. Attempt the grade fit when current body gap is below 32 m and radar is "
               "present, with camera acceptance assumed. Apply GradeObserver's last-nine raw-gap veto and "
               "last-three positive rule. Add archived event and phase labels after estimating grade.",
        interpretation="Confirmed pavement with a raw body gap below 24 m is the geometry-only upper bound. "
                       "Camera quality, RGB detections, signals, crossing history, speed caps and other control "
                       "eligibility checks can still force abstention. No speed requests or closed-loop runs were generated.",
        complete=summarize(samples),
        geometry_only_candidate_windows=list(contiguous_windows(samples, affected, include_samples=True)),
        raw_body_below_24_m_windows=list(contiguous_windows(samples, lambda r: r["raw_body_m"] < 24.)),
        negative_controls_by_station=report_windows(samples, CONTROL_STATIONS),
        negative_controls_by_native_event={event: summarize([r for r in samples if r["active_event"] == event])
                                           for event in sorted(CONTROL_EVENTS)},
    )
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["complete"]))


if __name__ == "__main__":
    main()
