"""Matched native Town03 progress and safety trials, with withheld actor geometry for scoring."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

from .city_drive import LoggedDriver, load_driver
from .city_progress_bus import BusCrossing, body_gap, crossing_score, footprint
from .drive import episode
from .experiment import PROJECT, fingerprint, save_arrays, write_json, write_trace
from .journey import JourneyScenario
from .overtake_transfer import EGO_STATION, SOURCE, second_corridor
from .overtake_trials import evaluate
from .roadworks_pace_study import run as run_workzone
from .runtime import producer_info, simulator
from .showcase import encode_and_check
from .world import _raised


# Geographic windows are fixed for both arms. The scored section starts after a run-up.
# Keep the bus approach, alongside, and recovery separate, including cautious segments.
WINDOWS = {
    "through": (620., 744., ((650., 688.), (701., 731.)), None, 621),
    "bus": (741., 909., ((770., 800.), (800., 825.), (825., 855.), (855., 890.),
                         (858., 872.), (872., 882.), (882., 892.)), "bus", 622),
    "occluded": (97., 171., ((124., 148.), (150., 168.)), "occluded", 623),
    "left": (995., 1085., ((1025., 1045.), (1046., 1077.)), "left", 624),
    "signal": (278., 363., ((299., 323.), (324., 352.)), "signal", 625),
    "delayed": (1228., 1333., ((1256., 1278.), (1286., 1325.)), "pedestrian-delayed", 626),
}
WINDOWS["bus-crossing"] = WINDOWS["bus"]
PASS_TRIALS = {"adjacent": "occupied", "adjacent-camera": "occupied", "no-lead": "no-lead",
               "departing-lead": "departing-lead"}
CASES = (*WINDOWS, *PASS_TRIALS, "reopened-board")


class Witness:
    primary_rig_only = True

    def __init__(self, stream, reference, target):
        self.stream, self.reference = stream, reference
        self.target = target
        self.scene = None
        self.closest = None

    def append(self, row, obs):
        score = row["observer_map"]
        measured = dict(frame=row["frame"], sim_time=row["sim_time"], ego=score["ego"],
                        ego_z=score["ego_z"], others=score["others"])
        if self.target:
            actor = next(event.actor for event in self.scene.events if event.kind == self.target)
            ego = self.scene.ego
            if ego.get_location().distance(actor.get_location()) < 40.:
                gap = round(body_gap(footprint(ego), footprint(actor)), 3)
                measured["target_polygon_gap_m"] = gap
                if self.closest is None or gap < self.closest["gap_m"]:
                    self.closest = dict(frame=row["frame"], sim_time=row["sim_time"], gap_m=gap,
                                        ego_kmh=round(row["speed_mps"] * 3.6, 2))
        self.reference.write(json.dumps(measured, separators=(",", ":"), allow_nan=False) + "\n")
        ok, jpeg = cv2.imencode(".jpg", obs["front"][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError(f"Missing front exposure at frame {row['frame']}")
        self.stream.write(jpeg.tobytes())


def interval(trace, a, b):
    rows = [r for r in trace if a <= r["progress_m"] < b]
    return dict(start_m=a, end_m=b, frames=len(rows), seconds=round(.1 * len(rows), 1),
                median_kmh=round(float(np.median([r["speed_mps"] * 3.6 for r in rows])), 2) if rows else None,
                median_request_kmh=round(float(np.median([r["speed_request_mps"] * 3.6 for r in rows])), 2) if rows else None,
                seconds_below_10_kmh=round(.1 * sum(r["speed_mps"] * 3.6 < 10 for r in rows), 1))


def audited_observer(model, folder, case, release_checks):
    """Observe numerical edges and the crossing release gate independently of commands."""
    observer = model.new_observer()
    observe = observer.observe
    saved = False

    def measured(obs, *args, **kwargs):
        nonlocal saved
        scan = obs["lidar"]
        if case == "signal" and not saved:
            from .lidar_motion import CELL, RADIUS
            xy = scan[:, :2]
            # Reproduce the superseded arithmetic exactly; this mask is evaluation-only.
            edge = (np.abs(xy) < RADIUS).all(axis=1) & (scan[:, 2] + 2.3 > .23) & (scan[:, 2] + 2.3 < 3.6)
            edge &= np.any((xy + np.float32(RADIUS)) / np.float32(CELL) >= 2 * RADIUS / CELL, axis=1)
            if np.any(edge):
                save_arrays(folder / "native-upper-edge.npz", frame=obs["progress_frame"], lidar=scan,
                            radar=obs["radar"], ego_xy_yaw=obs["progress_ego_pose"],
                            old_invalid_point_indices=np.flatnonzero(edge))
                saved = True
        result = observe(obs, *args, **kwargs)
        if case == "bus-crossing" and observer.crossing.anchor is not None and observer.crossing.sweeps:
            from .behavior_v2 import _checked_route
            crossing = observer.crossing
            current = crossing.sweeps[-1]
            path = _checked_route(args[1], kwargs.get("route_preview"))
            local = (crossing.anchor - current.origin) @ current.rotation
            frame = int(obs["progress_frame"])
            release_checks[frame] = dict(prediction_wait_remaining_s=round(max(0., crossing.release_after - frame*.1), 3),
                old_conflict_ground_observed=bool(crossing._open_at(scan, local, path)),
                consecutive_clear_scans=crossing.clear_frames)
        return result

    observer.observe = measured
    return observer


def route_trial(world, checkpoint, folder, raw, case):
    start, finish, windows, target, seed = WINDOWS[case]
    driver, names, schema = load_driver(checkpoint)
    model = LoggedDriver(driver)
    folder.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    release_checks = {}
    with (raw / "front.mjpeg").open("xb") as stream, gzip.open(
            folder / "reference.jsonl.gz", "xt", encoding="utf-8") as reference:
        witness = Witness(stream, reference, target)

        def factory(*args, **kwargs):
            scenario = BusCrossing if case == "bus-crossing" else JourneyScenario
            scene = scenario(*args, **kwargs, barriers=True)
            i = int(np.argmin(abs(np.asarray(scene._station) - start)))
            scene.ego.set_transform(_raised(scene._points[i].transform, .35))
            witness.scene = scene
            return scene

        def finished():
            scene = witness.scene
            return scene._coordinates(scene.ego.get_location())[0] >= finish

        summary, trace, _, _ = episode(world, "journey", seed, schema, model, scene_factory=factory,
            destination="market", weather="ClearNoon", crossing_delay=.5, recording=witness, tactical=True,
            change_request=finished, max_steps=700,
            observation_factory=(lambda: audited_observer(model, folder, case, release_checks)) if case in ("signal", "bus-crossing") and
                                getattr(driver, "confirmed_tactical_release", False) else None)
    if case == "bus-crossing":
        for row in trace:
            row.update(witness.scene.walk_rows[row["frame"]])
            if row["frame"] in release_checks:
                row["crossing_release_check"] = release_checks[row["frame"]]
        summary["crossing"] = crossing_score(trace)
    write_trace(folder / "trace.jsonl.gz", trace)
    save_arrays(folder / "observations.npz", frame=[r["frame"] for r in trace], features=model.observations,
                feature_names=names, requested_mps=model.requested_speed, posted_kmh=model.posted_limits)
    events = [e for e in summary["events"] if e["kind"] == target]
    passed = (summary["progress_m"] >= finish and not summary["collision_events"] and
              not summary["yield_violations"] and not summary["red_light_violations"] and
              all(e["handled"] for e in events) and (not target or len(events) == 1) and
              summary.get("crossing", {}).get("passed", True))
    summary.update(case=case, model_sha256=fingerprint(checkpoint), producer=producer_info(world),
                   start_m=start, end_m=finish, target_event=target, target_pass=bool(passed),
                   nearest_target_footprint=witness.closest,
                   windows=[interval(trace, a, b) for a, b in windows],
                   first_frame=trace[0]["frame"], last_frame=trace[-1]["frame"],
                   camera_contract="Frame zero in front.mp4 is first_frame. One exposure per 0.1 s step.")
    write_json(folder / "summary.json", summary)
    encode_and_check(raw / "front.mjpeg", folder / "front.mp4", (640, 360), 0, len(trace))
    print(case, summary["outcome"], passed, summary["windows"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--port", type=int, default=3518)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "checkpoint.npz"
    with args.model.open("rb") as source, checkpoint.open("xb") as copy:
        shutil.copyfileobj(source, copy)
    if fingerprint(args.model) != fingerprint(checkpoint):
        raise RuntimeError("Model changed while freezing the trial")
    snapshot = root / "study-source.zip"
    with ZipFile(snapshot, "x", ZIP_DEFLATED) as archive:
        for path in sorted(Path(__file__).parent.glob("*.py")):
            archive.write(path, f"crosswalk/{path.name}")
    write_json(root / "protocol.json", dict(cases=args.cases, windows=WINDOWS, model_sha256=fingerprint(checkpoint),
               execution_source_sha256=fingerprint(snapshot), quality="High", map="Town03_Opt", step_s=.1))
    with simulator(args.port, quality="High", map_name="Town03_Opt") as world:
        for case in args.cases:
            folder = root / case
            if case in PASS_TRIALS:
                folder.mkdir(parents=True, exist_ok=False)
                corridor, geometry = second_corridor(world.get_map())
                score = evaluate(world, PASS_TRIALS[case], checkpoint, folder, source_lane=SOURCE,
                                 ego_s=EGO_STATION, corridor=corridor, native_rgb=case != "adjacent")
                write_json(folder / "summary.json", dict(case=case, geometry=geometry, score=score,
                    model_sha256=fingerprint(checkpoint), producer=producer_info(world)))
            elif case == "reopened-board":
                run_workzone(world, checkpoint, folder, "blocked", 621)
            else:
                route_trial(world, checkpoint, folder, PROJECT / "out/city-progress-trials" / root.name / case, case)


if __name__ == "__main__":
    main()
