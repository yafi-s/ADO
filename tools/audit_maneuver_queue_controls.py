"""Inventory retained native source-vehicle motion labels for offline evaluation.

Reads telemetry and optional simulator reference traces. Evaluator filters and
reference fields must never enter inference. This script does not fit a model.
"""
from contextlib import ExitStack
import gzip
import json
from math import cos, radians, sin
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "city-maneuvers/passing/queue-preference-research/native-motion-controls.json"
FOLDERS = (
    "city-maneuvers/speed/final-primary-queue",
    "city-maneuvers/speed/final-long-queue",
    "city-maneuvers/trials/candidate/departing",
    "city-maneuvers/trials/candidate/slow",
    "city-maneuvers/trials/candidate/blocked",
    "city-maneuvers/passing/road65-pinned-r2/clear",
    "city-maneuvers/passing/road65-pinned-r2/occupied",
    "city-maneuvers/passing/road65-pinned-r2/rebrake",
    "city-maneuvers/qualification/market-loop/tail-continuous",
    "city-maneuvers/qualification/market-loop/tail-later-gap",
    "city-maneuvers/qualification/market-loop-9145/prefix-continuous",
)
JOURNEYS = (
    "city-cruise/seed-211",
    "city-progress/generalization/city-213-nz3y7mb4",
)


def road_lane(native):
    if isinstance(native, dict):
        return [native["road_id"], native["lane_id"]]
    return native[:2] if native else None


def front_reference(telemetry, reference):
    if reference is None:
        # These fixed-route fixtures record the source lead and a signed gap.
        gap = telemetry.get("lead_gap_m", telemetry.get("lead_bumper_longitudinal_gap_m"))
        if gap > 0 and telemetry.get("whole_ego_in_source", True):
            yield "lead", telemetry["lead_speed_mps"], telemetry["lead_body_gap_m"]
        return
    source_lane = telemetry["tactical_pass"]["source_lane"]
    ego_native = road_lane(reference.get("ego_native_lane", reference.get("ego_native")))
    if ego_native != source_lane:
        return
    if "ego_pose" in reference:
        ego = reference["ego_pose"]
        x, y, yaw = ego["x"], ego["y"], ego["yaw"]
    else:
        x, y, yaw = reference["ego"]
    direction = cos(radians(yaw)), sin(radians(yaw))
    for actor in reference.get("targets", reference.get("target_polygons", [])):
        if actor["role"] not in ("lead", "slow-traffic", "braking"):
            continue
        if road_lane(actor.get("native", actor.get("native_lane"))) != source_lane:
            continue
        if min((px-x)*direction[0] + (py-y)*direction[1] for px, py in actor["polygon_xy"]) <= 0:
            continue
        yield actor["role"], actor["speed_mps"], actor["body_gap_m"]


def count_motion(frames):
    if not frames:
        return {"frames": 0}
    longest = run = 0
    previous = None
    for frame, _ in frames:
        run = run + 1 if frame == previous else 1
        longest = max(run, longest)
        previous = frame + 1
    return dict(frames=len(frames), longest_consecutive_s=round(longest*.1, 1),
                first_frame=frames[0][0], last_frame=frames[-1][0],
                first_sim_time_s=frames[0][1], last_sim_time_s=frames[-1][1])


def inspect_trace(folder):
    root = RESULTS / folder
    roles = {}
    frames = 0
    with ExitStack() as files:
        trace = files.enter_context(gzip.open(root / "trace.jsonl.gz", "rt"))
        ref_path = root / "reference.jsonl.gz"
        ref = files.enter_context(gzip.open(ref_path, "rt")) if ref_path.exists() else None
        for line in trace:
            row = json.loads(line)
            reference = json.loads(next(ref)) if ref else None
            if reference:
                assert reference["frame"] == row["frame"], folder
            frames += 1
            seed = row["seed"]
            tactical = row["tactical_pass"]
            if tactical.get("source_age_scans") != 0 or abs(tactical["source_lateral_m"]) >= .55:
                continue
            raw = tactical["raw_source_gap_m"]
            if raw is None or raw >= 59:
                continue
            matches = [a for a in front_reference(row, reference) if abs(a[2] - (raw-2.5)) <= 2]
            if len(matches) != 1:
                continue
            role, speed, gap = matches[0]
            groups = roles.setdefault(role, {key: [] for key in
                ("all", "stationary", "close_settled_stationary", "sustained_speed_range", "above_5_mps")})
            item = row["frame"], row["sim_time"]
            groups["all"].append(item)
            if speed < .25:
                groups["stationary"].append(item)
                if row["speed_mps"] < .5 and gap < 15:
                    groups["close_settled_stationary"].append(item)
            if .8 <= speed <= 3.5:
                groups["sustained_speed_range"].append(item)
            if speed > 5:
                groups["above_5_mps"].append(item)
        if ref:
            assert next(ref, None) is None, folder
    return dict(folder=folder, seed=seed, frames=frames, matched_roles={
        role: {key: count_motion(items) for key, items in counters.items()}
        for role, counters in roles.items()})


def journey_events(folder):
    summary = json.loads((RESULTS / folder / "summary.json").read_text())
    events = []
    for event in summary["events"]:
        if event["kind"] not in ("lead", "braking"):
            continue
        events.append({k: event[k] for k in ("kind", "exposed_s", "lead_stopped_s",
                                            "release_command_s", "restarted_s") if k in event})
    return dict(folder=folder, seed=summary["seed"], events=events)


def main():
    report = dict(
        purpose="Reference-only coverage audit. Zero frames means no matching observations under this filter.",
        clock="Native simulator 10 Hz. Consecutive spans count observed frames including the first.",
        restrictions=[
            "Fresh source-lidar measurement (age zero, raw below 59 m), within 0.55 m of source lane center; exactly one forward source actor surface whose body gap matches raw lidar minus the nominal 2.5 m ego front offset within 2 m. In joined reference traces both native lanes must equal the planned source and the full target polygon must be forward of the ego origin.",
            "These proximity and role checks are evaluator-only correspondence, not production semantic association. They can omit valid observations on curves. Source tracking without a semantic witness cannot prove an identified passenger vehicle.",
            "Stationary means physical lead below 0.25 m/s. Close settled also requires physical gap below 15 m and ego below 0.5 m/s. The sustained-speed range (0.8 to 3.5 m/s inclusive) counts individual samples; inspect consecutive spans for duration. Above 5 m/s is a moving control, not a departure event.",
            "Seed 316 and 7401 repeats share scripts in their respective groups. Published candidate/baseline arms and related reruns of seed 212 are correlated. Hold out complete seed groups. All market-loop captures in this list report seed 212, including the folder named 9145.",
            "Boulevard seed 316 blocks both lanes and terminates after a 2 s settled stop. It cannot label the optimal time for an open-lane pass. Seed 7401 rebrake also blocks the adjacent lane. Native closed-loop counterfactual gain for a long stationary passenger vehicle in an open lane is not measured by these controls.",
            "City journeys 211 and 213 carry evaluator-timed exposures and physical restart (at least 1 m/s and 4 m displaced). Both use the same script: release after 4 s on exposure for a stopped lead or 3 s after a braking lead stops. The independent seed alone does not make these independent stop-duration distributions.",
        ],
        motion_controls=[inspect_trace(f) for f in FOLDERS],
        other_restart_evidence=[journey_events(f) for f in JOURNEYS],
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
