"""Generate the Town01 evidence report from the finished experiment artifacts."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from math import isclose
from pathlib import Path
from statistics import median


OUTCOMES = ("complete", "collision", "red_light", "lane_departure", "timeout")
EVENT_CASES = {"pedestrian", "pedestrian-delayed", "lead", "braking", "occluded", "signal", "left", "trip"}
COHORTS = ("primary", "camera_ablation", "supplemental")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def clean(trial):
    return (
        trial["outcome"] == "complete"
        and not trial["collision_events"]
        and not trial["red_light_violations"]
        and not trial["yield_violations"]
    )


def trial_key(trial):
    return trial["seed"], trial["case_id"]


def trace_index(path, trials):
    by_id = {trial["trial_id"]: trial for trial in trials}
    counts = Counter()
    for trial_id in by_id:
        counts[trial_id] = 0
    edges = {trial_id: {} for trial_id in by_id}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            trial_id = row["trial_id"]
            if trial_id not in by_id:
                raise ValueError(f"Unexpected trace trial {trial_id}")
            counts[trial_id] += 1
            for phase in ("event_triggered", "event_cleared"):
                if row[phase]:
                    edges[trial_id].setdefault(phase, row["sim_time"])
    expected = {trial_id: trial["frames"] for trial_id, trial in by_id.items()}
    if counts != expected:
        raise ValueError("Trace row counts differ from the scored trial frame counts")
    return edges


def id_set(trials, field):
    return {item for trial in trials for item in trial[field]}


def ids(values):
    return ", ".join(map(str, sorted(values))) or "∅"


def bins(rows):
    return ", ".join(f"`{key}` ({count})" for key, count in sorted(rows.items())) or "none"


def cohort_table(lines, primary, cases):
    groups = defaultdict(list)
    for trial in primary:
        groups[trial["case_id"]].append(trial)
    lines.extend([
        "| Primary case | Trials | Route reached | Clean finishes | Crashes | Red-light outcomes | Departures | Timeouts | Yield breaches | Triggered / cleared | Route layouts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for case, rows in [*((case, groups[case]) for case in cases), ("**Total**", primary)]:
        outcomes = Counter(trial["outcome"] for trial in rows)
        yielded = sum(trial["yield_violations"] > 0 for trial in rows)
        relevant = [trial for trial in rows if trial["case_id"] in EVENT_CASES]
        trigger = sum(trial["event_triggered"] for trial in relevant)
        cleared = sum(trial["event_cleared"] for trial in relevant)
        event_counts = f"{trigger}/{len(relevant)} / {cleared}/{len(relevant)}" if relevant else "—"
        values = [
            case, len(rows), outcomes["complete"], sum(map(clean, rows)), outcomes["collision"],
            outcomes["red_light"], outcomes["lane_departure"], outcomes["timeout"], yielded,
            event_counts, len({trial["route_id"] for trial in rows}),
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")


def matching_setup(first, second):
    fields = ("route_id", "road_ids", "junction_ids", "route_length_m", "destination", "weather",
              "spawn_lateral_m", "spawn_yaw_deg")
    return all(first[field] == second[field] for field in fields)


def ablation(lines, fusion, camera):
    by_case = defaultdict(list)
    mismatches = []
    for trial in camera:
        baseline = fusion[trial_key(trial)]
        if not matching_setup(baseline, trial) or baseline["crossing_delay_s"] != trial["crossing_delay_s"]:
            mismatches.append(trial["trial_id"])
            continue
        by_case[trial["case_id"]].append((baseline, trial))
    pairs = [pair for group in by_case.values() for pair in group]
    lines.extend([
        "", "## Paired camera comparison", "",
        f"Camera and fusion trials were joined by the exact seed and case ID. {len(pairs)}/{len(camera)} trials "
        "also matched the recorded route, destination, weather, initial pose perturbations and crossing delay. "
        "Actions can change when the distance-triggered encounters start. Counts summarize the sampled pairs.",
        "", "| Case | Pairs | Both clean | Fusion only clean | Camera only clean | Neither clean | Fusion / camera crashes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for case, rows in [*by_case.items(), ("**Total**", pairs)]:
        by_finish = Counter((clean(a), clean(b)) for a, b in rows)
        crashes = [sum(trial["outcome"] == "collision" for trial in policy) for policy in zip(*rows)]
        values = [case, len(rows), by_finish[True, True], by_finish[True, False],
                  by_finish[False, True], by_finish[False, False], " / ".join(map(str, crashes or [0, 0]))]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    if mismatches:
        lines.extend(["", "Setup differed for these camera trials: " + ", ".join(f"`{key}`" for key in mismatches) + "."])
    lines.extend(["", "[Camera cohort](../results/camera_ablation.json) · [Primary cohort](../results/primary.json)"])


def trip_pair(lines, fusion, supplemental, edges):
    def time(trial, field):
        value = edges[trial["trial_id"]].get(field)
        return f"{value:.1f} s" if value is not None else "—"

    lines.extend(["", "## City trips and crossing timing", ""])
    trips = defaultdict(list)
    for trial in fusion.values():
        if trial["case_id"] == "trip":
            trips[trial["destination"]].append(trial)
    for destination, rows in sorted(trips.items()):
        lengths = [trial["route_length_m"] for trial in rows]
        roads = sorted({tuple(trial["road_ids"]) for trial in rows})
        junctions = sorted({tuple(trial["junction_ids"]) for trial in rows})
        street_sequences = " / ".join(" → ".join(map(str, route)) for route in roads)
        intersection_sequences = " / ".join(" → ".join(map(str, route)) for route in junctions)
        distance = f"{min(lengths):.1f} m"
        if min(lengths) != max(lengths):
            distance = f"{min(lengths):.1f}–{max(lengths):.1f} m"
        lines.append(
            f"- **{destination.title()}** had {sum(map(clean, rows))}/{len(rows)} clean finishes on "
            f"{distance} routes. Planned road sequence `{street_sequences}`, junction sequence "
            f"`{intersection_sequences}`."
        )
    if trips:
        lines.append("")
    for changed in supplemental:
        original_case = changed["case_id"].removesuffix("-delayed")
        baseline = fusion[changed["seed"], original_case]
        lines.extend([
            f"Seed {changed['seed']} uses the {changed['destination'].title()} destination. Recorded setup "
            f"{'matches across both trials' if matching_setup(baseline, changed) else 'differs across the trials'}. "
            "Crossing start and clearance are the first scored frames where the respective trace flags are set.",
            "",
            "| Trial ID | Route ID | Configured delay | Crossing start | Crossing cleared | Outcome | Route progress |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: |",
        ])
        for trial in (baseline, changed):
            lines.append(
                f"| `{trial['trial_id']}` | `{trial['route_id']}` | {trial['crossing_delay_s']:g} s | "
                f"{time(trial, 'event_triggered')} | {time(trial, 'event_cleared')} | {trial['outcome']} | "
                f"{trial['progress_m']:.1f}/{trial['route_length_m']:.1f} m |"
            )
        lines.append("")
    lines.append("[Supplemental trial](../results/supplemental.json) · [Per-tick traces](../results/traces.jsonl.gz)")


def failure_examples(lines, trials):
    lines.extend(["", "## Observed failures", ""])
    selected = {}
    for trial in trials:
        if clean(trial):
            continue
        key = trial["outcome"]
        if key == "complete":
            key = "completion with a violation"
        elif key == "collision":
            key = "collision with " + (", ".join(trial["contacts"]) or "an unidentified actor")
        selected.setdefault(key, trial)
    if not selected:
        lines.append("Every scored trial in the three cohorts met the clean finish definition.")
    for key, trial in selected.items():
        command = (
            f"python -m crosswalk.drive --scenario {trial['scenario']} --seed {trial['seed']} "
            f"--policy {trial['policy']}"
        )
        if trial["destination"]:
            command += f" --destination {trial['destination']}"
        if trial["crossing_delay_s"]:
            command += f" --crossing-delay {trial['crossing_delay_s']:g}"
        lines.append(
            f"- `{trial['trial_id']}` ended with {key.replace('_', ' ')} at "
            f"{trial['progress_m']:.1f}/{trial['route_length_m']:.1f} m after {trial['duration_s']:.1f} "
            f"simulated seconds. Rerun the settings with `{command}`."
        )
    lines.extend(["", "[All scored trial summaries](../results/episodes.json) · [Per-tick traces keyed by trial ID](../results/traces.jsonl.gz)"])


def traction_section(lines, result):
    trials = result["trials"]
    parameter = result.get("friction_parameter", "tire_friction")
    carla_1 = parameter == "WheelPhysicsControl.friction_force_multiplier"
    physics = result.get("physics_engine", "PhysX")
    reference = ("https://github.com/carla-simulator/carla/blob/1.0/Docs/tuto_vehicle_physics.md"
                 if carla_1 else "https://carla.readthedocs.io/en/0.9.16/tuto_G_control_vehicle_physics/")
    lines.extend([
        "", "## Separate tire-friction trials", "",
        f"{len(trials)} straight-line trials commanded full braking from near {result['target_speed_mps']:g} m/s "
        f"in {result['simulator']} on {result['map']}. The trials scaled the {physics} wheel field "
        f"`{parameter}` relative to stock. The stopping distance is the along-route displacement when speed first reaches "
        f"{result['stop_speed_mps']:g} m/s or below. CARLA samples the threshold at {result['timestep_s']:g} s ticks.",
        "", "| Stock friction multiplier | Stops / trials | Median distance | Distance range | Median time |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for factor in result["friction_scales"]:
        group = [trial for trial in trials if trial["friction_scale"] == factor]
        stopped = [trial for trial in group if trial["outcome"] == "stopped"]
        distances = [trial["stopping_distance_m"] for trial in stopped]
        durations = [trial["stopping_duration_s"] for trial in stopped]
        measurements = (
            f"{median(distances):.2f} m | {min(distances):.2f}–{max(distances):.2f} m | {median(durations):.2f} s"
            if stopped else "— | — | —"
        )
        lines.append(f"| {factor:g}× | {len(stopped)}/{len(group)} | {measurements} |")
    speeds = [trial["initial_speed_mps"] for trial in trials if trial["initial_speed_mps"] is not None]
    verified = 0
    for trial in trials:
        actual, requested = trial["final_tire_friction"], trial["requested_tire_friction"]
        if actual and len(actual) == len(requested):
            verified += all(isclose(a, b, rel_tol=1e-5, abs_tol=1e-6) for a, b in zip(actual, requested))
    lines.append("")
    if speeds:
        lines.append(f"Brake-entry speeds ranged from {min(speeds):.2f} to {max(speeds):.2f} m/s.")
    lines.extend([
        f"Final wheel friction readback matched the requested setting in {verified}/{len(trials)} trials. "
        f"{sum(trial['matches_stock_setup'] for trial in trials)}/{len(trials)} trials matched the stock setup for their seed. "
        "These simulator settings describe one vehicle on straight routes.",
        "", f"[Tire-friction trial data](../results/traction.json) · [CARLA {physics} vehicle physics controls]({reference})",
    ])


def development_section(lines, results):
    baseline_path = results / "development-baseline" / "episodes.json"
    candidate_path = results / "dev" / "episodes.json"
    if not baseline_path.is_file() or not candidate_path.is_file():
        return

    baseline, candidate = (read_json(path) for path in (baseline_path, candidate_path))
    before = {(*trial_key(trial), trial["policy"]): trial for trial in baseline}
    after = {(*trial_key(trial), trial["policy"]): trial for trial in candidate}
    pair_fields = ("district", "crossing_delay_s", "producer", "render_profile")
    pairs = [
        (before[key], after[key]) for key in sorted(before.keys() & after.keys())
        if matching_setup(before[key], after[key])
        and all(before[key][field] == after[key][field] for field in pair_fields)
    ]
    if not pairs:
        return
    seeds = {initial["seed"] for initial, _ in pairs}
    lines.extend([
        "", "## Development comparison", "",
        f"The initial and updated checkpoints matched on {len(pairs)} drives by seed, exact case ID, policy "
        f"and recorded setup. Development used seed(s) {ids(seeds)} in the training district for iteration "
        "ahead of the held-out evaluation.",
        "", "| Policy | Paired drives | Clean finishes before → after | Timeouts before → after | Contact trials before → after |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for policy in ("fusion", "camera"):
        group = [(initial, updated) for initial, updated in pairs if initial["policy"] == policy]
        before_rows, after_rows = zip(*group) if group else ((), ())
        stats = [
            (sum(map(clean, rows)), sum(row["outcome"] == "timeout" for row in rows),
             sum(row["collision_events"] > 0 for row in rows))
            for rows in (before_rows, after_rows)
        ]
        comparison = " | ".join(f"{a} → {b}" for a, b in zip(*stats))
        lines.append(f"| {policy.title()} | {len(group)} | {comparison} |")
    if len(pairs) != len(baseline) or len(pairs) != len(candidate):
        lines.extend(["", f"The table covers {len(pairs)}/{len(baseline)} baseline records and "
                      f"{len(pairs)}/{len(candidate)} updated records that matched the paired setup."])
    lines.extend([
        "", "[Initial development drives](../results/development-baseline/episodes.json) "
        "· [Updated development drives](../results/dev/episodes.json) "
        "· [Updated development protocol and model hashes](../results/dev/manifest.json)",
    ])


def training_section(lines, collection, fits, manifest):
    lines.extend([
        "", "## Demonstrations and offline checks", "",
        f"{len(collection)} Traffic Manager demonstration drives covered {len({r['seed'] for r in collection})} seeds. "
        f"The collection outcomes were {bins(Counter(row['outcome'] for row in collection))}.",
        "",
    ])
    for policy, fit in fits.items():
        validation = fit.get("development_holdout_before_final_refit")
        diagnostic = ""
        if validation:
            errors = ", ".join(f"{action} {stats['mae']:.3f}" for action, stats in validation["actuators"].items())
            diagnostic = (
                f" The pre-refit comparison on {validation['frames']:,} demonstration frames from group(s) "
                f"{ids(validation['held_groups'])} measured mean absolute actuator errors of {errors}."
            )
        lines.append(
            f"- **{policy.title()}** trained on {fit['final_model_fit_frames']:,} synchronized demonstration rows"
            f" with {fit['features']} input features.{diagnostic} [Fit log](../results/fit-{policy}.json)"
        )
    if all(fit.get("pedal_training") == "local_cues" for fit in fits.values()):
        pedal_inputs = {policy: len(fit["head_features"]["drive"]) for policy, fit in fits.items()}
        lines.extend([
            "", f"The compact pedal heads used {pedal_inputs['fusion']} fusion inputs and "
            f"{pedal_inputs['camera']} camera inputs.",
        ])
    specialist = manifest.get("signal_specialist")
    if specialist:
        lines.extend([
            "", f"Both policies use the same mapped-RGB signal specialist. Static map geometry locates the lamps "
            f"and the RGB image supplies their color. The manifest records {specialist['lessons']} driving lessons, "
            f"{specialist['demonstrated_controls']} demonstrated controls and {specialist['static_pairs']} static color pairs "
            f"for the specialist. [Signal lessons and model](../data/signals/README.md)",
        ])
    lines.extend([
        "", "[Demonstration outcomes](../results/collection.json) · [Training features and labels](../data/training.npz) "
        "· [Methods and paper references](research.md)",
    ])


def rendering_section(lines, results, manifest, primary):
    profile = manifest["render_profile"]
    mode = profile["rgb"]
    percentage = profile["rgb_screen_percentage"]
    cameras = [camera for trial in primary for camera in trial["camera_rendering"]["cameras"].values()
               if camera["requested_attributes"]["enable_dlss"] == "true"]
    if mode == "mixed":
        description = (
            f"The benchmark used a native-resolution front RGB view and requested DLSS at {percentage:g}% "
            "screen percentage on auxiliary RGB views. Recorded benchmark episodes have five synchronized "
            "views: front, left, right, chase and modeled infrared."
        )
    else:
        description = f"The benchmark requested {mode} RGB rendering at {percentage:g}% screen percentage."
    lines.extend(["", "## Camera rendering", "", description,
                  f"The depth and semantic cameras used a `{profile['ground_truth']}` profile."])
    if cameras:
        matches = sum(camera["attributes_match_request"] for camera in cameras)
        confirmed = sum(camera["dlss_gpu_active"] is True for camera in cameras)
        unmeasured = sum(camera["dlss_gpu_active"] is None for camera in cameras)
        lines.append(
            f"Spawned actors returned matching settings for {matches}/{len(cameras)} DLSS-requested camera "
            "instances. CARLA can fall back to another upscaling path."
        )
        if unmeasured == len(cameras):
            lines.append("GPU DLSS activation was unmeasured in these trials.")
        else:
            lines.append(f"The evidence records {confirmed} GPU DLSS activations and {unmeasured} unmeasured instances.")
    lines.extend([
        "", "[Camera configuration and readback per trial](../results/primary.json) "
        "· [CARLA 1.0 RGB camera reference](https://github.com/carla-simulator/carla/blob/1.0/Docs/ref_sensors.md#rgb-camera)",
    ])
    path = results / "rendering.json"
    if path.is_file():
        rendering = read_json(path)
        lines.extend([
            "", f"A separate local timing profile selected `{rendering['selected_mode']}` RGB rendering.",
            "", "[Hardware, timings and rendering profile](../results/rendering.json)",
        ])
    root = results.parent
    showcases = sorted((root / "web/media/showcase").glob("*/manifest.json"))
    for path in showcases:
        record = read_json(path)
        if record["benchmark_member"]:
            continue
        video, front = record["video"], record["front_video"]
        lines.extend([
            "", f"Showcase `{record['trial_id']}` is a separate closed-loop drive. Its chase recording is "
            f"{video['width']}×{video['height']} with requested `{record['quality_requested']}` quality. The "
            f"policy's front camera video is {front['width']}×{front['height']}. "
            f"[Showcase capture manifest](../{path.relative_to(root).as_posix()})",
        ])


def build_report(root):
    results = root / "results"
    manifest = read_json(results / "manifest.json")
    primary, camera, supplemental = (read_json(results / f"{cohort}.json") for cohort in COHORTS)
    collection = read_json(results / "collection.json")
    fits = {policy: read_json(results / f"fit-{policy}.json") for policy in ("fusion", "camera")}
    trials = primary + camera + supplemental
    expected = Counter((trial["cohort"], trial["trial_id"]) for trial in manifest["plan"])
    actual = Counter((trial["cohort"], trial["trial_id"]) for trial in trials)
    if manifest["stage"] != "evaluate" or actual != expected or not all((primary, camera, supplemental, collection)):
        raise ValueError("The published cohort files must match the completed evaluation manifest")
    producer = manifest["evaluation_producer"]
    training_producer = manifest["training_producer"]
    if any(trial["producer"] != producer for trial in trials):
        raise ValueError("Evaluation trial producer differs from the published manifest")
    if any(trial["producer"] != training_producer for trial in collection):
        raise ValueError("Demonstration producer differs from the published manifest")
    if any(trial["render_profile"] != manifest["render_profile"] for trial in trials + collection):
        raise ValueError("Trial rendering profile differs from the published manifest")
    if any(trial["outcome"] not in OUTCOMES for trial in trials):
        raise ValueError("An evaluation trial has an unrecognized outcome")
    edges = trace_index(results / "traces.jsonl.gz", trials)
    seeds = {trial["seed"] for trial in primary}
    train_seeds = {trial["seed"] for trial in collection}
    layouts = {trial["route_id"] for trial in primary}
    fusion = {trial_key(trial): trial for trial in primary}
    failures = sum(not clean(trial) for trial in primary)
    planned = sum(trial["route_length_m"] for trial in primary) / 1000
    progress = sum(trial["progress_m"] for trial in primary) / 1000
    contacts = Counter(actor for trial in primary for actor in set(trial["contacts"]))
    case_names = manifest["test_encounters"]
    lines = [
        f"# {producer['map']} driving results", "",
        f"The fusion policy finished **{len(primary) - failures}/{len(primary)}** primary trials cleanly in "
        f"CARLA {producer['server_version']} on {producer['map']}. The primary suite exercised {len(case_names)} cases "
        f"across {len(seeds)} seeds ({ids(seeds)}). "
        f"The scored ego vehicle made {progress:.2f} km of route progress across {planned:.2f} km of planned routes.",
        "", f"The runtime recorded CARLA server version `{producer['server_version']}` and client version "
        f"`{producer['client_version']}`. The training archive records server version "
        f"`{training_producer['server_version']}`, client version `{training_producer['client_version']}` "
        f"and map `{training_producer['map']}`.",
        "", "[Primary trials](../results/primary.json) · [Evaluation protocol, environment and hashes](../results/manifest.json) "
        "· [CARLA 1.0 release](https://github.com/carla-simulator/carla/releases/tag/1.0)",
        "", "## Primary suite", "",
    ]
    cohort_table(lines, primary, case_names)
    red_flags = sum(row["red_light_violations"] > 0 for row in primary)
    lines.extend([
        "", "The runner assigns one terminal outcome per trial. Collision takes precedence over a red-light "
        "outcome, departure and completion. Route reaches are the runner's completion outcomes. A clean finish "
        "also requires zero recorded collisions, red-light violations and early entries into the oncoming turn. "
        f"In total {red_flags} primary trials recorded a red-light violation. The yield column counts trials with "
        "a breach even when another outcome ends the trial. The trigger and clearance fractions use the eligible "
        "trials in each row as the denominator.",
        "", f"Primary contact actors by number of trials: {bins(contacts)}. "
        f"The collision sensor logged {sum(row['collision_events'] for row in primary)} contact events.",
        "", f"Weather by primary trial: {bins(Counter(row['weather'] for row in primary))}. "
        "Weather presets select the simulated appearance. The driving study used stock tire friction.",
        "", "## Geographic holdout", "",
        f"The demonstration seeds ({ids(train_seeds)}) and primary seeds ({ids(seeds)}) share "
        f"{len(train_seeds & seeds)} seed values. The primary trials used {len(layouts)} unique route IDs "
        "across all cases. Route IDs describe the recorded start and end waypoints. A route can recur "
        "under several cases and seeds, including the two trip destinations.",
        "", "| Recorded route IDs by type | Demonstrations | Primary tests | Shared IDs |",
        "| --- | --- | --- | --- |",
    ])
    for label, field in (("OpenDRIVE roads outside junctions", "road_ids"), ("OpenDRIVE junctions", "junction_ids")):
        train_ids, test_ids = id_set(collection, field), id_set(primary, field)
        lines.append(f"| {label} | {ids(train_ids)} | {ids(test_ids)} | {ids(train_ids & test_ids)} |")
    lines.extend([
        "", "These sets use the planned ego routes stored in the episode records. Junction IDs identify the "
        "intersection traversals. The holdout covers a separate district of the same CARLA town, under the "
        "same sensor and vehicle setup with ideal supplied route localization. "
        f"`{manifest['held_out_encounter_family']}` is the protocol's withheld encounter family.",
    ])
    ablation(lines, fusion, camera)
    trip_pair(lines, fusion, supplemental, edges)
    failure_examples(lines, trials)
    lines.extend(["", "## Timing", ""])
    rate_rows = [trial for trial in primary if trial.get("wall_seconds", 0) > 0]
    control_hz = 1 / manifest["dt_seconds"]
    lines.append(f"The fixed control timestep is {manifest['dt_seconds']:g} s ({control_hz:g} simulated Hz).")
    if rate_rows:
        hz = sum(row["frames"] for row in rate_rows) / sum(row["wall_seconds"] for row in rate_rows)
        lines.append(
            f"The {len(rate_rows)} primary trial loops with wall times processed {hz:.2f} scored frames per "
            "wall-clock second in aggregate. The timer includes per-tick simulation, perception, control, any "
            "enabled recording and asynchronous collision draining after the final scored frame."
        )
    rendering_section(lines, results, manifest, primary)
    development_section(lines, results)
    training_section(lines, collection, fits, manifest)
    traction_section(lines, read_json(results / "traction.json"))
    lines.extend(["", "Regenerate this report with `python -m crosswalk.report --root .` using the published results."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Experiment root containing results")
    args = parser.parse_args()
    report = build_report(args.root)
    path = args.root / "docs" / "results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
