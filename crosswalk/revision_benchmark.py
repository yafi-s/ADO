"""Run the frozen v0.1 versus behavior revision CARLA comparison, one drive at a time."""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import platform

import cv2
import numpy as np

from . import experiment as ex
from . import report, runtime, sensors
from .district import DISTRICTS
from .drive import episode
from .policy import Model


PROJECT = Path(__file__).resolve().parents[1]
BASELINE_SHA = "0c438f310b8be5ca1e833984bb331074d6c2b2263d30436f97d19634e21b6b26"
PUBLISHED_SHA = "776d32b8d7167a368194e655e50292628e46dd4d67e509824e1fd7e69f1f7c2b"
SEEDS = {"dev": (31, 32), "regression": tuple(range(101, 111)), "fresh": tuple(range(2111, 2116))}
ARMS = ("baseline", "candidate")
PRODUCER = {"server_version": "1.0", "client_version": "1.0", "map": "Town01_Opt"}


def schedule(stage, seeds=None, cases=None, arms=None):
    if stage != "dev" and any(value is not None for value in (seeds, cases, arms)):
        raise ValueError("Only development drives support subsets.")
    def subset(requested, allowed):
        if requested is None:
            return allowed
        if not requested or len(requested) != len(set(requested)) or not set(requested) <= set(allowed):
            raise ValueError(f"Choose distinct values from {allowed}")
        return tuple(x for x in allowed if x in requested)
    allowed_arms = ARMS if stage != "regression" else ("candidate",)
    district = "train" if stage == "dev" else "heldout"
    selected_cases = subset(cases, ex.EVAL_KINDS)
    selected_arms = subset(arms, allowed_arms)
    for seed in subset(seeds, SEEDS[stage]):
        for index, case in enumerate(ex.EVAL_KINDS):
            if case not in selected_cases:
                continue
            order = selected_arms
            if (seed + index) % 2:
                order = tuple(reversed(order))
            for arm in order:
                yield ex.Trial(seed, case, arm, district, stage)


@contextmanager
def writer_lock(root):
    """A process exit releases the OS lock, including after an interrupted drive."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "writer.lock").open("a+b") as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"A benchmark writer already owns {root}") from error
        yield


def source_files(adapter):
    """Archive reachable project modules, including lazy and adapter imports."""
    pending = {"crosswalk.revision_benchmark", "crosswalk"}
    if adapter:
        pending.add(adapter.split(":")[0])
    files = {}
    while pending:
        module = pending.pop()
        path = PROJECT.joinpath(*module.split(".")).with_suffix(".py")
        if not path.exists():
            path = path.with_suffix("") / "__init__.py"
        name = path.relative_to(PROJECT).as_posix()
        if name in files or not path.is_file():
            continue
        files[name] = path
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                package = module.rpartition(".")[0]
                pending.update(f"{package}.{child}" for child in (
                    [node.module] if node.module else [alias.name for alias in node.names]))
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("crosswalk."):
                pending.add(node.module)
            elif isinstance(node, ast.Import):
                pending.update(a.name for a in node.names if a.name.startswith("crosswalk."))
    return files


def files_to_freeze(candidate, config, adapter):
    files = source_files(adapter)
    for name in ("data/signals/approach-gate.json", "vendor/nanodet/object_detection_nanodet_2022nov.onnx",
                 "models/fusion.npz", "results/primary.json"):
        files[name] = PROJECT / name
    files["candidate/checkpoint.npz"] = Path(candidate)
    if config:
        files["candidate/config.json"] = Path(config)
    return files


def freeze(root, stage, candidate, config_path, adapter, producer, profile, world_settings=None):
    if producer != PRODUCER:
        raise ValueError(f"Unexpected simulator producer {producer}")
    files = files_to_freeze(candidate, config_path, adapter)
    hashes = {name: ex.fingerprint(path) for name, path in files.items()}
    if hashes["models/fusion.npz"] != BASELINE_SHA or hashes["results/primary.json"] != PUBLISHED_SHA:
        raise ValueError("The release policy or historical result bytes have changed.")
    history = ex.read_json(files["results/primary.json"])
    if profile != history[0]["render_profile"]:
        raise ValueError("Use the published mixed 50 percent RGB profile.")
    manifest = dict(schema=1, producer=producer, render_profile=profile, world=world_settings, dt_s=0.1, range_only=False,
                    sources_sha256=hashes, candidate_adapter=adapter, python=platform.python_version(),
                    numpy=np.__version__, opencv=cv2.__version__,
                    stages={s: [asdict(t) for t in schedule(s)] for s in SEEDS})
    version = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    folder = root / ("development/" + version if stage == "dev" else "final")
    marker = folder / "manifest.json"
    if marker.exists() and ex.read_json(marker) != manifest:
        raise ValueError("Final study inputs changed. Resume with the original frozen revision.")
    for name, source in files.items():
        target = folder / "inputs" / name
        if not target.exists():
            ex.copy_atomic(source, target)
        if ex.fingerprint(target) != hashes[name]:
            raise ValueError(f"Frozen input differs: {name}")
    if not marker.exists():
        ex.write_json(marker, manifest)
    return folder, version, history


def load_model(arm, inputs, adapter):
    if arm == "baseline":
        return Model.load(inputs / "models/fusion.npz")
    checkpoint = inputs / "candidate/checkpoint.npz"
    config = inputs / "candidate/config.json"
    if adapter:
        module, function = adapter.split(":")
        return getattr(import_module(module), function)(checkpoint, ex.read_json(config) if config.exists() else {})
    if config.exists():
        raise ValueError("An external config requires an adapter that consumes it.")
    from .behavior_v2 import BehaviorModel
    return BehaviorModel.load(checkpoint)


def read_case(folder, version, trial):
    marker = folder / "checkpoint.json"
    if not marker.exists():
        return None
    data = ex.read_json(marker)
    if (data["study"] != version or data["trial"] != asdict(trial)
            or set(data["sha256"]) != {"summary.json", "trace.jsonl.gz"} or not ex.valid_checkpoint(folder)):
        raise ValueError(f"Invalid saved case {folder}")
    return ex.read_json(folder / "summary.json")


def run_case(world, trial, folder, model, version, producer, profile, verify_inputs=None):
    folder.mkdir(parents=True, exist_ok=True)
    row, trace, _, _ = episode(world, trial.kind, trial.seed, trial.policy, model,
                               run_id=trial.trial_id, **trial.settings)
    if not trace or row["frames"] != len(trace) or row["outcome"] not in report.OUTCOMES:
        raise ValueError(f"Invalid scored episode {trial.trial_id}")
    roads, junctions = DISTRICTS[trial.district]
    if (not row["road_ids"] or not set(row["road_ids"]) <= roads
            or not set(row["junction_ids"]) <= junctions or row["sensor_configuration"] != "three-camera-lidar-radar"
            or row["camera_rendering"]["requested"] != profile):
        raise ValueError(f"Unexpected course or sensors for {trial.trial_id}")
    if verify_inputs:
        verify_inputs()
    row.update(case_id=trial.case_id, trial_id=trial.trial_id, cohort=trial.cohort, study_id=version,
               producer=producer, render_profile=profile)
    ex.write_trace(folder / "trace.jsonl.gz", trace)
    ex.write_json(folder / "summary.json", row)
    ex.write_json(folder / "checkpoint.json", dict(study=version, trial=asdict(trial), sha256={
        name: ex.fingerprint(folder / name) for name in ("summary.json", "trace.jsonl.gz")}))
    # CARLA provides route identifiers as tuples; published and resumed JSON uses lists.
    # Compare the same representation on the initial run and every resume.
    return ex.read_json(folder / "summary.json")


def arm_stats(rows):
    return dict(drives=len(rows), clean=sum(map(report.clean, rows)), unique_layouts=len({r["route_id"] for r in rows}),
                outcomes=dict(Counter(r["outcome"] for r in rows)), contacts=sum(r["collision_events"] > 0 for r in rows),
                yield_breaches=sum(r["yield_violations"] > 0 for r in rows),
                red_breaches=sum(r["red_light_violations"] > 0 for r in rows))


def comparison(stage, rows, history):
    arms = {arm: [r for r in rows if r["policy"] == arm] for arm in ARMS}
    if stage == "regression":
        arms["baseline"] = history
    indexed = [{report.trial_key(r): r for r in arms[arm]} for arm in ARMS]
    paired = Counter()
    mismatches = []
    for key in sorted(indexed[0].keys() & indexed[1].keys()):
        old, new = (index[key] for index in indexed)
        fields = ("district", "crossing_delay_s", "start_xy", "background_actors",
                  "sensor_configuration", "producer", "render_profile")
        if not report.matching_setup(old, new) or any(old[k] != new[k] for k in fields):
            mismatches.append(list(key))
            continue
        label = {(True, True): "both_clean", (False, False): "neither_clean",
                 (True, False): "baseline_only", (False, True): "candidate_only"}
        paired[label[report.clean(old), report.clean(new)]] += 1
    expected = len(list(schedule(stage)))
    return dict(stage=stage, completed_drives=len(rows), expected_drives=expected,
                complete=len(rows) == expected, all_pairs_verified=(sum(paired.values()) == (expected if stage == "regression" else expected // 2)),
                baseline_source="published_v0.1" if stage == "regression" else "same_study_rerun",
                arms={arm: arm_stats(group) for arm, group in arms.items()},
                by_case={case: {arm: arm_stats([r for r in group if r["case_id"] == case])
                                for arm, group in arms.items()} for case in ex.EVAL_KINDS},
                matched_pairs=sum(paired.values()), paired_clean=dict(paired), setup_mismatches=mismatches,
                cases=[{k: r[k] for k in ("seed", "case_id", "policy", "route_id", "outcome",
                                          "collision_events", "yield_violations", "red_light_violations")}
                       | {"clean": report.clean(r)} for arm in ARMS for r in arms[arm]])


def study(world, root, stage, candidate, config=None, adapter=None, seeds=None, cases=None, arms=None):
    selected = list(schedule(stage, seeds, cases, arms))
    if adapter and (not adapter.startswith("crosswalk.") or len(adapter.split(":")) != 2):
        raise ValueError("Select a candidate adapter inside crosswalk using module:function.")
    producer = ex.validate_producer(runtime.producer_info(world))
    profile = sensors.render_profile()
    settings = world.get_settings()
    if (not settings.synchronous_mode or settings.no_rendering_mode
            or not np.isclose(settings.fixed_delta_seconds, .1) or not settings.substepping):
        raise ValueError("The full-rig study requires synchronous rendered 10 Hz physics.")
    world_settings = {key: getattr(settings, key) for key in ("synchronous_mode", "no_rendering_mode",
                      "fixed_delta_seconds", "substepping", "max_substeps", "max_substep_delta_time")}
    world_settings["opendrive_sha256"] = hashlib.sha256(world.get_map().to_opendrive().encode()).hexdigest()
    folder, version, history = freeze(root, stage, candidate, config, adapter, producer, profile, world_settings)
    live_hashes = {name: digest for name, digest in ex.read_json(folder / "manifest.json")["sources_sha256"].items()
                   if not name.startswith("candidate/")}

    def verify_inputs():
        for name, digest in live_hashes.items():
            if ex.fingerprint(PROJECT / name) != digest:
                raise ValueError(f"Benchmark input changed during the run: {name}")

    output = folder / stage
    rows = {}
    for trial in schedule(stage):
        row = read_case(output / trial.trial_id, version, trial)
        if row is not None:
            rows[trial.trial_id] = row
    def publish():
        stats = comparison(stage, list(rows.values()), history)
        ex.write_json(output / "report.json", stats)
        return stats
    publish()
    for trial in selected:
        if trial.trial_id in rows:
            continue
        print(f"Starting {stage} {trial.trial_id}", flush=True)
        model = load_model(trial.policy, folder / "inputs", adapter)
        verify_inputs()
        rows[trial.trial_id] = run_case(world, trial, output / trial.trial_id, model, version, producer, profile,
                                        verify_inputs)
        stats = publish()
        print(f"{stats['completed_drives']}/{stats['expected_drives']} {trial.trial_id} "
              f"{rows[trial.trial_id]['outcome']}", flush=True)
    return output / "report.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=SEEDS, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--adapter", help="Optional crosswalk module:function taking (checkpoint, config_dict)")
    parser.add_argument("--output", type=Path, default=PROJECT / "out/behavior-v2")
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--port", type=int, default=2468)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--cases", choices=ex.EVAL_KINDS, nargs="+")
    parser.add_argument("--arms", choices=ARMS, nargs="+")
    args = parser.parse_args()
    list(schedule(args.stage, args.seeds, args.cases, args.arms))
    with writer_lock(args.output), runtime.simulator(args.port, args.attach) as world:
        print(study(world, args.output, args.stage, args.candidate, args.config, args.adapter,
                    args.seeds, args.cases, args.arms), flush=True)


if __name__ == "__main__":
    main()
