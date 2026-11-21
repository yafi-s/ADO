"""Checkpoint an independent, camera-free CARLA 1.0 train-district driving diagnostic."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from . import policy, runtime
from .district import DISTRICTS
from .drive import episode
from .experiment import (PROJECT, Trial, fingerprint, identity, read_json, save_arrays,
                         validate_producer, write_json, write_trace)
from .train import train


CASES = ("pedestrian", "lead", "occluded", "clear", "turn", "trip")
TRAIN_SEEDS = (11, 12, 13, 14)
DEV_SEEDS = (31, 32)
PROFILE = {"no_rendering_mode": True, "camera_sensors": [], "inputs": "lidar, radar, ego speed, navigation"}
DATA_FORMAT = 1
STAGES = ("all", "collect", "fit", "eval")
TRIAL_FILES = ("summary.json", "trace.jsonl.gz")
FIT_FILES = ("training.npz", "model.npz", "imitation.npz", "fit.json", "provenance.json")


def schedule(collect):
    for seed in TRAIN_SEEDS if collect else DEV_SEEDS:
        for case in CASES:
            yield Trial(seed, case, "demonstrator" if collect else "range", "train",
                        "range-collection" if collect else "range-development")


def checked_session(world):
    producer = validate_producer(runtime.producer_info(world))
    settings = world.get_settings()
    if (world.get_map().name.rsplit("/", 1)[-1] != "Town01_Opt"
            or not settings.no_rendering_mode or not settings.synchronous_mode
            or settings.fixed_delta_seconds != .1):
        raise ValueError("Range development requires synchronous 0.1 s physics-only CARLA on Town01_Opt")
    return producer


def source_versions(names):
    return {name: fingerprint(Path(__file__).with_name(name + ".py")) for name in names}


def checkpoint(folder, files):
    write_json(folder / "checkpoint.json", {"sha256": {f: fingerprint(folder / f) for f in files}})


def verified(folder, files):
    try:
        digests = read_json(folder / "checkpoint.json")["sha256"]
        return set(digests) == set(files) and all(fingerprint(folder / f) == digests[f] for f in files)
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return False


def recovery(trace):
    """Advance at least 2 m after a triggered hazard clears, following a pre-clearance stop."""
    stopped = False
    cleared_progress = None
    peak_gain = 0.
    for row in trace:
        if (row["event_triggered"] and not row["event_cleared"] and row["speed_mps"] < .35
                and row["reference_gap_m"] < 12 and row["progress_m"] > 2):
            stopped = True
        if stopped and row["event_cleared"]:
            if cleared_progress is None:
                cleared_progress = row["progress_m"]
            peak_gain = max(peak_gain, row["progress_m"] - cleared_progress)
    return {"stopped_before_clearance": stopped, "clearance_observed_after_stop": cleared_progress is not None,
            "post_clearance_gain_m": round(peak_gain, 3) if cleared_progress is not None else None,
            "recovered": peak_gain >= 2 if cleared_progress is not None else None}


def run_trial(world, tm, model, folder, trial, producer, model_hash=None):
    collecting = trial.policy == "demonstrator"
    files = (*TRIAL_FILES, "samples.npz") if collecting else TRIAL_FILES
    if verified(folder, files):
        return read_json(folder / "summary.json")
    print(f"Starting {trial.cohort} {trial.trial_id}", flush=True)
    summary, rows, x, y = episode(world, trial.kind, trial.seed, trial.policy, model,
                                  teacher=tm if collecting else None, range_only=True,
                                  run_id=trial.trial_id, **trial.settings)
    roads, junctions = DISTRICTS["train"]
    if (not rows or summary["frames"] != len(rows) or not summary["road_ids"]
            or not set(summary["road_ids"]) <= roads or not set(summary["junction_ids"]) <= junctions):
        raise RuntimeError(f"Invalid trace or train-district route for {trial.trial_id}")
    if (summary.get("sensor_configuration") != "lidar-radar"
            or summary.get("camera_rendering", {}).get("cameras") != {}):
        raise RuntimeError(f"Unexpected sensor configuration for {trial.trial_id}")
    folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(trial_id=trial.trial_id, cohort=trial.cohort, district="train",
                    producer=producer, render_profile=PROFILE, model_sha256=model_hash)
    summary.update(metadata, recovery=recovery(rows))
    if collecting:
        points = np.asarray(summary["route_xy"], dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all()
                or not np.any(np.diff(points, axis=0))):
            raise ValueError(f"Missing demonstrated course polyline for {trial.trial_id}")
        if (x.shape != (len(rows)-1, len(policy.RANGE_FEATURE_NAMES))
                or y.shape != (len(x), len(policy.ACTION_NAMES)) or not len(y)
                or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError(f"Invalid packed range observations or controls for {trial.trial_id}")
        save_arrays(folder / "samples.npz", X=x, y=y, episode=np.repeat(trial.trial_id, len(y)),
                    seed=np.full(len(y), trial.seed, dtype=np.int32), data_format=DATA_FORMAT,
                    route_xy=points, route_seed=np.int32(trial.seed), producer=json.dumps(producer, sort_keys=True),
                    render_profile=json.dumps(PROFILE, sort_keys=True))
    write_json(folder / "summary.json", summary)
    write_trace(folder / "trace.jsonl.gz", (dict(row, trial_id=trial.trial_id) for row in rows))
    checkpoint(folder, files)
    return summary


def fit_dataset(base, producer, *, fit_allowed):
    trials = list(schedule(True))
    files = [base / "collection" / trial.trial_id / "samples.npz" for trial in trials]
    for f in files:
        if not verified(f.parent, (*TRIAL_FILES, "samples.npz")):
            raise RuntimeError(f"Collection checkpoint required before fitting or evaluation: {f.parent}")
    inputs = dict(collection_id=base.name, source=source_versions(("policy", "train", "steering_search")),
                  samples={t.trial_id: fingerprint(f) for t, f in zip(trials, files)})
    archive = base / "fits" / identity(inputs)
    if verified(archive, FIT_FILES):
        return archive
    if not fit_allowed:
        raise RuntimeError("Current verified model missing. Run the fit stage first.")
    parts = {key: [] for key in ("X", "y", "episode", "seed")}
    routes = []
    for trial, f in zip(trials, files):
        with np.load(f, allow_pickle=False) as sample:
            if (json.loads(str(sample["producer"])) != producer or int(sample["route_seed"]) != trial.seed
                    or not np.all(sample["seed"] == trial.seed) or not np.all(sample["episode"] == trial.trial_id)):
                raise ValueError(f"Inconsistent demonstration metadata: {f}")
            routes.append(sample["route_xy"])
            for key in parts:
                parts[key].append(sample[key])
    save_arrays(archive / "training.npz", **{k: np.concatenate(v) for k, v in parts.items()},
                route_xy=np.concatenate(routes), route_lengths=np.array([len(r) for r in routes], np.int32),
                route_seeds=np.array([t.seed for t in trials], np.int32), data_format=DATA_FORMAT,
                feature_names=policy.RANGE_FEATURE_NAMES, action_names=policy.ACTION_NAMES,
                producer=json.dumps(producer, sort_keys=True), render_profile=json.dumps(PROFILE, sort_keys=True))
    print(f"Training camera-free model on {len(files)} demonstrations", flush=True)
    train(archive / "training.npz", archive / "model.npz", archive / "fit.json",
          inputs="range", route_refinement=True, imitation_path=archive / "imitation.npz")
    write_json(archive / "provenance.json", inputs)
    checkpoint(archive, FIT_FILES)
    return archive


def report(results, demos, episodes, archive):
    outcomes = Counter(row["outcome"] for row in episodes)
    eligible = [row["recovery"] for row in episodes if row["recovery"]["recovered"] is not None]
    text = [
        "# Camera-free driving development results", "",
        "CARLA 1.0 on Town01_Opt, train district. The learned policy used lidar, radar, ego speed "
        "and route navigation. No camera sensors were used. These runs contribute zero episodes "
        "to the frozen 100-run primary benchmark.", "",
        f"The Traffic Manager supplied {len(demos)} demonstrations across seeds 11–14. "
        "Steering refinement used their recorded course polylines. The fitted policy drove "
        f"{len(episodes)} closed-loop episodes on development seeds 31 and 32.", "",
        f"Outcomes: {', '.join(f'{number} {name}' for name, number in sorted(outcomes.items()))}. "
        f"{sum(row['collision_events'] > 0 for row in episodes)} episodes had contact. "
        f"Aggregate route progress was {sum(row['progress_m'] for row in episodes):.1f} m "
        f"of {sum(row['route_length_m'] for row in episodes):.1f} m planned.", "",
        f"After stopping for a triggered hazard and observing clearance, the policy advanced "
        f"at least 2 m in {sum(r['recovered'] for r in eligible)} of {len(eligible)} eligible episodes.", "",
        "| Seed | Case | Outcome | Progress / route (m) | Contact events | Recovery after clearance |",
        "|---:|---|---|---:|---:|---|"]
    for row in episodes:
        status = {True: "yes", False: "no", None: "ineligible"}[row["recovery"]["recovered"]]
        text.append(f"| {row['seed']} | {row['scenario']} | {row['outcome']} | {row['progress_m']:.1f} / "
                    f"{row['route_length_m']:.1f} | {row['collision_events']} | {status} |")
    text.extend(("", f"Model SHA-256: `{fingerprint(archive / 'model.npz')}`.", ""))
    temporary = results / "report.md.partial"
    temporary.write_text("\n".join(text), encoding="utf-8")
    temporary.replace(results / "report.md")


def run(world, tm, root, stage="all"):
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage}")
    results = Path(root).resolve() / "results" / "range-development"
    index = results / "collection-id.json"
    if world is None and stage != "fit":
        raise ValueError("A verified active CARLA session is required")
    if stage in ("collect", "all") and tm is None:
        raise ValueError("Collection requires a Traffic Manager")
    producer = (checked_session(world) if world is not None else
                validate_producer(read_json(index)["producer"]))
    sources = source_versions(("range_development", "drive", "policy", "sensors", "world",
                               "district", "runtime", "control", "experiment"))
    manifest = dict(producer=producer, source_sha256=sources, numpy=np.__version__,
                    profile=PROFILE, cases=CASES, train_seeds=TRAIN_SEEDS, dev_seeds=DEV_SEEDS,
                    feature_names=policy.RANGE_FEATURE_NAMES, data_format=DATA_FORMAT)
    base = results / "archives" / identity(manifest)
    if stage in ("collect", "all"):
        write_json(index, dict(collection_id=base.name, producer=producer))
        write_json(base / "protocol.json", manifest)
        for trial in schedule(True):
            run_trial(world, tm, None, base / "collection" / trial.trial_id, trial, producer)
        if stage == "collect":
            return base
    archive = fit_dataset(base, producer, fit_allowed=stage != "eval")
    if stage == "fit":
        return archive
    model_path = archive / "model.npz"
    digest = fingerprint(model_path)
    model = policy.Model.load(model_path)
    if model.inputs != "range":
        raise ValueError("Expected a learned range policy")
    episodes = []
    for trial in schedule(False):
        folder = archive / "eval" / digest[:20] / trial.trial_id
        episodes.append(run_trial(world, None, model, folder, trial, producer, digest))
    demos = [read_json(base / "collection" / trial.trial_id / "summary.json") for trial in schedule(True)]
    write_json(results / "episodes.json", episodes)
    write_json(results / "manifest.json", dict(collection_id=base.name, fit_id=archive.name,
               model_sha256=digest, producer=producer, profile=PROFILE, district="train",
               fit_directory=archive.relative_to(results).as_posix(), primary_benchmark_contribution=0))
    report(results, demos, episodes, archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT)
    parser.add_argument("--port", type=int, default=2468)
    parser.add_argument("--stage", choices=STAGES, default="all")
    args = parser.parse_args()
    if args.stage == "fit":
        run(None, None, args.output, "fit")
    else:
        with runtime.simulator(args.port, with_manager=True, physics_only=True) as (world, tm):
            run(world, tm, args.output, args.stage)


if __name__ == "__main__":
    main()
