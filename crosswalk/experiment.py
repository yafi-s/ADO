"""Checkpoint demonstrations, training, development trials and the locked CARLA 1.0 test."""
import argparse
import ast
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import re
import shutil

import cv2
import numpy as np

from .drive import episode
from .district import DISTRICTS
from .policy import ACTION_NAMES, FEATURE_NAMES, Model
from . import runtime, sensors
from .runtime import simulator
from .train import train
from .viewio import CAMERAS, POSTERS, Recording

PROJECT = Path(__file__).resolve().parents[1]
TRAIN = tuple(range(11, 17))
DEV = (31, 32)
HELD_OUT = tuple(range(101, 111))
TRAIN_KINDS = ("pedestrian", "lead", "occluded", "signal", "left", "turn", "clear", "trip")
EVAL_KINDS = ("pedestrian", "pedestrian-delayed", "lead", "braking", "occluded",
              "signal", "left", "turn", "clear", "trip")
DEV_KINDS = tuple(case for case in EVAL_KINDS if case != "braking")
POLICIES = ("fusion", "camera")
VIDEO_CASES = ("pedestrian", "occluded", "signal", "left", "trip", "trip-delayed")
VIDEO_LABELS = dict(zip(VIDEO_CASES, ("Pedestrian crossing", "Occluded pedestrian", "Traffic signal",
                                      "Unprotected left turn", "Destination trip", "Destination trip · delayed crossing")))
DATA_FORMAT = 3
STAGES = ("collect", "fit", "dev", "evaluate", "all")
LAMP_SHA256 = "5d8479b212c386af7fa5df8128f468727b4a55abb1f8abd7c68fd40ac5d09d33"


@dataclass(frozen=True)
class Trial:
    seed: int
    case_id: str
    policy: str
    district: str
    cohort: str

    @property
    def trial_id(self):
        return f"{self.seed}-{self.case_id}-{self.policy}"

    @property
    def kind(self):
        return self.case_id.removesuffix("-delayed")

    @property
    def settings(self):
        return dict(district=self.district, crossing_delay=float(self.case_id.endswith("-delayed")),
                    destination=("market" if self.seed % 2 else "station") if self.kind == "trip" else None)

    @property
    def record(self):
        return (self.district == "heldout" and self.policy == "fusion"
                and self.seed == HELD_OUT[0] and self.case_id in VIDEO_CASES)


def schedule(stage, seeds=None):
    if stage == "evaluate":
        if seeds is not None:
            raise ValueError("The final evaluation uses its fixed seed list")
        for seed in HELD_OUT:
            for case in EVAL_KINDS:
                yield Trial(seed, case, "fusion", "heldout", "primary")
                if seed in HELD_OUT[:2]:
                    yield Trial(seed, case, "camera", "heldout", "camera_ablation")
            if seed == HELD_OUT[0]:
                yield Trial(seed, "trip-delayed", "fusion", "heldout", "supplemental")
        return
    if stage not in ("collect", "dev"):
        raise ValueError(f"No trials for {stage}")
    allowed = TRAIN if stage == "collect" else DEV
    seeds = allowed if seeds is None else tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(allowed):
        raise ValueError(f"Choose distinct {stage} seeds from {allowed}")
    for seed in seeds:
        for case in TRAIN_KINDS if stage == "collect" else DEV_KINDS:
            for policy in ("demonstrator",) if stage == "collect" else POLICIES:
                yield Trial(seed, case, policy, "train", stage)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


class CollectionPath(ast.NodeTransformer):
    """Project the shared drive loop onto its unrecorded Traffic Manager path."""

    def visit_If(self, node):
        condition = ast.unparse(node.test)
        if condition == "teacher is not None":
            return [self.visit(child) for child in node.body]
        if condition == "visuals":
            return [self.visit(child) for child in node.orelse]
        if condition in ("realtime", "change_request and change_request()"):
            return []
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id in {"row", "summary", "visuals"} for t in node.targets):
            return ast.Pass()
        return self.generic_visit(node)


def dependency_code(path, roots, *, collection=False):
    """Hash only selected entry points and module-local definitions they reference."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    if collection:
        tree = CollectionPath().visit(tree)
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
    selected = {}
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in selected or name not in definitions:
            continue
        node = definitions[name]
        selected[name] = ast.dump(node)
        pending.extend(child.id for child in ast.walk(node) if isinstance(child, ast.Name))
    return identity(selected)


def generator_sources():
    src = PROJECT / "crosswalk"
    names = ("world", "district", "ambient", "robotaxi", "sensors", "vision", "signal_vision", "control")
    versions = {n: fingerprint(src / f"{n}.py") for n in names if (src / f"{n}.py").exists()}
    versions["policy_features"] = dependency_code(src / "policy.py", ("features", "FEATURE_NAMES", "ACTION_NAMES"))
    versions["episode"] = dependency_code(src / "drive.py", ("episode",), collection=True)
    versions["renderer_launch"] = dependency_code(src / "runtime.py", ("_launch_args", "DEFAULT_QUALITY"))
    versions["detector_weights"] = fingerprint(PROJECT / "vendor/nanodet/object_detection_nanodet_2022nov.onnx")
    versions["collection_protocol"] = [dict(**asdict(t), **{"settings": t.settings}) for t in schedule("collect")]
    versions["runtime"] = {"carla": runtime.CARLA_VERSION, "map": runtime.MAP_NAME, "dt_seconds": 0.1,
                           "render_profile": sensors.render_profile(), "data_format": DATA_FORMAT,
                           "numpy": np.__version__, "opencv": cv2.__version__}
    return versions


def validate_producer(producer):
    signature = re.escape(runtime.CARLA_VERSION) + r"(?:[.+-].*)?"
    if (producer.get("map") != runtime.MAP_NAME
            or any(not re.fullmatch(signature, str(producer.get(key, "")))
                   for key in ("server_version", "client_version"))):
        raise ValueError(f"Unexpected CARLA study producer: {producer!r}")
    return producer


def copy_atomic(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".partial")
    shutil.copyfile(source, tmp)
    tmp.replace(target)


def save_arrays(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def write_trace(path, rows):
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())
    tmp.replace(path)


def valid_checkpoint(folder):
    marker = folder / "checkpoint.json"
    if not marker.is_file():
        return False
    checks = read_json(marker)["sha256"]
    return all((folder / name).is_file() and fingerprint(folder / name) == digest
               for name, digest in checks.items())


def run_trial(world, teacher, trial, folder, model=None, model_hash=None):
    producer = validate_producer(runtime.producer_info(world))
    profile = sensors.render_profile()
    if valid_checkpoint(folder):
        summary = read_json(folder / "summary.json")
        if (summary.get("producer") != producer or summary.get("render_profile") != profile
                or summary.get("data_format") != DATA_FORMAT):
            raise ValueError(f"Checkpoint producer or rendering settings differ for {trial.trial_id}")
        return summary, True
    folder.mkdir(parents=True, exist_ok=True)
    record = Recording(folder / "media") if trial.record else nullcontext()
    print(f"Starting {trial.cohort} {trial.trial_id}", flush=True)
    with record as rec:
        summary, trace, x, y = episode(world, trial.kind, trial.seed, trial.policy, model,
                                       teacher=teacher, recording=rec, run_id=trial.trial_id, **trial.settings)
    if not trace or summary["frames"] != len(trace):
        raise RuntimeError(f"Frame count mismatch for {trial.trial_id}")
    roads, junctions = DISTRICTS[trial.district]
    if (not summary["road_ids"] or not set(summary["road_ids"]) <= roads
            or not set(summary["junction_ids"]) <= junctions):
        raise RuntimeError(f"Route crossed the declared district boundary: {trial.trial_id}")
    metadata = dict(case_id=trial.case_id, trial_id=trial.trial_id, cohort=trial.cohort,
                    district=trial.district, crossing_delay_s=trial.settings["crossing_delay"])
    if model_hash:
        metadata["policy_version"] = model_hash
    if trial.settings["destination"]:
        metadata["destination"] = trial.settings["destination"]
    metadata.update(map=producer["map"], simulator_version=producer["server_version"])
    summary.update(metadata, producer=producer, render_profile=profile, data_format=DATA_FORMAT)
    write_json(folder / "summary.json", summary)
    write_trace(folder / "trace.jsonl.gz", (dict(row, **metadata) for row in trace))
    files = ["summary.json", "trace.jsonl.gz"]
    if teacher is not None:
        if x.shape != (len(y), len(FEATURE_NAMES)) or y.shape != (len(x), len(ACTION_NAMES)) or not len(y):
            raise RuntimeError(f"Invalid demonstration arrays for {trial.trial_id}: {x.shape}, {y.shape}")
        points = np.asarray(summary["route_xy"], dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or not np.isfinite(points).all() or not np.any(np.diff(points, axis=0))):
            raise RuntimeError(f"Invalid demonstrated route geometry for {trial.trial_id}")
        save_arrays(folder / "samples.npz", X=x, y=y, episode=np.repeat(trial.trial_id, len(y)),
                    seed=np.full(len(y), trial.seed, dtype=np.int32), data_format=DATA_FORMAT,
                    route_xy=points, route_seed=np.int32(trial.seed),
                    producer=json.dumps(producer, sort_keys=True), render_profile=json.dumps(profile, sort_keys=True))
        files.append("samples.npz")
    if trial.record:
        files.extend(f"media/{name}.mp4" for name in CAMERAS)
        files.extend(f"media/{name}.jpg" for name in POSTERS)
        files.append("media/replay.json")
    write_json(folder / "checkpoint.json", {"sha256": {name: fingerprint(folder / name) for name in files}})
    return summary, False


def collect(world, tm, base, seeds):
    if tm is None:
        raise ValueError("Collection requires a CARLA Traffic Manager")
    trials = list(schedule("collect", seeds))
    for index, trial in enumerate(trials, 1):
        summary, reused = run_trial(world, tm, trial, base / "collection" / trial.trial_id)
        print(f"Collection {index}/{len(trials)} {trial.trial_id} {summary['outcome']} "
              f"{summary['frames']} frames" + (" (checkpoint)" if reused else ""), flush=True)


def bundle(files, path):
    arrays = {key: [] for key in ("X", "y", "episode", "seed")}
    routes, seeds = [], []
    producer = None
    profile = sensors.render_profile()
    for file in files:
        with np.load(file, allow_pickle=False) as records:
            if int(records["data_format"]) != DATA_FORMAT or json.loads(str(records["render_profile"])) != profile:
                raise ValueError(f"Incompatible demonstration data format or render profile: {file}")
            observed = validate_producer(json.loads(str(records["producer"])))
            if producer is not None and producer != observed:
                raise ValueError(f"Mixed simulator producers in the demonstration set: {file}")
            producer = observed
            route = records["route_xy"]
            route_seed = int(records["route_seed"])
            if route_seed not in TRAIN or not np.all(records["seed"] == route_seed):
                raise ValueError(f"Unexpected route seed in training input: {file}")
            routes.append(route)
            seeds.append(route_seed)
            for key in arrays:
                arrays[key].append(records[key])
    save_arrays(path, feature_names=FEATURE_NAMES, action_names=ACTION_NAMES,
                route_xy=np.concatenate(routes), route_lengths=np.asarray([len(r) for r in routes], dtype=np.int32),
                route_seeds=np.asarray(seeds, dtype=np.int32),
                data_format=DATA_FORMAT, producer=json.dumps(producer, sort_keys=True),
                render_profile=json.dumps(profile, sort_keys=True),
                **{key: np.concatenate(values) for key, values in arrays.items()})


def signal_specialist(root):
    path = root / "data/signals/lamp.npz"
    if not path.is_file():
        return None, {}
    if fingerprint(path) != LAMP_SHA256:
        raise ValueError(f"Signal expert differs from the validated calibration artifact: {path}")
    from .lamp import LampExpert
    arrays = LampExpert.load(path).arrays("lamp_")
    info = dict(source="data/signals/lamp.npz", sha256=LAMP_SHA256, fit_seeds=[11, 12, 13, 14],
                lessons=8, demonstrated_controls=len(arrays["lamp_pedals"]), static_pairs=16)
    report = path.with_name("fit.json")
    if report.is_file():
        calibration = read_json(report)
        if (len(calibration["lessons"]) != info["lessons"] or calibration["fit_frames"] != info["demonstrated_controls"]
                or calibration["fit_seeds"] != info["fit_seeds"] or calibration["fit_static_pairs"] != info["static_pairs"]):
            raise ValueError(f"Signal calibration report disagrees with validated specialist: {report}")
        info["fit_report_sha256"] = fingerprint(report)
    return info, arrays


def fit_archive(root, base):
    seeds = []
    folders = []
    for seed in TRAIN:
        cohort = [base / "collection" / t.trial_id for t in schedule("collect", (seed,))]
        if all(valid_checkpoint(f) for f in cohort):
            seeds.append(seed)
            folders.extend(cohort)
    if not seeds:
        raise RuntimeError("Collect at least one complete training seed cohort before fitting")
    if len(seeds) != len(TRAIN):
        print(f"Development fit on complete training seeds {seeds}; final evaluation requires all {TRAIN}", flush=True)
    src = PROJECT / "crosswalk"
    source = {n: fingerprint(src / f"{n}.py") for n in ("policy", "train", "steering_search")}
    sample_hashes = {f.name: read_json(f / "checkpoint.json")["sha256"]["samples.npz"] for f in folders}
    specialist, lamp_arrays = signal_specialist(root)
    ingredients = dict(source=source, samples=sample_hashes, dataset_id=base.name)
    if specialist:
        ingredients["signal_specialist"] = specialist["sha256"]
    fit_id = identity(ingredients)
    archive = base / "fits" / fit_id
    archive.mkdir(parents=True, exist_ok=True)
    combined = archive / "training.npz"
    if not valid_checkpoint(archive):
        bundle([f / "samples.npz" for f in folders], combined)
        for policy in POLICIES:
            print(f"Fitting {policy} on {len(folders)} drives", flush=True)
            train(combined, archive / f"{policy}.npz", archive / f"fit-{policy}.json",
                  inputs=policy, imitation_path=archive / f"{policy}-imitation.npz")
            if lamp_arrays:
                for name in (f"{policy}.npz", f"{policy}-imitation.npz"):
                    path = archive / name
                    with np.load(path, allow_pickle=False) as weights:
                        merged = dict(weights)
                    merged.update(lamp_arrays)
                    save_arrays(path, **merged)
        files = ["training.npz"] + [f"{p}.npz" for p in POLICIES] + [f"fit-{p}.json" for p in POLICIES]
        files.extend(f"{p}-imitation.npz" for p in POLICIES)
        write_json(archive / "checkpoint.json", {"sha256": {n: fingerprint(archive / n) for n in files}})
    hashes = read_json(archive / "checkpoint.json")["sha256"]
    with np.load(combined, allow_pickle=False) as records:
        producer = validate_producer(json.loads(str(records["producer"])))
        profile = json.loads(str(records["render_profile"]))
    provenance = dict(dataset_id=base.name, fit_id=fit_id, training_seeds=seeds, training_sources=source,
                      signal_specialist=specialist,
                      training_producer=producer, render_profile=profile, data_format=DATA_FORMAT,
                      training_samples=sample_hashes, training_data_sha256=hashes["training.npz"],
                      model_sha256={p: hashes[f"{p}.npz"] for p in POLICIES},
                      imitation_sha256={p: hashes[f"{p}-imitation.npz"] for p in POLICIES})
    write_json(archive / "provenance.json", provenance)
    copy_atomic(combined, root / "data/training.npz")
    for p in POLICIES:
        copy_atomic(archive / f"{p}.npz", root / "models" / f"{p}.npz")
        copy_atomic(archive / f"{p}-imitation.npz", root / "models" / f"{p}-imitation.npz")
        copy_atomic(archive / f"fit-{p}.json", root / "results" / f"fit-{p}.json")
    write_json(root / "results/collection.json", [read_json(f / "summary.json") for f in folders])
    write_json(root / "models/fit.json", provenance)
    return provenance


def merge_traces(paths, target):
    # Concatenated gzip members form one standard, streaming-decodable gzip file.
    tmp = target.with_name(target.name + ".partial")
    with tmp.open("wb") as out:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, out)
    tmp.replace(target)


def evaluate(world, root, base, stage, seeds):
    fit = read_json(root / "models/fit.json")
    if fit["dataset_id"] != base.name or (stage == "evaluate" and tuple(fit["training_seeds"]) != TRAIN):
        raise RuntimeError("Fit models with the current collection protocol and required training seeds first")
    producer = validate_producer(runtime.producer_info(world))
    if (fit.get("training_producer") != producer or fit.get("render_profile") != sensors.render_profile()
            or fit.get("data_format") != DATA_FORMAT):
        raise ValueError("Fitted demonstration producer, data format or rendering profile differs from this simulation")
    archive = base / "fits" / fit["fit_id"]
    if not valid_checkpoint(archive):
        raise RuntimeError(f"Incomplete or damaged fit archive: {archive}")
    specialist, lamp_arrays = signal_specialist(root)
    if specialist:
        if (fit.get("signal_specialist") or {}).get("sha256") != specialist["sha256"]:
            raise RuntimeError("Fit the models with the current canonical signal specialist before evaluation")
        for policy in POLICIES:
            with np.load(archive / f"{policy}.npz", allow_pickle=False) as weights:
                if any(key not in weights or not np.array_equal(weights[key], value)
                       for key, value in lamp_arrays.items()):
                    raise ValueError(f"{policy} model lacks the validated signal expert")
    src = PROJECT / "crosswalk"
    if any(fingerprint(src / f"{name}.py") != digest for name, digest in fit["training_sources"].items()):
        raise RuntimeError("Policy or training code changed since this fit. Run --stage fit first.")
    models = {p: Model.load(archive / f"{p}.npz") for p in POLICIES}
    trials = list(schedule(stage, seeds))
    source = {name: fingerprint(src / f"{name}.py") for name in
              ("drive", "world", "district", "policy", "sensors", "control", "vision", "signal_vision",
               "robotaxi", "thermal", "runtime", "viewio", "lamp", "tracking") if (src / f"{name}.py").is_file()}
    gate = None
    gate_path = PROJECT / "data/signals/approach-gate.json"
    if specialist and gate_path.is_file():
        from .lamp import APPROACH_GATE_SHA256
        if fingerprint(gate_path) != APPROACH_GATE_SHA256:
            raise ValueError("Invalid learned approach gate")
        gate = {"path":"data/signals/approach-gate.json", "sha256":APPROACH_GATE_SHA256}
    version = identity(dict(fit=fit["fit_id"], scoring=source, signal_gate=gate,
                            protocol=[(asdict(t), t.settings, t.record) for t in schedule(stage)]))
    runs = base / stage / version
    results = root / "results" if stage == "evaluate" else root / "results/dev"
    code_revision = subprocess.run(["git","rev-parse","HEAD"], cwd=PROJECT, capture_output=True,
                                   text=True, check=False).stdout.strip() if (PROJECT / ".git").exists() else None
    text_hashes = {name: hashlib.sha256((src/f"{name}.py").read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()
                   for name in source}
    manifest = dict(**fit, simulator=f"CARLA {producer['server_version']}", map=producer["map"], dt_seconds=0.1,
                    evaluation_producer=producer, signal_gate=gate, scoring_revision=code_revision,
                    normalized_source_sha256=text_hashes,
                    evaluation_id=version, source_sha256=source, python_version=platform.python_version(),
                    numpy_version=np.__version__, opencv_version=cv2.__version__, demonstration_seeds=fit["training_seeds"],
                    test_seeds=sorted({t.seed for t in trials}), training_encounters=TRAIN_KINDS,
                    held_out_encounter_family="braking",
                    test_encounters=EVAL_KINDS if stage == "evaluate" else DEV_KINDS, policies=POLICIES, stage=stage,
                    plan=[dict(**asdict(t), trial_id=t.trial_id, settings=t.settings, replay=t.record) for t in trials])
    write_json(runs / "protocol.json", manifest)
    episodes, traces, recordings = [], [], []
    for index, trial in enumerate(trials, 1):
        folder = runs / trial.trial_id
        summary, reused = run_trial(world, None, trial, folder, models[trial.policy], fit["model_sha256"][trial.policy])
        if trial.record:
            target = root / "web/media"
            if trial.case_id != "pedestrian":
                target /= trial.case_id
            for file in (folder / "media").iterdir():
                if file.is_file():
                    copy_atomic(file, target / file.name)
            recordings.append(dict(label=VIDEO_LABELS[trial.case_id],
                                   path="" if trial.case_id == "pedestrian" else trial.case_id,
                                   cameras=list(CAMERAS), trial_id=trial.trial_id, evaluation_id=version))
            write_json(root / "web/media/catalog.json", recordings)
        episodes.append(summary)
        traces.append(folder / "trace.jsonl.gz")
        print(f"{stage} {index}/{len(trials)} {trial.trial_id} {summary['outcome']} "
              f"{summary['progress_m']} m" + (" (checkpoint)" if reused else ""), flush=True)
        write_json(runs / "progress.json", dict(completed=index, total=len(trials), last_trial=trial.trial_id))
    write_json(results / "episodes.json", episodes)
    write_json(results / "manifest.json", manifest)
    for cohort in sorted({t.cohort for t in trials}):
        write_json(results / f"{cohort}.json", [e for e in episodes if e["cohort"] == cohort])
    write_json(results / "failures.json", [e for e in episodes if e["outcome"] != "complete"
                                          or e.get("yield_violations") or e.get("red_light_violations")])
    merge_traces(traces, results / "traces.jsonl.gz")
    if stage == "evaluate":
        copy_atomic(results / "episodes.json", root / "web/media/episodes.json")


def study(world, tm, root, stage="all", seeds=None):
    if stage not in STAGES or (seeds is not None and stage not in ("collect", "dev")):
        raise ValueError("Seed selection is available only for collect and dev")
    root = Path(root).resolve()
    sources = generator_sources()
    base = root / "out/city" / identity(sources)
    write_json(base / "generator.json", sources)
    if stage in ("all", "collect"):
        collect(world, tm, base, seeds)
    if stage in ("all", "fit"):
        fit_archive(root, base)
    if stage in ("all", "dev"):
        evaluate(world, root, base, "dev", seeds)
    if stage in ("all", "evaluate"):
        evaluate(world, root, base, "evaluate", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--port", type=int, default=2468)
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT)
    parser.add_argument("--seeds", nargs="+", help="Training or development seed subset, separated by spaces or commas")
    args = parser.parse_args()
    seeds = None
    if args.seeds:
        try:
            seeds = tuple(int(seed) for group in args.seeds for seed in group.split(","))
            if args.stage not in ("collect", "dev"):
                raise ValueError("--seeds applies only to collect or dev")
            list(schedule(args.stage, seeds))
        except ValueError as error:
            parser.error(str(error))
    if args.stage == "fit":
        study(None, None, args.output, args.stage)
    elif args.stage in ("all", "collect"):
        with simulator(args.port, args.attach, with_manager=True) as (world, tm):
            study(world, tm, args.output, args.stage, seeds)
    else:
        with simulator(args.port, args.attach) as world:
            study(world, None, args.output, args.stage, seeds)


if __name__ == "__main__":
    main()
