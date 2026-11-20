"""Reproduce the staged imitation-learning study, one CARLA world at a time."""
import argparse
from contextlib import nullcontext
import gzip
import hashlib
import json
from pathlib import Path
import platform
import shutil

import cv2
import numpy as np

from .drive import episode
from .policy import ACTION_NAMES, FEATURE_NAMES, Model
from .runtime import simulator
from .train import train

TRAIN = (11, 12, 13, 14, 15, 16, 17, 18)
HELD_OUT = (101, 102, 103, 104)
TRAIN_KINDS = ("pedestrian", "lead", "clear", "turn")
EVAL_KINDS = TRAIN_KINDS + ("braking",)
POLICIES = ("fusion", "camera")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def collect(world, teacher, directory, seeds):
    files = []
    for seed in seeds:
        for kind in TRAIN_KINDS:
            path = directory / f"town01-{seed}-{kind}.npz"
            files.append(path)
            if path.exists():
                continue
            summary, _, x, y = episode(world, kind, seed, "demonstrator", teacher=teacher)
            print("Collection", seed, kind, summary["outcome"], len(y), flush=True)
            write_json(path.with_suffix(".json"), summary)
            temp = path.with_suffix(".partial.npz")
            np.savez_compressed(temp, X=x, y=y, episode=np.repeat(path.stem, len(y)),
                                seed=np.full(len(y), seed, dtype=np.int32))
            temp.replace(path)
    return files


def bundle(files, path):
    arrays = {key: [] for key in ("X", "y", "episode", "seed")}
    for file in files:
        with np.load(file, allow_pickle=False) as sample:
            for key, rows in arrays.items():
                rows.append(sample[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, feature_names=FEATURE_NAMES, action_names=ACTION_NAMES,
                        **{key: np.concatenate(rows) for key, rows in arrays.items()})


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def study(world, tm, root):
    work = root / "out"
    models = root / "models"
    results = root / "results"
    data = root / "data"
    source = Path(__file__).resolve().parent
    hasher = hashlib.sha256()
    for name in ("experiment", "drive", "sensors", "thermal", "vision", "control", "world", "policy", "train", "runtime"):
        hasher.update((source / f"{name}.py").read_bytes())
    hasher.update((source.parent / "requirements.txt").read_bytes())
    experiment_id = hasher.hexdigest()[:12]
    work = work / experiment_id
    demos = work / "collection"
    demos.mkdir(parents=True, exist_ok=True)
    demonstrations = collect(world, tm, demos, TRAIN)
    combined = data / "training.npz"
    bundle(demonstrations, combined)
    write_json(results / "collection.json", [json.loads(p.with_suffix(".json").read_text()) for p in demonstrations])
    for policy in POLICIES:
        train(combined, models / f"{policy}.npz", results / f"fit-{policy}.json", inputs=policy)
    eval_models = {p: Model.load(models / f"{p}.npz") for p in POLICIES}
    hashes = {p: fingerprint(models / f"{p}.npz") for p in eval_models}
    write_json(results / "manifest.json", dict(
        simulator="CARLA 0.9.16", map="Town01", dt_seconds=0.1, tick_limit=340, code_id=experiment_id,
        python_version=platform.python_version(), numpy_version=np.__version__, opencv_version=cv2.__version__,
        demonstration_seeds=TRAIN, test_seeds=HELD_OUT,
        training_encounters=TRAIN_KINDS, held_out_encounter_family="braking",
        test_encounters=EVAL_KINDS, policies=POLICIES, model_sha256=hashes,
        training_data_sha256=fingerprint(combined)))
    episodes = []
    for seed in HELD_OUT:
        for kind in EVAL_KINDS:
            for policy in POLICIES:
                name = f"{seed}-{kind}-{policy}"
                path = work / "evaluation" / f"{name}.json"
                digest = hashes[policy]
                if path.exists() and (saved := json.loads(path.read_text()))["policy_version"] == digest:
                    summary = saved
                else:
                    # Record fixed, prespecified trials, including any failure in those trials.
                    replay = seed == HELD_OUT[0] and kind in ("pedestrian", "braking", "turn") and policy == "fusion"
                    if replay:
                        from .viewio import Recording
                    folder = root / "web" / "media"
                    recording = Recording(folder if kind == "pedestrian" else folder / kind) if replay else nullcontext()
                    with recording as rec:
                        summary, trace, _, _ = episode(world, kind, seed, policy, eval_models.get(policy),
                                                       recording=rec)
                    summary["policy_version"] = digest
                    write_json(work / "traces" / f"{name}.json", trace)
                    write_json(path, summary)
                episodes.append(summary)
                print("Evaluation", name, summary["outcome"], summary["progress_m"], flush=True)
    write_json(results / "episodes.json", episodes)
    public = root / "web" / "media"
    public.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(results / "episodes.json", public / "episodes.json")
    with (results / "traces.jsonl.gz").open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as archive:
        for item in episodes:
            stem = f"{item['seed']}-{item['scenario']}-{item['policy']}"
            for row in json.loads((work / "traces" / f"{stem}.json").read_text()):
                archive.write((json.dumps(row, separators=(",", ":")) + "\n").encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=2468)
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("."), help="Experiment root. Reuse to resume the same code version.")
    args = parser.parse_args()
    with simulator(args.port, args.attach, with_manager=True) as (world, tm):
        study(world, tm, args.output)


if __name__ == "__main__":
    main()
