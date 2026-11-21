"""Build the public evidence index from completed local CARLA experiments."""
import argparse
from dataclasses import asdict
import gzip
from itertools import groupby
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from .experiment import (CAMERAS, POSTERS, TRAIN, POLICIES, fingerprint, read_json,
                         schedule, validate_producer, write_json)
from .policy import Model

REPO = "https://github.com/yafi-s/ADO"
RAW = "https://raw.githubusercontent.com/yafi-s/ADO"
COMMISSIONING_FILES = (
    ("Initial scenario commissioning", "results/commissioning-initial.json"),
    ("Scenario commissioning", "results/commissioning.json"),
    ("Native rendering measurements", "results/rendering.json"),
    ("Traction measurements", "results/traction.json"),
    ("Simulator vehicle verification", "assets/robotaxi/simulator-verification.json"),
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def indexed(rows):
    require(isinstance(rows, list), "Expected a JSON array of trials")
    keyed = {row["trial_id"]: row for row in rows}
    require(len(keyed) == len(rows), "Duplicate trial ID")
    return keyed


def commissioning(root, logs):
    evidence = []
    producer = None
    for label, rel in COMMISSIONING_FILES:
        path = root / rel
        record = read_json(path)
        if rel.endswith("verification.json"):
            observed = dict(server_version=record["server"], client_version=record["client"],
                            map=record["map"].split("/")[-1])
            require(record["stock"] and record["retrofit"], "Missing vehicle comparison")
        else:
            observed = record["producer"]
            if rel.endswith("rendering.json"):
                require(any(p["mode"] == "native" and p["frames"] > 0 for p in record["profiles"]),
                        "Missing native rendering measurements")
            else:
                require(record["trials"], f"Missing trials in {rel}")
        validate_producer(observed)
        require(producer is None or producer == observed, f"Mixed commissioning producers in {rel}")
        producer = observed
        shutil.copyfile(path, logs / path.name)
        evidence.append(dict(label=label, url=f"./logs/{path.name}"))
    return dict(simulator=producer, evidence=evidence)


def verified_episodes(root, manifest):
    producer = validate_producer(manifest["evaluation_producer"])
    require(manifest["stage"] == "evaluate" and manifest["training_producer"] == producer
            and manifest["map"] == producer["map"] and manifest["training_seeds"] == list(TRAIN),
            "Expected completed held-out evaluation with matching training producer")
    trials = {t.trial_id: t for t in schedule("evaluate")}
    plan = indexed(manifest["plan"])
    episodes = indexed(read_json(root / "results/episodes.json"))
    require(plan.keys() == episodes.keys() == trials.keys(), "Incomplete or unexpected evaluation trials")
    require(sorted({t.seed for t in trials.values()}) == manifest["test_seeds"],
            "Wrong held-out seeds")
    for key, trial in trials.items():
        protocol = dict(**asdict(trial), settings=trial.settings, replay=trial.record)
        require(all(plan[key].get(k) == v for k, v in protocol.items()), f"Wrong protocol for {key}")
        episode = episodes[key]
        require(all(episode.get(k) == v for k, v in asdict(trial).items())
                and episode["producer"] == producer and episode["simulator_version"] == producer["server_version"]
                and episode["map"] == producer["map"] and episode["policy_version"] == manifest["model_sha256"][trial.policy]
                and type(episode["frames"]) is int and episode["frames"] > 0
                and episode["duration_s"] == round(episode["frames"] * manifest["dt_seconds"], 2),
                f"Inconsistent summary for {key}")
    return episodes, trials


def split_traces(source, logs, episodes):
    frames = {}
    with gzip.open(source, "rb") as stream:
        rows = ((json.loads(line), line) for line in stream)
        for key, records in groupby(rows, key=lambda pair: pair[0]["trial_id"]):
            require(key in episodes and key not in frames, f"Unexpected trace group {key}")
            summary = episodes[key]
            expected = {k: summary[k] for k in ("seed", "policy", "case_id", "cohort", "district",
                                                 "map", "simulator_version", "policy_version")}
            frame_ids = []
            with (logs / f"{key}.jsonl.gz").open("wb") as raw, gzip.GzipFile(
                    filename="", fileobj=raw, mode="wb", mtime=0) as target:
                for row, line in records:
                    frame = row["frame"]
                    require(all(row.get(k) == v for k, v in expected.items()) and type(frame) is int
                            and (not frame_ids or frame > frame_ids[-1]), f"Unexpected trace frame for {key}")
                    target.write(line)
                    frame_ids.append(frame)
            require(len(frame_ids) == summary["frames"] and row["progress_m"] == summary["progress_m"],
                    f"Trace and summary disagree for {key}")
            frames[key] = frame_ids
    require(frames.keys() == episodes.keys(), "Missing trace group")
    return frames


def check_video(video, frame_count):
    info = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-threads", "1",
                                    "-count_frames", "-show_entries", "stream=nb_read_frames",
                                    "-of", "csv=p=0", str(video)], text=True)
    require(info.strip() == str(frame_count), f"Video length differs for {video}")


def training_clip(web, build):
    video = web / "media/training-drive.mp4"
    log = web / "logs/training-drive.json"
    if not (log.is_file() and video.is_file()):
        return
    record = read_json(log)
    ids = [r["frame"] for r in record["frames"]]
    require(record["policy"] == "demonstrator" and record["subject"] == "Traffic Manager training demonstration"
            and record["seed"] in TRAIN and validate_producer(record["producer"]) == build["simulator"]
            and ids and ids == sorted(set(ids)), "Inconsistent training demonstration log")
    check_video(video, len(ids))
    build["evidence"].append(dict(label=record["subject"], url="./logs/training-drive.json",
                                  producer=record["producer"], subject=record["subject"]))
    if build["status"] == "commissioning":
        build.update(preview_video="./media/training-drive.mp4", preview_log="./logs/training-drive.json",
                     preview_label="Training demonstration · CARLA 1.0")
    else:
        build["evidence"].append(dict(label="Traffic Manager training footage", url="./media/training-drive.mp4"))


def recordings(media, trials, evaluation_id, frames):
    import cv2

    catalog = indexed(read_json(media / "catalog.json"))
    require(catalog.keys() == {k for k, t in trials.items() if t.record}, "Unexpected recording catalog")
    paths = {}
    for key, entry in catalog.items():
        case = trials[key].case_id
        path = "" if case == "pedestrian" else case
        require(entry["path"] == path and entry["evaluation_id"] == evaluation_id
                and entry["cameras"] == list(CAMERAS), f"Wrong recording metadata for {key}")
        directory = media / path
        replay = read_json(directory / "replay.json")
        require([r["frame"] for r in replay] == frames[key] and all(r["run_id"] == key for r in replay),
                f"Replay frames differ from telemetry for {key}")
        for name in CAMERAS:
            check_video(directory / f"{name}.mp4", len(replay))
        for name in POSTERS:
            require(cv2.imread(str(directory / f"{name}.jpg")) is not None, f"Missing poster for {key}/{name}")
        paths[key] = path
    return paths


def evaluated(root, logs, tag):
    manifest = read_json(root / "results/manifest.json")
    episodes, trials = verified_episodes(root, manifest)
    checkpoints = {}
    for policy in POLICIES:
        path = root / "models" / f"{policy}.npz"
        sha = fingerprint(path)
        require(sha == manifest["model_sha256"][policy] and Model.load(path).inputs == policy,
                f"Wrong {policy} checkpoint")
        checkpoints[policy] = dict(sha256=sha, url=f"{RAW}/{tag}/models/{policy}.npz")
    frames = split_traces(root / "results/traces.jsonl.gz", logs, episodes)
    paths = recordings(root / "web/media", trials, manifest["evaluation_id"], frames)
    runs = []
    for key, e in episodes.items():
        fields = {k: e[k] for k in ("trial_id", "case_id", "seed", "policy", "cohort",
                                    "outcome", "progress_m", "duration_s")}
        runs.append(dict(**fields, log_url=f"./logs/{key}.jsonl.gz", recording_path=paths.get(key),
                         model_sha256=e["policy_version"]))
    download = f"{REPO}/releases/download/{tag}"
    release = dict(label=tag, url=f"{REPO}/releases/tag/{tag}",
                   zip_url=f"{download}/ADO-{tag}-Windows.zip", sha256_url=f"{download}/SHA256SUMS.txt")
    evidence = [dict(label=label, url=f"{RAW}/{tag}/{rel}") for label, rel in (
        ("Held-out evaluation protocol", "results/manifest.json"), ("Episode summaries", "results/episodes.json"),
        ("All held-out frame logs", "results/traces.jsonl.gz"), ("Recording catalog", "web/media/catalog.json"))]
    return dict(status="evaluated", title="Held-out evaluation", simulator=manifest["evaluation_producer"],
                release=release, checkpoints=checkpoints, evidence=evidence, runs=runs)


def publish(root, tag, source_commit, commissioning_only=False):
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit), "Use a full source commit SHA")
    require(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag), "Expected a version tag such as v0.1.0")
    web = Path(root) / "web"
    web.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=".publish-", dir=web) as tmp:
        stage = Path(tmp)
        logs = stage / "logs"
        logs.mkdir()
        if commissioning_only:
            build = dict(status="commissioning", title="Simulator commissioning", release=None,
                         checkpoints={}, runs=[], **commissioning(root, logs))
        else:
            build = evaluated(root, logs, tag)
        training_clip(web, build)
        preview = "commissioning-chase.jpg" if commissioning_only else "chase.jpg"
        build.update(source=dict(commit=source_commit, url=f"{REPO}/tree/{source_commit}"),
                     preview_image=f"./media/{preview}" if (web / "media" / preview).is_file() else None)
        write_json(stage / "build.json", build)
        (web / "logs").mkdir(exist_ok=True)
        (web / "media").mkdir(exist_ok=True)
        for path in logs.iterdir():
            path.replace(web / "logs" / path.name)
        (stage / "build.json").replace(web / "media/build.json")
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--tag", default="v0.1.0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--commissioning", action="store_true")
    args = parser.parse_args()
    result = publish(args.root, args.tag, args.source_commit, args.commissioning)
    print(f"Indexed {len(result['runs'])} drives ({result['status']}).")


if __name__ == "__main__":
    main()
