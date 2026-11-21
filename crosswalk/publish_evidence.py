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
                                    "-count_frames", "-show_entries",
                                    "stream=nb_read_frames,width,height,codec_name,r_frame_rate,pix_fmt",
                                    "-of", "json", str(video)], text=True)
    stream, = json.loads(info)["streams"]
    require(int(stream["nb_read_frames"]) == frame_count, f"Video length differs for {video}")
    return stream


def showcase_entry(directory, build):
    manifest = read_json(directory / "manifest.json")
    scenario = directory.name
    path = f"showcase/{scenario}"
    key = f"showcase-211-{scenario}-fusion"
    base = f"./media/{path}"
    summary = manifest["summary"]
    entry = manifest["catalog_entry"]
    require(manifest["benchmark_member"] is False and entry["benchmark_member"] is False
            and entry["path"] == path and entry["trial_id"] == manifest["trial_id"] == key
            and entry["manifest"] == f"{base}/manifest.json" and entry["cameras"] == ["chase", "front"]
            and manifest["replay"] == "replay.json" and summary["scenario"] == scenario
            and summary["seed"] == 211 and summary["policy"] == "fusion" and summary["district"] == "heldout"
            and manifest["model_sha256"] == build["checkpoints"]["fusion"]["sha256"]
            and validate_producer(manifest["producer"]) == build["simulator"],
            f"Unexpected showcase identity or provenance for {key}")
    files = {"front.mp4", "chase.mp4", "poster.jpg", "front.jpg", "chase.jpg", "telemetry.json", "replay.json"}
    require(manifest["sha256"].keys() == files
            and all(fingerprint(directory / name) == sha for name, sha in manifest["sha256"].items()),
            f"Showcase file digest mismatch for {key}")
    trace, replay = (read_json(directory / name) for name in ("telemetry.json", "replay.json"))
    frames = len(trace)
    dt = manifest["frame_period_s"]
    require(frames > 0 and frames == len(replay) == summary["frames"] and dt == 0.1
            and summary["duration_s"] == round(frames * dt, 2) and summary["outcome"] == manifest["outcome"]
            and trace[-1]["progress_m"] == summary["progress_m"] and trace[-1]["frame"] == manifest["last_frame"]
            and manifest["first_frame"] <= manifest["poster_frame"] <= manifest["last_frame"],
            f"Inconsistent showcase summary for {key}")
    for i, (row, recorded) in enumerate(zip(trace, replay)):
        require(row["frame"] == row["camera_frame"] == manifest["first_frame"] + i
                and row["video_index"] == i and row["video_time"] == i / 10 and recorded["run_id"] == key
                and row["scenario"] == scenario and row["seed"] == 211 and row["policy"] == "fusion"
                and row["map"] == build["simulator"]["map"] and -1 <= row["steer"] <= 1
                and 0 <= row["throttle"] <= 1 and 0 <= row["brake"] <= 1
                and all(recorded[k] == v for k, v in row.items())
                and all(isinstance(recorded[k], list) for k in ("lidar", "radar", "tracks", "detections"))
                and "ego" in recorded["observer_map"], f"Inconsistent showcase replay at frame {i} in {key}")
    quality = manifest["quality_requested"]
    require(quality in {"Medium", "High", "Epic"}, f"Unknown capture quality for {key}")
    for camera, field in (("chase", "video"), ("front", "front_video")):
        info = check_video(directory / f"{camera}.mp4", frames)
        require(info == manifest[field], f"Showcase video metadata mismatch for {key}/{camera}")
    video = manifest["video"]
    show = dict(trial_id=key, case_id=scenario, policy="fusion", seed=summary["seed"], cohort="showcase",
                benchmark_member=False, quality_requested=quality, width=video["width"], height=video["height"],
                frames=frames, model_sha256=manifest["model_sha256"], outcome=summary["outcome"],
                duration_s=summary["duration_s"], progress_m=summary["progress_m"], recording_path=path,
                manifest_url=entry["manifest"], log_url=f"{base}/telemetry.json", video_url=f"{base}/chase.mp4")
    for label, file in (("capture manifest", "manifest.json"), ("chase camera", "chase.mp4"),
                        ("front camera", "front.mp4"), ("frame telemetry", "telemetry.json")):
        build["evidence"].append(dict(label=f"{entry['label']} · {quality} · {label}", url=f"{base}/{file}"))
    build["showcases"].append(show)
    return dict(entry, cohort="showcase")


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


def check_recording(directory, key, frames):
    import cv2

    replay = read_json(directory / "replay.json")
    require([r["frame"] for r in replay] == frames and all(r["run_id"] == key for r in replay),
            f"Replay frames differ from telemetry for {key}")
    for name in CAMERAS:
        check_video(directory / f"{name}.mp4", len(replay))
    for name in POSTERS:
        require(cv2.imread(str(directory / f"{name}.jpg")) is not None, f"Missing poster for {key}/{name}")
    return replay


def recordings(media, trials, evaluation_id, frames):
    catalog = indexed(read_json(media / "catalog.json"))
    require(catalog.keys() == {k for k, t in trials.items() if t.record}, "Unexpected recording catalog")
    paths = {}
    for key, entry in catalog.items():
        case = trials[key].case_id
        path = "" if case == "pedestrian" else case
        require(entry["path"] == path and entry["evaluation_id"] == evaluation_id
                and entry["cameras"] == list(CAMERAS), f"Wrong recording metadata for {key}")
        check_recording(media / path, key, frames[key])
        paths[key] = path
    return paths


def development(root, logs):
    key, path = "dev31-crossing", "dev-crossing"
    web = root / "web"
    media = web / "media"
    build = commissioning(root, logs)
    data = read_json(web / f"logs/{key}.json")
    episode, trace = data["summary"], data["trace"]
    expected = dict(seed=31, policy="fusion", scenario="pedestrian")
    require(all(episode[k] == v for k, v in expected.items()) and episode["district"] == "train"
            and episode["cohort"] == "development" and validate_producer(episode["producer"]) == build["simulator"],
            "Unexpected development drive, cohort or producer")
    digest = episode["model_sha256"]
    weights = web / "logs/dev31-fusion.npz"
    require(fingerprint(weights) == digest and Model.load(weights).inputs == "fusion", "Wrong development checkpoint")
    frames = [r["frame"] for r in trace]
    require(frames and all(type(f) is int for f in frames) and frames == sorted(set(frames))
            and len(frames) == episode["frames"] and round(len(frames) * 0.1, 2) == episode["duration_s"]
            and trace[-1]["progress_m"] == episode["progress_m"], "Development trace and summary disagree")
    entry = indexed(read_json(media / "catalog.json"))[key]
    require(entry["path"] == path and sorted(entry["cameras"]) == sorted(CAMERAS),
            "Development recording catalog mismatch")
    replay = check_recording(media / path, key, frames)
    for index, (row, recorded) in enumerate(zip(trace, replay)):
        require(all(row[k] == v for k, v in expected.items()) and row["map"] == build["simulator"]["map"]
                and row["sim_time"] == round(index * 0.1, 2)
                and -1 <= row["steer"] <= 1 and 0 <= row["throttle"] <= 1 and 0 <= row["brake"] <= 1
                and all(recorded[k] == row[k] for k in (*expected, "map", "steer", "throttle", "brake",
                                                         "sim_time", "progress_m", "speed_mps")),
                f"Development controls or telemetry disagree at frame {row['frame']}")
    run = {k: episode[k] for k in ("seed", "policy", "cohort", "district", "outcome", "progress_m", "duration_s")}
    run.update(trial_id=key, case_id=episode["scenario"], model_sha256=digest,
               log_url=f"./logs/{key}.json", recording_path=path)
    for name in CAMERAS:
        build["evidence"].append(dict(label=f"Development drive · {name} camera", url=f"./media/{path}/{name}.mp4"))
    build.update(status="development", title="Learned development drive", release=None,
                 checkpoints=dict(fusion=dict(sha256=digest, url="./logs/dev31-fusion.npz")), runs=[run],
                 preview_video=f"./media/{path}/chase.mp4", preview_image=f"./media/{path}/chase.jpg",
                 preview_label="Learned development drive · CARLA 1.0", preview_log=run["log_url"])
    return build


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


def publish(root, tag, source_commit, commissioning_only=False, development_only=False):
    require(not (commissioning_only and development_only), "Choose only one interim publication mode")
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
        elif development_only:
            build = development(root, logs)
        else:
            build = evaluated(root, logs, tag)
        catalog = []
        if build["status"] == "evaluated":
            build["showcases"] = []
            for scenario in ("trip", "pedestrian"):
                directory = web / "media/showcase" / scenario
                if directory.exists():
                    catalog.append(showcase_entry(directory, build))
        if not commissioning_only:
            runs = {run["trial_id"] for run in build["runs"]}
            catalog.extend(entry for entry in read_json(web / "media/catalog.json") if entry["trial_id"] in runs)
        write_json(stage / "viewer-catalog.json", catalog)
        training_clip(web, build)
        preview = "commissioning-chase.jpg" if commissioning_only else "chase.jpg"
        build["source"] = dict(commit=source_commit, url=f"{REPO}/tree/{source_commit}")
        build.setdefault("preview_image", f"./media/{preview}" if (web / "media" / preview).is_file() else None)
        write_json(stage / "build.json", build)
        (web / "logs").mkdir(exist_ok=True)
        (web / "media").mkdir(exist_ok=True)
        for path in logs.iterdir():
            path.replace(web / "logs" / path.name)
        (stage / "viewer-catalog.json").replace(web / "media/viewer-catalog.json")
        (stage / "build.json").replace(web / "media/build.json")
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--tag", default="v0.1.0")
    parser.add_argument("--source-commit", required=True)
    interim = parser.add_mutually_exclusive_group()
    interim.add_argument("--commissioning", action="store_true")
    interim.add_argument("--development", action="store_true")
    args = parser.parse_args()
    result = publish(args.root, args.tag, args.source_commit, args.commissioning, args.development)
    print(f"Indexed {len(result['runs'])} drives ({result['status']}).")


if __name__ == "__main__":
    main()
