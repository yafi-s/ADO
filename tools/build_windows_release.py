"""Build and verify an ADO Windows ZIP from a committed release tree."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile


REPO = Path(__file__).resolve().parents[1]
PUBLIC_REPO = "https://github.com/yafi-s/ADO"
PREFIX = "ADO/"
PACKAGE_PATHS = (
    "LICENSE", "README.md", "Start-ADO.cmd", "Watch-Live.cmd", "requirements.txt", "setup.ps1",
    "ado", "assets", "crosswalk", "data", "docs", "models", "results", "tests", "tools", "vendor", "web",
)
REQUIRED = (
    "Start-ADO.cmd", "Watch-Live.cmd", "setup.ps1", "requirements.txt", "tools/fetch_carla_client.py",
    "ado/__init__.py", "ado/__main__.py", "crosswalk/drive.py", "crosswalk/policy.py", "crosswalk/signal_vision.py",
    "vendor/nanodet/object_detection_nanodet_2022nov.onnx",
    "assets/robotaxi/vehicle/meridian.blend", "assets/waymo-ipace/waymo-ipace.blend",
    "assets/robotaxi/SM_RobotaxiCarrier.fbx", "assets/robotaxi/vehicle/meridian.glb",
    "assets/waymo-ipace/waymo-ipace.glb", "models/fit.json", "data/training.npz",
    "results/manifest.json", "results/episodes.json", "results/traces.jsonl.gz",
    "web/index.html", "web/ado-home.css", "web/ado-home.js", "web/builds.html", "web/drive.html",
    "web/media/build.json", "web/media/catalog.json", "web/media/viewer-catalog.json",
    "web/media/training-drive.mp4", "web/logs/training-drive.json",
)
BEHAVIOR_ROOT = "results/behavior-v2"
STUDY_FILES = ("fresh.json", "regression.json", "study.json", "closed-loop-evidence.zip", "README.md")
ORIGINAL_BEHAVIOR_ARCHIVE = {
    "url": f"{PUBLIC_REPO}/raw/refs/tags/v0.2.0/{BEHAVIOR_ROOT}/closed-loop-evidence.zip",
    "bytes": 6115585,
    "sha256": "08b4288f26c4aaefca5b0522bbc093418d778d9f65a0b905f97fcd32c76ab0b5",
}
ORIGINAL_BEHAVIOR_MANIFEST = "f3742e11e373b8ebe545aa32023d30c6b0f53672b22ae501f01500a55677876e"
ORIGINAL_REPORT_WRITER = "5d83b9fc3cf6ffaf7c04fea9b119893069fbefd403972ffb079c19ed0443e979"
WAYMO_MEDIA = "web/media/waymo-research-pilot"
BEHAVIOR_REQUIRED = (
    "ado/__main__.py", "crosswalk/behavior_v2.py", "crosswalk/city_drive.py", "crosswalk/journey.py",
    "crosswalk/journey_capture.py", "crosswalk/revision_benchmark.py", "docs/city-driving.md",
    "data/behavior-v2/fit.ipynb", "data/behavior-v2/provenance.json", "data/behavior-v2/README.md",
    "models/behavior-v2/candidate.npz", "waymo_training/ADO_Waymo_Colab.ipynb", "waymo_training/NOTICE.md",
    "waymo_training/WAYMO_DATASET_AGREEMENT.txt", "waymo_training/schema/LICENSE", "waymo_training/schema/NOTICE",
    f"{WAYMO_MEDIA}/NOTICE.md", f"{WAYMO_MEDIA}/WAYMO_DATASET_AGREEMENT.txt", f"{WAYMO_MEDIA}/SHA256SUMS.txt",
)
CITY_PRODUCER = {"server_version": "1.0", "client_version": "1.0", "map": "Town03_Opt"}
CITY_ARTIFACTS = {"chase.mp4", "front.mp4", "chase.jpg", "front.jpg", "poster.jpg",
                  "telemetry.json", "replay.json", "town.json"}
SURROUND_MEDIA = "web/media/surround-city"
SURROUND_MASTER = "ADO-Surround-City-1080p.mp4"
SURROUND_SCHEMA = "ado-city-cruise-v1"
PACE_MEDIA = "web/media/urban-pace"
PACE_MASTER = "ADO-Urban-Pace-1080p.mp4"
PACE_SCHEMA = "ado-city-pace-v1"
BARRIER_MODEL = "models/barrier-appearance.json"
BARRIER_DATA = "data/barrier-recognition/town03-native.zip"
BARRIER_VALIDATION = "results/barrier-recognition/validation.json"
BARRIER_REVIEW = "barrier-review.json"
BARRIER_WITNESS = "data/barrier-recognition/hero-post-proposals.zip"
BARRIER_POST_MODEL = "models/barrier-post-appearance.npz"
BARRIER_POST_FIT = "results/barrier-recognition/post-appearance-fit.json"
BARRIER_POST_SOURCE = "crosswalk/barrier_posts.py"
BARRIER_REVIEW_SOURCE = "crosswalk/barrier_review.py"
PERIMETER_VIEWS = ("left", "right", "rear_left", "rear_right")
SURROUND_CAMERAS = ("chase", "front", *PERIMETER_VIEWS)
TACTICAL_MODEL = "models/tactical-pass/choice.json"
TACTICAL_RECORDINGS = {"overtake-clear": "clear", "overtake-yield": "occupied"}
SURROUND_DETECTOR = "vendor/yolox/object_detection_yolox_2022nov.onnx"
SURROUND_DETECTOR_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
SURROUND_REQUIRED = (
    "Watch-City.cmd", "crosswalk/camera_geometry.py", "crosswalk/city_cruise.py", "crosswalk/lidar_motion.py",
    "crosswalk/overtake.py", "crosswalk/overtake_replay.py", "crosswalk/overtake_trials.py",
    "crosswalk/radar_geometry.py", "crosswalk/signal_observer.py",
    "crosswalk/surround.py", "crosswalk/surround_vision.py", "crosswalk/tracking.py",
    "models/city-cruise/candidate.npz", "models/city-cruise/fit.json", TACTICAL_MODEL,
    SURROUND_DETECTOR, "vendor/yolox/LICENSE", "vendor/yolox/README.md",
    f"{SURROUND_MEDIA}/manifest.json", *(f"web/media/{name}/manifest.json" for name in TACTICAL_RECORDINGS),
)
PACE_REQUIRED = (
    "crosswalk/city_pace.py", "crosswalk/speed_signs.py", "crosswalk/cycle_association.py",
    "crosswalk/town01_interaction.py", "crosswalk/town01_interaction_gate.py",
    "models/town01-yield-gated-v04/calibration.json", "results/town01-yield-gated-v04/audit.json",
    "crosswalk/barrier_chain.py", "crosswalk/barrier.py", "crosswalk/barrier_collect.py", "crosswalk/barrier_scene.py",
    "crosswalk/barrier_study.py", "models/city-pace/candidate.npz", "models/city-pace/fit.json",
    BARRIER_MODEL, BARRIER_DATA, BARRIER_VALIDATION,
    BARRIER_POST_SOURCE, BARRIER_REVIEW_SOURCE, BARRIER_POST_MODEL, BARRIER_POST_FIT, BARRIER_WITNESS,
    "web/barrier-review.mjs", "web/follow-heading.mjs", "web/inspect.mjs",
    f"{PACE_MEDIA}/manifest.json", f"{PACE_MEDIA}/{BARRIER_REVIEW}",
)
CITY_EVENTS = {"pedestrian", "pedestrian-delayed", "occluded", "lead", "braking", "left",
               "signal", "cyclist", "roadworks", "bus"}
PRIVATE_NAMES = {"credentials.json", "secrets.json", "__pycache__"}
PRIVATE_SUFFIXES = {".exe", ".dll", ".whl", ".pem", ".key", ".pfx", ".pyc", ".log", ".partial"}
CAMERAS = {"front", "chase", "infrared", "left", "right"}
PRODUCER = {"server_version": "1.0", "client_version": "1.0", "map": "Town01_Opt"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def resolve_commit(ref, tag):
    commit = git("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").decode().strip()
    if git("tag", "--list", tag):
        tagged = git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}").decode().strip()
        require(commit == tagged, f"{tag} points to {tagged}, not {commit}")
    return commit


def behavior_release(tag):
    return tuple(map(int, tag[1:].split("."))) >= (0, 2, 0)


def surround_release(tag):
    return tuple(map(int, tag[1:].split("."))) >= (0, 3, 0)


def pace_release(tag):
    return tuple(map(int, tag[1:].split("."))) >= (0, 4, 0)


def featured_recording(tag):
    if pace_release(tag):
        return PACE_MEDIA, PACE_MASTER, PACE_SCHEMA
    return SURROUND_MEDIA, SURROUND_MASTER, SURROUND_SCHEMA


def release_paths(tag):
    return (*PACKAGE_PATHS, "waymo_training", "Watch-City.cmd") if behavior_release(tag) else PACKAGE_PATHS


def required_files(tag):
    if pace_release(tag):
        return (*REQUIRED, *BEHAVIOR_REQUIRED, *SURROUND_REQUIRED, *PACE_REQUIRED)
    if surround_release(tag):
        return (*REQUIRED, *BEHAVIOR_REQUIRED, *SURROUND_REQUIRED)
    return (*REQUIRED, *BEHAVIOR_REQUIRED) if behavior_release(tag) else REQUIRED


def package_inventory(commit, tag):
    records = git("ls-tree", "-r", "-z", commit, "--", *release_paths(tag))
    files = set()
    for record in records.decode("utf-8").split("\0")[:-1]:
        attributes, name = record.split("\t", 1)
        mode, _, _ = attributes.split()
        path = PurePosixPath(name)
        require(mode in {"100644", "100755"}, f"Release file is not an ordinary Git blob: {name}")
        require(not any(part.startswith(".") or part.lower() in PRIVATE_NAMES for part in path.parts)
                and not re.search(r'[<>:"\\|?*\x00-\x1f]', name)
                and all(not part.endswith((".", " ")) for part in path.parts)
                and path.suffix.lower() not in PRIVATE_SUFFIXES, f"Unexpected release path: {name}")
        files.add(name)
    require(len(files) == len({name.casefold() for name in files}), "Release paths collide on Windows")
    missing = sorted(set(required_files(tag)) - files)
    require(not missing, f"Release files missing from {commit}: {', '.join(missing)}")
    return files


def recorded_frames(archive, episodes):
    frames = {}
    with archive.open(PREFIX + "results/traces.jsonl.gz") as compressed, gzip.GzipFile(fileobj=compressed) as trace:
        for record in trace:
            row = json.loads(record)
            key = row["trial_id"]
            frames.setdefault(key, []).append(row["frame"])
            require(row["policy_version"] == episodes[key]["policy_version"], f"Wrong trace checkpoint for {key}")
    require(frames.keys() == episodes.keys(), "Missing evaluated telemetry")
    for key, ids in frames.items():
        require(len(ids) == episodes[key]["frames"] and ids and ids == sorted(set(ids)), f"Wrong trace frames for {key}")
    return frames


def verify_video_file(video, frame_count, label, recorded=None):
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-threads", "1", "-count_frames",
        "-show_entries", ("stream=width,height,codec_name,codec_tag_string,nb_read_frames,pix_fmt,r_frame_rate"
                          ":stream_tags=encoder"), "-of", "json", str(video),
    ]))
    stream = next(iter(info["streams"]), {})
    require(stream.get("codec_name") == "h264" and stream.get("codec_tag_string") == "avc1"
            and "libx264" in stream.get("tags", {}).get("encoder", "")
            and stream.get("nb_read_frames") == str(frame_count), f"Wrong x264 encoding or frame count for {label}")
    if recorded is not None:
        require(all(stream.get(key) == recorded[key] for key in ("width", "height", "pix_fmt", "r_frame_rate")),
                f"Wrong video properties for {label}")


def verify_video(archive, relative, frame_count, scratch, recorded=None):
    video = scratch / "preview.mp4"
    with archive.open(PREFIX + relative) as source, video.open("wb") as target:
        shutil.copyfileobj(source, target)
    verify_video_file(video, frame_count, relative, recorded)


def read_json(archive, relative):
    return json.loads(archive.read(PREFIX + relative))


def verify_digest(archive, relative, expected):
    with archive.open(PREFIX + relative) as item:
        require(hashlib.file_digest(item, "sha256").hexdigest() == expected, f"Wrong SHA-256 for {relative}")


def verify_behavior_release(archive, files, build, *, derivative_required=False):
    provenance = read_json(archive, "data/behavior-v2/provenance.json")["artifacts"]
    for relative, artifact in provenance.items():
        require(archive.getinfo(PREFIX + relative).file_size == artifact["bytes"], f"Wrong size for {relative}")
        verify_digest(archive, relative, artifact["sha256"])
    require(build["checkpoints"]["behavior-v2"]["sha256"]
            == provenance["models/behavior-v2/candidate.npz"]["sha256"],
            "Builds & logs links the wrong behavior checkpoint")

    checksums = archive.read(f"{PREFIX}{WAYMO_MEDIA}/SHA256SUMS.txt").decode("ascii").splitlines()
    covered = set()
    for line in checksums:
        match = re.fullmatch(r"([0-9a-f]{64})  ([a-zA-Z0-9_.-]+)", line)
        require(match is not None, f"Invalid Waymo research checksum: {line}")
        digest, name = match.groups()
        require(name not in covered, f"Duplicate Waymo research checksum: {name}")
        covered.add(name)
        verify_digest(archive, f"{WAYMO_MEDIA}/{name}", digest)
    require(covered == {name.removeprefix(f"{WAYMO_MEDIA}/") for name in files
                        if name.startswith(f"{WAYMO_MEDIA}/")} - {"SHA256SUMS.txt"},
            "Waymo research clip checksum coverage is incomplete")

    paths = {f"{BEHAVIOR_ROOT}/{name}" for name in STUDY_FILES}
    if derivative_required or not files.isdisjoint(paths):
        missing = sorted(paths - files)
        require(not missing, f"Incomplete Behavior v2 benchmark bundle: {', '.join(missing)}")
        verify_behavior_study(archive, derivative_required=derivative_required)


def verify_behavior_study(archive, *, derivative_required=False):
    study = read_json(archive, f"{BEHAVIOR_ROOT}/study.json")
    for stage in ("fresh", "regression"):
        relative = f"{BEHAVIOR_ROOT}/{stage}.json"
        report = read_json(archive, relative)
        require(report["complete"] is True and report["completed_drives"] == report["expected_drives"] == 100,
                f"Incomplete {stage} benchmark cohort")
        verify_digest(archive, relative, study["cohorts"][stage]["report_sha256"])
    artifact = study["archive"]
    relative = f"{BEHAVIOR_ROOT}/closed-loop-evidence.zip"
    require(artifact["file"] == "closed-loop-evidence.zip"
            and archive.getinfo(PREFIX + relative).file_size == artifact["bytes"]
            and artifact["new_trials"] == 200 and artifact["historical_primary_trials"] == 100,
            "Wrong Behavior v2 benchmark archive inventory")
    verify_digest(archive, relative, artifact["sha256"])
    with archive.open(PREFIX + relative) as stream, ZipFile(stream) as evidence:
        entries = [item for item in evidence.infolist() if not item.is_dir()]
        require(len(entries) == len({item.filename for item in entries}), "Duplicate Behavior v2 benchmark member")
        members = {item.filename for item in entries if item.file_size > 0}
        require({"SHA256SUMS", "manifest.json", "archive-info.json", "protocol.md",
                 "execution/results/primary.json", "published-v0.1/traces.jsonl.gz"} <= members,
                "Missing Behavior v2 benchmark provenance or historical logs")
        if derivative_required or "derivation.json" in members or "derivation" in artifact:
            verify_behavior_derivative(evidence, artifact, len(entries))
        manifest = json.loads(evidence.read("manifest.json"))
        for stage in ("fresh", "regression"):
            trials = {f"{stage}/{trial['seed']}-{trial['case_id']}-{trial['policy']}"
                      for trial in manifest["stages"][stage]}
            required = {f"{root}/{name}" for root in trials
                        for name in ("summary.json", "trace.jsonl.gz", "checkpoint.json")}
            required.update((f"{stage}/report.json", f"{stage}/verification.json"))
            require(len(trials) == len(manifest["stages"][stage]) == 100 and required <= members,
                    f"Missing Behavior v2 {stage} trial logs")


def verify_behavior_derivative(evidence, artifact, member_count):
    require(artifact.get("verified_members") == member_count and "derivation.json" in evidence.namelist(),
            "Missing Behavior v2 derivative provenance or member inventory")
    derivation = json.loads(evidence.read("derivation.json"))
    require(artifact.get("derivation") == derivation and derivation["original_archive"] == ORIGINAL_BEHAVIOR_ARCHIVE
            and derivation["original_manifest_sha256"] == ORIGINAL_BEHAVIOR_MANIFEST
            and hashlib.sha256(evidence.read("manifest.json")).hexdigest() == ORIGINAL_BEHAVIOR_MANIFEST,
            "Wrong original Behavior v2 evidence provenance")
    writer = derivation["modified_member"]
    require(writer["path"] == "execution/crosswalk/report.py" and writer["original_sha256"] == ORIGINAL_REPORT_WRITER
            and writer["derivative_sha256"] != ORIGINAL_REPORT_WRITER,
            "Wrong Behavior v2 derivative member")
    checks = dict(line.split("  ", 1)[::-1] for line in evidence.read("SHA256SUMS").decode().splitlines())
    require(checks.keys() == set(evidence.namelist()) - {"SHA256SUMS"}
            and checks[writer["path"]] == writer["derivative_sha256"], "Incomplete Behavior v2 derivative checksums")
    for name, expected in checks.items():
        require(hashlib.sha256(evidence.read(name)).hexdigest() == expected,
                f"Wrong Behavior v2 derivative SHA-256 for {name}")


def verify_packed_replay(archive, media, replay):
    with archive.open(f"{PREFIX}{media}/replay-pack.json.gz") as compressed, gzip.GzipFile(fileobj=compressed) as stream:
        pack = json.load(stream)
    frames, ranges, path = pack["replay"], pack["path_ranges"], pack["shared_path"]
    require(pack["format"] == "ado-replay-pack-v1" and isinstance(frames, list) and isinstance(ranges, list)
            and (isinstance(path, list) if ranges else path is None), f"Invalid replay pack at {media}")
    previous = 0
    for begin, end in ranges:
        require(type(begin) is int and type(end) is int and previous <= begin < end <= len(frames),
                f"Invalid route reference in replay pack at {media}")
        for index in range(begin, end):
            observer = frames[index]["observer_map"]
            require("path" not in observer, f"Conflicting route in replay pack at {media}")
            observer["path"] = path
        previous = end
    require(frames == replay, f"Replay pack differs from the original recording at {media}")


def verify_city_film(archive, media, capture, scratch, tactical_case=None):
    require(capture["producer"] == CITY_PRODUCER and capture["benchmark_member"] is False
            and capture["teacher"] is False and capture["map_asset"] == "town.json"
            and capture["source_mode"] == "one closed-loop simulation with simultaneous camera capture",
            f"Wrong Town03 film provenance at {media}")
    policy = capture["policy"]
    pace = policy["schema"] == PACE_SCHEMA
    surround = policy["schema"] in {SURROUND_SCHEMA, PACE_SCHEMA}
    cameras = SURROUND_CAMERAS if surround else ("chase", "front")
    artifacts = CITY_ARTIFACTS | {f"{camera}.mp4" for camera in cameras}
    if tactical_case is not None:
        require(policy["schema"] == SURROUND_SCHEMA, f"Wrong tactical study policy at {media}")
    if tactical_case is not None or pace:
        artifacts.add("replay-pack.json.gz")
    require(artifacts <= capture["sha256"].keys(), f"Incomplete Town03 film at {media}")
    if surround:
        require(tuple(capture.get("perimeter_views", ())) == PERIMETER_VIEWS
                and capture["camera_videos"].keys() == set(SURROUND_CAMERAS),
                f"Wrong surround cameras at {media}")
        verify_digest(archive, TACTICAL_MODEL, policy["tactical_sha256"])
        require(read_json(archive, TACTICAL_MODEL)["format"] == "ado-tactical-pass-v1",
                f"Invalid tactical model at {media}")
        fit = read_json(archive, f"models/{'city-pace' if pace else 'city-cruise'}/fit.json")
        require(fit["model_sha256"] == policy["sha256"] and fit["schema"] == policy["schema"],
                f"Wrong city model fit at {media}")
    if pace:
        verify_digest(archive, BARRIER_MODEL, policy["barrier_sha256"])
    for name, digest in capture["sha256"].items():
        require(re.fullmatch(r"[a-zA-Z0-9_.-]+", name), f"Invalid Town03 film artifact at {media}: {name}")
        verify_digest(archive, f"{media}/{name}", digest)

    folder = {"ado-behavior-v2": "behavior-v2", "ado-city-policy-v1": "city-v1",
              SURROUND_SCHEMA: "city-cruise", PACE_SCHEMA: "city-pace"}.get(policy["schema"])
    require(folder is not None and re.fullmatch(r"[a-zA-Z0-9_-]+\.npz", policy["path"]),
            f"Invalid Town03 policy path at {media}")
    verify_digest(archive, f"models/{folder}/{policy['path']}", policy["sha256"])
    town = read_json(archive, f"{media}/town.json")
    require(town["producer"] == CITY_PRODUCER and town["roads"], f"Wrong map geometry at {media}")

    summary = capture["summary"]
    if tactical_case is None:
        verify_journey_result(capture, media, CITY_EVENTS | {"barrier"} if pace else CITY_EVENTS)
    else:
        verify_tactical_result(capture, media, tactical_case)

    telemetry = read_json(archive, f"{media}/telemetry.json")
    replay = read_json(archive, f"{media}/replay.json")
    if pace:
        verify_packed_replay(archive, media, replay)
    require(len(telemetry) == len(replay) == summary["frames"] > 0
            and capture["last_frame"] - capture["first_frame"] + 1 == len(telemetry),
            f"Wrong Town03 film frame count at {media}")
    for index, (score, exposure) in enumerate(zip(telemetry, replay, strict=True)):
        frame = capture["first_frame"] + index
        for row in (score, exposure):
            require(row["frame"] == row["camera_frame"] == frame and row["video_index"] == index
                    and row["map"].split(".", 1)[0] == CITY_PRODUCER["map"],
                    f"Unsynchronized Town03 film at {media} frame {frame}")
        require(exposure["run_id"] == capture["trial_id"], f"Wrong Town03 replay at {media} frame {frame}")
    for camera in cameras:
        relative = f"{media}/{camera}.mp4"
        if surround:
            recorded = capture["camera_videos"][camera]
            require(recorded["nb_read_frames"] == str(len(telemetry))
                    and recorded["codec_name"] == "h264" and recorded["pix_fmt"] == "yuv420p"
                    and recorded["r_frame_rate"] == "10/1", f"Wrong capture metadata for {relative}")
            verify_video(archive, relative, len(telemetry), scratch, recorded)
        else:
            verify_video(archive, relative, len(telemetry), scratch)


def verify_journey_result(capture, media, required_events=CITY_EVENTS):
    summary = capture["summary"]
    events = capture["events"]
    event_kinds = {event["kind"] for event in events}
    require(summary["outcome"] == capture["outcome"] and summary["events"] == events
            and set(capture["required_event_kinds"]) == required_events
            and len(capture["required_event_kinds"]) == len(required_events)
            and event_kinds <= required_events and len(events) == len(event_kinds),
            f"Inconsistent encounter results at {media}")
    require(isinstance(capture["passed"], bool) and capture["passed"] == (not capture["failure_reasons"]),
            f"Inconsistent Town03 film outcome at {media}")
    if capture["passed"]:
        milestones = ("triggered", "exposed", "cleared", "passed", "handled")
        require(summary["outcome"] == "complete" and event_kinds == required_events
                and summary["progress_m"] >= summary["route_length_m"] - 0.6 and summary["duration_s"] >= 120
                and all(summary[key] == 0 for key in ("collision_events", "red_light_violations", "yield_violations"))
                and all(not event["violation"] and all(event[key] for key in milestones) for event in events),
                f"Unsubstantiated complete Town03 film at {media}")


def verify_tactical_result(capture, media, case):
    summary, score = capture["summary"], capture["score"]
    require(capture["cohort"] == "tactical-study" and capture["case"] == score["case"] == case
            and summary["scenario"] == "overtake" and capture["outcome"] == summary["outcome"] == "complete"
            and summary["collision_events"] == score["collisions"] == 0
            and capture["passed"] is True and score["passed"] is True
            and capture["count"] == summary["frames"] == score["frames"]
            and 0 <= score["first_commit_s"] < score["first_alongside_s"] < score["completed_pass_s"]
            <= summary["duration_s"]
            and score["lead_speed_during_pass_mps"] > 0 and score["minimum_passing_clearance_m"] > 0,
            f"Inconsistent completed moving-lead study at {media}")
    require((case == "clear" and score["yield_s"] == 0)
            or (case == "occupied" and score["yield_s"] > 0
                and score["first_yield_s"] < score["first_commit_s"]),
            f"Wrong yield timing for the {case} study at {media}")


def verify_tactical_recordings(archive, build, viewer_records, scratch):
    for trial, case in TACTICAL_RECORDINGS.items():
        media = f"web/media/{trial}"
        capture = read_json(archive, f"{media}/manifest.json")
        require(capture["trial_id"] == trial and any(
            row["trial_id"] == trial and row["recording_path"] == trial and row["case_id"] == "overtake"
            and row["benchmark_member"] is False for row in build["development_runs"]
        ), f"Missing supplemental tactical study in the build catalog: {trial}")
        require(any(
            row["trial_id"] == trial and row["path"] == trial and tuple(row["cameras"]) == SURROUND_CAMERAS
            and row["benchmark_member"] is False and row["replay_pack_sha256"] == capture["sha256"]["replay-pack.json.gz"]
            for row in viewer_records
        ), f"Wrong supplemental tactical study in the viewer catalog: {trial}")
        verify_city_film(archive, media, capture, scratch, tactical_case=case)


def verify_barrier_review(archive, capture, showcase, entry):
    relative = f"{PACE_MEDIA}/{BARRIER_REVIEW}"
    url = f"./{relative.removeprefix('web/')}"
    expected = capture["sha256"][BARRIER_REVIEW]
    require(all(ref["barrier_review_url"] == url and ref["barrier_review_sha256"] == expected
                for ref in (capture, showcase, entry)), "Barrier review links differ from the recorded artifact")
    verify_digest(archive, relative, expected)
    review = read_json(archive, relative)
    first, last = capture["first_frame"], capture["last_frame"]
    frames = capture["summary"]["frames"]
    raw = capture["raw_sha256"]
    cameras = {camera: {"sha256": raw[f"{camera}.mjpeg"], "images": frames}
               for camera in SURROUND_CAMERAS if camera != "chase"}
    require(review["schema"] == "ado-barrier-post-review-v1" and review["partial"] is False
            and (review["first_frame"], review["last_frame"], review["capture_frames"]) == (first, last, frames)
            and review["native_journal_sha256"] == raw["perception.jsonl"] and review["native_mjpeg"] == cameras
            and all(first <= int(frame) <= last for frame in review["frames"]),
            "Barrier review does not match the complete native urban recording")
    verify_digest(archive, BARRIER_POST_MODEL, review["model_sha256"])
    fit = read_json(archive, BARRIER_POST_FIT)
    require(fit["schema"] == "ado-post-appearance-fit-v1"
            and fit["model_sha256"] == review["model_sha256"] and fit["threshold"] == review["threshold"],
            "The barrier review uses a different post appearance fit")
    for path, key in ((BARRIER_POST_SOURCE, "post_classifier_lf_sha256"),
                      (BARRIER_REVIEW_SOURCE, "review_code_lf_sha256")):
        source = archive.read(PREFIX + path).replace(b"\r\n", b"\n")
        require(hashlib.sha256(source).hexdigest() == review[key], f"Wrong barrier review source: {path}")
    verify_digest(archive, BARRIER_WITNESS, review["witness_sha256"])
    with archive.open(PREFIX + BARRIER_WITNESS) as stream, ZipFile(stream) as witness:
        inputs = json.loads(witness.read("inputs.json"))
        require(inputs["schema"] == "ado-barrier-post-witness-v1" and inputs["partial"] is False
                and (inputs["first_frame"], inputs["last_frame"], inputs["frames"]) == (first, last, frames)
                and inputs["raw_journal_sha256"] == raw["perception.jsonl"] and inputs["native_mjpeg"] == cameras,
                "Barrier witness does not match the complete native urban recording")
        for observation in inputs["packets"]:
            name = observation["source_jpeg"]
            require(hashlib.sha256(witness.read(name)).hexdigest() == observation["source_jpeg_sha256"],
                    f"Wrong native barrier witness exposure: {name}")


def verify_surround_release(archive, build, viewer_records, tag):
    media, asset, schema = featured_recording(tag)
    directory = PurePosixPath(media).name
    capture = read_json(archive, f"{media}/manifest.json")
    require(capture["policy"]["schema"] == schema, f"Featured drive is missing the {schema} policy")
    showcase = next((row for row in build["showcases"]
                     if row["trial_id"] == capture["trial_id"] and row["recording_path"] == directory), None)
    require(showcase is not None, f"{directory} is missing from the build catalog")
    require(any(row["trial_id"] == capture["trial_id"] and row["path"] == directory
                and tuple(row["cameras"]) == SURROUND_CAMERAS for row in viewer_records),
            f"{directory} or its cameras are missing from the viewer catalog")
    if pace_release(tag):
        entry = viewer_records[0]
        require(entry["path"] == directory and entry["benchmark_member"] is False
                and entry["replay_pack_url"] == f"./media/{directory}/replay-pack.json.gz"
                and entry["replay_pack_sha256"] == capture["sha256"]["replay-pack.json.gz"],
                "Featured urban drive or its replay pack is missing from the viewer catalog")
        report = read_json(archive, BARRIER_VALIDATION)
        require(report["schema"] == "ado-barrier-validation-v1"
                and report["model_sha256"] == capture["policy"]["barrier_sha256"],
                "The barrier validation uses a different recognition model")
        verify_digest(archive, BARRIER_DATA, report["archive_sha256"])
        verify_barrier_review(archive, capture, showcase, entry)
    master = capture["master"]["chase"]
    require(master["path"] == "master/chase.mp4" and (master["width"], master["height"]) == (1920, 1080)
            and master["file_bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", master["sha256"])
            and master["nb_read_frames"] == str(capture["summary"]["frames"])
            and master["codec_name"] == "h264" and master["pix_fmt"] == "yuv420p"
            and master["r_frame_rate"] == "10/1", "Wrong surround master metadata")
    require(any(item["url"] == f"{PUBLIC_REPO}/releases/download/{tag}/{asset}"
                for item in build["evidence"]), "Builds & logs is missing the master release asset")
    verify_digest(archive, SURROUND_DETECTOR, SURROUND_DETECTOR_SHA256)
    requirements = archive.read(PREFIX + "requirements.txt").decode()
    for dependency in ("numpy", "opencv-python-headless", "onnxruntime"):
        require(re.search(rf"(?m)^{dependency}==[^#\s]+", requirements),
                f"The Windows installer is missing the pinned {dependency} dependency")


def stage_release_master(source, archive, target, media=SURROUND_MEDIA):
    with ZipFile(archive) as package:
        capture = read_json(package, f"{media}/manifest.json")
    master = capture["master"]["chase"]
    shutil.copyfile(source, target)
    require(target.stat().st_size == master["file_bytes"], "Wrong surround master size")
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(digest == master["sha256"], "Wrong surround master SHA-256")
    verify_video_file(target, capture["summary"]["frames"], target.name, master)
    return digest


def validate_zip(path, files, commit, tag, scratch):
    with ZipFile(path) as archive:
        entries = [member for member in archive.infolist() if not member.is_dir()]
        names = {entry.filename for entry in entries}
        require(archive.comment == commit.encode("ascii"), "ZIP does not identify the expected Git commit")
        require(len(entries) == len(names) and names == {PREFIX + name for name in files},
                "ZIP file inventory differs from the selected commit")
        damaged = archive.testzip()
        require(damaged is None, f"Damaged ZIP file: {damaged}")

        def read(path):
            return archive.read(PREFIX + path)

        def load(path):
            return read_json(archive, path)

        for required in required_files(tag):
            with archive.open(PREFIX + required) as item:
                header = item.read(64)
                require((header or required == "ado/__init__.py")
                        and not header.startswith(b"version https://git-lfs.github.com/spec/v1"),
                        f"Missing Git contents for {required}")

        build = load("web/media/build.json")
        manifest = load("results/manifest.json")
        release_url = f"{PUBLIC_REPO}/releases/download/{tag}"
        require(build["status"] == "evaluated" and manifest["stage"] == "evaluate"
                and build["simulator"] == manifest["evaluation_producer"] == PRODUCER,
                "The committed release tree must contain a completed CARLA 1.0 evaluation")
        require(build["release"] == dict(label=tag, url=f"{PUBLIC_REPO}/releases/tag/{tag}",
                                         zip_url=f"{release_url}/{path.name}", sha256_url=f"{release_url}/SHA256SUMS.txt"),
                "Builds & logs release links do not match the ZIP or tag")
        for item in build["evidence"]:
            if item["url"].startswith("./"):
                require(f"web/{item['url'][2:]}" in files, f"Missing published evidence: {item['url']}")
            if behavior_release(tag) and item["url"].startswith("../results/"):
                require(item["url"][3:] in files, f"Missing published evidence: {item['url']}")

        fit = load("models/fit.json")
        require(fit["fit_id"] == manifest["fit_id"], "Wrong training fit")
        for policy in ("camera", "fusion"):
            for hashes, suffix in (("model_sha256", ""), ("imitation_sha256", "-imitation")):
                model = f"models/{policy}{suffix}.npz"
                actual = hashlib.sha256(read(model)).hexdigest()
                require(actual == fit[hashes][policy] == manifest[hashes][policy], f"Wrong checkpoint: {model}")
            require(build["checkpoints"][policy]["sha256"] == fit["model_sha256"][policy],
                    f"Builds & logs links the wrong {policy} checkpoint")
        require(hashlib.sha256(read("data/training.npz")).hexdigest()
                == fit["training_data_sha256"] == manifest["training_data_sha256"], "Wrong training dataset")

        summary_rows = load("results/episodes.json")
        episodes = {row["trial_id"]: row for row in summary_rows}
        runs = {row["trial_id"]: row for row in build["runs"]}
        plan = {row["trial_id"]: row for row in manifest["plan"]}
        require(episodes and episodes.keys() == plan.keys() == runs.keys()
                and len(episodes) == len(summary_rows) == len(build["runs"]) == len(manifest["plan"]),
                "Incomplete published evaluation")
        for key, episode in episodes.items():
            require(episode["producer"] == PRODUCER and episode["district"] == "heldout"
                    and episode["policy_version"] == runs[key]["model_sha256"] == fit["model_sha256"][episode["policy"]]
                    and runs[key]["log_url"] == f"./logs/{key}.jsonl.gz" and f"web/logs/{key}.jsonl.gz" in files,
                    f"Wrong published evidence for {key}")
        frames = recorded_frames(archive, episodes)

        catalog = load("web/media/catalog.json")
        require(catalog and {row["trial_id"] for row in catalog} == {key for key, row in plan.items() if row["replay"]},
                "Recording catalog differs from the evaluation plan")
        for entry in catalog:
            key = entry["trial_id"]
            relative = entry["path"]
            require(entry["evaluation_id"] == manifest["evaluation_id"] and set(entry["cameras"]) == CAMERAS
                    and relative == runs[key]["recording_path"]
                    and (not relative or re.fullmatch(r"[a-z0-9-]+", relative)), f"Wrong recording location for {key}")
            media = f"web/media/{relative}".rstrip("/")
            require(all(f"{media}/{camera}.mp4" in files for camera in CAMERAS), f"Missing camera footage for {key}")
            replay = load(f"{media}/replay.json")
            require([row["frame"] for row in replay] == frames[key] and all(row["run_id"] == key for row in replay),
                    f"Replay and telemetry differ for {key}")
            if not relative:
                verify_video(archive, f"{media}/chase.mp4", len(replay), scratch)
        require(any(entry["path"] == "" for entry in catalog), "Missing representative recording")

        showcases = build.get("showcases", [])
        viewer_records = load("web/media/viewer-catalog.json")
        extras = [row for key in ("native_replays", "development_runs") for row in build.get(key, [])
                  if row.get("recording_path") is not None]
        expected_ids = {row["trial_id"] for row in [*catalog, *showcases, *extras]}
        if behavior_release(tag):
            verify_behavior_release(archive, files, build, derivative_required=pace_release(tag))
            expected_ids.add(load(f"{WAYMO_MEDIA}/entry.json")["trial_id"])
        require(len(viewer_records) == len(expected_ids) and
                {row["trial_id"] for row in viewer_records} == expected_ids,
                "Incomplete viewer recording catalog")
        if surround_release(tag):
            verify_surround_release(archive, build, viewer_records, tag)
            verify_tactical_recordings(archive, build, viewer_records, scratch)
        city_media = set()
        for showcase in showcases:
            relative = showcase["recording_path"]
            require(re.fullmatch(r"[a-z0-9-]+(?:/[a-z0-9-]+)*", relative), f"Invalid showcase path: {relative}")
            media = f"web/media/{relative}"
            capture = load(f"{media}/manifest.json")
            if behavior_release(tag) and capture["producer"] == CITY_PRODUCER:
                require(capture["trial_id"] == showcase["trial_id"], f"Wrong showcase at {relative}")
                verify_city_film(archive, media, capture, scratch)
                city_media.add(media)
                continue
            require(re.fullmatch(r"showcase/[a-z0-9-]+", relative), f"Invalid showcase path: {relative}")
            require(capture["benchmark_member"] is False and capture["trial_id"] == showcase["trial_id"]
                    and capture["model_sha256"] == fit["model_sha256"]["fusion"], f"Wrong showcase at {relative}")
            for name, expected in capture["sha256"].items():
                with archive.open(PREFIX + f"{media}/{name}") as artifact:
                    require(hashlib.file_digest(artifact, "sha256").hexdigest() == expected,
                            f"Wrong showcase file: {relative}/{name}")

        if behavior_release(tag):
            for media in ("web/media/long-city-drive", "web/media/city-attempt"):
                if media not in city_media and any(name.startswith(f"{media}/") for name in files):
                    verify_city_film(archive, media, load(f"{media}/manifest.json"), scratch)

        training = load("web/logs/training-drive.json")
        require(training["policy"] == "demonstrator" and training["producer"] == PRODUCER and training["frames"],
                "Missing CARLA training demonstration")
        verify_video(archive, "web/media/training-drive.mp4", len(training["frames"]), scratch)


def build_release(ref, tag, output, film_master=None):
    require(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag), "Use a version tag such as v0.1.0")
    require(film_master is None or surround_release(tag), "--film-master applies to v0.3.0 and newer releases")
    commit = resolve_commit(ref, tag)
    files = package_inventory(commit, tag)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ado-release-", dir=output) as folder:
        scratch = Path(folder)
        archive = scratch / f"ADO-{tag}-Windows.zip"
        git("archive", "--format=zip", f"--prefix={PREFIX}", "-1", f"--output={archive}",
            commit, "--", *release_paths(tag))
        validate_zip(archive, files, commit, tag, scratch)
        with archive.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        checksums = f"{sha256}  {archive.name}\n"
        artifacts = [archive]
        if film_master is not None:
            media, asset, _ = featured_recording(tag)
            film = scratch / asset
            film_sha256 = stage_release_master(film_master, archive, film, media)
            checksums += f"{film_sha256}  {film.name}\n"
            artifacts.append(film)
        sums = scratch / "SHA256SUMS.txt"
        sums.write_bytes(checksums.encode("ascii"))
        metadata = scratch / "release-metadata.json"
        metadata.write_bytes((json.dumps(dict(tag=tag, commit=commit, archive=archive.name,
                                               bytes=archive.stat().st_size, sha256=sha256), indent=2) + "\n").encode("ascii"))
        for artifact in (*artifacts, metadata, sums):
            artifact.replace(output / artifact.name)
    return output / archive.name, commit, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Git commit SHA or tag for the release tree")
    parser.add_argument("--tag", required=True, help="Release version tag")
    parser.add_argument("--output", type=Path, default=Path(".local/release"))
    parser.add_argument("--film-master", type=Path, help="Include the verified standalone featured chase master")
    args = parser.parse_args()
    try:
        archive, commit, sha256 = build_release(args.ref, args.tag, args.output, args.film_master)
    except (OSError, ValueError, KeyError, BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release build failed: {error}\n")
    print(f"{archive}\nGit commit {commit}\nSHA-256 {sha256}")


if __name__ == "__main__":
    main()
