"""Build and verify an ADO Windows ZIP from a committed release tree."""

import argparse
from collections import Counter, deque
from contextlib import ExitStack, contextmanager
import gzip
import hashlib
from io import BytesIO
from itertools import groupby
import json
from math import dist, isfinite
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import quote
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo


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
LIFE_MEDIA = "web/media/city-life"
LIFE_MASTER = "ADO-City-Life-1080p.mp4"
LIFE_SCHEMA = "ado-city-clearance-v1"
LIFE_RESULTS = "results/city-life"
PROGRESS_MEDIA = "web/media/city-progress"
PROGRESS_MASTER = "ADO-City-Progress-1080p.mp4"
PROGRESS_SCHEMA = "ado-city-progress-v1"
PROGRESS_RESULTS = "results/city-progress"
MANEUVERS_MEDIA = "web/media/city-maneuvers"
MANEUVERS_MASTER = "ADO-City-Maneuvers-1080p.mp4"
MANEUVERS_SCHEMA = "ado-city-maneuvers-v1"
MANEUVERS_RESULTS = "results/city-maneuvers"
MANEUVERS_CAPTURE = f"{MANEUVERS_RESULTS}/capture"
MANEUVERS_AUDIT = f"{MANEUVERS_RESULTS}/audit/v07.json"
MANEUVERS_ROUTE = f"{MANEUVERS_RESULTS}/audit/route-212.json"
MARKET_ROUTE_SHA256 = "9e0628e15f6a27b8d8ed0d3cb06bcc144a97860556028f0f04bda8015ffb9847"
MARKET_LOOP_ROUTE = f"{MANEUVERS_RESULTS}/audit/route-market-loop-212.json"
MARKET_LOOP_ROUTE_SHA256 = "f73c2afaa46d66e5d94b90b9df525074f3e60ab2d76aa77b8a96789f03145f14"
MARKET_LOOP_SURVEY = f"{MANEUVERS_RESULTS}/audit/market-loop-survey.json"
MARKET_PREFIX_SAMPLES = 1439
MARKET_LOOP_SAMPLES = 1737
MARKET_LOOP_LANES = {"in-lane-obstruction": ([67, -2], [67, -1]), "slow-traffic": ([67, 3], [67, 2])}
MANEUVERS_CHOICE = "models/route-maneuvers/choice.json"
MANEUVERS_STOPPING = f"{MANEUVERS_RESULTS}/speed/stopping-envelope.json"
STATIONARY_INTENT_MODEL = "models/city-maneuvers/stationary-intent.npz"
STATIONARY_INTENT_CAPTURE = f"{MANEUVERS_CAPTURE}/stationary-intent.npz"
STATIONARY_INTENT_FIT = "models/city-maneuvers/stationary-intent-fit.json"
STATIONARY_INTENT_SCHEMA = "ado-stationary-vehicle-intent-v1"
STATIONARY_INTENT_FIELD = "stationary_intent_sha256"
STATIONARY_INTENT_CATALOG_KEY = "stationary-vehicle-intent"
DRIVER_BUNDLE_NAME = "stationary-driver.json"
DRIVER_BUNDLE_MODEL = f"models/city-maneuvers/{DRIVER_BUNDLE_NAME}"
DRIVER_BUNDLE_CAPTURE = f"{MANEUVERS_CAPTURE}/{DRIVER_BUNDLE_NAME}"
DRIVER_BUNDLE_FORMAT = "ado-city-driver-bundle-v1"
DRIVER_BUNDLE_FIELD = "driver_bundle_sha256"
DRIVER_BUNDLE_CATALOG_KEY = "city-driver-bundle"
MANEUVERS_TRIALS = f"{MANEUVERS_RESULTS}/trials"
# The City Maneuvers fit ran from a working tree with mixed line endings. The native bend
# trial archived those exact source bytes.
MANEUVERS_FIT_SNAPSHOT = f"{MANEUVERS_RESULTS}/pace/native-v07-bend/source.zip"
MANEUVERS_TRIAL_CASES = {"cones", "board", "slow", "blocked", "no-lead", "departing", "clear-bend", "inside-curb",
                         "outside-curb", "person", "reopened", "closing-rejoin", "uncooperative-rear"}
AMENDED_TRIAL_CASES = ("slow", "blocked")
TRIAL_PRODUCTION_PIN = "9bd6dafa0348d37e163a3acf8de553f642bebf45"
TRIAL_EVALUATOR_PIN = "5510f55c2115f7600b4d56816ce66ffc31121775"
TRIAL_AMENDMENT_PIN = "0e9fa722d117a1efce85bc3e2eeaeac6ffd62ccb"
TRIAL_AMENDMENT = "isolated-fixture-amendment"
CURB_TRIAL_CASES = ("inside-curb", "outside-curb")
CURB_TRIAL_AMENDMENT = "curb-fixture-amendment"
CURB_TRIAL_PIN = "ee6a6582c072a018161cec7c79cb6ad4d4d647bf"
TRIAL_AMENDMENTS = (
    (TRIAL_AMENDMENT, TRIAL_AMENDMENT_PIN, AMENDED_TRIAL_CASES, (67, -2, 94., 231.)),
    (CURB_TRIAL_AMENDMENT, CURB_TRIAL_PIN, CURB_TRIAL_CASES, (67, -2, 48., 145.)),
)
TRIAL_RECEIPTS = ("protocol.json", "qualification.json", "capture-status.json", "checkpoint.npz", "study-source.zip",
                  "tactical-model.json",
                  *(f"{name}.{ext}" for name, *_ in TRIAL_AMENDMENTS for ext in ("json", "zip")))
TRIAL_ARTIFACTS = ("trace.jsonl.gz", "reference.jsonl.gz", "observations.npz")
# The published v0.6 Git tree retains the optional historical raw sweeps.
PROGRESS_ARCHIVE_COMMIT = "51b1cbb25837eb43693b775ccb14e68bffc4084b"
PROGRESS_CROSSING_SOURCE = "city_crossing.py"
PROGRESS_GRADE_SOURCE = "city_grade.py"
POPULATION_EVIDENCE = {
    "population-reference.jsonl.gz": "population_reference_sha256",
    "population-score.json": "population_score_sha256",
    "capture-source.zip": "capture_source_sha256",
}
POPULATED_POLICIES = {LIFE_SCHEMA: (LIFE_RESULTS, "city_clearance.py"),
                      PROGRESS_SCHEMA: (PROGRESS_RESULTS, "city_progress.py"),
                      MANEUVERS_SCHEMA: (MANEUVERS_CAPTURE, "city_maneuvers.py")}
LIFE_BASELINE = f"{LIFE_RESULTS}/baseline-population-score.json"
LIFE_CALIBRATION = "results/roadworks-pace/stopping-calibration.json"
POPULATION_THRESHOLDS = {"car": 1., "pedestrian": .5}
POPULATION_RADII = ("35", "50", "65", "100")
MAX_UPLOAD_BYTES = 2 * 1024**3
EXTERNAL_SWEEPS_INDEX = f"{PROGRESS_RESULTS}/EXTERNAL-SWEEPS.md"
EXTERNAL_RESEARCH_INDEX = f"{PROGRESS_RESULTS}/EXTERNAL-RESEARCH.md"
COMPANION_EVIDENCE_INDEX = f"{MANEUVERS_RESULTS}/COMPANION-EVIDENCE.json"
COMPANION_EVIDENCE_SCHEMA = "ado-companion-evidence-v1"
SWEEP_INDEX_RESERVE = 1024**2
BARRIER_MODEL = "models/barrier-appearance.json"
BARRIER_DATA = "data/barrier-recognition/town03-native.zip"
BARRIER_VALIDATION = "results/barrier-recognition/validation.json"
BARRIER_REVIEW = "barrier-review.json"
BARRIER_WITNESS = "data/barrier-recognition/hero-post-proposals.zip"
BARRIER_POST_MODEL = "models/barrier-post-appearance.npz"
BARRIER_POST_FIT = "results/barrier-recognition/post-appearance-fit.json"
BARRIER_POST_SOURCE = "crosswalk/barrier_posts.py"
BARRIER_REVIEW_SOURCE = "crosswalk/barrier_review.py"
CONE_REVIEW = "cone-observations.json"
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
LIFE_REQUIRED = (
    "crosswalk/city_clearance.py", "crosswalk/city_population.py", "crosswalk/population_score.py",
    "crosswalk/traffic_cones.py",
    "models/city-clearance/candidate.npz", "models/city-clearance/fit.json", "web/cone-observations.mjs",
    f"{LIFE_MEDIA}/manifest.json", f"{LIFE_MEDIA}/replay-pack.json.gz", f"{PACE_MEDIA}/{CONE_REVIEW}",
    LIFE_BASELINE, LIFE_CALIBRATION, *(f"{LIFE_RESULTS}/{name}" for name in POPULATION_EVIDENCE),
)
PROGRESS_REQUIRED = (
    "crosswalk/city_progress.py", f"crosswalk/{PROGRESS_CROSSING_SOURCE}", "crosswalk/city_progress_fit.py",
    "models/city-progress/candidate.npz", "models/city-progress/fit.json",
    f"{PROGRESS_MEDIA}/manifest.json", f"{PROGRESS_MEDIA}/replay-pack.json.gz",
    *(f"{PROGRESS_RESULTS}/{name}" for name in POPULATION_EVIDENCE),
)
MANEUVERS_REQUIRED = (
    "crosswalk/city_maneuvers.py", "crosswalk/city_maneuvers_fit.py", "crosswalk/city_open_road.py",
    "models/city-maneuvers/candidate.npz", "models/city-maneuvers/fit.json",
    MANEUVERS_CHOICE, MANEUVERS_AUDIT, MANEUVERS_ROUTE, MANEUVERS_STOPPING,
    "web/media/city-maneuvers-tests/comparison.mp4",
    f"{MANEUVERS_MEDIA}/manifest.json", f"{MANEUVERS_MEDIA}/replay-pack.json.gz",
    *(f"{MANEUVERS_CAPTURE}/{name}" for name in (*POPULATION_EVIDENCE, "capture.json", "source-checkout.json",
                                                "policy-observations.npz", "tactical-model.json")),
    *(f"{MANEUVERS_TRIALS}/{arm}/{name}" for arm in ("baseline", "candidate") for name in TRIAL_RECEIPTS),
    *(f"{MANEUVERS_TRIALS}/{arm}/{case}/{name}" for arm in ("baseline", "candidate")
      for case in sorted(MANEUVERS_TRIAL_CASES) for name in ("summary.json", "front.mp4", *TRIAL_ARTIFACTS)),
)
CITY_MODEL_FOLDERS = {
    "ado-behavior-v2": "behavior-v2", "ado-city-policy-v1": "city-v1",
    SURROUND_SCHEMA: "city-cruise", PACE_SCHEMA: "city-pace", LIFE_SCHEMA: "city-clearance",
    PROGRESS_SCHEMA: "city-progress", MANEUVERS_SCHEMA: "city-maneuvers",
}
SAFETY_COUNTS = {
    "collision_events": "collision_count", "red_light_violations": "red_light_violations",
    "yield_violations": "yield_violations",
}
CITY_EVENTS = {"pedestrian", "pedestrian-delayed", "occluded", "lead", "braking", "left",
               "signal", "cyclist", "roadworks", "bus"}
MANEUVER_EVENTS = {"in-lane-obstruction", "slow-traffic"}
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


def life_release(tag):
    return tuple(map(int, tag[1:].split("."))) >= (0, 5, 0)


def progress_release(tag):
    return tuple(map(int, tag[1:].split("."))) >= (0, 6, 0)


def maneuvers_release(tag):
    return tuple(map(int, tag[1:].split("."))) >= (0, 7, 0)


def featured_recording(tag):
    if maneuvers_release(tag):
        return MANEUVERS_MEDIA, MANEUVERS_MASTER, MANEUVERS_SCHEMA
    if progress_release(tag):
        return PROGRESS_MEDIA, PROGRESS_MASTER, PROGRESS_SCHEMA
    if life_release(tag):
        return LIFE_MEDIA, LIFE_MASTER, LIFE_SCHEMA
    if pace_release(tag):
        return PACE_MEDIA, PACE_MASTER, PACE_SCHEMA
    return SURROUND_MEDIA, SURROUND_MASTER, SURROUND_SCHEMA


def release_paths(tag):
    return (*PACKAGE_PATHS, "waymo_training", "Watch-City.cmd") if behavior_release(tag) else PACKAGE_PATHS


def required_files(tag):
    if progress_release(tag):
        files = (*REQUIRED, *BEHAVIOR_REQUIRED, *SURROUND_REQUIRED, *PACE_REQUIRED, *LIFE_REQUIRED, *PROGRESS_REQUIRED)
        return (*files, *MANEUVERS_REQUIRED) if maneuvers_release(tag) else files
    if life_release(tag):
        return (*REQUIRED, *BEHAVIOR_REQUIRED, *SURROUND_REQUIRED, *PACE_REQUIRED, *LIFE_REQUIRED)
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


def progress_fit_sources(fit):
    revision = fit.get("revision", 3)
    require(type(revision) is int and revision >= 3, "Unsupported City Progress checkpoint revision")
    require((revision >= 4) == ("grade_source_sha256" in fit), "City Progress grade fit differs from the checkpoint revision")
    sources = {"city_progress.py": fit["source_sha256"], PROGRESS_CROSSING_SOURCE: fit["crossing_source_sha256"]}
    if revision >= 4:
        sources[PROGRESS_GRADE_SOURCE] = fit["grade_source_sha256"]
    return sources


def fit_time_source(archive, name, digest):
    """Return the exact bytes the City Maneuvers fit recorded for one source, from the archived snapshot."""
    recorded = b""
    if PREFIX + MANEUVERS_FIT_SNAPSHOT in archive.namelist():
        with archive.open(PREFIX + MANEUVERS_FIT_SNAPSHOT) as item, ZipFile(item) as snapshot:
            if name in snapshot.namelist():
                recorded = snapshot.read(name)
    require(hashlib.sha256(recorded).hexdigest() == digest,
            f"The recorded City Maneuvers fit used a different execution source: {name}")
    return recorded


def maneuver_fit_sources(archive, fit, policy):
    base = "models/city-progress/candidate.npz"
    inputs = fit["inputs"]
    require(inputs[base] == fit["base_sha256"], "City Maneuvers fit refers to different baseline checkpoints")
    require({"crosswalk/city_maneuvers.py", "crosswalk/city_maneuvers_fit.py"} <= inputs.keys(),
            "City Maneuvers fit is missing its execution and training sources")
    parent = read_json(archive, "models/city-progress/fit.json")
    require(parent["checkpoint_sha256"] == fit["base_sha256"], "Wrong City Maneuvers baseline fit")
    # Both published captures compare inherited modules to the same Git source after line-ending normalization.
    sources = set(progress_fit_sources(parent))
    for name, digest in inputs.items():
        path = PurePosixPath(name)
        require(re.fullmatch(r"[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_.-]+)+", name) and not {".", ".."}.intersection(path.parts),
                f"Invalid City Maneuvers fit input: {name}")
        if path.parent == PurePosixPath("crosswalk") and path.suffix == ".py":
            if policy["code_sha256"].get(path.name) != digest:
                # Same code, different line endings: the fitted bytes must equal the released
                # source after normalization. verify_population_sources ties the capture to it.
                require(fit_time_source(archive, name, digest).replace(b"\r\n", b"\n")
                        == archive.read(PREFIX + name).replace(b"\r\n", b"\n"),
                        f"The recorded City Maneuvers fit used a different execution source: {name}")
            sources.add(path.name)
        else:
            verify_digest(archive, name, digest)
    if MANEUVERS_STOPPING in inputs:
        calibration = read_json(archive, MANEUVERS_STOPPING)
        stopping_fields = ("delay_s", "quadratic_s2_per_m", "guard_m", "max_fit_speed_mps")
        require(calibration["producer"] == CITY_PRODUCER
                and fit["high_speed_stopping"] == [calibration[field] for field in stopping_fields]
                and {"fit", "held_out", "grip_stress"} <= {trial["split"] for trial in calibration["trials"]},
                "City Maneuvers stopping calibration differs from the recorded fit")
        for trial in calibration["trials"]:
            name = trial["trace"]
            margin = trial.get("guarded_margin_m")
            require(re.fullmatch(r"[a-z0-9-]+\.jsonl\.gz", name) and (margin is None or margin >= 0),
                    f"Invalid native City Maneuvers stopping trial: {name}")
            verify_digest(archive, f"{MANEUVERS_RESULTS}/speed/{name}", trial["trace_sha256"])
    return sources


def verify_city_film(archive, media, capture, scratch, tactical_case=None, *, population_baseline=False):
    require(capture["producer"] == CITY_PRODUCER and capture["benchmark_member"] is False
            and capture["teacher"] is False and capture["map_asset"] == "town.json"
            and capture["source_mode"] == "one closed-loop simulation with simultaneous camera capture",
            f"Wrong Town03 film provenance at {media}")
    policy = capture["policy"]
    populated = policy["schema"] in POPULATED_POLICIES
    pace = populated or policy["schema"] == PACE_SCHEMA
    surround = pace or policy["schema"] == SURROUND_SCHEMA
    progress_sources = {}
    folder = CITY_MODEL_FOLDERS.get(policy["schema"])
    require(folder is not None and re.fullmatch(r"[a-zA-Z0-9_-]+\.npz", policy["path"]),
            f"Invalid Town03 policy path at {media}")
    cameras = SURROUND_CAMERAS if surround else ("chase", "front")
    artifacts = CITY_ARTIFACTS | {f"{camera}.mp4" for camera in cameras}
    replay_name = capture.get("replay", "replay.json") if populated else "replay.json"
    require(replay_name in {"replay.json", "replay.json.gz"}, f"Invalid city replay filename: {replay_name}")
    if replay_name == "replay.json.gz":
        artifacts = (artifacts - {"replay.json"}) | {replay_name}
    if tactical_case is not None:
        require(policy["schema"] == SURROUND_SCHEMA, f"Wrong tactical study policy at {media}")
    if tactical_case is not None or pace:
        artifacts.add("replay-pack.json.gz")
    require(artifacts <= capture["sha256"].keys(), f"Incomplete Town03 film at {media}")
    if surround:
        require(tuple(capture.get("perimeter_views", ())) == PERIMETER_VIEWS
                and capture["camera_videos"].keys() == set(SURROUND_CAMERAS),
                f"Wrong surround cameras at {media}")
        tactical_model, tactical_format = (MANEUVERS_CHOICE, "ado-route-maneuvers-1") if policy["schema"] == MANEUVERS_SCHEMA else (
            TACTICAL_MODEL, "ado-tactical-pass-v1")
        verify_digest(archive, tactical_model, policy["tactical_sha256"])
        require(read_json(archive, tactical_model)["format"] == tactical_format,
                f"Invalid tactical model at {media}")
        fit = read_json(archive, f"models/{folder}/fit.json")
        checkpoint_key = "checkpoint_sha256" if populated else "model_sha256"
        require(fit[checkpoint_key] == policy["sha256"] and fit["schema"] == policy["schema"],
                f"Wrong city model fit at {media}")
        if policy["schema"] == LIFE_SCHEMA:
            verify_digest(archive, "models/city-pace/candidate.npz", fit["base_sha256"])
            verify_digest(archive, LIFE_CALIBRATION, fit["calibration_sha256"])
        if policy["schema"] == PROGRESS_SCHEMA:
            verify_digest(archive, "models/city-clearance/candidate.npz", fit["base_sha256"])
            progress_sources = progress_fit_sources(fit)
            for source, digest in progress_sources.items():
                require(digest == policy["code_sha256"].get(source),
                        f"The recording captured a different fitted module at {media}: {source}")
        if policy["schema"] == MANEUVERS_SCHEMA:
            progress_sources = maneuver_fit_sources(archive, fit, policy)
        if policy["schema"] == LIFE_SCHEMA:
            _, source = POPULATED_POLICIES[policy["schema"]]
            require(fit["source_sha256"] == policy["code_sha256"][source],
                    f"The recording captured a different fit source at {media}")
    if pace:
        verify_digest(archive, BARRIER_MODEL, policy["barrier_sha256"])
    for name, digest in capture["sha256"].items():
        require(re.fullmatch(r"[a-zA-Z0-9_.-]+", name), f"Invalid Town03 film artifact at {media}: {name}")
        verify_digest(archive, f"{media}/{name}", digest)

    verify_digest(archive, f"models/{folder}/{policy['path']}", policy["sha256"])
    town = read_json(archive, f"{media}/town.json")
    require(town["producer"] == CITY_PRODUCER and town["roads"], f"Wrong map geometry at {media}")

    summary = capture["summary"]
    if tactical_case is None:
        required_events = CITY_EVENTS | {"barrier"} if pace else CITY_EVENTS
        if policy["schema"] == MANEUVERS_SCHEMA:
            required_events |= MANEUVER_EVENTS
        verify_journey_result(capture, media, required_events)
    else:
        verify_tactical_result(capture, media, tactical_case)

    telemetry = read_json(archive, f"{media}/telemetry.json")
    if replay_name == "replay.json.gz":
        with archive.open(f"{PREFIX}{media}/{replay_name}") as item, gzip.GzipFile(fileobj=item) as decoded:
            contents = decoded.read()
        require(hashlib.sha256(contents).hexdigest() == capture["raw_sha256"]["replay.json"],
                f"Compressed replay differs from the native recording at {media}")
        replay = json.loads(contents)
        del contents
    else:
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
    if populated:
        verify_city_life_score(capture, telemetry, replay)
        verify_city_life_evidence(archive, capture, replay, progress_sources)
    if policy["schema"] == MANEUVERS_SCHEMA and capture["passed"]:
        verify_maneuver_events(capture)
        verify_maneuver_sequence(capture, telemetry)
        verify_maneuver_capture(archive, capture)
        verify_maneuver_audit(archive, capture, telemetry)
        verify_maneuver_trials(archive, capture, scratch)
    if population_baseline:
        score = read_json(archive, LIFE_BASELINE)
        verify_population_metadata(score, capture, len(replay))
        require(score["radii_m"] == population_density(population_rows(replay)),
                "Baseline density claims differ from the archived Urban Pace actor positions")
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


def verify_maneuver_events(capture):
    events = {event["kind"]: event for event in capture["events"]}
    for kind in sorted(MANEUVER_EVENTS):
        event = events[kind]
        before, alongside, rear_clear, returned = (event[key] for key in (
            "before_target_s", "alternate_alongside_s", "rear_cleared_target_s", "returned_to_route_s"))
        require(0 <= before < alongside < rear_clear <= returned <= event["passed_s"] <= capture["summary"]["duration_s"]
                and event["rejoin_station_m"] > event["station_m"] and event["rejoin_rear_clear_m"] > 1.
                and 0.1 <= event["minimum_body_clearance_m"] <= event["alongside_minimum_body_gap_m"]
                and isfinite(event["alongside_minimum_body_gap_m"])
                and event["alongside_ego_lane"] == event["passing_lane"]
                and event["source_lane"][0] == event["passing_lane"][0]
                and event["source_lane"][1] != event["passing_lane"][1],
                f"The native frame sequence does not substantiate a clear pass and return for {kind}")
        if kind == "in-lane-obstruction":
            require(event["target_count"] > 0 and 0 <= event["maximum_target_displacement_m"] <= .12,
                    "The in-lane obstruction moved during the purported stationary bypass")
        else:
            require(before <= event["occupied_gap_wait_s"] < event["adjacent_traffic_cleared_s"] <= alongside
                    and event["occupied_gap_frames"] > 0 and event["moving_target_s"] <= event["moving_alongside_s"] < rear_clear
                    and alongside <= event["moving_alongside_s"]
                    and event["alongside_target_mps"] > .65 and event["target_traveled_m"] > event["target_maximum_step_m"] > 0,
                    "The slow-traffic result lacks a moving target or a wait for the occupied passing lane")


def verify_maneuver_sequence(capture, telemetry):
    events = {event["kind"]: event for event in capture["events"]}
    require(events["in-lane-obstruction"]["returned_to_route_s"] < events["slow-traffic"]["occupied_gap_wait_s"],
            "City Maneuvers needs separate stationary-bypass and moving-pass sequences")
    waited = []
    previous = 0
    for row in telemetry:
        event = next((event for event in row["events"] if event["kind"] == "slow-traffic"), None)
        require(event is not None, f"Missing slow-traffic state at frame {row['frame']}")
        count = event.get("occupied_gap_frames", 0)
        choice = row.get("tactical_pass", {})
        if count > previous and choice.get("maneuver_phase") == "yield" and choice.get("committed") is False:
            waited.append(row["frame"])
        previous = count
    require(waited and capture["maneuver_sequence"] == dict(passed=True, obstruction_rejoin_precedes_occupied_wait=True,
                                                            actual_wait_policy_frames=len(waited),
                                                            first_actual_wait_frame=waited[0],
                                                            last_actual_wait_frame=waited[-1]),
            "The frame-aligned City Maneuvers controls do not show a wait in the occupied passing gap")


def verify_stationary_intent(archive, capture):
    policy = capture["policy"]
    digest = policy.get(STATIONARY_INTENT_FIELD)
    require(capture["summary"].get(STATIONARY_INTENT_FIELD) == digest,
            "City Maneuvers used a different stationary vehicle intent checkpoint from the declared capture policy")
    if STATIONARY_INTENT_FIELD not in policy:
        return
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            "Invalid declared stationary vehicle intent checkpoint SHA-256")
    verify_digest(archive, STATIONARY_INTENT_MODEL, digest)
    verify_digest(archive, STATIONARY_INTENT_CAPTURE, digest)
    fit = read_json(archive, STATIONARY_INTENT_FIT)
    require(fit["schema"] == STATIONARY_INTENT_SCHEMA and fit["checkpoint_sha256"] == digest,
            "Stationary vehicle intent fit describes a different checkpoint")


def verify_driver_bundle(archive, capture):
    policy = capture["policy"]
    digest = policy.get(DRIVER_BUNDLE_FIELD)
    require(capture["summary"].get(DRIVER_BUNDLE_FIELD) == digest,
            "City Maneuvers used a different driver bundle from the declared capture policy")
    if DRIVER_BUNDLE_FIELD not in policy:
        require("driver_bundle_path" not in policy, "City Maneuvers driver bundle path is missing its checksum")
        return
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
            and policy.get("driver_bundle_path") == DRIVER_BUNDLE_NAME,
            "Invalid declared City Maneuvers driver bundle path or SHA-256")
    require(policy["schema"] == MANEUVERS_SCHEMA and policy["path"] == "candidate.npz"
            and policy.get(STATIONARY_INTENT_FIELD) and policy["tactical_sha256"] == policy["maneuver_choice_sha256"],
            "City Maneuvers driver bundle must identify the original pace, active lane choice, and stationary intent")
    verify_digest(archive, DRIVER_BUNDLE_MODEL, digest)
    verify_digest(archive, DRIVER_BUNDLE_CAPTURE, digest)
    bundle = read_json(archive, DRIVER_BUNDLE_MODEL)
    require(bundle == dict(format=DRIVER_BUNDLE_FORMAT, schema=MANEUVERS_SCHEMA,
                           pace=dict(path="candidate.npz", sha256=policy["sha256"]),
                           maneuver_choice=dict(path="../route-maneuvers/choice.json", sha256=policy["tactical_sha256"]),
                           stationary_intent=dict(path="stationary-intent.npz", sha256=policy[STATIONARY_INTENT_FIELD])),
            "City Maneuvers driver bundle components differ from the native capture")
    verify_digest(archive, "models/city-maneuvers/candidate.npz", bundle["pace"]["sha256"])
    verify_digest(archive, MANEUVERS_CHOICE, bundle["maneuver_choice"]["sha256"])


def verify_maneuver_capture(archive, capture):
    source = read_json(archive, f"{MANEUVERS_CAPTURE}/capture.json")
    fields = (
        "capture_id", "trial_id", "producer", "policy", "summary", "passed", "raw_sha256", "first_frame", "last_frame",
        "population_reference_sha256", "population_score_sha256", "capture_source_sha256", "policy_observations_sha256",
        "maneuver_sweeps", "maneuver_sequence", "capture_checkout")
    if capture["summary"].get("destination") == "market-loop":
        fields += ("maneuver_sweep_windows_m",)
    require(all(source[key] == capture[key] for key in fields), "City Maneuvers does not match the sealed native capture")
    require(source["raw_sha256"]["telemetry.json"] == capture["sha256"]["telemetry.json"],
            "City Maneuvers telemetry differs from the sealed capture")
    checkout = read_json(archive, f"{MANEUVERS_CAPTURE}/source-checkout.json")
    require(checkout == capture["capture_checkout"] and checkout["source_sha256"] == capture["capture_source_sha256"],
            "City Maneuvers checksum for the execution sources differs from the capture checkout")
    require(capture["policy"]["maneuver_choice_sha256"] == capture["policy"]["tactical_sha256"],
            "City Maneuvers must identify the active lane-choice checkpoint")
    verify_digest(archive, f"{MANEUVERS_CAPTURE}/tactical-model.json", capture["policy"]["tactical_sha256"])
    verify_stationary_intent(archive, capture)
    verify_driver_bundle(archive, capture)
    verify_digest(archive, f"{MANEUVERS_CAPTURE}/policy-observations.npz", capture["policy_observations_sha256"])
    sweeps = capture["maneuver_sweeps"]
    require(sweeps, "City Maneuvers is missing native passing sensor observations")
    prefix = f"{MANEUVERS_CAPTURE}/maneuver-sweeps/"
    recorded = {info.filename.removeprefix(PREFIX + prefix) for info in archive.infolist()
                if not info.is_dir() and info.filename.startswith(PREFIX + prefix)}
    require(recorded == sweeps.keys(), "City Maneuvers sensor observations differ from the capture inventory")
    for name, digest in sweeps.items():
        match = re.fullmatch(r"frame-([0-9]+)\.npz", name)
        require(match and capture["first_frame"] <= int(match[1]) <= capture["last_frame"],
                f"City Maneuvers sensor sample is outside the native capture: {name}")
        verify_digest(archive, f"{prefix}{name}", digest)


def verify_market_loop_route(archive, source, summary):
    require(source["route_sha256"] == MARKET_LOOP_ROUTE_SHA256, "Market Loop audit uses a different registered route")
    verify_digest(archive, MARKET_LOOP_ROUTE, MARKET_LOOP_ROUTE_SHA256)
    verify_digest(archive, MANEUVERS_ROUTE, MARKET_ROUTE_SHA256)
    route = read_json(archive, MARKET_LOOP_ROUTE)
    original = read_json(archive, MANEUVERS_ROUTE)
    prefix_end = original["route_length_m"]
    samples = route["samples"]
    require(route["destination"] == "market-loop" and route["route_id"] == source["route_id"]
            and route["opendrive_sha256"] == original["opendrive_sha256"] == source["map_sha256"]
            and route["seed"] == original["seed"] == summary["seed"] == 212
            and len(original["samples"]) == route["shared_market_prefix_count"] == MARKET_PREFIX_SAMPLES
            and len(samples) == MARKET_LOOP_SAMPLES and original["samples"] == samples[:MARKET_PREFIX_SAMPLES]
            and route["shared_market_prefix_to_m"] == prefix_end == original["samples"][-1]["station_m"]
            and abs(route["route_length_m"] - summary["route_length_m"]) <= .001,
            "Market Loop does not contain the exact surveyed Market route prefix and registered continuation")

    survey = read_json(archive, MARKET_LOOP_SURVEY)
    extension, prefix = survey["proposed_survey"], survey["identical_original_prefix"]
    require(extension["file"] == PurePosixPath(MARKET_LOOP_ROUTE).name and extension["sha256"] == MARKET_LOOP_ROUTE_SHA256
            and extension["route_id"] == route["route_id"] and extension["route_length_m"] == route["route_length_m"]
            and extension["samples"] == MARKET_LOOP_SAMPLES and extension["junction_traversals"] == 14
            and prefix["file"] == PurePosixPath(MANEUVERS_ROUTE).name and prefix["sha256"] == MARKET_ROUTE_SHA256
            and prefix["route_id"] == original["route_id"] and prefix["samples"] == MARKET_PREFIX_SAMPLES
            and prefix["cutoff_m"] == prefix_end and prefix["junction_traversals"] == 11
            and prefix["every_sample_and_lane_marking_identical"] is True,
            "Market Loop survey receipt differs from the archived route and prefix")

    starts, stops = [], []
    for i, (left, right) in enumerate(zip(samples, samples[1:]), start=1):
        if left["junction"] is None and right["junction"] is not None:
            starts.append(i)
        if left["junction"] is not None and right["junction"] is None:
            stops.append(i)
    traversals = summary["junction_traversals"]
    ids = {samples[i]["junction"] for i in starts}
    require(len(starts) == len(stops) == len(traversals) == 14 and len(ids) == 11
            and sorted(ids) == extension["unique_junction_ids"],
            "Market Loop must complete 14 registered traversals over 11 distinct native junctions")
    previous_exit = 0.
    for recorded, start, stop in zip(traversals, starts, stops, strict=True):
        entry, exit = samples[start], samples[stop]
        yaw = (exit["heading_deg"] - samples[start - 1]["heading_deg"] + 180.) % 360. - 180.
        direction = "left" if yaw < -50. else "right" if yaw > 50. else "straight"
        require(recorded["junction_id"] == entry["junction"] and recorded["direction"] == direction
                and recorded["entry_m"] == round(entry["station_m"], 2) and recorded["exit_m"] == round(exit["station_m"], 2)
                and 0 <= previous_exit <= recorded["entered_s"] < recorded["completed_s"] <= summary["duration_s"],
                "Market Loop native junction sequence differs from the surveyed geometry or has incomplete traversal")
        previous_exit = recorded["completed_s"]
    return route, original


def verify_market_loop_stats(statistics, rows, period):
    require(statistics["frames"] == len(rows), "Market Loop statistics omit or duplicate native frames")
    if not rows:
        return
    require(statistics["seconds"] == round(period * len(rows), 3),
            "Market Loop statistics disagree with the native exposure duration")
    for point, row in ((statistics["entry"], rows[0]), (statistics["exit"], rows[-1])):
        require(point["frame"] == row["frame"] and point["time_s"] == row["sim_time"]
                and point["station_m"] == row["progress_m"]
                and point["actual_kmh"] == round(3.6 * row["speed_mps"], 3)
                and point["target_kmh"] == round(3.6 * row["speed_request_mps"], 3),
                "Market Loop speed island endpoints differ from the native time, station or speed")


def verify_market_loop_timeline(audit, telemetry, original, period):
    cutoff = original["route_length_m"]
    comparison = audit["route_comparison"]
    require(comparison["original_route_id"] == original["route_id"]
            and comparison["original_route_sha256"] == MARKET_ROUTE_SHA256 and comparison["shared_route_to_m"] == cutoff,
            "Market Loop audit comparison uses a different original route or prefix boundary")
    portions = {name: [row for row in telemetry if (row["progress_m"] <= cutoff) == shared]
                for name, shared in (("shared_prefix", True), ("extension", False))}
    for name, rows in portions.items():
        verify_market_loop_stats(comparison[name]["whole"], rows, period)
        verify_market_loop_stats(comparison[name]["posted_90"], [row for row in rows if row["posted_limit_kmh"] == 90], period)

    observed = [(posted, list(rows)) for posted, rows in groupby(telemetry, key=lambda row: row["posted_limit_kmh"])]
    require(len(audit["signed_limit_islands"]) == len(observed), "Market Loop audit must preserve each native signed-limit island")
    native_ninety = []
    for report, (posted, rows) in zip(audit["signed_limit_islands"], observed, strict=True):
        require(report["posted_kmh"] == posted, "Market Loop signed-limit island disagrees with native telemetry")
        verify_market_loop_stats(report, rows, period)
        shared = [row for row in rows if row["progress_m"] <= cutoff]
        extension = [row for row in rows if row["progress_m"] > cutoff]
        verify_market_loop_stats(report["shared_prefix"], shared, period)
        verify_market_loop_stats(report["extension"], extension, period)
        if posted == 90:
            native_ninety.append((bool(shared), bool(extension)))
            speeds = [row["speed_mps"] * 3.6 for row in rows]
            require(report["kmh_p10_p50_p95_max"][-1] == round(max(speeds), 3)
                    and report["over_40_kmh_seconds"] == round(period * sum(speed > 40 for speed in speeds), 3),
                    "Market Loop 90 km/h island pace differs from native speed")
            for level in (45, 50, 55, 60):
                observed_level = report["higher_thresholds"][str(level)]
                require(observed_level["actual_seconds"] == round(period * sum(speed >= level for speed in speeds), 3)
                        and observed_level["requested_seconds"] == round(period * sum(
                            row["speed_request_mps"] * 3.6 >= level for row in rows), 3),
                        "Market Loop measured and requested speed durations differ from a native 90 km/h island")
    require(native_ninety == [(True, False), (False, True)],
            "Market Loop needs two distinct measured 90 km/h islands on the shared route and the extension")


def verify_market_loop_sweep_coverage(capture, telemetry, route):
    reverse = [sample["station_m"] for sample in route["samples"] if (sample["road"], sample["lane"]) == (67, 3)]
    require(reverse and reverse[-1] == route["route_length_m"], "Wrong registered Market Loop boulevard extension")
    windows = [[399., 548.], [round(reverse[0], 3), round(reverse[-1], 3) + .001]]
    require(capture["maneuver_sweep_windows_m"] == windows, "Market Loop sensor witness windows differ from the route")
    expected = {}
    for start, end in windows:
        window_count = len(expected)
        for row in telemetry:
            station = row["progress_m"]
            if start <= station < end:
                key = (int(station // 5), row.get("tactical_pass", {}).get("maneuver_phase"))
                expected.setdefault(key, row["frame"])
        require(len(expected) > window_count, "Market Loop capture is missing a native sensor witness region")
    frames = set(expected.values())
    require({f"frame-{frame}.npz" for frame in frames} == capture["maneuver_sweeps"].keys(),
            "Market Loop sensor sweeps must cover the first native exposure of every occupied station and maneuver bin")
    return frames


def verify_market_loop_lane_move(move, event, native_rows, witness_frames):
    source, target = MARKET_LOOP_LANES[event["kind"]]
    require(event["source_lane"] == source and event["passing_lane"] == target
            and [move["road"], move["source_lane"]] == source and [move["road"], move["alternate_lane"]] == target
            and move["side"] == "left", "Market Loop pass used a different signed lane pair")
    start, finish = move["footprint_departs"]["frame"], move["restored_source"]["frame"]
    exposures = move["lane_exposures"]
    states = ("body_in_source", "center_in_alternate", "body_in_alternate", "body_within_legal_union")
    require(start < finish and [row["frame"] for row in exposures] == list(range(start, finish + 1))
            and all(type(row[field]) is bool for row in exposures for field in states)
            and all(row["body_within_legal_union"] for row in exposures)
            and all(not row["body_in_source"] for row in exposures[:-1]) and exposures[-1]["body_in_source"]
            and not exposures[-1]["center_in_alternate"]
            and all(not row["body_in_alternate"] or row["center_in_alternate"] for row in exposures)
            and move["last_alternate_sample"]["frame"] == finish - 1
            and next((row["frame"] for row in exposures if row["center_in_alternate"]), None) == move["center_crosses"]["frame"]
            and next((row["frame"] for row in exposures if row["body_in_alternate"]), None)
            == move["whole_body_in_alternate"]["frame"]
            and any(start <= frame <= finish for frame in witness_frames),
            "Market Loop pass is missing contiguous legal lane and native sensor evidence through the return")
    committed_frames = 0
    for exposure in exposures:
        row = native_rows[exposure["frame"]]
        tactical = row.get("tactical_pass", {})
        native_legal = row.get("passing_footprint_in_legal_lanes")
        require(native_legal is not False, "Market Loop native lane telemetry contradicts the independent footprint audit")
        if tactical.get("committed") is True:
            require(tactical["source_lane"] == source and tactical["target_lane"] == target and native_legal is True,
                    "Market Loop committed-action telemetry uses a different lane pair or footprint")
            committed_frames += 1
    require(committed_frames, "Market Loop pass is missing committed native controller frames")


def verify_maneuver_audit(archive, capture, telemetry):
    audit = read_json(archive, MANEUVERS_AUDIT)
    source, whole, grade = (audit[key] for key in ("source", "whole", "grade"))
    summary = capture["summary"]
    require(audit["schema"] == "ado-city-maneuver-audit-v1" and source["seal_verified"] is True
            and source["quality"] == capture["quality_requested"] == "High" and source["producer"] == CITY_PRODUCER
            and source["capture_id"] == capture["capture_id"] and source["route_id"] == summary["route_id"]
            and source["policy_sha256"] == capture["policy"]["sha256"]
            and source["telemetry_sha256"] == capture["sha256"]["telemetry.json"]
            and source["population_sha256"] == capture["population_reference_sha256"],
            "City Maneuvers audit refers to a different recording")
    destination = summary.get("destination")
    require(destination in ("market", "market-loop"), "City Maneuvers needs a registered Market destination")
    loop = destination == "market-loop"
    if loop:
        route, original = verify_market_loop_route(archive, source, summary)
        verify_market_loop_timeline(audit, telemetry, original, capture["frame_period_s"])
        witness_frames = verify_market_loop_sweep_coverage(capture, telemetry, route)
        native_rows = {row["frame"]: row for row in telemetry}
    else:
        require(source["route_sha256"] == MARKET_ROUTE_SHA256, "City Maneuvers audit uses a different registered Market route")
        verify_digest(archive, MANEUVERS_ROUTE, MARKET_ROUTE_SHA256)
        route = read_json(archive, MANEUVERS_ROUTE)
        require(route["route_id"] == source["route_id"] and route["opendrive_sha256"] == source["map_sha256"],
                "City Maneuvers audit uses different OpenDRIVE geometry")
    require(whole["entry"]["frame"] == capture["first_frame"] and whole["exit"]["frame"] == capture["last_frame"]
            and source["first_frame"] == capture["first_frame"] and source["last_frame"] == capture["last_frame"]
            and whole["frames"] == len(telemetry) and grade["participant_contacts"] == []
            and all(grade[key] == summary[key] for key in ("frames", "outcome", "progress_m", "route_length_m", "events",
                                                           *SAFETY_COUNTS))
            and all(counter["overlap_samples"] == 0 for counter in audit["independent_clearance"].values()),
            "The independent City Maneuvers safety audit differs from the synchronized native evidence")
    separation = audit["straddling_moving_separation"]
    require(separation["minimum_required_straddle_separation_m"] == 1.
            and separation["scored_frames_with_moving_counterpart"] > 0 and separation["sampled_pass"] is True
            and separation["sub_1m_samples"] == separation["overlap_samples"] == 0,
            "City Maneuvers has insufficient measured separation from moving traffic during a lane change")

    period = capture["frame_period_s"]
    fast_rows = [row for row in telemetry if row["posted_limit_kmh"] == 90]
    fast_zone = [row["speed_mps"] * 3.6 for row in fast_rows]
    posted = audit["posted_90"]
    require(fast_zone and posted["frames"] == len(fast_zone) and posted["seconds"] == round(period * len(fast_zone), 3)
            and posted["kmh_p10_p50_p95_max"][-1] == round(max(fast_zone), 3)
            and posted["seconds_at_or_above_35"] == round(period * sum(speed >= 35 for speed in fast_zone), 3)
            and posted["seconds_at_or_above_37"] == round(period * sum(speed >= 37 for speed in fast_zone), 3)
            and posted["over_40_kmh_seconds"] == round(period * sum(speed > 40 for speed in fast_zone), 3)
            and not any(row["speed_mps"] * 3.6 > row["posted_limit_kmh"] + 1 for row in telemetry),
            "City Maneuvers pace scores differ from the native speed or posted-limit measurements")
    for level in (45, 50, 55, 60):
        measure = posted["higher_thresholds"][str(level)]
        require(measure["actual_seconds"] == round(period * sum(speed >= level for speed in fast_zone), 3)
                and measure["requested_seconds"] == round(period * sum(row["speed_request_mps"] * 3.6 >= level
                                                                     for row in fast_rows), 3),
                f"City Maneuvers timing above {level} km/h differs from the native track")

    needed = {(event["kind"], tuple(event["alongside_ego_lane"])): event for event in summary["events"]
              if event["kind"] in MANEUVER_EVENTS}
    witnessed = set()
    for move in audit["actual_lateral_departures"]:
        alternate = move["whole_body_in_alternate"]
        restored = move["restored_source"]
        if (alternate is None or restored is None or move["shoulder_or_departure_samples"]
                or move["first_native_legality_failure"] is not None):
            continue
        for actor, groups in move["alongside"].items():
            first = groups["first"]
            ahead = move["ahead_with_sampled_clearance"].get(actor)
            if first["role"] == "slow-traffic" and move["moving_vehicle_clearance"]["scored_frames_with_moving_counterpart"] <= 0:
                continue
            if (ahead and groups["nearest"]["body_clearance_m"] > 0
                    and capture["first_frame"] <= first["frame"] < ahead["frame"] <= restored["frame"]
                    <= capture["last_frame"]):
                key = (first["role"], (move["road"], move["alternate_lane"]))
                if loop and key in needed and key not in witnessed:
                    verify_market_loop_lane_move(move, needed[key], native_rows, witness_frames)
                witnessed.add(key)
    require(needed.keys() <= witnessed, "The independent audit does not show both actual passes in the legal alternate lane")


def pinned_trial_sources(revision):
    renderer = "tools/render_maneuver_trials.py"
    contents = git("archive", "--format=zip", revision, "--", "crosswalk", renderer)
    with ZipFile(BytesIO(contents)) as source:
        return {name: source.read(name).replace(b"\r\n", b"\n") for name in source.namelist()
                if name == renderer or re.fullmatch(r"crosswalk/[a-zA-Z0-9_]+\.py", name)}


def verify_trial_sources(archive, name, digest, pinned):
    verify_digest(archive, name, digest)
    with archive.open(PREFIX + name) as blob, ZipFile(blob) as source:
        names = source.namelist()
        require(len(names) == len(pinned) and set(names) == pinned.keys(), f"Wrong pinned source inventory at {name}")
        hashes = {}
        for member in names:
            data = source.read(member)
            require(data.replace(b"\r\n", b"\n") == pinned[member], f"Wrong pinned source at {name}: {member}")
            hashes[member] = hashlib.sha256(data).hexdigest()
    return hashes


def verify_trial_amendments(archive, folder, protocol, sources):
    registered = {}
    archives = {}
    same_conditions = ("seed", "weather", "map", "step_s", "requested_quality", "policy_schema", "model_sha256",
                       "tactical_model_sha256")
    launch_stations = protocol["launch_route_stations_m"]
    for name, pin, cases, window in TRIAL_AMENDMENTS:
        amendment = read_json(archive, f"{folder}/{name}.json")
        windows = dict.fromkeys(cases, list(window))
        require(amendment["registration_pin"] == pin and amendment["base_evaluator_pin"] == TRIAL_EVALUATOR_PIN
                and amendment["production_pin"] == TRIAL_PRODUCTION_PIN and amendment["cases"] == list(cases)
                and amendment["seed"] == 212 and not (registered.keys() & set(cases))
                and amendment["windows"] == windows == {case: protocol["windows"][case] for case in cases}
                and amendment.get("launch_route_stations_m", {}) == {case: launch_stations[case] for case in cases
                                                                    if case in launch_stations}
                and all(amendment[key] == protocol[key] for key in same_conditions),
                f"Wrong registered {name} at {folder}")
        archives[name] = verify_trial_sources(archive, f"{folder}/{name}.zip", amendment["execution_source_sha256"],
                                              sources[pin])
        registered.update(dict.fromkeys(cases, amendment))
    return registered, archives


def verify_curb_trial_placement(summary, first_reference):
    case, geometry = summary["case"], summary["geometry"]
    placement = summary["curb_placement"]
    require(placement == first_reference.get("curb_placement") and placement["verified"] is True
            and placement["frame"] == first_reference["frame"] == summary["first_frame"]
            and placement["blueprint"] == geometry["board_blueprint"] == "static.prop.streetbarrier"
            and type(placement["actor_id"]) is int and placement["actor_id"] > 0,
            f"Missing verified first-exposure curb placement for {case}")
    footprint = placement["polygon_xy"]
    observed = [target for target in first_reference["target_polygons"]
                if target["role"] == case and target["actor_id"] == placement["actor_id"]]
    require(len(footprint) >= 2 and all(len(point) == 2 and all(map(isfinite, point)) for point in footprint)
            and len({tuple(point) for point in footprint}) >= 2 and len(observed) == 1
            and observed[0]["polygon_xy"] == footprint,
            f"Missing the physical curb board in the first native reference for {case}")

    position = placement["actual_center_xy"]
    error = dist(position, geometry["requested_center_xy"])
    lo, hi = placement["physical_lateral_extent_m"]
    planned_lo, planned_hi = geometry["board_lateral_extent_m"]
    edge = geometry["sidewalk_edge_m"]
    correct_side = lo > geometry["driving_edge_m"] and hi < edge - .05 if case == "inside-curb" else lo > edge + .05
    require(geometry["board_native_s"] == 84. and isfinite(lo) and isfinite(hi) and lo <= hi
            and error < .05 and abs(error - placement["center_error_m"]) <= .0000051
            and max(abs(lo - planned_lo), abs(hi - planned_hi)) < .05 and correct_side,
            f"Native curb board placement differs from the registered geometry for {case}")
    return position, (lo, hi), sorted(map(tuple, footprint))


def verify_maneuver_trials(archive, capture, scratch):
    fit = read_json(archive, "models/city-maneuvers/fit.json")
    baseline = read_json(archive, f"{PROGRESS_MEDIA}/manifest.json")["policy"]
    candidate = capture["policy"]
    arms = (("baseline", PROGRESS_SCHEMA, fit["base_sha256"], baseline["tactical_sha256"]),
            ("candidate", MANEUVERS_SCHEMA, candidate["sha256"], candidate["tactical_sha256"]))
    original_sources = pinned_trial_sources(TRIAL_EVALUATOR_PIN)
    amended_sources = {pin: pinned_trial_sources(pin) for _, pin, _, _ in TRIAL_AMENDMENTS}
    common_settings = ("seed", "weather", "map", "step_s", "requested_quality")
    matched = {}
    for arm, schema, checkpoint, tactical in arms:
        folder = f"{MANEUVERS_TRIALS}/{arm}"
        protocol_path = f"{folder}/protocol.json"
        protocol = read_json(archive, protocol_path)
        statuses = read_json(archive, f"{folder}/capture-status.json")
        require(protocol["policy_schema"] == schema and protocol["model_sha256"] == checkpoint
                and protocol["tactical_model_sha256"] == tactical
                and len(protocol["cases"]) == len(MANEUVERS_TRIAL_CASES)
                and set(protocol["cases"]) == protocol["windows"].keys() == statuses.keys() == MANEUVERS_TRIAL_CASES
                and protocol["step_s"] == capture["frame_period_s"] and protocol["map"] == CITY_PRODUCER["map"]
                and protocol["requested_quality"] == capture["quality_requested"],
                f"Wrong City Maneuvers trial protocol for {arm}")
        qualification = read_json(archive, f"{folder}/qualification.json")
        require(qualification["pin"] == TRIAL_PRODUCTION_PIN and qualification["evaluator_pin"] == TRIAL_EVALUATOR_PIN
                and qualification["arm"] == arm and qualification["model_sha256"] == checkpoint,
                f"Wrong City Maneuvers trial qualification for {arm}")
        verify_digest(archive, protocol_path, qualification["native_protocol_sha256"])
        verify_digest(archive, f"{folder}/checkpoint.npz", checkpoint)
        verify_digest(archive, f"{folder}/tactical-model.json", protocol["tactical_model_sha256"])
        original = verify_trial_sources(archive, f"{folder}/study-source.zip", protocol["execution_source_sha256"],
                                        original_sources)

        amended_protocols, revised = verify_trial_amendments(archive, folder, protocol, amended_sources)
        trial_geometries = {}
        curb_placements = {}
        for case in protocol["cases"]:
            trial = f"{folder}/{case}"
            summary = read_json(archive, f"{trial}/summary.json")
            geometry = summary["geometry"]
            trial_geometries[case] = geometry
            trial_protocol = amended_protocols.get(case, protocol)
            require(statuses[case] is summary["capture_complete"] is True and summary["case"] == case
                    and summary["policy_schema"] == schema and summary["model_sha256"] == checkpoint
                    and summary["producer"] == CITY_PRODUCER and summary["seed"] == protocol["seed"]
                    and summary["execution_source_sha256"] == trial_protocol["execution_source_sha256"]
                    and [geometry[key] for key in ("road_id", "lane_id", "start_s", "end_s")] == trial_protocol["windows"][case]
                    and summary["last_frame"] - summary["first_frame"] + 1 == summary["frames"] > 0,
                    f"Wrong native trial identity or exposure interval at {trial}")
            for name in TRIAL_ARTIFACTS:
                verify_digest(archive, f"{trial}/{name}", summary["artifact_sha256"][name])
                if name.endswith(".gz"):
                    with archive.open(f"{PREFIX}{trial}/{name}") as compressed, gzip.GzipFile(fileobj=compressed) as rows:
                        first = json.loads(next(rows, b"{}"))
                        frames = [first.get("frame"), *(json.loads(row)["frame"] for row in rows)]
                    require(frames == list(range(summary["first_frame"], summary["last_frame"] + 1)),
                            f"Unsynchronized {name} exposures at {trial}")
                    if name == "reference.jsonl.gz" and case in CURB_TRIAL_CASES:
                        curb_placements[case] = verify_curb_trial_placement(summary, first)
            verify_digest(archive, f"{trial}/front.mp4", summary["front_sha256"])
            verify_video(archive, f"{trial}/front.mp4", summary["frames"], scratch)
        matched[arm] = (tuple(protocol[key] for key in common_settings), protocol["launch_route_stations_m"],
                        trial_geometries, curb_placements, original, revised)
    require(matched["baseline"] == matched["candidate"],
            "City Maneuvers benchmarks used different native fixture geometry, conditions or evaluator sources")


def verify_city_life_score(capture, telemetry, replay):
    summary = capture["summary"]
    schema = capture["policy"]["schema"]
    profile = "city-maneuvers" if schema == MANEUVERS_SCHEMA else "city-life"
    require(capture["population_profile"] == profile and summary["policy"] == schema
            and summary["scenario"] == "journey" and capture["frame_period_s"] == 0.1
            and summary["duration_s"] == round(len(telemetry) * capture["frame_period_s"], 2),
            "City Life summary does not match the recorded route and timeline")
    for index, (score, exposure) in enumerate(zip(telemetry, replay, strict=True)):
        require(all(type(score[key]) is int and score[key] >= 0 and score[key] == exposure[key]
                    for key in SAFETY_COUNTS.values())
                and score["sim_time"] == exposure["sim_time"] == score["video_time"] == exposure["video_time"]
                == round(index * capture["frame_period_s"], 6),
                f"City Life safety telemetry differs from its camera exposure at frame {score['frame']}")
    require(all(summary[key] == max(row[field] for row in telemetry) for key, field in SAFETY_COUNTS.items())
            and all(summary[key] == telemetry[-1][key] for key in ("progress_m", "route_length_m", "events")),
            "City Life route or safety totals differ from the scored telemetry")


def verify_city_life_population(capture, archive_capture, baseline):
    summary = capture["summary"]
    previous = archive_capture["summary"]
    walkers, cars = summary["sidewalk_activity"], summary["background_traffic_activity"]
    counts = summary["background_actors"]
    prior_walkers = previous["sidewalk_activity"]["moved_on_approach"]
    prior_cars = previous["background_traffic_activity"]["moving_near_ego"]
    require(counts["walkers"] == walkers["spawned"] == len(walkers["motion_on_approach_m"])
            and counts["vehicles"] == cars["spawned"] == len(cars["vehicles"])
            and prior_walkers < walkers["moved_on_approach"] <= counts["walkers"]
            and prior_cars < cars["moving_near_ego"] <= counts["vehicles"],
            "City Life does not substantiate more moving background pedestrians and vehicles than Urban Pace")
    require(cars["participant_contacts"] == [] and all(not row["left_corridor"] for row in cars["vehicles"]),
            "City Life background traffic has participant contacts or left a driving corridor")
    old = baseline["radii_m"]["50"]
    new = capture["population_density"]["50"]
    require(new["both_frames"] * old["scored_frames"] > old["both_frames"] * new["scored_frames"],
            "City Life does not improve the share of frames with both types of moving actors within 50 m")


def verify_population_sources(archive, capture, results, policy_source, progress_sources=()):
    source_hashes = {f"crosswalk/{name}": digest for name, digest in capture["policy"]["code_sha256"].items()
                     if re.fullmatch(r"[a-zA-Z0-9_]+\.py", name)}
    fitted_sources = {f"crosswalk/{name}" for name in progress_sources}
    if capture["policy"]["schema"] == MANEUVERS_SCHEMA:
        fitted_sources.update(f"crosswalk/{name}" for name in (
            "city_drive.py", "drive.py", "journey.py", "journey_capture.py", "maneuver_choice.py", "maneuver_map.py",
            "maneuver_scene.py", "route_maneuvers.py", "city_population.py", "population_score.py"))
        if capture.get("summary", {}).get("destination") == "market-loop":
            fitted_sources.update(("crosswalk/urban_map.py", "crosswalk/market_loop_traffic.py"))
        if STATIONARY_INTENT_FIELD in capture["policy"]:
            fitted_sources.add("crosswalk/stationary_intent.py")
        if DRIVER_BUNDLE_FIELD in capture["policy"]:
            fitted_sources.update(("crosswalk/driver_bundle.py", "crosswalk/motor_queue.py", "crosswalk/source_vehicle.py"))
    required_sources = fitted_sources | {f"crosswalk/{policy_source}", "crosswalk/city_clearance.py",
                                         "crosswalk/city_population.py", "crosswalk/journey_capture.py",
                                         "crosswalk/population_score.py"}
    with archive.open(f"{PREFIX}{results}/capture-source.zip") as source_file, ZipFile(source_file) as source:
        require(required_sources <= source_hashes.keys()
                and len(source.namelist()) == len(source_hashes) == len(capture["policy"]["code_sha256"])
                and set(source.namelist()) == source_hashes.keys(), "Incomplete City Life execution sources")
        for name, digest in source_hashes.items():
            recorded = source.read(name)
            require(hashlib.sha256(recorded).hexdigest() == digest, f"Wrong recorded execution source: {name}")
            if name in fitted_sources:
                # Git archives use repository line endings. The simulator snapshot keeps checkout bytes.
                committed = archive.read(PREFIX + name)
                require(committed.replace(b"\r\n", b"\n") == recorded.replace(b"\r\n", b"\n"),
                        f"The release and capture use different fit sources: {name}")


def verify_city_life_evidence(archive, capture, replay, progress_sources=()):
    results, policy_source = POPULATED_POLICIES[capture["policy"]["schema"]]
    for name, digest_field in POPULATION_EVIDENCE.items():
        verify_digest(archive, f"{results}/{name}", capture[digest_field])
    verify_population_sources(archive, capture, results, policy_source, progress_sources)

    score = read_json(archive, f"{results}/population-score.json")
    verify_population_metadata(score, capture, len(replay))
    with archive.open(f"{PREFIX}{results}/population-reference.jsonl.gz") as stream, gzip.GzipFile(fileobj=stream) as refs:
        for ref, row in zip(refs, population_rows(replay), strict=True):
            require(json.loads(ref) == row, f"Population reference differs from City Life exposure at frame {row['frame']}")
    density = population_density(population_rows(replay))
    require(density == score["radii_m"] == capture["population_density"],
            "Population density claims differ from the recorded actor positions")
    if capture["passed"]:
        require(density["35"]["two_of_each_frames"] > 0, "The populated drive has no qualifying frames within 35 m")
        require(capture["summary"]["background_traffic_activity"]["participant_contacts"] == [],
                "The populated drive recorded background participant contacts")


def verify_population_metadata(score, capture, count):
    require(score["schema"] == "ado-city-population-score-v1" and score["speed_threshold_mps"] == POPULATION_THRESHOLDS
            and (score["first_frame"], score["last_frame"]) == (capture["first_frame"], capture["last_frame"])
            and score["capture_frames"] == count and score["motion_window_s"] == 0.5,
            "Population score differs from the captured timeline")


def population_rows(replay):
    for frame in replay:
        observer = frame["observer_map"]
        yield {"frame": frame["frame"], "sim_time": frame["sim_time"], "progress_m": frame["progress_m"],
               "ego": observer["ego"], "others": observer["others"], "signal_reference": observer.get("signal_reference")}


def population_density(rows):
    histories = {}
    stats = {radius: Counter(scored_frames=0, moving_car_frames=0, moving_pedestrian_frames=0,
                             both_frames=0, two_of_each_frames=0) for radius in POPULATION_RADII}
    peaks = {radius: Counter(cars=0, pedestrians=0) for radius in POPULATION_RADII}
    representatives = {radius: ((-1, -1), None, 0, 0) for radius in POPULATION_RADII}
    for index, row in enumerate(rows):
        present = set()
        counts = {radius: Counter() for radius in POPULATION_RADII}
        for actor in row["others"]:
            kind = {"car": "car", "van": "car", "bus": "car", "pedestrian": "pedestrian"}.get(actor["type"])
            if kind is None:
                continue
            key, xy = actor["actor_id"], (actor["x"], actor["y"])
            require(key not in present, f"Duplicate actor at native frame {row['frame']}")
            present.add(key)
            history = histories.setdefault(key, deque(maxlen=6))
            if history and dist(xy, history[-1][1]) > 3:
                history.clear()
            history.append((row["sim_time"], xy))
            if len(history) == 6:
                old_time, old_xy = history[0]
                if dist(xy, old_xy) >= POPULATION_THRESHOLDS[kind] * (row["sim_time"] - old_time):
                    distance = dist(row["ego"][:2], xy)
                    for radius, count in counts.items():
                        count[kind] += distance <= int(radius)
        for key in histories.keys() - present:
            del histories[key]
        if index < 5:
            continue
        for radius, count in counts.items():
            tally = stats[radius]
            cars, people = count["car"], count["pedestrian"]
            tally["scored_frames"] += 1
            tally["moving_car_frames"] += cars > 0
            tally["moving_pedestrian_frames"] += people > 0
            tally["both_frames"] += min(cars, people) >= 1
            tally["two_of_each_frames"] += min(cars, people) >= 2
            peak = peaks[radius]
            peak["cars"] = max(peak["cars"], cars)
            peak["pedestrians"] = max(peak["pedestrians"], people)
            rank = (min(cars, people), cars + people)
            if rank > representatives[radius][0]:
                representatives[radius] = (rank, row["frame"], cars, people)
    density = {}
    for radius, (_, frame, cars, people) in representatives.items():
        peak = {**peaks[radius], "simultaneous_frame": frame, "simultaneous_cars": cars, "simultaneous_pedestrians": people}
        density[radius] = {**stats[radius], "peak": peak}
    return density


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


def verify_cone_review(archive, capture, showcase, entry):
    relative = f"{PACE_MEDIA}/{CONE_REVIEW}"
    url = f"./{relative.removeprefix('web/')}"
    expected = capture["sha256"][CONE_REVIEW]
    require(all(ref["cone_review_url"] == url and ref["cone_review_sha256"] == expected
                for ref in (capture, capture["catalog_entry"], showcase, entry)),
            "Cone review links differ from the original Urban Pace capture")
    verify_digest(archive, relative, expected)
    review = read_json(archive, relative)
    first, last = capture["first_frame"], capture["last_frame"]
    frames = capture["summary"]["frames"]
    raw = capture["raw_sha256"]
    cameras = {camera: {"sha256": raw[f"{camera}.mjpeg"], "images": frames}
               for camera in SURROUND_CAMERAS if camera != "chase"}
    require(review["schema_version"] == 1 and review["trial_id"] == capture["trial_id"]
            and (review["first_frame"], review["last_frame"], review["capture_frames"]) == (first, last, frames)
            and review["source_journal_sha256"] == raw["perception.jsonl"] and review["camera_streams"] == cameras,
            "Cone review does not match the original Urban Pace exposures")
    ids = [row["frame"] for row in review["frames"]]
    require(ids and ids == sorted(set(ids)) and first <= ids[0] <= ids[-1] <= last
            and all(row["cone_observations"] for row in review["frames"]), "Invalid native cone frame references")
    for row in review["frames"]:
        for observation in row["cone_observations"]:
            box = observation["box"]
            require(observation["label"] == "traffic_cone" and observation["camera"] in cameras
                    and len(box) == 4 and all(0 <= value <= 1 for value in box)
                    and box[0] < box[2] and box[1] < box[3]
                    and observation["sources"] == ["rgb", "lidar"] and observation["lidar_hits"] > 0
                    and observation["range_source"] in {"projected_xyz", "ground_gated_xy"},
                    f"Invalid cone observation at native frame {row['frame']}")


def catalog_film(archive, build, viewer_records, media, schema):
    directory = PurePosixPath(media).name
    capture = read_json(archive, f"{media}/manifest.json")
    require(capture["policy"]["schema"] == schema, f"{directory} is missing the {schema} policy")
    showcase = next((row for row in build["showcases"]
                     if row["trial_id"] == capture["trial_id"] and row["recording_path"] == directory), None)
    require(showcase is not None, f"{directory} is missing from the build catalog")
    entry = next((row for row in viewer_records if row["trial_id"] == capture["trial_id"]
                  and row["path"] == directory), None)
    require(entry is not None and tuple(entry["cameras"]) == SURROUND_CAMERAS,
            f"{directory} or its cameras are missing from the viewer catalog")
    if schema == PACE_SCHEMA or schema in POPULATED_POLICIES:
        require(entry["benchmark_member"] is False
                and entry["replay_pack_url"] == showcase["replay_pack_url"] == f"./media/{directory}/replay-pack.json.gz"
                and entry["replay_pack_sha256"] == capture["sha256"]["replay-pack.json.gz"],
                f"{directory} or its replay pack is missing from the viewer catalog")
    return capture, showcase, entry


def verify_master_metadata(capture):
    master = capture["master"]["chase"]
    require(master["path"] == "master/chase.mp4" and (master["width"], master["height"]) == (1920, 1080)
            and 0 < master["file_bytes"] < MAX_UPLOAD_BYTES and re.fullmatch(r"[0-9a-f]{64}", master["sha256"])
            and master["nb_read_frames"] == str(capture["summary"]["frames"])
            and master["codec_name"] == "h264" and master["pix_fmt"] == "yuv420p"
            and master["r_frame_rate"] == "10/1", "Wrong surround master metadata")


def verify_populated_catalog(capture, showcase, entry, media):
    directory = PurePosixPath(media).name
    summary_keys = ("outcome", "frames", "duration_s", "progress_m", "route_length_m", *SAFETY_COUNTS)
    require(capture["trial_id"] == directory and (capture["video"]["width"], capture["video"]["height"]) == (1920, 1080)
            and capture["video"] == capture["camera_videos"]["chase"],
            f"{directory} requires a synchronized 1080p chase capture")
    require(capture["passed"] is True and capture["outcome"] == "complete",
            f"{directory} must pass the native route grade to be a featured drive")
    require(all(showcase[key] == capture["summary"][key] for key in summary_keys)
            and showcase["model_sha256"] == capture["policy"]["sha256"]
            and entry["replay_url"] == showcase["replay_url"] == f"./media/{directory}/{capture['replay']}",
            f"{directory} catalog differs from the measured capture")


def verify_maneuver_catalog_checkpoints(build, capture):
    checkpoints = build["checkpoints"]
    policy = capture["policy"]
    require(checkpoints["city-maneuvers"]["sha256"] == policy["sha256"],
            "Builds & logs links a different City Maneuvers checkpoint")
    if STATIONARY_INTENT_FIELD in policy:
        entry = checkpoints.get(STATIONARY_INTENT_CATALOG_KEY, {})
        require("sha256" in entry and entry["sha256"] == policy[STATIONARY_INTENT_FIELD],
                "Builds & logs must identify the selected stationary vehicle intent checkpoint")
    if DRIVER_BUNDLE_FIELD in policy:
        entry = checkpoints.get(DRIVER_BUNDLE_CATALOG_KEY, {})
        require("sha256" in entry and entry["sha256"] == policy[DRIVER_BUNDLE_FIELD],
                "Builds & logs must identify the selected City Maneuvers driver bundle")


def verify_surround_release(archive, build, viewer_records, tag):
    media, asset, schema = featured_recording(tag)
    capture, showcase, entry = catalog_film(archive, build, viewer_records, media, schema)
    if pace_release(tag):
        require(viewer_records[0] is entry, "The featured city drive is not the default recording")
        if life_release(tag):
            urban, urban_showcase, urban_entry = catalog_film(archive, build, viewer_records, PACE_MEDIA, PACE_SCHEMA)
            verify_master_metadata(urban)
            verify_cone_review(archive, urban, urban_showcase, urban_entry)
            verify_populated_catalog(capture, showcase, entry, media)
            life = capture
            if progress_release(tag):
                life, life_showcase, life_entry = catalog_film(archive, build, viewer_records, LIFE_MEDIA, LIFE_SCHEMA)
                verify_populated_catalog(life, life_showcase, life_entry, LIFE_MEDIA)
                verify_master_metadata(life)
            verify_city_life_population(life, urban, read_json(archive, LIFE_BASELINE))
            if maneuvers_release(tag):
                progress, progress_showcase, progress_entry = catalog_film(archive, build, viewer_records,
                                                                          PROGRESS_MEDIA, PROGRESS_SCHEMA)
                verify_populated_catalog(progress, progress_showcase, progress_entry, PROGRESS_MEDIA)
                verify_master_metadata(progress)
                verify_maneuver_catalog_checkpoints(build, capture)
        else:
            urban, urban_showcase, urban_entry = capture, showcase, entry
        report = read_json(archive, BARRIER_VALIDATION)
        require(report["schema"] == "ado-barrier-validation-v1"
                and report["model_sha256"] == urban["policy"]["barrier_sha256"],
                "The barrier validation uses a different recognition model")
        verify_digest(archive, BARRIER_DATA, report["archive_sha256"])
        verify_barrier_review(archive, urban, urban_showcase, urban_entry)
    verify_master_metadata(capture)
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


class ReleaseArchivePair:
    """Read disjoint release ZIPs through the same paths as an unsplit package."""

    def __init__(self, core, evidence, evidence_names):
        self.core = core
        self.evidence = evidence
        self.evidence_names = evidence_names

    def source(self, name):
        return self.evidence if name in self.evidence_names else self.core

    def open(self, name):
        return self.source(name).open(name)

    def read(self, name):
        return self.source(name).read(name)

    def getinfo(self, name):
        return self.source(name).getinfo(name)

    def infolist(self):
        return [*self.core.infolist(), *self.evidence.infolist()]


@contextmanager
def verified_release_archives(path, files, commit, companion=None, companion_evidence=None):
    require((companion is None) == (companion_evidence is None), "Companion ZIP and its release index must be supplied together")

    def checked_asset(stack, asset):
        require(asset.stat().st_size < MAX_UPLOAD_BYTES, f"{asset.name} exceeds the release asset size limit")
        package = stack.enter_context(ZipFile(asset))
        require(package.comment == commit.encode("ascii"), "ZIP does not identify the expected Git commit")
        names = [member.filename for member in package.infolist() if not member.is_dir()]
        require(len(names) == len(set(names)), f"Duplicate file in {asset.name}")
        return package, set(names)

    with ExitStack() as stack:
        core, core_names = checked_asset(stack, path)
        archive = core
        names = core_names
        packages = [core]
        if companion is not None:
            evidence, evidence_names = checked_asset(stack, companion)
            expected = {PREFIX + record["path"] for record in companion_evidence["files"]}
            require(expected and len(expected) == len(companion_evidence["files"])
                    and core_names.isdisjoint(evidence_names) and evidence_names == expected,
                    "Companion ZIP file inventory differs from its index or overlaps the Windows ZIP")
            require(read_json(core, COMPANION_EVIDENCE_INDEX) == companion_evidence
                    and companion_evidence["schema"] == COMPANION_EVIDENCE_SCHEMA
                    and companion_evidence["commit"] == commit and companion_evidence["archive"] == companion.name
                    and companion_evidence["bytes"] == companion.stat().st_size,
                    "Companion evidence index does not match the release asset")
            with companion.open("rb") as blob:
                require(hashlib.file_digest(blob, "sha256").hexdigest() == companion_evidence["sha256"],
                        "Wrong companion ZIP SHA-256")
            for record in companion_evidence["files"]:
                relative = record["path"]
                require(evidence.getinfo(PREFIX + relative).file_size == record["bytes"],
                        f"Wrong companion evidence size for {relative}")
                verify_digest(evidence, relative, record["sha256"])
            archive = ReleaseArchivePair(core, evidence, evidence_names)
            packages.append(evidence)
            names |= evidence_names
        require(names == {PREFIX + name for name in files}, "ZIP file inventory differs from the selected commit")
        for package in packages:
            damaged = package.testzip()
            require(damaged is None, f"Damaged ZIP file: {damaged}")
        yield archive


def validate_zip(path, files, commit, tag, scratch, external_sweeps=(), companion=None, companion_evidence=None):
    with verified_release_archives(path, files, commit, companion, companion_evidence) as archive:

        def read(path):
            return archive.read(PREFIX + path)

        def load(path):
            return read_json(archive, path)

        if external_sweeps:
            historical = maneuvers_release(tag)
            index = EXTERNAL_RESEARCH_INDEX if historical else EXTERNAL_SWEEPS_INDEX
            require(read(index) == sweep_index(commit, external_sweeps, historical=historical),
                    "External download index differs from the verified archival inventory")

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
                verify_city_film(archive, media, capture, scratch,
                                 population_baseline=life_release(tag) and media == PACE_MEDIA)
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


def archive_release(path, commit, tag, external_sweeps=()):
    excluded = [f":(exclude,top,literal){item['path']}" for item in external_sweeps]
    git("archive", "--format=zip", f"--prefix={PREFIX}", "-1", f"--output={path}",
        commit, "--", *release_paths(tag), *excluded)


def external_sweep_inventory(path, commit, *, historical=False):
    external, _ = release_evidence_inventory(path, commit, historical=historical)
    return external


def companion_movie_group(relative):
    parts = PurePosixPath(relative).parts
    if len(parts) < 4 or parts[:2] != ("results", "city-maneuvers") or not parts[-1].endswith(".mp4"):
        return None
    if parts[2] in ("attempts", "pace", "passing", "qualification", "speed"):
        return "supplemental"
    if (len(parts) == 6 and parts[2] == "trials" and parts[3] in ("baseline", "candidate")
            and parts[4] in MANEUVERS_TRIAL_CASES and parts[5] == "front.mp4"):
        return "trials"
    return None


def release_evidence_inventory(path, commit, *, historical=False, allow_companion=False):
    size = path.stat().st_size
    if size < MAX_UPLOAD_BYTES:
        return [], []
    revision = PROGRESS_ARCHIVE_COMMIT if historical else commit
    pinned = historical_research_paths(commit) if historical else None
    required_savings = size - MAX_UPLOAD_BYTES + SWEEP_INDEX_RESERVE
    external, companion = [], []
    with ZipFile(path) as archive:
        build = read_json(archive, "web/media/build.json")
        linked = {item["url"][3:] for item in build["evidence"] if item["url"].startswith("../results/")}
        groups = {group: [] for group in ("sweeps", "clips", "supplemental", "trials")}
        for member in archive.infolist():
            relative = member.filename.removeprefix(PREFIX)
            if member.is_dir() or relative in linked:
                continue
            old = relative in pinned if historical else (
                relative.startswith(f"{PROGRESS_RESULTS}/") and PurePosixPath(relative).name == "sensor-sweeps.zip")
            if old:
                group = "clips" if relative.endswith(".mp4") else "sweeps"
                groups[group].append(member)
            elif allow_companion:
                group = companion_movie_group(relative)
                if group is not None:
                    groups[group].append(member)

        for group, candidates in groups.items():
            candidates.sort(key=lambda item: (item.compress_size, item.filename))
            while required_savings > 0 and candidates:
                index = next((i for i, member in enumerate(candidates) if member.compress_size >= required_savings),
                             len(candidates) - 1)
                member = candidates.pop(index)
                relative = member.filename.removeprefix(PREFIX)
                with archive.open(member) as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                record = dict(path=relative, bytes=member.file_size, sha256=digest)
                if group in ("sweeps", "clips"):
                    record["url"] = f"https://raw.githubusercontent.com/yafi-s/ADO/{revision}/{quote(relative, safe='/')}"
                    external.append(record)
                else:
                    companion.append(record)
                required_savings -= member.compress_size
        require(required_savings <= 0,
                "Windows ZIP exceeds 2 GiB after allowing for eligible historical and companion evidence")
    return (sorted(external, key=lambda item: item["path"]), sorted(companion, key=lambda item: item["path"]))


def historical_research_paths(commit):
    def tree(revision):
        records = git("ls-tree", "-r", "-z", revision, "--", PROGRESS_RESULTS)
        entries = {}
        for record in records.decode("utf8").split("\0")[:-1]:
            attributes, name = record.split("\t", 1)
            path = PurePosixPath(name)
            if (path.name == "sensor-sweeps.zip"
                    or name.startswith(f"{PROGRESS_RESULTS}/trials/") and path.suffix == ".mp4"):
                _, kind, oid = attributes.split()
                require(kind == "blob", f"Historical evidence is not a Git blob: {name}")
                entries[name] = oid
        return entries

    old = tree(PROGRESS_ARCHIVE_COMMIT)
    return {name for name, oid in tree(commit).items() if old.get(name) == oid}


def sweep_index(commit, external_sweeps, *, historical=False):
    title, contents = ("Historical City Progress research", "Development clips and raw sensor sweeps") if historical else (
        "Optional native sensor sweeps", "Raw development sweep archives")
    lines = [f"# {title}", "", f"{contents} accompany the source at Git commit `{commit}`.",
             "Each download link is pinned to a Git commit containing the exact archived file.",
             "Save a downloaded file to the listed path relative to `ADO`.",
             "Verify the downloaded file with `Get-FileHash -Algorithm SHA256` before using it.", ""]
    lines.extend(f"- {item['bytes']:,} bytes · SHA-256 `{item['sha256']}`. [{item['path']}]({item['url']})"
                 for item in external_sweeps)
    return ("\n".join(lines) + "\n").encode("utf8")


def add_release_index(path, relative, contents):
    with ZipFile(path, "a") as package:
        timestamp = package.getinfo(PREFIX + "docs/release.md").date_time
        index = ZipInfo(PREFIX + relative, date_time=timestamp)
        index.external_attr = 0o100644 << 16
        package.writestr(index, contents, compress_type=ZIP_DEFLATED, compresslevel=1)


def add_sweep_index(path, commit, external_sweeps, *, historical=False):
    relative = EXTERNAL_RESEARCH_INDEX if historical else EXTERNAL_SWEEPS_INDEX
    add_release_index(path, relative, sweep_index(commit, external_sweeps, historical=historical))


def stage_companion_evidence(path, commit, tag, files):
    included = [f":(top,literal){record['path']}" for record in files]
    git("archive", "--format=zip", f"--prefix={PREFIX}", "-1", f"--output={path}", commit, "--", *included)
    size = path.stat().st_size
    require(size < MAX_UPLOAD_BYTES, f"{path.name} exceeds the release asset size limit")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(schema=COMPANION_EVIDENCE_SCHEMA, commit=commit, archive=path.name,
                url=f"{PUBLIC_REPO}/releases/download/{tag}/{path.name}", bytes=size, sha256=digest, files=files)


def build_release(ref, tag, output, film_master=None):
    require(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag), "Use a version tag such as v0.1.0")
    require(film_master is None or surround_release(tag), "--film-master applies to v0.3.0 and newer releases")
    require(not life_release(tag) or film_master is not None, "The featured drive requires the native 1080p --film-master")
    commit = resolve_commit(ref, tag)
    files = package_inventory(commit, tag)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ado-release-", dir=output) as folder:
        scratch = Path(folder)
        archive = scratch / f"ADO-{tag}-Windows.zip"
        historical = maneuvers_release(tag)
        archive_release(archive, commit, tag)
        external, companion_files = [], []
        if historical:
            require(COMPANION_EVIDENCE_INDEX not in files, f"Conflicting generated path {COMPANION_EVIDENCE_INDEX}")
            external, companion_files = release_evidence_inventory(archive, commit, historical=True, allow_companion=True)
        elif progress_release(tag):
            external = external_sweep_inventory(archive, commit)
        if external or companion_files:
            archive_release(archive, commit, tag, (*external, *companion_files))
        if external:
            index = EXTERNAL_RESEARCH_INDEX if historical else EXTERNAL_SWEEPS_INDEX
            require(index not in files, f"Conflicting generated path {index}")
            add_sweep_index(archive, commit, external, historical=historical)
            files.difference_update(item["path"] for item in external)
            files.add(index)
        companion = companion_evidence = None
        if companion_files:
            companion = scratch / f"ADO-{tag}-Maneuver-Evidence.zip"
            companion_evidence = stage_companion_evidence(companion, commit, tag, companion_files)
            contents = (json.dumps(companion_evidence, indent=2) + "\n").encode("utf8")
            add_release_index(archive, COMPANION_EVIDENCE_INDEX, contents)
            files.add(COMPANION_EVIDENCE_INDEX)
        validate_zip(archive, files, commit, tag, scratch, external, companion, companion_evidence)
        with archive.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        checksums = f"{sha256}  {archive.name}\n"
        artifacts = [archive]
        metadata = dict(tag=tag, commit=commit, archive=archive.name, bytes=archive.stat().st_size, sha256=sha256)
        if external:
            field = "external_research" if historical else "external_sweeps"
            metadata[field] = dict(index=index, archives=external)
        if companion is not None:
            checksums += f"{companion_evidence['sha256']}  {companion.name}\n"
            artifacts.append(companion)
            metadata["companion_evidence"] = dict(index=COMPANION_EVIDENCE_INDEX, **companion_evidence)
        if film_master is not None:
            media, asset, _ = featured_recording(tag)
            film = scratch / asset
            film_sha256 = stage_release_master(film_master, archive, film, media)
            checksums += f"{film_sha256}  {film.name}\n"
            artifacts.append(film)
            if life_release(tag):
                metadata["film"] = dict(file=film.name, bytes=film.stat().st_size, sha256=film_sha256)
        sums = scratch / "SHA256SUMS.txt"
        sums.write_bytes(checksums.encode("ascii"))
        provenance = scratch / "release-metadata.json"
        provenance.write_bytes((json.dumps(metadata, indent=2) + "\n").encode("ascii"))
        for artifact in (*artifacts, provenance, sums):
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
