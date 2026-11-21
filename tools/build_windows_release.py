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


def package_inventory(commit):
    records = git("ls-tree", "-r", "-z", commit, "--", *PACKAGE_PATHS)
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
    missing = sorted(set(REQUIRED) - files)
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


def verify_video(archive, relative, frame_count, scratch):
    video = scratch / "preview.mp4"
    with archive.open(PREFIX + relative) as source, video.open("wb") as target:
        shutil.copyfileobj(source, target)
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-threads", "1", "-count_frames",
        "-show_entries", "stream=codec_name,codec_tag_string,nb_read_frames:stream_tags=encoder", "-of", "json", str(video),
    ]))
    stream = next(iter(info["streams"]), {})
    require(stream.get("codec_name") == "h264" and stream.get("codec_tag_string") == "avc1"
            and "libx264" in stream.get("tags", {}).get("encoder", "")
            and stream.get("nb_read_frames") == str(frame_count), f"Wrong x264 encoding or frame count for {relative}")


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

        def read_json(path):
            return json.loads(read(path))

        for required in REQUIRED:
            with archive.open(PREFIX + required) as item:
                header = item.read(64)
                require((header or required == "ado/__init__.py")
                        and not header.startswith(b"version https://git-lfs.github.com/spec/v1"),
                        f"Missing Git contents for {required}")

        build = read_json("web/media/build.json")
        manifest = read_json("results/manifest.json")
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

        fit = read_json("models/fit.json")
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

        summary_rows = read_json("results/episodes.json")
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

        catalog = read_json("web/media/catalog.json")
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
            replay = read_json(f"{media}/replay.json")
            require([row["frame"] for row in replay] == frames[key] and all(row["run_id"] == key for row in replay),
                    f"Replay and telemetry differ for {key}")
            if not relative:
                verify_video(archive, f"{media}/chase.mp4", len(replay), scratch)
        require(any(entry["path"] == "" for entry in catalog), "Missing representative recording")

        showcases = build.get("showcases", [])
        viewer_records = read_json("web/media/viewer-catalog.json")
        require({row["trial_id"] for row in viewer_records}
                == {row["trial_id"] for row in [*catalog, *showcases]}, "Incomplete viewer recording catalog")
        for showcase in showcases:
            relative = showcase["recording_path"]
            require(re.fullmatch(r"showcase/[a-z0-9-]+", relative), f"Invalid showcase path: {relative}")
            media = f"web/media/{relative}"
            capture = read_json(f"{media}/manifest.json")
            require(capture["benchmark_member"] is False and capture["trial_id"] == showcase["trial_id"]
                    and capture["model_sha256"] == fit["model_sha256"]["fusion"], f"Wrong showcase at {relative}")
            for name, expected in capture["sha256"].items():
                with archive.open(PREFIX + f"{media}/{name}") as artifact:
                    require(hashlib.file_digest(artifact, "sha256").hexdigest() == expected,
                            f"Wrong showcase file: {relative}/{name}")

        training = read_json("web/logs/training-drive.json")
        require(training["policy"] == "demonstrator" and training["producer"] == PRODUCER and training["frames"],
                "Missing CARLA training demonstration")
        verify_video(archive, "web/media/training-drive.mp4", len(training["frames"]), scratch)


def build_release(ref, tag, output):
    require(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag), "Use a version tag such as v0.1.0")
    commit = resolve_commit(ref, tag)
    files = package_inventory(commit)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ado-release-", dir=output) as folder:
        scratch = Path(folder)
        archive = scratch / f"ADO-{tag}-Windows.zip"
        git("archive", "--format=zip", f"--prefix={PREFIX}", "-1", f"--output={archive}",
            commit, "--", *PACKAGE_PATHS)
        validate_zip(archive, files, commit, tag, scratch)
        with archive.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        sums = scratch / "SHA256SUMS.txt"
        sums.write_bytes(f"{sha256}  {archive.name}\n".encode("ascii"))
        metadata = scratch / "release-metadata.json"
        metadata.write_bytes((json.dumps(dict(tag=tag, commit=commit, archive=archive.name,
                                               bytes=archive.stat().st_size, sha256=sha256), indent=2) + "\n").encode("ascii"))
        for artifact in (archive, sums, metadata):
            artifact.replace(output / artifact.name)
    return output / archive.name, commit, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Git commit SHA or tag for the release tree")
    parser.add_argument("--tag", required=True, help="Release version tag")
    parser.add_argument("--output", type=Path, default=Path(".local/release"))
    args = parser.parse_args()
    try:
        archive, commit, sha256 = build_release(args.ref, args.tag, args.output)
    except (OSError, ValueError, KeyError, BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release build failed: {error}\n")
    print(f"{archive}\nGit commit {commit}\nSHA-256 {sha256}")


if __name__ == "__main__":
    main()
