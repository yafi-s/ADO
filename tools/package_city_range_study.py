"""Package and verify the original City Life range and front-horizon evidence."""
import argparse
import hashlib
from io import BytesIO
import json
from math import isclose
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ASSET = "ADO-City-Life-Range-Sweeps.zip"
MAX_BYTES = 2 * 1024**3
SCHEMA = "ado-city-native-range-study-v1"
RAW = ("perception.jsonl", "town.json", "telemetry.json")
BASELINE = "reference/object-memory-before.mjs"
BASELINE_REVISION = "251ae057c4e6b1a8fa06ce6ca91959e4caee3916"
BASELINE_PATH = "web/object-memory.mjs"
BASELINE_SHA256 = "602884d383e8c56f0991356ef731b5875f9f968711e3b7ed07ce705bcf69a8ff"
SWEEP_ARRAYS = {"lidar", "ego_xy_yaw", "radar", "radar_front_left", "radar_front_right", "radar_rear_left", "radar_rear_right"}
OPTICAL_ARRAYS = {"horizon_rgb", "horizon_xyxy", "front_size_wh"}
CITY_MANIFEST = "ADO/web/media/city-life/manifest.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(data):
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def capture_provenance(capture):
    summary = capture["summary"]
    policy = capture["policy"]
    require(capture["trial_id"] == "city-life" and capture["population_profile"] == "city-life"
            and capture["producer"]["map"] == "Town03_Opt" and policy["schema"] == "ado-city-clearance-v1"
            and capture["last_frame"] - capture["first_frame"] + 1 == summary["frames"] > 0,
            "Expected the sealed native Town03 City Life capture")
    return {**{key: capture[key] for key in ("trial_id", "producer", "first_frame", "last_frame", "capture_source_sha256")},
            "frames": summary["frames"], "policy_schema": policy["schema"], "policy_sha256": policy["sha256"],
            "raw_sha256": {name: capture["raw_sha256"][name] for name in RAW}}


def frame_ids(recording):
    return range(recording["first_frame"], recording["last_frame"] + 1)


def sweep_member(frame):
    return f"sweeps/{frame}.npz"


def sweep_files(directory, recording):
    files = {}
    for path in directory.iterdir():
        require(path.is_file() and not path.is_symlink() and re.fullmatch(r"(?:0|[1-9][0-9]*)\.npz", path.name),
                f"Unexpected file in the capture sweep directory: {path.name}")
        frame = int(path.stem)
        require(frame not in files, f"Duplicate sweep frame {frame}")
        files[frame] = path
    expected = set(frame_ids(recording))
    require(files.keys() == expected, f"Sweep inventory differs from the capture. Missing: {len(expected - files.keys())}. "
                                      f"Extra: {len(files.keys() - expected)}.")
    return files


def write_file(archive, name, path, compression):
    require(path.is_file() and not path.is_symlink(), f"Expected an ordinary capture file: {path}")
    member = ZipInfo(name)
    member.compress_type = compression
    member.file_size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as source, archive.open(member, "w") as output:
        while chunk := source.read(1024**2):
            digest.update(chunk)
            output.write(chunk)
    require(member.file_size > 0, f"Empty capture file: {path}")
    return {"bytes": member.file_size, "sha256": digest.hexdigest()}


def verify_file(archive, name, artifact, compression, *, contents=False):
    item = archive.getinfo(name)
    require(item.compress_type == compression and item.file_size == artifact["bytes"] > 0,
            f"Wrong archive size or compression for {name}")
    with archive.open(item) as source:
        if contents:
            data = source.read()
            digest = hashlib.sha256(data)
        else:
            data = None
            digest = hashlib.file_digest(source, "sha256")
    require(digest.hexdigest() == artifact["sha256"], f"Wrong SHA-256 for {name}")
    return data


def verify_archive(path):
    require(0 < path.stat().st_size < MAX_BYTES, "Range sweep ZIP must be below 2 GiB")
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        recording = manifest["capture"]
        files = manifest["files"]
        baseline = manifest["baseline_renderer"]
        ids = frame_ids(recording)
        names = {"manifest.json", BASELINE, *(f"raw/{name}" for name in RAW), *(sweep_member(frame) for frame in ids)}
        require(manifest["schema"] == SCHEMA and recording["trial_id"] == "city-life" and len(ids) == recording["frames"] > 0
                and len(archive.namelist()) == len(names) and set(archive.namelist()) == names
                and files.keys() == names - {"manifest.json"}, "Unexpected range study archive inventory")
        require(baseline == {"git_revision": BASELINE_REVISION, "source_path": BASELINE_PATH, "member": BASELINE,
                             "sha256": BASELINE_SHA256} and files[BASELINE]["sha256"] == BASELINE_SHA256,
                "Wrong reference renderer provenance")
        verify_file(archive, BASELINE, files[BASELINE], ZIP_DEFLATED)
        for name in RAW:
            member = f"raw/{name}"
            require(recording["raw_sha256"][name] == files[member]["sha256"], f"Wrong capture SHA-256 for {member}")
            verify_file(archive, member, files[member], ZIP_DEFLATED)
        telemetry = json.loads(archive.read("raw/telemetry.json"))
        require(len(telemetry) == len(ids), "Wrong number of scored telemetry frames")
        with archive.open("raw/perception.jsonl") as observations:
            for index, (frame, line, score) in enumerate(zip(ids, observations, telemetry, strict=True)):
                row = json.loads(line)
                require(row["frame"] == score["frame"] == score["camera_frame"] == frame
                        and row["run_id"] == recording["trial_id"] and score["video_index"] == index,
                        f"Capture telemetry or journal is out of sync at frame {frame}")
                name = sweep_member(frame)
                contents = verify_file(archive, name, files[name], ZIP_STORED, contents=True)
                with np.load(BytesIO(contents), allow_pickle=False) as sweep:
                    arrays = set(sweep.files)
                    require(len(arrays) == len(sweep.files) and arrays in (SWEEP_ARRAYS, SWEEP_ARRAYS | OPTICAL_ARRAYS),
                            f"Unexpected native sensor arrays in {name}")
                    pose = sweep["ego_xy_yaw"]
                    require(pose.shape == (3,) and all(isclose(float(actual), expected, rel_tol=0, abs_tol=.006)
                            for actual, expected in zip(pose, row["observer_map"]["ego"], strict=True)),
                            f"Sweep pose differs from the native journal at frame {frame}")
    return manifest


def build_archive(capture_dir, sweeps_dir, output):
    require(not output.exists(), f"Use a new output location to preserve the existing file: {output}")
    manifest_path = capture_dir / "capture.json"
    native = manifest_path.read_bytes()
    capture = capture_provenance(json.loads(native))
    source = json.loads((capture_dir / "source-checkout.json").read_bytes())
    require(source["source_sha256"] == capture["capture_source_sha256"] == sha256(capture_dir / "capture-source.zip")
            and re.fullmatch(r"[0-9a-f]{40}", source["head_at_capture_start"]),
            "The capture execution source archive does not match its provenance")
    capture.update(native_manifest_sha256=hashlib.sha256(native).hexdigest(),
                   checkout_at_capture_start=source["head_at_capture_start"])
    sweeps = sweep_files(sweeps_dir, capture)
    require(sum(path.stat().st_size for path in sweeps.values()) < MAX_BYTES, "Native sweeps alone meet or exceed 2 GiB")

    reference = subprocess.check_output(["git", "-C", str(REPO), "show", f"{BASELINE_REVISION}:{BASELINE_PATH}"])
    require(hashlib.sha256(reference).hexdigest() == BASELINE_SHA256, "The Git baseline renderer differs from its pinned hash")
    files = {BASELINE: {"bytes": len(reference), "sha256": BASELINE_SHA256}}
    manifest = {"schema": SCHEMA, "capture": capture, "files": files,
                "baseline_renderer": {"git_revision": BASELINE_REVISION, "source_path": BASELINE_PATH,
                                      "member": BASELINE, "sha256": BASELINE_SHA256}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ado-range-study-", dir=output.parent) as folder:
        pending = Path(folder) / ASSET
        with ZipFile(pending, "x", compresslevel=5) as archive:
            for name in RAW:
                member = f"raw/{name}"
                files[member] = write_file(archive, member, capture_dir / member, ZIP_DEFLATED)
                require(files[member]["sha256"] == capture["raw_sha256"][name], f"Capture source changed: {member}")
            for frame in frame_ids(capture):
                name = sweep_member(frame)
                files[name] = write_file(archive, name, sweeps[frame], ZIP_STORED)
            baseline = ZipInfo(BASELINE)
            baseline.compress_type = ZIP_DEFLATED
            archive.writestr(baseline, reference)
            archive.writestr(ZipInfo("manifest.json"), json_bytes(manifest))
        require(verify_archive(pending) == manifest, "Range ZIP manifest changed during verification")
        require(not output.exists(), f"Range sweep ZIP output appeared during the build: {output}")
        pending.replace(output)
    return manifest


def attach_release(path, release_dir):
    manifest = verify_archive(path)
    metadata_path = release_dir / "release-metadata.json"
    checksums = release_dir / "SHA256SUMS.txt"
    metadata = json.loads(metadata_path.read_bytes())
    require(metadata["tag"] == "v0.5.0" and metadata["archive"] == "ADO-v0.5.0-Windows.zip"
            and metadata["film"]["file"] == "ADO-City-Life-1080p.mp4", "Expected the built v0.5.0 Windows ZIP and master")
    zip_path = release_dir / metadata["archive"]
    film = metadata["film"]
    require(zip_path.stat().st_size == metadata["bytes"] and (release_dir / film["file"]).stat().st_size == film["bytes"],
            "The core release assets do not match their verified metadata")
    with ZipFile(zip_path) as package:
        require(package.comment.decode("ascii") == metadata["commit"], "Windows ZIP commit does not match the release metadata")
        capture = capture_provenance(json.loads(package.read(CITY_MANIFEST)))
    require(all(manifest["capture"][key] == value for key, value in capture.items()),
            "Range sweeps belong to a different capture than the Windows release")
    asset = {"file": ASSET, "bytes": path.stat().st_size, "sha256": sha256(path)}
    base = f"{metadata['sha256']}  {metadata['archive']}\n{film['sha256']}  {film['file']}\n"
    all_sums = f"{base}{asset['sha256']}  {ASSET}\n"
    require(checksums.read_text(encoding="ascii") in (base, all_sums)
            and metadata.get("range_study", asset) == asset, "Release already records different checksums")
    target = release_dir / ASSET
    with TemporaryDirectory(prefix=".ado-range-attach-", dir=release_dir) as folder:
        scratch = Path(folder)
        if not target.exists():
            pending = scratch / ASSET
            shutil.copyfile(path, pending)
            require(sha256(pending) == asset["sha256"], "Range sweep ZIP changed while being staged")
            pending.replace(target)
        elif target.resolve() != path.resolve():
            require(sha256(target) == asset["sha256"], "Release already contains a different sweep archive")
        metadata["range_study"] = asset
        (scratch / metadata_path.name).write_bytes(json_bytes(metadata))
        (scratch / checksums.name).write_bytes(all_sums.encode("ascii"))
        (scratch / metadata_path.name).replace(metadata_path)
        (scratch / checksums.name).replace(checksums)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_subparsers(dest="action", required=True)
    build = action.add_parser("build", help="Build from a sealed capture and its matching sweep directory")
    build.add_argument("capture", type=Path)
    build.add_argument("sweeps", type=Path)
    build.add_argument("output", type=Path)
    verify = action.add_parser("verify", help="Check all members, checksums and frame alignment")
    verify.add_argument("output", type=Path)
    attach = action.add_parser("attach", help="Add the verified archive to the completed v0.5.0 release checksums")
    attach.add_argument("output", type=Path)
    attach.add_argument("release_dir", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "build":
            manifest = build_archive(args.capture, args.sweeps, args.output)
        elif args.action == "attach":
            manifest = attach_release(args.output, args.release_dir)
        else:
            manifest = verify_archive(args.output)
    except (BadZipFile, KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Cannot prepare the native range study: {exc}\n")
    capture = manifest["capture"]
    print(json_bytes({"asset": args.output.name, "bytes": args.output.stat().st_size, "sha256": sha256(args.output),
                      "first_frame": capture["first_frame"], "last_frame": capture["last_frame"], "frames": capture["frames"],
                      "journal_sha256": capture["raw_sha256"]["perception.jsonl"],
                      "execution_sources_sha256": capture["capture_source_sha256"],
                      "baseline_source": manifest["baseline_renderer"]}).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
