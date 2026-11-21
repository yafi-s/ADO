"""Verify the installed native City Progress drive and retain its independent score files."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from crosswalk.experiment import fingerprint
from crosswalk.journey_capture import PERIMETER_VIEWS
from crosswalk.replay_pack import pack


def read(path):
    return json.loads(path.read_bytes())


def preserve(source, target):
    if not target.exists():
        shutil.copyfile(source, target)
    if fingerprint(source) != fingerprint(target):
        raise ValueError(f"Existing evidence differs from {source.name}")


def verify_ranges(directory, frames, telemetry, summary, folder):
    files = {int(path.stem): path for path in directory.glob("*.npz")}
    if set(files) != set(map(int, frames)):
        raise ValueError("Full-resolution range tape does not cover every native camera exposure")
    person = next(e for e in summary["events"] if e["kind"] == "occluded")
    positive = next(i for i, row in enumerate(telemetry) if row["sim_time"] >= person["exposed_s"])
    junction = next(j for j in summary["junction_traversals"] if j["junction_id"] == 1932)
    midpoint = (junction["entry_m"] + junction["exit_m"]) / 2
    negative = min((i for i, row in enumerate(telemetry) if junction["entry_m"] < row["progress_m"] <
                   junction["exit_m"] and row["reference_gap_m"] >= 59. and row.get("active_event") is None),
                   key=lambda i: abs(telemetry[i]["progress_m"] - midpoint), default=None)
    examples = {"occluded-person-exposure": positive}
    if negative is None:
        negative = next(i for i,r in enumerate(telemetry) if r["progress_m"] > 200. and r["speed_mps"] > 2.
                        and r["reference_gap_m"] >= 59. and r.get("active_event") is None)
    examples["no-staged-hazard"] = negative
    selected = {int(frames[i]): label for label, index in examples.items()
                for i in range(max(0, index - 6), min(len(frames), index + 2))}
    rows = []
    samples = {}
    for frame, row in zip(frames, telemetry, strict=True):
        frame = int(frame)
        path = files[frame]
        with np.load(path, allow_pickle=False) as sweep:
            pose = row["ego_pose"]
            if not np.allclose(sweep["ego_xy_yaw"], [pose[k] for k in ("x", "y", "yaw")], atol=.015, rtol=0.):
                raise ValueError(f"Raw range pose disagrees with native camera at {frame}")
            if frame in selected:
                raw = BytesIO()
                np.savez(raw, **{k: sweep[k] for k in sweep.files if k == "lidar" or k == "ego_xy_yaw" or k.startswith("radar")})
                samples[f"{selected[frame]}/{frame}.npz"] = raw.getvalue()
        rows.append(dict(frame=frame, bytes=path.stat().st_size, sha256=fingerprint(path)))
    target = folder / "range-samples.zip"
    if not target.exists():
        with ZipFile(target, "x", ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in samples.items():
                archive.writestr(name, data)
    with ZipFile(target) as archive:
        if set(archive.namelist()) != set(samples) or any(archive.read(k) != v for k,v in samples.items()):
            raise ValueError("Selected native range samples do not match the retained raw sweep arrays")
    return dict(local_sweep_session=directory.name, frames=len(rows),
                total_bytes=sum(row["bytes"] for row in rows), files=rows,
                samples=dict(sha256=fingerprint(target), frames=selected))


def verify(stage, media, results, checkpoint, ranges=None):
    manifest = read(media / "manifest.json")
    request, source = (read(stage / name) for name in ("request.json", "capture.json"))
    cameras = ("chase", "front", *PERIMETER_VIEWS)
    contract = (request["seed"], request["destination"], request["crossing_delay_s"], request["quality"],
                request["population_profile"], source["trial_id"], manifest["policy"]["schema"])
    if contract != (212, "market", .5, "High", "city-life", "city-progress", "ado-city-progress-v1"):
        raise ValueError(f"Unexpected conditions {contract}")
    if not manifest["passed"] or manifest["teacher"] or manifest["producer"] != {
        "server_version": "1.0", "client_version": "1.0", "map": "Town03_Opt"}:
        raise ValueError("The installed drive did not meet the CARLA capture contract")
    model_sha = fingerprint(checkpoint)
    if not all(m["policy"]["sha256"] == model_sha for m in (request, source, manifest)):
        raise ValueError("The retained checkpoint differs from the recorded driver")
    with ZipFile(stage / "capture-source.zip") as archive:
        for name, sha in manifest["policy"]["code_sha256"].items():
            if hashlib.sha256(archive.read("crosswalk/" + name)).hexdigest() != sha:
                raise ValueError(f"Execution archive mismatch: {name}")
    for name, sha in manifest["sha256"].items():
        if fingerprint(media / name) != sha:
            raise ValueError(f"The installed {name} differs from the capture")
    summary = source["summary"]
    if len(summary["events"]) != 11 or not all(e["handled"] for e in summary["events"]):
        raise ValueError("Incomplete encounter contract")
    junctions = [e for e in summary["junction_traversals"] if "completed_s" in e]
    if len(junctions) != 11 or any(summary[k] for k in ("collision_events", "yield_violations", "red_light_violations")):
        raise ValueError("Incomplete route or recorded violation")
    telemetry = read(media / "telemetry.json")
    frames = np.asarray([r["frame"] for r in telemetry])
    with np.load(stage / "policy-observations.npz", allow_pickle=False) as tape:
        np.testing.assert_array_equal(frames, tape["frame"])
        np.testing.assert_allclose(tape["speed_request_mps"], [r["speed_request_mps"] for r in telemetry], atol=5.1e-6)
        if np.any(~np.isfinite(tape["features"])) or tape["features"].shape != (len(frames), len(tape["feature_names"])):
            raise ValueError("Incomplete model feature tape")
    if len(frames) != manifest["summary"]["frames"] or np.any(np.diff(frames) != 1):
        raise ValueError("Dropped policy exposure")
    for i, row in enumerate(telemetry):
        if row["video_index"] != i or row["camera_frame"] != row["frame"] or "progress_evidence" not in row:
            raise ValueError(f"Bad telemetry at exposure {i}")
    timings = {}
    for name in cameras:
        video = manifest["camera_videos"][name]
        if video["r_frame_rate"] != "10/1" or int(video["nb_read_frames"]) != len(frames):
            raise ValueError(f"Unregistered video clock: {name}")
        timings[name] = {k: video[k] for k in ("r_frame_rate", "nb_read_frames", "width", "height")}
    results.mkdir(parents=True, exist_ok=True)
    folder = results / "capture"
    folder.mkdir(exist_ok=True)
    for name, field in (("capture-source.zip", "capture_source_sha256"),
                        ("population-reference.jsonl.gz", "population_reference_sha256"),
                        ("population-score.json", "population_score_sha256"),
                        ("policy-observations.npz", "policy_observations_sha256")):
        if fingerprint(stage / name) != source[field]:
            raise ValueError(f"Source file checksum failed for {name}")
        preserve(stage / name, (folder if name.endswith(".npz") else results) / name)
    for name in ("source-checkout.json", "request.json", "capture.json"):
        preserve(stage / name, folder / name)
    if ranges is not None:
        inventory = verify_ranges(ranges, frames, telemetry, summary, folder)
        (folder / "range-inventory.json").write_text(json.dumps(inventory, separators=(",", ":")) + "\n", encoding="utf-8")
    compact = media / "replay-pack.json.gz"
    if not compact.exists():
        pack(stage / "raw/replay.json", compact)
    data = dict(trial_id=manifest["trial_id"], seed=212, producer=source["producer"], camera_timelines=timings,
                model_sha256=model_sha, source_sha256=source["capture_source_sha256"], native_frames=len(frames),
                first_frame=int(frames[0]), last_frame=int(frames[-1]), completed_junctions=len(junctions),
                encounters_handled=11, outcome=summary["outcome"], native_duration_s=summary["duration_s"],
                native_progress_m=summary["progress_m"], sensor_configuration=summary["sensor_configuration"],
                master_sha256=manifest["master"]["chase"]["sha256"], compact_sha256=fingerprint(compact),
                manifest_sha256=fingerprint(media / "manifest.json"))
    if ranges is not None:
        data.update(full_range_frames=inventory["frames"], full_range_bytes=inventory["total_bytes"],
                    range_inventory_sha256=fingerprint(folder / "range-inventory.json"),
                    range_samples_sha256=inventory["samples"]["sha256"])
    target = folder / "synchronization.json"
    if target.exists() and read(target) != data:
        raise ValueError("Existing synchronization file belongs to a different recording")
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", type=Path)
    p.add_argument("--media", type=Path, default=Path("web/media/city-progress"))
    p.add_argument("--results", type=Path, default=Path("results/city-progress"))
    p.add_argument("--checkpoint", type=Path, default=Path("models/city-progress/candidate.npz"))
    p.add_argument("--ranges", type=Path, help="Native timestamp-aligned full-resolution sweep directory")
    a = p.parse_args()
    print(json.dumps(verify(a.stage, a.media, a.results, a.checkpoint, a.ranges), indent=2))
