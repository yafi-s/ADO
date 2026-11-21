"""Reprocess synchronized native camera JPEGs and ego-frame laser returns, then score independently."""

import argparse
import io
import json
import math
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from .barrier_review import jpeg_images
from .traffic_cones import detect_cones, detect_views


def _file_hash(path):
    with path.open("rb") as stream:
        return __import__("hashlib").file_digest(stream, "sha256").hexdigest()


def _decode(jpeg):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def write_json(path, value, compact=False):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(
        json.dumps(
            value, allow_nan=False, indent=None if compact else 2, separators=(",", ":") if compact else None
        )
        + "\n",
        encoding="utf8",
        newline="\n",
    )


def observe(raw):
    """The first scored RGB in every stream corresponds to the first native journal frame."""
    capture = json.loads((raw.parent / "capture.json").read_bytes())
    streams = {}
    views = ["front", *capture["perimeter_views"]]
    sources = [jpeg_images(raw / (v + ".mjpeg"), None, streams) for v in views]
    first, last, frames = None, None, []
    journal = sha256()
    with (raw / "perception.jsonl").open("rb") as lines:
        for i, (line, *images) in enumerate(zip(lines, *sources, strict=True)):
            journal.update(line)
            row = json.loads(line)
            frame = row["frame"]
            if (last is not None and frame != last + 1) or any(n != i for n, b in images):
                raise ValueError("Nonconsecutive frame or JPEG")
            first = frame if first is None else first
            last = frame
            observations = detect_views(
                {v: _decode(b) for v, (_, b) in zip(views, images)}, row["lidar_xyz"], row["lidar"]
            )
            if observations:
                frames.append({"frame": frame, "cone_observations": observations})
    if (first, last) != (capture["first_frame"], capture["last_frame"]) or i + 1 != capture["summary"][
        "frames"
    ]:
        raise ValueError("Original capture and synchronized rows differ")
    for name, record in streams.items():
        if record["images"] != i + 1 or record["sha256"] != capture["raw_sha256"][name + ".mjpeg"]:
            raise ValueError("Original camera fingerprint or image count differs: " + name)
    if journal.hexdigest() != capture["raw_sha256"]["perception.jsonl"]:
        raise ValueError("Original sensor journal fingerprint differs")
    return {
        "schema_version": 1,
        "trial_id": capture["trial_id"],
        "first_frame": first,
        "last_frame": last,
        "capture_frames": i + 1,
        "frames": frames,
        "source_journal_sha256": journal.hexdigest(),
        "camera_streams": streams,
    }


def _targets(row):
    pose = row["observer_map"]
    ex, ey, yaw = pose["ego"]
    rad = math.radians(yaw)
    targets = []
    for actor in pose["others"]:
        if actor["type"] != "cone":
            continue
        dx, dy = actor["x"] - ex, actor["y"] - ey
        targets.append(
            (
                actor["actor_id"],
                dx * math.cos(rad) + dy * math.sin(rad),
                dy * math.cos(rad) - dx * math.sin(rad),
            )
        )
    return targets


def _errors(values):
    return (
        {
            "median_m": round(float(np.median(values)), 4),
            "p95_m": round(float(np.percentile(values, 95)), 4),
            "max_m": round(float(np.max(values)), 4),
        }
        if values
        else {}
    )


def evaluate(raw, sidecar):
    frames = {f["frame"]: f["cone_observations"] for f in sidecar["frames"]}
    nearest = Counter()
    matched = Counter()
    by_camera = Counter()
    by_range = Counter()
    center_errors, range_errors, unmatched = [], [], []
    observations = inside = 0
    with (raw / "perception.jsonl").open(encoding="utf8") as handle:
        for line in handle:
            row = json.loads(line)
            refs = _targets(row)
            for key, x, y in refs:
                if math.hypot(x, y) <= 20.0:
                    nearest[key] += 1
            seen = frames.get(row["frame"], ())
            observations += len(seen)
            pairs = sorted(
                (math.hypot(o["x"] - x, o["y"] - y), i, k, x, y)
                for i, o in enumerate(seen)
                for k, x, y in refs
            )
            used = set()
            owned = set()
            for error, i, k, x, y in pairs:
                if i in used or k in owned or error > 0.65:
                    continue
                used.add(i)
                owned.add(k)
                o = seen[i]
                center_errors.append(error)
                range_errors.append(abs(math.hypot(o["x"], o["y"]) - math.hypot(x, y)))
                inside += int(math.hypot(x, y) <= 20.0)
                matched[k] += 1
                by_camera[o["camera"]] += 1
                by_range[o["range_source"]] += 1
            unmatched.extend(dict(frame=row["frame"], **o) for i, o in enumerate(seen) if i not in used)
    return {
        "capture_frames": sidecar["capture_frames"],
        "frames_with_fresh_measurements": len(frames),
        "observations": observations,
        "matched": observations - len(unmatched),
        "unmatched": unmatched,
        "match_radius_m": 0.65,
        "within_20m_reference_actor_frames": sum(nearest.values()),
        "matched_within_20m": inside,
        "missed_within_20m": sum(nearest.values()) - inside,
        "actor_fresh_measurements": matched,
        "matched_by_view": by_camera,
        "matched_by_range_source": by_range,
        "horizontal_center_error": _errors(center_errors),
        "absolute_range_error": _errors(range_errors),
    }


def evaluate_archive(path):
    rows = []
    with ZipFile(path) as archive:
        for sample in json.loads(archive.read("manifest.json")):
            name = sample["name"]
            rgb = _decode(archive.read(name + ".jpg"))
            with np.load(io.BytesIO(archive.read(name + ".npz"))) as pkt:
                xyz = pkt["lidar"][:, :3].copy()
            xyz[:, 2] += 2.30
            obs = detect_cones(rgb, "front", xyz)
            positive = sample["condition"] == "cone"
            target = sample["target_xy_m"] if positive else None
            distances = [math.hypot(o["x"] - target[0], o["y"] - target[1]) for o in obs] if positive else []
            best = min(distances, default=float("inf"))
            match = best <= 0.65
            rows.append(
                {
                    "name": name,
                    "split": sample["split"],
                    "positive": positive,
                    "true_positive": int(match),
                    "false_negative": int(positive) - int(match),
                    "false_positive": len(obs) - int(match),
                    "center_error_m": round(best, 4) if match else None,
                }
            )
    return {
        "source_sha256": _file_hash(path),
        "samples": rows,
        "groups": {
            group: {
                k: sum(r[k] for r in rows if r["split"] == group)
                for k in ("positive", "true_positive", "false_positive", "false_negative")
            }
            for group in ("fit", "holdout")
        },
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("raw", type=Path)
    cli.add_argument("--sidecar", type=Path, required=True)
    cli.add_argument("--metrics", type=Path, required=True)
    cli.add_argument("--archive", type=Path, default=Path("data/barrier-recognition/town03-native.zip"))
    args = cli.parse_args()
    cv2.setNumThreads(1)
    measured = observe(args.raw)
    write_json(args.sidecar, measured, compact=True)
    report = {"journey": evaluate(args.raw, measured), "archive": evaluate_archive(args.archive)}
    write_json(args.metrics, report)
    print({k: v for k, v in report["journey"].items() if k != "unmatched"})
    print("Unmatched", len(report["journey"]["unmatched"]))


if __name__ == "__main__":
    main()
