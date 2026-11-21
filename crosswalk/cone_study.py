"""Reprocess synchronized native camera JPEGs and ego-frame laser returns, then score independently."""

import argparse
import io
import json
import math
from collections import Counter
from hashlib import file_digest, sha256
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from .barrier_review import jpeg_images
from .traffic_cones import detect_cones, detect_views


def _file_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


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
    count = 0
    with (raw / "perception.jsonl").open("rb") as lines:
        for i, (line, *images) in enumerate(zip(lines, *sources, strict=True)):
            count += 1
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
    if (first, last) != (capture["first_frame"], capture["last_frame"]) or count != capture["summary"][
        "frames"
    ]:
        raise ValueError("Original capture and synchronized rows differ")
    for name, record in streams.items():
        if record["images"] != count or record["sha256"] != capture["raw_sha256"][name + ".mjpeg"]:
            raise ValueError("Original camera fingerprint or image count differs: " + name)
    if journal.hexdigest() != capture["raw_sha256"]["perception.jsonl"]:
        raise ValueError("Original sensor journal fingerprint differs")
    return {
        "schema_version": 1,
        "trial_id": capture["trial_id"],
        "first_frame": first,
        "last_frame": last,
        "capture_frames": count,
        "frames": frames,
        "source_journal_sha256": journal.hexdigest(),
        "camera_streams": streams,
    }


def live_observations(raw):
    """Audit measurements emitted directly from the full live lidar sweep before thinning."""
    capture = json.loads((raw.parent / "capture.json").read_bytes())
    journal = sha256()
    frames = []
    count = 0
    with (raw / "perception.jsonl").open("rb") as lines:
        for line in lines:
            journal.update(line)
            row = json.loads(line)
            if row["frame"] != capture["first_frame"] + count:
                raise ValueError("Out-of-order live cone observations")
            count += 1
            if row["cone_observations"]:
                frames.append({"frame": row["frame"], "cone_observations": row["cone_observations"]})
    if (
        count != capture["summary"]["frames"]
        or count != capture["last_frame"] - capture["first_frame"] + 1
        or journal.hexdigest() != capture["raw_sha256"]["perception.jsonl"]
    ):
        raise ValueError("Live journal differs from its completed capture")
    source = {
        k: capture[k] for k in ("trial_id", "capture_id", "first_frame", "last_frame", "quality_requested")
    }
    source["source_journal_sha256"] = journal.hexdigest()
    source["detector_sha256"] = capture["policy"]["code_sha256"]["traffic_cones.py"]
    return {"frames": frames, "capture_frames": count, "source": source}


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
    with (raw / "perception.jsonl").open(encoding="utf8") as stream:
        return score_rows((json.loads(line) for line in stream), sidecar, range_bands="source" in sidecar)


def score_rows(rows, sidecar, *, range_bands=False):
    frames = {f["frame"]: f["cone_observations"] for f in sidecar["frames"]}
    nearest = Counter()
    matched = Counter()
    by_camera = Counter()
    by_range = Counter()
    center_errors, range_errors, unmatched = [], [], []
    observations = inside = 0
    opportunities, matches = Counter(), Counter()
    for row in rows:
        refs = _targets(row)
        for key, x, y in refs:
            if math.hypot(x, y) <= 20.0:
                nearest[key] += 1
                if range_bands:
                    opportunities[max(1, math.ceil(math.hypot(x, y) / 5))] += 1
        seen = frames.get(row["frame"], ())
        observations += len(seen)
        pairs = sorted(
            (math.hypot(o["x"] - x, o["y"] - y), i, k, x, y) for i, o in enumerate(seen) for k, x, y in refs
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
            if range_bands and math.hypot(x, y) <= 20:
                matches[max(1, math.ceil(math.hypot(x, y) / 5))] += 1
            matched[k] += 1
            by_camera[o["camera"]] += 1
            by_range[o["range_source"]] += 1
        unmatched.extend(dict(frame=row["frame"], **o) for i, o in enumerate(seen) if i not in used)
    report = {
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
    if range_bands:
        report["nearby_actor_frames_by_distance"] = [
            {
                "upper_bound_m": 5 * k,
                "reference": opportunities[k],
                "matched": matches[k],
                "missed": opportunities[k] - matches[k],
            }
            for k in range(1, 5)
        ]
    return report


def evaluate_packets(raw):
    """Score saved native close-range trials with full lidar and available cameras."""
    rows = []
    measured = []
    fingerprint = sha256()
    for folder in sorted(raw.glob("frame-*"), key=lambda p: int(p.name.split("-")[-1])):
        image_files = sorted(folder.glob("*.jpg"))
        files = [*image_files, folder / "range.npz", folder / "frame.json"]
        for path in files:
            fingerprint.update((path.relative_to(raw).as_posix() + " " + _file_hash(path) + "\n").encode())
        images = {p.stem: _decode(p.read_bytes()) for p in image_files}
        with np.load(folder / "range.npz") as scan:
            xyz = scan["lidar"][:, :3].copy()
        xyz[:, 2] += 2.30
        predictions = detect_views(images, xyz)
        # Only the independent scorer gets the car and actor poses.
        truth = json.loads((folder / "frame.json").read_bytes())
        if predictions:
            measured.append({"frame": truth["frame"], "cone_observations": predictions})
        x, y, _, _, yaw, _ = truth["ego"]
        rows.append(
            {"frame": truth["frame"], "observer_map": {"ego": [x, y, yaw], "others": truth["others"]}}
        )
    if not rows:
        raise ValueError("No native packet folders in " + str(raw))
    return {
        "input_files_sha256": fingerprint.hexdigest(),
        **score_rows(rows, {"capture_frames": len(rows), "frames": measured}),
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


def review_image(raw, sidecar, output):
    """Show the original camera exposures with the most simultaneous fresh measurements."""
    row = max(sidecar["frames"], key=lambda r: len(r["cone_observations"]))
    offset = row["frame"] - sidecar["first_frame"]
    views = sorted({o["camera"] for o in row["cone_observations"]})
    panels = []
    for view in views:
        blob = next(blob for i, blob in jpeg_images(raw / (view + ".mjpeg"), offset + 1, {}) if i == offset)
        bgr = cv2.resize(cv2.cvtColor(_decode(blob), cv2.COLOR_RGB2BGR), (640, 360))
        observations = [o for o in row["cone_observations"] if o["camera"] == view]
        for o in observations:
            left, top, right, bottom = np.rint(np.asarray(o["box"]) * [640, 360, 640, 360]).astype(int)
            cv2.rectangle(bgr, (left, top), (min(right, 639), min(bottom, 359)), (190, 235, 90), 2)
        panel = np.full((420, 640, 3), (35, 30, 27), np.uint8)
        panel[:360] = bgr
        cv2.putText(
            panel,
            view.replace("_", " ").title(),
            (14, 386),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        detail = f"Frame {row['frame']}   |   {len(observations)} fresh RGB + lidar measurements"
        cv2.putText(panel, detail, (14, 409), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (205, 205, 205), 1, cv2.LINE_AA)
        panels.append(panel)
    output.parent.mkdir(exist_ok=True, parents=True)
    if not cv2.imwrite(str(output), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError("Could not save sensor review image")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("raw", type=Path)
    cli.add_argument("--sidecar", type=Path, help="Export fresh measurements for an older camera recording")
    cli.add_argument(
        "--live-measurements", action="store_true", help="Score captured full-sweep measurements"
    )
    cli.add_argument("--sampled-packets", action="store_true", help="Score frame-* native packet directories")
    cli.add_argument("--metrics", type=Path, required=True)
    cli.add_argument(
        "--review-image", type=Path, help="Save an annotated native frame from retrospective review"
    )
    cli.add_argument("--archive", type=Path, default=Path("data/barrier-recognition/town03-native.zip"))
    args = cli.parse_args()
    cv2.setNumThreads(1)
    if args.sampled_packets:
        if args.live_measurements or args.sidecar or args.review_image:
            cli.error("Sampled native packets take only a --metrics output")
        write_json(args.metrics, evaluate_packets(args.raw))
        return
    if args.live_measurements:
        if args.sidecar or args.review_image:
            cli.error("Live observations are already embedded in the replay; use --metrics")
        measured = live_observations(args.raw)
    else:
        if not args.sidecar:
            cli.error("Reprocessing recorded images requires an output --sidecar")
        measured = observe(args.raw)
        write_json(args.sidecar, measured, compact=True)
        if args.review_image:
            review_image(args.raw, measured, args.review_image)
    report = {"journey": evaluate(args.raw, measured), "archive": evaluate_archive(args.archive)}
    if args.live_measurements:
        report["source"] = measured["source"]
    write_json(args.metrics, report)
    print({k: v for k, v in report["journey"].items() if k != "unmatched"})
    print("Unmatched", len(report["journey"]["unmatched"]))


if __name__ == "__main__":
    main()
