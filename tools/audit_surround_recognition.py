"""Replay the archived CARLA camera frames and score raw observer boxes."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from crosswalk.surround_vision import SurroundDetector
from crosswalk.vision import Detector


SAMPLE = Path(__file__).resolve().parents[1] / "results" / "surround-recognition"
VEHICLES = {"car", "bus", "truck", "motorcycle"}


def _iou(a, b):
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return intersection / (np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)


def _group(objects, detections, group):
    targets = {i: item for i, item in enumerate(objects)
               if (item["label"] == "motor_vehicle") == (group == "motor_vehicle")}
    matched = {}
    false_positives = []
    explicit_bikes = set()
    for det in sorted(detections, key=lambda det: -det["score"]):
        motor = det["label"] in VEHICLES
        if motor != (group == "motor_vehicle"):
            continue
        b = np.asarray(det["box"]) * [640, 360, 640, 360]
        overlaps = [(_iou(np.asarray(item["box"]), b), i) for i, item in targets.items()
                    if det["label"] != "bicycle" or item["label"] == "cyclist"]
        overlap, index = max(overlaps, default=(0, -1))
        if overlap >= 0.30:
            hits = matched.setdefault(index, set())
            # Person and bicycle may independently support the same annotated cyclist.
            companion = targets[index]["label"] == "cyclist" and det["label"] not in hits
            if hits and not companion:
                false_positives.append(det)
            hits.add(det["label"])
            if det["label"] == "bicycle":
                explicit_bikes.add(index)
        elif not motor or b[3] - b[1] >= 40:
            false_positives.append(det)
    return {"true_positive": len(matched), "false_positive": len(false_positives),
            "false_negative": len(targets) - len(matched), "explicit_bicycle": len(explicit_bikes),
            "cyclist_targets": sum(t["label"] == "cyclist" for t in targets.values()),
            "missed": [targets[i] for i in targets.keys() - matched.keys()],
            "spurious": false_positives}


def _timed(fn, *args):
    start = perf_counter()
    result = fn(*args)
    return round(1000 * (perf_counter() - start), 2), result


def audit():
    cv2.setNumThreads(2)
    labels = json.loads((SAMPLE / "labels.json").read_text())
    archive = SAMPLE / "sample.mjpeg"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != labels["sample_sha256"]:
        raise ValueError("Camera sample digest mismatch")
    cap = cv2.VideoCapture(str(archive), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    images = []
    try:
        for _ in labels["frames"]:
            ok, bgr = cap.read()
            if not ok:
                raise ValueError("Incomplete camera sample")
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if cap.grab():
            raise ValueError("Extra frames in camera sample")
    finally:
        cap.release()
    baseline = Detector()
    selected = SurroundDetector()
    frames = []
    for j, (rgb, source) in enumerate(zip(images, labels["frames"], strict=True)):
        timing, raw = {}, {}
        timing["nanodet"], raw["nanodet"] = _timed(baseline.detect, rgb)
        timing["yolox"], raw["yolox"] = _timed(selected.detect, rgb)
        # Rotate the second input by half the audit to exercise neighboring camera
        # interference. Each audited image is seen in both positions across the run.
        other_index = (j + len(images) // 2) % len(images)
        first = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)
        other = cv2.resize(images[other_index], (320, 180), interpolation=cv2.INTER_AREA)
        ms, (top, bottom) = _timed(selected.detect_pair, first, other)
        timing["yolox_pair"] = ms
        raw["yolox_pair_top"] = top
        frame = {"video_index": source["video_index"], "ms": timing, "boxes": raw,
                 "pair_bottom_index": labels["frames"][other_index]["video_index"],
                 "pair_bottom": bottom}
        frames.append(frame)
    by_index = {row["video_index"]: row for row in frames}
    for row in frames:
        by_index[row.pop("pair_bottom_index")]["boxes"]["yolox_pair_bottom"] = row.pop("pair_bottom")
    metrics = {}
    for name in frames[0]["boxes"]:
        metrics[name] = {}
        for group in ("vulnerable_road_user", "motor_vehicle"):
            results = []
            for frame, source in zip(frames, labels["frames"], strict=True):
                score = _group(source["objects"], frame["boxes"][name], group)
                results.append(score)
                frame.setdefault("matches", {}).setdefault(name, {})[group] = score
            totals = {k: sum(row[k] for row in results)
                      for k in ["true_positive", "false_positive", "false_negative",
                                "explicit_bicycle", "cyclist_targets"]}
            tp = totals["true_positive"]
            totals["recall"] = round(tp / max(1, tp + totals["false_negative"]), 4)
            totals["precision"] = round(tp / max(1, tp + totals["false_positive"]), 4)
            metrics[name][group] = totals
    median_ms = {k: round(float(np.median([f["ms"][k] for f in frames[1:]])), 1)
                 for k in frames[0]["ms"]}
    return {"source": labels["sample_sha256"], "frames": frames, "metrics": metrics,
            "opencv": cv2.__version__, "backend": selected.backend,
            "median_ms_excluding_first": median_ms}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save per-frame boxes, timings and matches to JSON")
    args = parser.parse_args()
    result = audit()
    if args.output:
        args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
