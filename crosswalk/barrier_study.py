"""Reproduce the native striped-board appearance fit and blind RGB/lidar scoring."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from .barrier import MODEL, detect, stripe_image
from .camera_geometry import CAMERAS
from .surround_vision import SurroundDetector
from .tracking import Tracker

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/barrier-recognition/town03-native.zip"
RESULT = ROOT / "results/barrier-recognition/validation.json"
# Human-annotated rectangles around the striped panels in the two close training views.
ANNOTATIONS = {"fit_0_barricade_5": (428, 234, 116, 72), "fit_1_barricade_5": (428, 240, 119, 72)}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_rgb(data):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def fit(archive):
    templates = []
    for name, (x, y, width, height) in ANNOTATIONS.items():
        image = read_rgb(archive.read(name + ".jpg"))
        stripe = stripe_image(image)[y:y + height, x:x + width]
        tile = cv2.resize(stripe, (32, 20), interpolation=cv2.INTER_AREA).astype(float)
        templates.append(tile / tile.max())
    return {"schema": 1, "name": "striped native street barricade", "threshold": .60,
            "stripe_template": np.rint(np.mean(templates, axis=0)*255).astype(int).tolist()}


def validate(source=ARCHIVE):
    detector = SurroundDetector()
    empty = np.empty((0, 4))
    details, by_group = [], defaultdict(list)
    with ZipFile(source) as archive:
        if source.resolve() == ARCHIVE.resolve() and fit(archive) != json.loads(MODEL.read_text()):
            raise ValueError("Published model differs from fit on the two annotated training views")
        for entry in json.loads(archive.read("manifest.json")):
            name = entry["name"]
            rgb = read_rgb(archive.read(name + ".jpg"))
            with np.load(io.BytesIO(archive.read(name + ".npz"))) as packet:
                lidar = packet["lidar"].astype(np.float32)
            # Use the same production competition for laser hits. Each exposure is
            # scored with fresh tracker state, so a remembered label cannot earn credit.
            boxes = detector.detect(rgb) + detect(rgb, lidar, "front")
            tracks = Tracker(surround=True).update(lidar, empty, {"front": boxes}, (0., 0., 0.), 0.)
            predictions = [t for t in tracks if t["label"] == "barrier" and t["lidar_hits"] >= 3 and t["cameras"]]
            positive = entry["condition"] == "barricade"
            target = entry["target_xy_m"] if positive else None
            measured = [{"center_xy_m": [round(t["x"], 4), round(t["y"], 4)], "lidar_hits": t["lidar_hits"],
                         "rgb_score": round(t["confidence"], 4)} for t in predictions]
            errors = [float(np.hypot(t["x"]-target[0], t["y"]-target[1])) for t in predictions] if positive else []
            error = min(errors, default=None)
            tp = int(error is not None and error <= .65)
            output = dict(name=name, split=entry["split"], condition=entry["condition"], distance_m=entry["distance_m"],
                          positive=positive, true_positive=tp, false_positive=len(predictions)-tp,
                          false_negative=int(positive)-tp, closest_center_error_m=None if error is None else round(error, 4),
                          measurements=measured)
            details.append(output)
            by_group[entry["split"]].append(output)
    summaries = {}
    for split, rows in by_group.items():
        counts = {field: sum(row[field] for row in rows) for field in
                  ("positive", "true_positive", "false_positive", "false_negative")}
        errors = [row["closest_center_error_m"] for row in rows if row["true_positive"]]
        summaries[split] = dict(exposures=len(rows), negative_exposures=len(rows)-counts["positive"],
                                **counts, median_center_error_m=round(float(np.median(errors)), 4),
                                max_center_error_m=max(errors))
    hedge_root = ROOT / "results/surround-rig/rear-right-edge"
    with np.load(hedge_root / "range.npz") as packet:
        hedge = {view: len(detect(read_rgb((hedge_root / (view + ".png")).read_bytes()), packet["lidar"], view))
                 for view in CAMERAS}
    return {"schema": "ado-barrier-validation-v1", "archive_sha256": digest(source.read_bytes()),
            "model_sha256": digest(MODEL.read_bytes()), "labels_used_at_inference": False,
            "split_unit": "road/lane/station, four Town03 capture sites; independent held-out approach angles",
            "match_radius_m": .65, "fit_annotations_xywh": ANNOTATIONS, "totals": summaries,
            "seed212_hedge_control_detections_by_view": hedge, "frames": details}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--fit", action="store_true", help="refit and write the appearance model")
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.fit:
        with ZipFile(args.archive) as archive:
            MODEL.write_bytes((json.dumps(fit(archive), separators=(",", ":")) + "\n").encode("utf-8"))
    report = validate(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["totals"], indent=2))
