"""Replay the same native optical, lidar and Doppler exposures through two tracker revisions."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import types

import cv2
import numpy as np

from .surround import _ImageAssociations
from .surround_vision import SurroundDetector
from .tracking import Tracker

BASELINE = "ae31697ea43aa9f773c7e73a5bcd3114b3547175"


def baseline():
    module = types.ModuleType("crosswalk._cycle_reference")
    module.__package__ = "crosswalk"
    sys.modules[module.__name__] = module
    source = subprocess.check_output(["git", "show", f"{BASELINE}:crosswalk/tracking.py"])
    exec(compile(source, "original_tracking.py", "exec"), module.__dict__)
    return module.Tracker


def detect(directory):
    net = SurroundDetector()
    rows = []
    for line in (directory / "manifest.jsonl").read_text().splitlines():
        name = json.loads(line)["stem"]
        bgr = cv2.imread(str(directory / f"{name}.png"))
        boxes = net.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        rows.append(dict(stem=name, boxes=boxes))
    (directory / "boxes.json").write_text(json.dumps(rows))


def score(directory, report, phase=0):
    if not (directory / "boxes.json").exists():
        detect(directory)
    boxes = {row["stem"]: row["boxes"] for row in json.loads((directory / "boxes.json").read_text())}
    old = baseline()
    metrics = {}
    examples = []
    for line in (directory / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = row["split"] + "/" + row["scene"]
        if key not in metrics:
            metrics[key] = defaultdict(int)
            trackers = {"before": old(surround=True), "after": Tracker(surround=True)}
            images = _ImageAssociations()
        stat = metrics[key]
        stat["frames"] += 1
        name = row["stem"]
        now = row["index"] * 0.1
        # The production front classifier is sampled every two 10 Hz lidar sweeps.
        fresh = {"front": images.update("front", boxes[name], now)} if row["index"] % 2 == phase else {}
        if any(b["label"] == "bicycle" for b in fresh.get("front", [])):
            stat["rgb_bicycle_frames"] += 1
        truth = row["truth"]
        with np.load(directory / f"{name}.npz") as ranges:
            for mode, tracker in trackers.items():
                result = tracker.update(ranges["lidar"], ranges["radar"], fresh, row["ego"], now)
                cycle = truth.get("cycle")
                near = lambda track, point: (
                    point is not None and np.hypot(track["x"] - point[0], track["y"] - point[1]) < 1.5
                )
                bicycles = [
                    t
                    for t in result
                    if t["label"] == "bicycle" and near(t, cycle or truth.get("parked"))
                ]
                false_bikes = [
                    t
                    for t in result
                    if t["label"] == "bicycle" and not near(t, cycle or truth.get("parked"))
                ]
                if false_bikes:
                    stat[mode + "_false_bicycle_frames"] += 1
                if bicycles:
                    stat[mode + "_cycle_frames"] += 1
                    if len(bicycles) > 1 or any(
                        t["label"] == "person" and near(t, cycle) and not near(t, truth.get("walker"))
                        for t in result
                    ):
                        stat[mode + "_duplicates"] += 1
                if any(t["label"] == "person" and near(t, truth.get("walker")) for t in result):
                    stat[mode + "_walk_frames"] += 1
                for t in result:
                    info = t.get("cycle_association")
                    if not info:
                        continue
                    stat["associated_frames"] += 1
                    if not near(t, cycle) or (
                        truth.get("walker")
                        and np.linalg.norm(np.array([t["x"], t["y"]]) - truth["walker"])
                        < np.linalg.norm(np.array([t["x"], t["y"]]) - cycle)
                    ):
                        stat["false_association_frames"] += 1
                    if info["bicycle_exclusive_hits"]:
                        stat["fresh_exclusive_frames"] += 1
                    if info["age_s"] == 0 and t["radar_sensors"]:
                        stat["fresh_association_with_doppler"] += 1
                    error = (
                        float(np.linalg.norm(np.array([t["x"], t["y"]]) - cycle))
                        if cycle is not None
                        else None
                    )
                    examples.append(
                        dict(
                            stem=name,
                            id=t["id"],
                            xy=[t["x"], t["y"]],
                            center_error_m=error,
                            doppler=t["doppler"],
                            **info,
                        )
                    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            dict(
                baseline=BASELINE,
                camera_phase=phase,
                match_radius_m=1.5,
                results=metrics,
                associations=examples,
            ),
            indent=2,
        )
    )
    for k, v in metrics.items():
        print(k, dict(v))


def parked_controls(directory):
    """Native-derived counterfactual. CARLA's bicycle blueprint includes a seated rider."""
    from .camera_geometry import CAMERAS
    from .cycle_association import _pixels, pairs
    from .tracking import _project, _measure

    manifest = {
        r["stem"]: r for r in map(json.loads, (directory / "manifest.jsonl").read_text().splitlines())
    }
    boxes = {r["stem"]: r["boxes"] for r in json.loads((directory / "boxes.json").read_text())}
    controls = {}
    camera = CAMERAS["front"]
    for split in ("development", "heldout"):
        for walker in (False, True):
            counts = defaultdict(int)
            tracker = Tracker(surround=True)
            images = _ImageAssociations()
            for i in range(15):
                name = f"{split}-stationary_cycle-{i:02d}"
                walk_name = f"{split}-walk-{i:02d}"
                parked_boxes = boxes[name]
                opt = [b for b in parked_boxes if b["label"] == "bicycle"]
                with np.load(directory / f"{name}.npz") as archive:
                    raw = archive["lidar"]
                    pts, ids, dep, u, v = _project(raw, camera)
                    # Remove all returns that could belong to the baked-in seated rider.
                    bm = np.zeros(len(pts), bool)
                    for bike in opt:
                        bm |= _pixels(u, v, bike["box"])
                    for rider in parked_boxes:
                        if rider["label"] == "person":
                            bm &= ~_pixels(u, v, rider["box"], 1 / camera.width, 1 / camera.height)
                    if not any(d["label"] == "person" for d in parked_boxes):
                        bm[:] = False
                    bm &= pts[:, 2] < 0.95
                    # Restrict fixture construction to the scored native parked
                    # actor. The tracker below receives only boxes, ranges and ego pose.
                    bm &= np.linalg.norm(pts[:, :2] - manifest[name]["truth"]["cycle"], axis=1) < 1.4
                    scan = raw[ids[bm]]
                counts["bike_exclusive_returns"] += len(scan)
                if walker:
                    people = [d for d in boxes[walk_name] if d["label"] == "person"]
                    opt += people
                    with np.load(directory / f"{walk_name}.npz") as archive:
                        raw = archive["lidar"]
                        items = _measure(raw, people)
                        hits = sorted({idx for item in items for idx in item["hit_ids"]})
                        scan = np.concatenate([scan, raw[hits]])
                now = i * 0.1
                fresh = {"front": images.update("front", opt, now)} if i % 2 == 0 else {}
                if pairs(fresh.get("front", [])):
                    counts["aligned_camera_pairs"] += 1
                tracks = tracker.update(scan, np.empty((0, 4)), fresh, manifest[name]["ego"], now)
                counts["frames"] += 1
                counts["standalone_bicycle_frames"] += any(t["label"] == "bicycle" for t in tracks)
                counts["person_frames"] += any(t["label"] == "person" for t in tracks)
                counts["false_mount_frames"] += any("cycle_association" in t for t in tracks)
            controls[f"{split}/parked_" + ("with_walker" if walker else "alone")] = counts
    return controls


def audit_city(tape):
    """Count spatial support preserved in the High film's thinned sweep + measured track samples."""
    import hashlib
    from .camera_geometry import CAMERAS
    from .cycle_association import _pixels, pairs, pair_surface
    from .tracking import _project

    result = dict(
        sha256=hashlib.file_digest(tape.open("rb"), "sha256").hexdigest(),
        frames=0,
        bicycle_boxes=0,
        measured_bicycle_track_frames=0,
        exposures=[],
    )
    for line in tape.open(encoding="utf-8"):
        frame = json.loads(line)
        result["frames"] += 1
        if any(t["label"] == "bicycle" for t in frame["tracks"]):
            result["measured_bicycle_track_frames"] += 1
        for view, boxes in frame["camera_detections"].items():
            bikes = [b for b in boxes if b["label"] == "bicycle"]
            if not bikes:
                continue
            result["bicycle_boxes"] += len(bikes)
            # Published streams round their common samples differently. Deduplicate
            # at 2 cm before counting so a return is not corroborated by its own copy.
            xyzi = np.asarray(
                frame["lidar_xyz"]
                + [pt for t in frame["tracks"] if not t["coasting"] for pt in t["surface_xyz"]],
                dtype=float,
            )
            seen = set()
            unique = []
            for point in xyzi:
                key = tuple(np.round(point / 0.02).astype(int))
                if key not in seen:
                    seen.add(key)
                    unique.append(point.copy())
            cloud = np.array(unique)
            cloud[:, 2] -= 2.3
            pts, _, dep, u, v = _project(np.column_stack([cloud, np.ones(len(cloud))]), CAMERAS[view])
            for p, b in pairs(boxes):
                cam = CAMERAS[view]
                exclusive = _pixels(u, v, b["box"]) & ~_pixels(
                    u, v, p["box"], 1 / cam.width, 1 / cam.height
                )
                exclusive &= (pts[:, 2] > 0.22) & (pts[:, 2] < 0.95)
                for other in boxes:
                    if other is not p and other is not b:
                        exclusive &= ~_pixels(u, v, other["box"])
                joint = pair_surface(
                    pts, dep, u, v, [(d, _pixels(u, v, d["box"])) for d in boxes], p, b, cam
                )
                result["exposures"].append(
                    dict(
                        frame=frame["frame"],
                        view=view,
                        shared_surface=joint is not None,
                        exclusive_low_samples=pts[exclusive].round(3).tolist(),
                    )
                )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--phase", choices=(0, 1), default=0, type=int)
    parser.add_argument("--archived-city", type=Path)
    parser.add_argument("--parked-controls", action="store_true")
    args = parser.parse_args()
    if args.archived_city:
        args.report.write_text(json.dumps(audit_city(args.archived_city), indent=2))
    elif args.parked_controls:
        args.report.write_text(json.dumps(parked_controls(args.directory), indent=2))
    else:
        score(args.directory, args.report, args.phase)
