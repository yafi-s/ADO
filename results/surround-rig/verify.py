"""Verify the retained native samples without CARLA, a GPU or detector inference."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from crosswalk.tracking import Tracker, _across_views, _match_radar


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def projection_count(xyz, camera, box):
    """Independent pinhole projection of recorded ego-frame surface points."""
    delta = xyz - [camera["x"], camera["y"], camera["z"]]
    yaw = math.radians(camera["yaw"])
    forward = delta[:, 0]*math.cos(yaw) + delta[:, 1]*math.sin(yaw)
    right = delta[:, 1]*math.cos(yaw) - delta[:, 0]*math.sin(yaw)
    fx = .5 / math.tan(math.radians(camera["hfov"])/2)
    u = .5 + fx*right/forward
    v = .5 - fx*(camera["width"]/camera["height"])*delta[:, 2]/forward
    # Published samples are quantized to millimetres.
    return int(np.sum((forward > 0) & (box[0]-.002 <= u) & (u <= box[2]+.002)
                      & (box[1]-.002 <= v) & (v <= box[3]+.002)))


def inspect_snapshot(entry):
    folder = HERE / entry["folder"]
    frame = json.loads((folder/"frame.json").read_text())
    target, = [track for track in frame["tracks"] if track["id"] == entry["target_track"]]
    for name, camera in frame["cameras"].items():
        rgb = cv2.imread(str(folder/f"{name}.png"))
        assert rgb.shape == (camera["height"], camera["width"], 3), name
    with np.load(folder/"range.npz", allow_pickle=False) as data:
        lidar = data["lidar"]
        assert set(data.files) == {"lidar", *frame["radar_mounts"]}
        assert all(data[k].shape[1] == 4 and np.isfinite(data[k]).all() for k in data.files)
        xyz = lidar[:, :3] + frame["lidar_mount_xyz"]
        samples = np.asarray(target["surface_xyz"])
        nearest = [np.linalg.norm(xyz - sample, axis=1).min() for sample in samples]
        assert max(nearest) <= .0009, "Recorded points must exist in the native scan"
        counts = {name: projection_count(samples, frame["cameras"][name], box)
                  for name, box in target["rgb_boxes"].items()}
        assert all(n == len(samples) for n in counts.values())
        # Rebuild every fresh optical candidate from its original selected 2D boxes.
        # Re-associate native per-mount Doppler samples, including unique-hit gating.
        optical = _across_views(lidar, frame["fresh_boxes"])
        for name in frame["radar_mounts"]:
            _match_radar(optical, data[name], name)
        measured, = [item for item in optical if item["boxes"] == target["rgb_boxes"]]
        assert len(measured["hit_ids"]) == target["lidar_hits"]
        assert measured["doppler"].keys() == target["doppler"].keys()
        for name, recorded in target["doppler"].items():
            rebuilt = measured["doppler"][name]
            assert recorded["hits"] == rebuilt["hits"]
            assert math.isclose(recorded["radial_mps"], rebuilt["radial_mps"], abs_tol=1e-5)
        corners = {name: data[name] for name in frame["radar_mounts"] if name != "radar"}
        updated = Tracker(surround=True).update(lidar, data["radar"], frame["fresh_boxes"],
                                                frame["ego_xy_yaw"], frame["sim_time_s"], corner_radars=corners)
        would_start = any(t["rgb_boxes"] == target["rgb_boxes"] for t in updated)
        assert would_start == (entry["seed"] == 211), "Optical edge regression"
        return dict(folder=entry["folder"], time_s=frame["sim_time_s"], recorded_id=target["id"],
                    recorded_label=target["label"], ego_xyz=[target[k] for k in ("x", "y", "z")],
                    verified_surface_points=len(samples), projected_points=counts, lidar_hits=target["lidar_hits"],
                    doppler=target["doppler"], current_tracker_starts_target=would_start,
                    lidar_octant_counts=np.histogram(np.arctan2(xyz[:, 1], xyz[:, 0]),
                                                     bins=np.linspace(-math.pi, math.pi, 9))[0].tolist(),
                    native_return_counts={k: len(data[k]) for k in data.files})


def verify():
    manifest = json.loads((HERE/"manifest.json").read_text())
    for path, expected in manifest["sha256"].items():
        assert digest(HERE/path) == expected, path
    assert digest(HERE.parents[1]/"vendor/yolox/object_detection_yolox_2022nov.onnx") == manifest["model_sha256"]
    handoff = json.loads((HERE/"crossing-handoff.json").read_text())
    assert all(b["frame"] == a["frame"]+1 for a, b in zip(handoff, handoff[1:]))
    seen = Counter()
    errors = []
    for row in handoff:
        assert row["actor_id"] == 433 and row["track_id"] == 5
        x, y = row["track_xy"]
        origin_x, origin_y, heading = row["ego_xy_yaw"]
        yaw = math.radians(heading)
        wx, wy = origin_x+x*math.cos(yaw)-y*math.sin(yaw), origin_y+x*math.sin(yaw)+y*math.cos(yaw)
        if not row["coasting"]:
            errors.append(math.dist((wx, wy), row["actor_world_xy"]))
            seen.update(row["cameras"])
    assert set(seen) == {"front", "left", "rear_left"}
    return dict(snapshots=[inspect_snapshot(entry) for entry in manifest["snapshots"]],
                crossing=dict(first_s=handoff[0]["time_s"], last_s=handoff[-1]["time_s"],
                              consecutive_frames=len(handoff), measured_frames=len(errors), measured_rgb_views=dict(seen),
                              median_world_position_error_m=round(float(np.median(errors)), 3),
                              max_world_position_error_m=round(max(errors), 3)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    output = json.dumps(verify(), indent=2, allow_nan=False) + "\n"
    if args.write_report:
        (HERE/"audit.json").write_text(output, encoding="utf-8")
    print(output, end="")
