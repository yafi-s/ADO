"""Five synchronized CARLA RGB cameras classified at 5 Hz per view, and 10 Hz full-sweep lidar."""

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from tempfile import mkdtemp

import cv2
import numpy as np

from .camera_geometry import CAMERAS
from .radar_geometry import CORNER_RADARS, RADARS, ego_returns
from .tracking import Tracker


class _ImageAssociations:
    """Short per-camera box histories used only to re-identify independently re-ranged objects."""

    def __init__(self):
        self.recent = {}
        self.next_id = 1

    def update(self, name, boxes, now):
        recent = {k: entry for k, entry in self.recent.get(name, {}).items() if now - entry[0] <= .45}
        assigned = [dict(box) for box in boxes]
        pairs = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box["box"]
            for key, (observed, label, prior, velocity) in recent.items():
                if label != box["label"]:
                    continue
                elapsed = now - observed
                a, b, c, d = prior
                a += velocity[0] * elapsed
                c += velocity[0] * elapsed
                b += velocity[1] * elapsed
                d += velocity[1] * elapsed
                overlap = max(0., min(x2,c)-max(x1,a)) * max(0., min(y2,d)-max(y1,b))
                union = (x2-x1)*(y2-y1) + (c-a)*(d-b) - overlap
                if union > 0 and overlap / union >= .18:
                    pairs.append((overlap / union, idx, key))
        used = set()
        for score, idx, key in sorted(pairs, reverse=True):
            if key in used or "image_id" in assigned[idx]:
                continue
            # Nearly tied overlaps during a crossing are insufficient evidence of identity.
            if any(other != (score, idx, key) and (other[1] == idx or other[2] == key)
                   and other[0] >= score - .10 for other in pairs):
                continue
            assigned[idx]["image_id"] = key
            used.add(key)
        for box in assigned:
            if "image_id" not in box:
                box["image_id"] = self.next_id
                self.next_id += 1
            key = box["image_id"]
            velocity = (0., 0.)
            if key in recent:
                stamp, _, prior, old_velocity = recent[key]
                elapsed = now - stamp
                if elapsed > .001:
                    velocity = tuple(.25*old_velocity[i] + .75*(
                        box["box"][i]+box["box"][i+2]-prior[i]-prior[i+2])/(2*elapsed) for i in (0, 1))
            recent[key] = (now, box["label"], box["box"], velocity)
        self.recent[name] = recent
        return assigned


class SurroundPerception:
    """Infer only on current images. A skipped image cannot support a fresh track."""

    def __init__(self, evidence_dir=None):
        self.tracker = Tracker(surround=True)
        self._images = _ImageAssociations()
        self.view_detections = {}
        self.radar_points = {}
        self.view_ages = dict.fromkeys(CAMERAS)
        self._last_inference = {}
        self._evidence_root = evidence_dir or os.environ.get("CROSSWALK_SURROUND_EVIDENCE")
        self._evidence_session = None
        self._saved = 0
        self._examples = set()
        self._special_examples = set()

    def update(self, detector, obs, front_detections, ego_pose, sim_time, step):
        fresh = {}
        if step % 2 == 0:
            fresh["front"] = front_detections
            auxiliary = ("left", "rear_left")
        else:
            auxiliary = ("right", "rear_right")
        if hasattr(type(detector), "detect_pair"):
            fresh.update(zip(auxiliary, detector.detect_pair(*(obs[name] for name in auxiliary)), strict=True))
        else:
            for name in auxiliary:
                fresh[name] = detector.detect(obs[name])
        fresh = {name: self._images.update(name, boxes, sim_time) for name, boxes in fresh.items()}
        self._last_inference.update(dict.fromkeys(fresh, sim_time))
        self.view_ages = {name: round(sim_time - self._last_inference[name], 4)
                          if name in self._last_inference else None for name in CAMERAS}
        corners = {name: obs[name] for name in CORNER_RADARS if name in obs}
        self.radar_points = {}
        for name, samples in corners.items():
            returns = ego_returns(samples, name)
            self.radar_points[name] = returns[::max(1, math.ceil(len(returns)/24))].round(3).tolist()
        tracks = self.tracker.update(obs["lidar"], obs["radar"], fresh, ego_pose, sim_time, corner_radars=corners)
        self.view_detections = {}
        for view, boxes in fresh.items():
            camera = CAMERAS[view]
            supported = {tuple(track["rgb_boxes"][view]): track["id"]
                         for track in tracks if view in track["rgb_boxes"]}
            tagged = []
            for detection in boxes:
                box = tuple(detection["box"])
                midpoint = (box[0] + box[2]) / 2
                bearing = camera.yaw + math.degrees(math.atan((midpoint - .5) / camera.fx))
                track_id = supported.get(box)
                tagged.append(dict(detection, bearing_deg=(bearing + 180) % 360 - 180,
                                   range_supported=track_id is not None, track_id=track_id))
            self.view_detections[view] = tagged
        if self._evidence_root and self._saved < 8:
            self._record_examples(obs, ego_pose, sim_time, step, tracks)
        return tracks

    def _record_examples(self, obs, pose, sim_time, step, tracks):
        # Reserve capacity for late encounters even when the first hundred metres
        # contain many front-facing cars. Baseline + five directions + two classes.
        new = {view for track in tracks for view in track["cameras"]} - self._examples
        special = {"cycle" for track in tracks if track["cameras"] and track["label"] in ("bicycle", "motorcycle")}
        if any(t["label"] == "unknown" and "lidar" in t["sources"] for t in tracks):
            special.add("unclassified_mover")
        special -= self._special_examples
        if self._saved and not (new or special):
            return
        if self._evidence_session is None:
            root = Path(self._evidence_root)
            root.mkdir(parents=True, exist_ok=True)
            self._evidence_session = Path(mkdtemp(prefix="surround-", dir=root))
        folder = self._evidence_session / f"tick-{step:05d}"
        folder.mkdir()
        for name in CAMERAS:
            if not cv2.imwrite(str(folder / f"{name}.png"), cv2.cvtColor(obs[name], cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not save {name} evidence image")
        names = ("lidar", *RADARS)
        np.savez_compressed(folder / "range.npz", **{name: obs[name] for name in names if name in obs})
        metadata = dict(step=step, sim_time_s=sim_time, ego_xy_yaw=list(pose),
                        cameras={name: asdict(cam) for name, cam in CAMERAS.items()},
                        lidar_mount_xyz=[0, 0, 2.3], radar_mounts=RADARS,
                        fresh_boxes=self.view_detections, inference_age_s=self.view_ages, tracks=tracks)
        (folder / "frame.json").write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")
        self._examples.update(new)
        self._special_examples.update(special)
        self._saved += 1
