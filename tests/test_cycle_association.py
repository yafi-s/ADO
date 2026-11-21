"""Provenance and identity checks on the native synchronized CARLA range fixtures."""

import io
import json
from pathlib import Path
import unittest
import zipfile

import numpy as np

from crosswalk.camera_geometry import CAMERAS
from crosswalk.cycle_association import pairs
from crosswalk.surround import _ImageAssociations
from crosswalk.tracking import Tracker, _project


DATA = Path(__file__).resolve().parents[1] / "results/perception/cycle-range-v04/native-range.zip"


def replay(scene):
    tracker = Tracker(surround=True)
    images = _ImageAssociations()
    with zipfile.ZipFile(DATA) as archive:
        manifest = [json.loads(line) for line in archive.read("manifest.jsonl").splitlines()]
        all_boxes = {r["stem"]: r["boxes"] for r in json.loads(archive.read("boxes.json"))}
        for row in manifest:
            if row["split"] + "-" + row["scene"] != scene:
                continue
            now = row["index"] * 0.1
            boxes = all_boxes[row["stem"]]
            fresh = {"front": images.update("front", boxes, now)} if row["index"] % 2 == 0 else {}
            with np.load(io.BytesIO(archive.read(row["stem"] + ".npz"))) as scan:
                tracks = tracker.update(scan["lidar"], scan["radar"], fresh, row["ego"], now)
            yield row, tracks


class NativeCycleTests(unittest.TestCase):
    def test_exclusive_points_project_outside_rider_and_coast_does_not_refresh_them(self):
        exclusive = []
        for row, tracks in replay("development-side"):
            for track in tracks:
                info = track.get("cycle_association", {})
                if info.get("age_s", 0):
                    self.assertEqual(info["bicycle_exclusive_hits"], 0)
                    self.assertEqual(info["surface_hits"], 0)
                if not info.get("bicycle_exclusive_hits"):
                    continue
                exclusive.append(track["id"])
                self.assertGreaterEqual(info["bicycle_exclusive_hits"], 3)
                self.assertEqual(track["label"], "bicycle")
                points = np.asarray(info["bicycle_surface_xyz"])
                cloud = np.column_stack([points - [0, 0, 2.3], np.ones(len(points))])
                _, _, _, u, v = _project(cloud, CAMERAS["front"])
                a, b, c, d = info["rider_box"]
                self.assertFalse(np.any((u >= a) & (u <= c) & (v >= b) & (v <= d)))
                a, b, c, d = info["bicycle_box"]
                self.assertTrue(
                    np.all((u >= a - 0.0001) & (u <= c + 0.0001) & (v >= b - 0.0001) & (v <= d + 0.0001))
                )
                self.assertEqual(
                    sum(
                        np.hypot(t["x"] - track["x"], t["y"] - track["y"]) < 1.5
                        for t in tracks
                        if t["label"] in ("person", "bicycle")
                    ),
                    1,
                )
        self.assertGreaterEqual(len(exclusive), 2)
        self.assertEqual(len(set(exclusive)), 1)

    def test_separate_walker_and_shared_range_at_unseen_site(self):
        mounted = 0
        for row, tracks in replay("heldout-near_walker"):
            walker = row["truth"]["walker"]
            self.assertTrue(
                any(
                    t["label"] == "person" and np.hypot(t["x"] - walker[0], t["y"] - walker[1]) < 1.0
                    for t in tracks
                )
            )
            for t in tracks:
                if "cycle_association" in t:
                    mounted += 1
                    self.assertEqual(
                        t["cycle_association"]["range_source"], "shared rider/bicycle lidar"
                    )
                    self.assertFalse(t["cycle_association"]["bicycle_surface_xyz"])
        self.assertGreaterEqual(mounted, 4)

    def test_stationary_mounted_actor_does_not_pass_motion_gate(self):
        self.assertFalse(
            any(
                "cycle_association" in t
                for _, tracks in replay("heldout-stationary_cycle")
                for t in tracks
            )
        )

    def test_two_eligible_people_make_pair_ambiguous(self):
        person = dict(label="person", box=[0.42, 0.4, 0.54, 0.7])
        bike = dict(label="bicycle", box=[0.43, 0.54, 0.54, 0.72])
        self.assertEqual(len(pairs([person, bike])), 1)
        self.assertFalse(pairs([person, dict(person), bike]))
