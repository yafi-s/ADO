"""Independent pixel geometry and causal multi-view fixtures, without a simulator server."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np

from crosswalk.surround import SurroundPerception
from crosswalk.tracking import Tracker


EMPTY = np.empty((0, 4))
MOUNTS = {"front": (1.3, 0, 1.65, 0), "left": (1.3, -.75, 1.65, -70),
          "right": (1.3, .75, 1.65, 70), "rear_left": (-1.1, -.7, 1.72, -145),
          "rear_right": (-1.1, .7, 1.72, 145)}


def target(x, y):
    angle = math.atan2(y, x)
    return np.array([[x - math.sin(angle)*w, y + math.cos(angle)*w, h - 2.3, .8]
                     for w in (-.21, 0, .21) for h in (.4, .8, 1.2, 1.65)])


def pixel_box(points, name):
    x, y, z, yaw = MOUNTS[name]
    dx = points[:, 0] - x
    dy = points[:, 1] - y
    yaw = math.radians(yaw)
    depth = dx * math.cos(yaw) + dy * math.sin(yaw)
    if min(depth) <= .15:
        return []
    # Square pixels, horizontal FOV of 90 degrees.
    width, height = (640, 360) if name == "front" else (320, 180)
    px = width/2 + (dy * math.cos(yaw) - dx * math.sin(yaw)) * width/2/depth
    py = height/2 - (points[:, 2] + 2.3 - z) * width/2/depth
    if min(px) < 1 or max(px) > width - 1 or min(py) < 1 or max(py) > height - 1:
        return []
    return [dict(label="person", score=.9, box=[(min(px)-1)/width, (min(py)-1)/height,
                                               (max(px)+1)/width, (max(py)+1)/height])]


class SurroundTests(unittest.TestCase):
    def test_full_ring_with_measured_depth_and_overlap_deduplication(self):
        for degree in range(-180, 180):
            with self.subTest(degree=degree):
                yaw = math.radians(degree)
                x, y = 12 * math.cos(yaw), 12 * math.sin(yaw)
                points = target(x, y)
                boxes = {name: pixel_box(points, name) for name in MOUNTS}
                tracks = Tracker(surround=True).update(points, EMPTY, boxes, (0, 0, 180), 0)
                self.assertEqual(len(tracks), 1)
                item = tracks[0]
                self.assertEqual(set(item["cameras"]), {name for name in boxes if boxes[name]})
                self.assertAlmostEqual(item["x"], x, delta=.28)
                self.assertAlmostEqual(item["y"], y, delta=.28)
                self.assertEqual(item["lidar_hits"], len(points))
                self.assertEqual(len(item["surface_xyz"]), len(points))
                self.assertEqual(item["sources"], ["rgb", "lidar"])

    def test_passing_stationary_person_front_side_rear_and_brief_loss(self):
        tracker, ids, cameras = Tracker(surround=True), set(), set()
        for step in range(41):
            # Ego passes a person on its left at 8 m/s. Hide the RGB boxes for half a second.
            ego_x = .8 * step
            x, y = 15 - ego_x, -5
            points = target(x, y)
            views = ("front", "left", "rear_left") if step % 2 == 0 else ("right", "rear_right")
            boxes = {view: pixel_box(points, view) for view in views} if step not in range(14, 19) else {}
            item, = tracker.update(points, EMPTY, boxes, (ego_x, 0, 0), .1 * step)
            ids.add(item["id"])
            cameras.update(item["cameras"])
            self.assertLess(abs(ego_x + item["x"] - 15), .38)
            if step in range(14, 19):
                self.assertFalse(item["coasting"])
                self.assertEqual(item["sources"], ["lidar"])
                self.assertEqual(item["lidar_hits"], len(points))
                self.assertGreater(item["class_age_s"], 0)
        self.assertEqual(len(ids), 1)
        self.assertEqual(cameras, {"front", "left", "rear_left"})
        self.assertEqual(tracker.update(EMPTY, EMPTY, {}, (33, 0, 0), 4.81), [])

    def test_rear_neighbours_are_separate_and_rgb_alone_creates_no_tracks(self):
        points = [target(-12., y) for y in (-.8, .8)]
        boxes = {view: sum([pixel_box(p, view) for p in points], []) for view in ("rear_left", "rear_right")}
        tracker = Tracker(surround=True)
        self.assertFalse(tracker.update(EMPTY, EMPTY, boxes, (0, 0, 0), 0))
        fresh = tracker.update(np.vstack(points), EMPTY, boxes, (0, 0, 0), .1)
        self.assertEqual(len(fresh), 2)
        self.assertEqual(len({item["id"] for item in fresh}), 2)
        self.assertGreater(abs(fresh[0]["y"] - fresh[1]["y"]), 1.4)
        self.assertFalse(tracker.update(EMPTY, EMPTY, {}, (0, 0, 0), .41))

    def test_lidar_only_mover_requires_positive_free_rays_in_both_sweeps(self):
        old, new = target(-12., -.7), target(-12., .7)
        def beyond(points):
            far = points.copy()
            far[:, :3] *= 3.
            return far
        tracker = Tracker(surround=True)
        self.assertFalse(tracker.update(np.vstack([old, beyond(new)]), EMPTY, {}, (0, 0, 0), 0.))
        item, = tracker.update(np.vstack([new, beyond(old)]), EMPTY, {}, (0, 0, 0), .4)
        self.assertEqual(item["label"], "unknown")
        self.assertEqual(item["sources"], ["lidar"])
        self.assertIsNone(item["class_age_s"])
        tracker = Tracker(surround=True)
        tracker.update(old, EMPTY, {}, (0, 0, 0), 0.)
        self.assertFalse(tracker.update(new, EMPTY, {}, (0, 0, 0), .4))

    def test_each_corner_radar_matches_its_own_lidar_supported_target(self):
        from crosswalk.radar_geometry import RADARS, CORNER_RADARS
        for name in CORNER_RADARS:
            with self.subTest(name=name):
                x, y, _, yaw = RADARS[name]
                angle = math.radians(yaw)
                x += 10 * math.cos(angle)
                y += 10 * math.sin(angle)
                points = target(x, y)
                boxes = {view: pixel_box(points, view) for view in MOUNTS}
                item, = Tracker(surround=True).update(points, EMPTY, boxes, (0, 0, 0), 0,
                            corner_radars={name: [[-4.25, 0., 0., 10.], [8., .6, 0., 24.]]})
                self.assertEqual(item["radar_sensors"], [name])
                self.assertEqual(item["radial_mps"], -4.25)
                self.assertEqual(item["doppler"][name]["hits"], 1)

    def test_camera_only_keeps_bearing_without_invented_depth_or_identity(self):
        perception = SurroundPerception()
        obs = {view: np.zeros((18, 32, 3), np.uint8) for view in MOUNTS}
        obs.update(lidar=EMPTY, radar=EMPTY)
        left_box = dict(label="person", score=.75, box=[.45,.1,.55,.8])
        detector = Mock()
        detector.detect.side_effect = [[left_box], []]
        self.assertFalse(perception.update(detector, obs, [], (0,0,0), 0., 0))
        item, = perception.view_detections["left"]
        self.assertFalse(item["range_supported"])
        self.assertIsNone(item["track_id"])
        self.assertAlmostEqual(item["bearing_deg"], -70.)

    def test_actual_occlusion_coasts_but_free_rays_erase_ghost(self):
        points = target(-12., 1.5)
        boxes = {name: pixel_box(points, name) for name in MOUNTS}
        tracker = Tracker(surround=True)
        tracker.update(points, EMPTY, boxes, (0, 0, 0), 0.)
        closer = points.copy()
        closer[:, :3] *= .45
        held, = tracker.update(closer, EMPTY, {}, (0, 0, 0), .1)
        self.assertTrue(held["coasting"])
        beyond = points.copy()
        beyond[:, :3] *= 2.5
        self.assertFalse(tracker.update(beyond, EMPTY, {}, (0, 0, 0), .2))

    def test_continuous_image_can_recover_dormant_id_only_after_new_depth(self):
        obs = {name: np.zeros((18, 32, 3), np.uint8) for name in MOUNTS}
        obs["radar"] = EMPTY
        points = target(18., 1.)
        detector = Mock()
        detector.detect.return_value = []
        perception = SurroundPerception()
        last_image = None
        for step in range(19):
            obs["lidar"] = points if step in (0, 18) else EMPTY
            front = pixel_box(points, "front") if step % 2 == 0 and step != 8 else []
            tracks = perception.update(detector, obs, front, (0, 0, 0), step*.1, step)
            if 4 <= step < 18:
                self.assertFalse(tracks)
            if step == 0:
                first_id = tracks[0]["id"]
            if front:
                fresh, = perception.view_detections["front"]
                self.assertEqual(fresh["range_supported"], step in (0, 18))
                if last_image is not None:
                    self.assertEqual(last_image, fresh["image_id"])
                last_image = fresh["image_id"]
        self.assertEqual(tracks[0]["id"], first_id)

    def test_skipped_cameras_never_reuse_old_boxes_and_native_samples_reopen(self):
        with TemporaryDirectory() as directory:
            perception = SurroundPerception(directory)
            obs = {view: np.zeros((18, 32, 3), np.uint8) for view in MOUNTS}
            obs.update(lidar=target(-12., 0.), radar=EMPTY)
            detector = Mock()
            detector.detect.side_effect = [[], pixel_box(obs["lidar"], "rear_left"),
                                           [], pixel_box(obs["lidar"], "rear_right")]
            first, = perception.update(detector, obs, [], (0, 0, 0), 0., 0)
            second, = perception.update(detector, obs, [], (0, 0, 0), .1, 1)
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(second["cameras"], ["rear_right"])
            self.assertEqual(perception.view_ages["rear_left"], .1)
            self.assertNotIn("rear_left", perception.view_detections)
            saved = list(Path(directory).glob("*/tick-*/frame.json"))
            self.assertEqual(len(saved), 2)
            for path in saved:
                meta = json.loads(path.read_text())
                self.assertEqual(meta["tracks"][0]["lidar_hits"], 12)
                with np.load(path.parent / "range.npz", allow_pickle=False) as archive:
                    np.testing.assert_array_equal(archive["lidar"], obs["lidar"])
                self.assertEqual(len(list(path.parent.glob("*.png"))), 5)


if __name__ == "__main__":
    unittest.main()
