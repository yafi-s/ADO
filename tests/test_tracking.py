import json
import math
import unittest

import numpy as np

from crosswalk.tracking import Tracker


EMPTY = np.empty((0, 4))


def person(x=11.3, y=0.):
    # An independent pinhole fixture in pixel coordinates, before normalization.
    d = x - 1.3
    left = 320 + 320 * (y - .28) / d
    right = 320 + 320 * (y + .28) / d
    top = 180 - 320 * (.10 / d)
    bottom = 180 + 320 * (1.65 / d)
    detection = dict(label="person", score=.87, box=[left/640, top/360, right/640, bottom/360])
    returns = np.array([[x, y+dy, z-2.3, .9] for dy in (-.19, .0, .19) for z in (.4, .8, 1.3, 1.65)])
    return returns, detection


class TrackingTests(unittest.TestCase):
    def test_extrinsic_projection_and_ground_removal(self):
        # At this range the camera/lidar offset shifts the box by dozens of pixels.
        lidar, detection = person(4.3, 1.7)
        track, = Tracker().update(lidar, EMPTY, [detection], (0, 0, 0), 0.)
        self.assertGreater(track["x"], 4.3)
        self.assertLess(track["x"], 4.56)
        self.assertAlmostEqual(track["y"] / track["x"], 1.7 / 4.3)
        self.assertGreater(track["z"], .6)
        self.assertEqual(track["sources"], ["rgb", "lidar"])
        self.assertIn("heading", track["estimated_fields"])
        json.dumps(track, allow_nan=False)
        ground = lidar.copy()
        ground[:, 2] = -2.29
        self.assertEqual(Tracker().update(ground, EMPTY, [detection], (0, 0, 0), 0.), [])

    def test_pose_compensation_coasting_and_expiry(self):
        tracker = Tracker()
        lidar, detection = person(11.3)
        original, = tracker.update(lidar, EMPTY, [detection], (0, 0, 0), 0.)
        lidar, detection = person(9.8)
        second, = tracker.update(lidar, EMPTY, [detection], (1.5, 0, 0), .1)
        self.assertEqual(original["id"], second["id"])
        self.assertAlmostEqual(second["speed_mps"], 0.)
        coast, = tracker.update(EMPTY, EMPTY, [], (1.5, 0, 90), .2)
        self.assertTrue(coast["coasting"])
        self.assertEqual(coast["sources"], [])
        self.assertLess(coast["confidence"], second["confidence"])
        self.assertAlmostEqual(coast["x"], second["y"])
        self.assertAlmostEqual(coast["y"], -second["x"])
        self.assertEqual(tracker.update(EMPTY, EMPTY, [], (0, 0, 0), .61), [])

    def test_no_measurement_from_unsupported_or_ambiguous_boxes(self):
        lidar, detection = person()
        behind = lidar.copy()
        behind[:, 0] = -4
        outside = lidar.copy()
        outside[:, 1] = 30
        background = lidar.copy()
        background[:, 0] += 3
        for points, boxes in [(behind, [detection]), (outside, [detection]),
                              (EMPTY, [detection]), (lidar, []),
                              (np.vstack([lidar, background]), [detection]),
                              (lidar, [detection, detection])]:
            self.assertEqual(Tracker().update(points, EMPTY, boxes, (0, 0, 0), 0.), [])

    def test_signed_radar_velocity_matches_in_three_dimensions(self):
        lidar, detection = person(11.3, 1.5)
        dx, dy = 9.3, 1.5
        az = math.atan2(dy, dx)
        d = math.hypot(dx, dy)
        radar = [[-2.75, az, 0, d], [8., -az, 0, d], [9., az, .55, d], [4., az, 0, d + 3]]
        track, = Tracker().update(lidar, radar, [detection], (0, 0, 0), 0.)
        self.assertEqual(track["radial_mps"], -2.75)
        self.assertIn("radar", track["sources"])
        self.assertIsNone(track["speed_mps"])

    def test_world_motion_while_ego_follows_at_same_speed(self):
        tracker = Tracker()
        lidar, detection = person(13.)
        track, = tracker.update(lidar, EMPTY, [detection], (0, 0, 30), 0.)
        dx = math.cos(math.radians(30)) * .2
        dy = math.sin(math.radians(30)) * .2
        moved, = tracker.update(lidar, EMPTY, [detection], (dx, dy, 30), .1)
        self.assertEqual(moved["id"], track["id"])
        self.assertAlmostEqual(moved["speed_mps"], 2.)
        self.assertAlmostEqual(moved["heading"], 0.)


if __name__ == "__main__":
    unittest.main()
