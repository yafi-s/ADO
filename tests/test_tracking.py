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


def car(x, y, side=0., broadside=False):
    """Lidar returns and an image box for a car whose nearest face is at ego-frame (x, y).

    `side` adds that many metres of the flank beyond the rear face, as an oblique view would.
    `broadside` turns the car across the road, so its 4.4 m flank faces the ego vehicle.
    """
    heights = (.5, .8, 1.1, 1.4)
    across = np.linspace(-2.2, 2.2, 12) if broadside else np.linspace(-.9, .9, 6)
    surface = [(x, y + dy, z) for dy in across for z in heights]
    edge = y + (.9 if y < 0 else -.9)   # the flank nearer the ego vehicle's axis
    surface += [(x + dx, edge, z) for dx in np.arange(.4, side + .01, .4) for z in heights]
    points = np.array(surface)
    depth = points[:, 0] - 1.3
    u = .5 + .5 * points[:, 1] / depth
    v = .5 - (320 / 360) * (points[:, 2] - 1.65) / depth
    box = [u.min() - .004, v.min() - .004, u.max() + .004, v.max() + .004]
    returns = np.column_stack((points[:, 0], points[:, 1], points[:, 2] - 2.3, np.full(len(points), .9)))
    return returns, dict(label="car", score=.9, box=box)


def world(track, pose):
    """World position of a track reported relative to an ego pose (x, y, yaw degrees)."""
    yaw = math.radians(pose[2])
    return (pose[0] + math.cos(yaw) * track["x"] - math.sin(yaw) * track["y"],
            pose[1] + math.sin(yaw) * track["x"] + math.cos(yaw) * track["y"])


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

    def test_new_vehicle_faces_along_the_ego_heading_not_the_sightline(self):
        # A car in the next lane is seen off-axis. Its heading is not the direction it is seen from.
        lidar, detection = car(20., 4.)
        track, = Tracker().update(lidar, EMPTY, [detection], (0, 0, 0), 0.)
        self.assertAlmostEqual(track["heading"], 0.)
        lidar, detection = car(14., 0., broadside=True)
        track, = Tracker().update(lidar, EMPTY, [detection], (0, 0, 0), 0.)
        self.assertAlmostEqual(abs(track["heading"]), 90.)

    def test_stopped_vehicle_holds_heading_and_size_while_the_view_changes(self):
        # The ego vehicle closes on a stopped car in the oncoming lane. Alternate frames
        # see only its rear face, then its rear face and three metres of flank, which
        # changes the single-view length estimate and the apparent centre.
        tracker = Tracker()
        tracks = []
        poses = []
        for step in range(21):
            pose = (.5 * step, 0., 0.)
            lidar, detection = car(30. - pose[0], -4., side=3. if step % 2 else 0.)
            track, = tracker.update(lidar, EMPTY, [detection], pose, .1 * step)
            tracks.append(track)
            poses.append(pose)
        self.assertEqual(len({track["id"] for track in tracks}), 1)
        for before, after in zip(tracks, tracks[1:]):
            self.assertLess(abs(after["heading"]), 3.)
            self.assertLess(abs(after["length"] - before["length"]), .2)
        settled = [world(track, pose) for track, pose in zip(tracks[6:], poses[6:])]
        self.assertLess(np.ptp([x for x, _ in settled]), .3)
        self.assertLess(np.ptp([y for _, y in settled]), .3)
        self.assertLess(max(track["speed_mps"] for track in tracks[6:]), .9)
        # Losing sight of it must not slide it along whatever velocity the changing view suggested.
        held = world(tracks[-1], poses[-1])
        for step in range(21, 25):
            pose = (.5 * step, 0., 0.)
            coast, = tracker.update(EMPTY, EMPTY, [], pose, .1 * step)
            self.assertTrue(coast["coasting"])
            self.assertAlmostEqual(world(coast, pose)[0], held[0])
            self.assertAlmostEqual(world(coast, pose)[1], held[1])

    def test_heading_follows_sustained_travel(self):
        # A car crosses from left to right at 4 m/s. It is first assumed to face the other way.
        tracker = Tracker()
        for step in range(11):
            lidar, detection = car(14., 3. - .4 * step, broadside=True)
            track, = tracker.update(lidar, EMPTY, [detection], (0, 0, 0), .1 * step)
            if step == 0:
                self.assertAlmostEqual(track["heading"], 90.)
        self.assertAlmostEqual(track["heading"], -90., delta=6.)
        self.assertAlmostEqual(track["speed_mps"], 4., delta=.3)
        # A travelling object is still carried forward while it is briefly unseen.
        before = track["y"]
        coast, = tracker.update(EMPTY, EMPTY, [], (0, 0, 0), 1.2)
        self.assertAlmostEqual(coast["y"] - before, -.8, delta=.1)


if __name__ == "__main__":
    unittest.main()
