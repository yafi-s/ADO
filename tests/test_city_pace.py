import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from crosswalk.city_drive import load_driver
from crosswalk.city_policy import INDEX


ROOT = Path(__file__).resolve().parents[1]


class CityPaceTests(unittest.TestCase):
    def setUp(self):
        self.model, _, _ = load_driver(ROOT / "models/city-pace/candidate.npz")
        with np.load(ROOT / "results/city-cruise/seed-212/observations.npz") as data:
            self.clear = data["features"][1100].copy()

    def test_native_limit_and_speed_regulation(self):
        self.clear[INDEX["ego_speed_mps"]] = 10.
        _, throttle, brake = self.model.predict(self.clear)
        self.assertGreater(self.model.last_speed_request * 3.6, 38.)
        self.assertGreater(throttle, 0.)
        self.assertEqual(brake, 0.)
        for posted in (30., float("nan")):
            self.model.posted_limit_kmh = posted
            _, throttle, brake = self.model.predict(self.clear)
            self.assertLessEqual(self.model.last_speed_request * 3.6, 30.0001)
            self.assertEqual(throttle, 0.)
            self.assertGreater(brake, 0.)

    def test_observed_obstacle_brakes_and_holds(self):
        for field in ("path_body_lidar", "path_body_lidar_recent", "path_body_radar"):
            self.clear[INDEX[field]] = 1.
        for speed in (9., 0.):
            self.clear[INDEX["ego_speed_mps"]] = speed
            _, throttle, brake = self.model.predict(self.clear)
            self.assertLess(self.model.last_speed_request, .15)
            self.assertEqual(throttle, 0.)
            self.assertGreater(brake, .4)

    def test_distinguishes_no_return_from_a_real_obstacle(self):
        first = self.model._plan(self.clear)
        self.clear[INDEX["path_body_lidar_recent"]] = 20.
        second = self.model._plan(self.clear)
        self.assertLess(second, first)
        self.assertLess(second, 8.)

    def test_nearby_curb_return_slows_approach_and_clearance_recovers_gradually(self):
        near_curb = self.clear.copy()
        near_curb[INDEX["right_edge_lidar"]] = self.model.standoff - .2
        self.model.predict(near_curb)
        slow = self.model.last_speed_request
        self.assertAlmostEqual(slow, self.model.side_speed)
        self.model.predict(self.clear)
        self.assertGreater(self.model.last_speed_request, slow)
        self.assertLessEqual(self.model.last_speed_request - slow, .185001)

    def test_late_in_lane_observation_preserves_braking_effort(self):
        x = self.clear.copy()
        x[INDEX["ego_speed_mps"]] = 4.9
        for field in ("path_body_lidar", "path_body_radar"):
            x[INDEX[field]] = 3.7
        _, throttle, brake = self.model.predict(x)
        self.assertEqual(throttle, 0.)
        self.assertGreater(brake, .9)

    def test_signed_preview_applies_to_speed_request_and_resets_between_episodes(self):
        self.model.signed_speed_limit = 7.9
        self.model.predict(self.clear)
        self.assertEqual(self.model.last_speed_request, 7.9)
        self.model.new_observer()
        self.assertIsNone(self.model.signed_speed_limit)
        self.model.predict(self.clear)
        self.assertGreater(self.model.last_speed_request, 10.)

    def test_normal_city_entry_selects_verified_checkpoint(self):
        from crosswalk import city_drive
        class BeforeServerLaunch(Exception):
            pass
        with TemporaryDirectory() as output, patch("sys.argv", ["city-drive","--output",output]), \
                patch.object(city_drive, "load_driver", wraps=load_driver) as load, \
                patch.object(city_drive, "simulator", side_effect=BeforeServerLaunch):
            with self.assertRaises(BeforeServerLaunch):
                city_drive.main()
        load.assert_called_once_with(ROOT/"models/city-maneuvers/candidate.npz")


if __name__ == "__main__":
    unittest.main()
