import json
from unittest import TestCase

import numpy as np

from crosswalk.behavior_v2 import FEATURE_NAMES
from crosswalk.town01_interaction import COL, MODEL, InteractionModel, calibrate
from crosswalk.town01_interaction_gate import GatedInteractionModel, MODEL as GATED_MODEL, fit
from crosswalk.town01_interaction_study import GATED_SCHEDULE, SCHEDULE


class InteractionTests(TestCase):
    def setUp(self):
        self.driver = InteractionModel.load()
        self.driver.new_observer()
        self.row = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        for name in ("path_body_lidar", "path_body_radar", "right_edge_lidar", "oncoming_gap"):
            self.row[COL[name]] = 36.
        self.row[COL["junction_gap"]] = 60.
        self.row[COL["ego_speed_mps"]] = 4.
        self.row[COL["bend_13_25"]] = -1.

    def test_reproducible_fit_disjoint_schedule(self):
        self.assertEqual(json.loads(MODEL.read_text()), calibrate())
        seeds = {seed for cases in SCHEDULE.values() for seed, _ in cases}
        self.assertTrue(seeds.isdisjoint({43, 44, 47, 48}))

    def test_doppler_hold_and_release(self):
        self.row[COL["junction_gap"]] = 4.
        self.row[COL["oncoming_gap"]] = 15.
        self.assertGreater(self.driver.predict(self.row)[2], 0.)
        self.row[COL["oncoming_gap"]] = 36.
        for _ in range(10):
            self.driver.predict(self.row)
            self.assertEqual(self.driver.observer.last_info["target_mps"], 0.)
        for _ in range(5):
            result = self.driver.predict(self.row)
        self.assertEqual(result, self.driver.base.predict(self.row))

    def test_intermittent_body_protection_and_episode_reset(self):
        self.row[COL["path_body_lidar"]] = 3.
        self.assertGreater(self.driver.predict(self.row)[2], .25)
        self.row[COL["path_body_lidar"]] = 36.
        for _ in range(5):
            self.assertGreater(self.driver.predict(self.row)[2], .25)
        self.driver.new_observer()
        self.assertEqual(self.driver.predict(self.row), self.driver.base.predict(self.row))

    def test_curb_only_observation_allows_forward_motion(self):
        self.row[COL["ego_speed_mps"]] = .1
        self.row[COL["right_edge_lidar"]] = 2.
        self.driver.predict(self.row)
        self.assertGreater(self.driver.observer.last_info["target_mps"], 0.)

    def test_gated_curb_requires_corroboration(self):
        driver = GatedInteractionModel.load()
        driver.new_observer()
        self.row[COL["right_edge_lidar"]] = 2.
        self.assertEqual(driver.edge_distance(self.row, 5.), 36.)
        self.row[COL["rgb_person_height"]] = .2
        self.assertEqual(driver.edge_distance(self.row, 5.), 2.)
        self.row[COL["rgb_person_height"]] = 0.
        for _ in range(driver.config["gate_hold_frames"] - 1):
            self.assertEqual(driver.edge_distance(self.row, 5.), 2.)
        self.assertEqual(driver.edge_distance(self.row, 5.), 36.)

    def test_gate_fit_and_new_challenge(self):
        self.assertEqual(json.loads(GATED_MODEL.read_text()), fit())
        self.assertTrue(set(SCHEDULE["fresh_challenge"]).isdisjoint(GATED_SCHEDULE["fresh_challenge"]))
