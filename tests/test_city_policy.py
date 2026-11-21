import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np

from crosswalk.city_policy import CityModel, FEATURE_NAMES, INDEX, VERSION, clearance, fit, town_records
from crosswalk.behavior_v2 import BehaviorModel, Observer


ROOT = Path(__file__).resolve().parents[1]


def clear_frame():
    x = np.asarray(Observer().observe({"lidar": [], "radar": []}, [], [3,0,7,0,13,0], 0.))
    return np.r_[x, 0., 60.]


class CityPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "models/city-v1/candidate.npz"
        cls.model = CityModel.load(cls.path)

    def test_paired_lidar_radar_close_approach_and_release(self):
        x = clear_frame()
        facts = []
        for depth in (20., 12., 8., 5., 3., 1.):
            x[[INDEX["path_core_lidar"], INDEX["path_body_lidar"], INDEX["path_body_radar"]]] = depth
            facts.append(self.model._plan(x))
        self.assertGreater(facts[0], 4.)
        self.assertTrue(np.all(np.diff(facts) <= 1e-6))
        self.assertLess(facts[-1], .15)
        x[INDEX["ego_speed_mps"]] = 3.9
        self.assertGreater(self.model.predict(x)[2], .4)
        empty = clear_frame()
        self.assertGreater(self.model.predict(empty)[1], .25)

    def test_head_cannot_raise_obstacle_request_with_green_or_red(self):
        x = clear_frame()
        x[INDEX["path_body_lidar"]] = x[INDEX["path_body_radar"]] = .5
        x[INDEX["mapped_lamp_gap_m"]] = 9.
        for red in (0., .5, 1.):
            x[INDEX["rgb_lamp_red_probability"]] = red
            self.assertLess(self.model._plan(x), .15)
        self.assertIsNone(self.model.lamp_expert)
        self.assertIsNotNone(self.model.sensor_lamp_classifier)

    def test_episode_memory_resets_and_model_roundtrips(self):
        x=clear_frame()
        x[INDEX["rgb_lamp_red_probability"]]=1.
        self.model._plan(x)
        self.model.new_observer()
        self.assertEqual(len(self.model.light_window),0)
        self.assertIsNone(self.model.last_speed_request)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"copy.npz"
            self.model.save(output)
            restored=CityModel.load(output)
            np.testing.assert_allclose(self.model.predict(x),restored.predict(x),atol=1e-6)
            self.assertIsNotNone(self.model.last_speed_request)
            self.assertEqual(len(self.model.light_window),1)
        with self.assertRaises(ValueError):
            self.model.predict(x[:-1])

    def test_two_sensor_fusion_and_width(self):
        row=clear_frame()
        row[INDEX["path_body_lidar"]]=9.
        row[INDEX["path_body_radar"]]=4.
        self.assertAlmostEqual(clearance(row,.4),7.)
        self.assertEqual(len(row),len(FEATURE_NAMES))
        self.assertEqual(VERSION,"ado-city-policy-v1")

    def test_reference_town_input_has_no_observed_signal(self):
        episodes=town_records(ROOT/"data/behavior-v2/lessons.zip")
        self.assertEqual(len(episodes),32)
        self.assertEqual(sum(len(r["X"]) for r in episodes),7150)
        for rec in episodes:
            self.assertTrue(np.all(rec["X"][:,-2]==0.))
            self.assertTrue(np.all(rec["X"][:,-1]==60.))

    @unittest.skipUnless(importlib.util.find_spec("sklearn"),"optional fitting dependencies")
    def test_fitting_uses_target_clearance_and_preserves_unseen_near_behavior(self):
        depth=np.repeat([4.,8.,12.,36.],16)
        inputs=np.tile(clear_frame(),(len(depth),1))
        for name in ("path_core_lidar","path_body_lidar","path_body_radar"):
            inputs[:,INDEX[name]]=depth
        rec=dict(X=inputs,kind=np.full(len(depth),"lead"),goal=np.repeat([0.,2.,4.,5.],16),
                 approach=np.full(len(depth),5.),city=True)
        base=BehaviorModel.load(ROOT/"models/behavior-v2/candidate.npz")
        fitted=fit([rec],base)
        queries=[]
        for d in (1.,4.,8.,12.,36.):
            x=clear_frame()
            x[[INDEX["path_body_lidar"],INDEX["path_body_radar"]]]=d
            queries.append(fitted._plan(x))
        np.testing.assert_allclose(queries,[0.,0.,2.,4.,5.],atol=1.e-5)
