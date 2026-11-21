import tempfile
import unittest
from pathlib import Path

import numpy as np

from crosswalk.behavior_v2 import (
    BehaviorModel, FEATURE_NAMES, HORIZON, Observer, PLANNER_NAMES, _feedback_features,
    _ground_plane, _preview, _steering_basis, _targets,
)


NAV = [3., 0., 7., 0., 13., 0.]
EMPTY = np.empty((0, 4), dtype=np.float32)


def read(lidar=EMPTY, radar=EMPTY, route=NAV, speed=0., observer=None):
    observer = observer if observer is not None else Observer()
    row = observer.observe(dict(lidar=lidar, radar=radar), [], route, speed)
    return dict(zip(FEATURE_NAMES, row))


class ObservationTests(unittest.TestCase):
    def test_height_above_inclined_road(self):
        x, y = np.meshgrid(np.arange(1.1, 26., .25), np.arange(-4.9, 5., .25))
        z = .025*x - .02*y - .10
        measured = _ground_plane(x.ravel(), y.ravel(), z.ravel())
        np.testing.assert_allclose(measured[:2], [.025, -.02], atol=.002)
        cloud = np.column_stack((x.ravel(), y.ravel(), z.ravel()-2.30, np.ones(x.size)))
        self.assertEqual(read(lidar=cloud)["path_body_lidar"], HORIZON)
        above = [[8., .2, -.85, 1.], [8.03, .2, -.87, 1.]]
        self.assertAlmostEqual(read(lidar=np.vstack((cloud, above)))["path_body_lidar"], 5.75, places=1)

    def test_off_axis_parked_vehicle_and_planned_curve(self):
        van = np.array([[6., 1.65, -.8, 1.], [6.05, 1.66, -.85, 1.]])
        straight = read(lidar=van)
        self.assertEqual(straight["path_body_lidar"], HORIZON)
        self.assertLess(straight["right_edge_lidar"], 4.)
        curve = read(lidar=van, route=[3., .75, 7., 2., 12., 5.])
        self.assertLess(curve["path_core_lidar"], 5.)

    def test_doppler_sign_and_ego_compensation(self):
        radar = np.array([[-10., -.12, 0., 16.]])
        row = read(radar=radar, speed=4.)
        self.assertLess(row["oncoming_gap"], 17.)
        self.assertAlmostEqual(row["oncoming_speed"], -6., delta=.1)
        radar[0, 0] = -4.*np.cos(.12)
        self.assertEqual(read(radar=radar, speed=4.)["oncoming_gap"], HORIZON)

    def test_history_is_causal_episode_local_and_expires(self):
        obstacle = np.array([[5., 0., -.7, 1.], [5.02, 0., -.65, 1.]])
        observer = Observer()
        first = read(lidar=obstacle, observer=observer)
        self.assertEqual(first["path_body_lidar"], first["path_body_lidar_recent"])
        self.assertLess(read(observer=observer)["path_body_lidar_recent"], HORIZON)
        for _ in range(8):
            result = read(observer=observer)
        self.assertEqual(result["path_body_lidar_recent"], HORIZON)
        self.assertEqual(read()["path_body_lidar_recent"], HORIZON)

    def test_preview_reconstructs_ego_frame_including_end_extension(self):
        route = np.array([[10., 20.], [10., 25.], [10., 30.]])
        close = np.array([[3., -.5, 7., -.5, 13., -.5]])
        world = _preview(route, np.array([8.]), close)
        np.testing.assert_allclose(world[0,:,0], [18., 25., 32., 40.], atol=1e-5)
        np.testing.assert_allclose(world[0,:,1], -.5, atol=1e-5)


def tiny_model():
    gap = PLANNER_NAMES.index("path_body_lidar")
    return BehaviorModel(np.array([0]), np.array([gap, -2, -2]), np.array([5., -2., -2.]),
                         np.array([1, -1, -1]), np.array([2, -1, -1]), np.array([4., 0., 6.]),
                         np.array([.8, .3, .1]), np.array([.08, .2, .4, .6, .4, -.8]), None)


class PolicyTests(unittest.TestCase):
    def test_saved_control_roundtrip_with_learned_leaf(self):
        model = tiny_model()
        row = np.array(list(read().values()))
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/"candidate.npz"
            model.save(p)
            loaded = BehaviorModel.load(p)
            self.assertGreater(loaded.predict(row)[1], .3)
            row[FEATURE_NAMES.index("path_body_lidar")] = 2.
            self.assertGreater(loaded.predict(row)[2], .6)
            np.testing.assert_allclose(loaded.predict(row), model.predict(row))
            self.assertIsNot(loaded.new_observer(), loaded.new_observer())

    def test_mirrored_navigation_and_monotone_pedal(self):
        nav = np.array([4., 3., .6, 6.8, 1.2, 12.7, 2.2])
        mirror = nav.copy()
        mirror[[2, 4, 6]] *= -1
        np.testing.assert_allclose(_steering_basis(nav), -_steering_basis(mirror))
        target = np.linspace(0., 6., 60)
        model = tiny_model()
        for speed in (0., 2., 4., 6.):
            pedal = _feedback_features(target, np.full_like(target,speed)) @ model.feedback
            self.assertTrue(np.all(np.diff(pedal) >= -1e-8))

    def test_pedestrian_lookahead_stays_within_demonstration(self):
        X = np.zeros((12,len(FEATURE_NAMES)))
        rec = dict(X=X,expert_gap_m=np.r_[np.full(8,60.),np.full(4,1.5)],
                   progress_m=np.zeros(12),scenario=np.array(["pedestrian"]*12),
                   episode_id=np.array(["a"]*4 + ["b"]*8))
        plan = _targets(rec)
        np.testing.assert_array_equal(plan[:4], 6.)
        np.testing.assert_array_equal(plan[4:], 0.)
