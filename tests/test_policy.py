import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from crosswalk.policy import (BUMPER_X, FEATURE_NAMES, GRID_NAMES, LEGACY_FEATURE_NAMES, NAV_NAMES,
                              RADAR_X, RANGE_FEATURE_NAMES, Model, expand_range, features,
                              head_columns, input_mask, range_features)
from crosswalk.signal_vision import SIGNAL_NAMES, SIDE_NAMES, signal_candidates
from unittest.mock import patch

from crosswalk.steering_search import refine
from crosswalk.train import _fit_head
from crosswalk.train import fit, train
from crosswalk.vision import VISION_NAMES


IMAGE = np.zeros((40, 60, 3), dtype=np.uint8)
EMPTY = np.empty((0, 4), dtype=np.float32)
NAV = (3, 0, 7, 0, 13, 0)
NAV_COLS = [FEATURE_NAMES.index(name) for name in NAV_NAMES]
SIDES = IMAGE, IMAGE


class PerceptionTest(unittest.TestCase):
    def test_rgb_range_navigation_and_detection_encoding(self):
        lidar = np.array([[3, 0, -2.28, 1], [4, -2.0, -1.1, .9], [10, .5, -1.5, .7]])
        radar = np.array([[3, 0, -0.15, 6], [1.5, 0, 0, 6]])
        item = {"label": "person", "score": 0.8, "box": [0.45, 0.1, 0.55, 0.6]}
        x = features(IMAGE, lidar, radar, 4, [item], NAV, side_views=SIDES)
        self.assertEqual(len(x), len(FEATURE_NAMES))
        np.testing.assert_allclose(x[[0, 1, 2, 3, 5, 8]], [5.75, 7.75, 5.75, 1.75, 1.5, 4])
        np.testing.assert_allclose(x[[FEATURE_NAMES.index(k) for k in VISION_NAMES]], [.8, .5, 0, 0, 0, 0])
        np.testing.assert_equal(x[NAV_COLS], NAV)
        b = features(np.full_like(IMAGE, 255), EMPTY, EMPTY, 0, navigation=NAV, side_views=SIDES)
        self.assertEqual(b[0], 60)
        self.assertGreater(np.linalg.norm(x[9:33] - b[9:33]), 1)
        self.assertEqual(FEATURE_NAMES[:45], LEGACY_FEATURE_NAMES)
        with self.assertRaisesRegex(ValueError, "route points"):
            features(IMAGE, EMPTY, EMPTY, 0, navigation=[3, 0], side_views=SIDES)

    def test_lamps_measured_from_actual_images_and_side_swap(self):
        left = np.zeros((180, 320, 3), dtype=np.uint8)
        right = left.copy()
        cv2.circle(left, (224, 70), 3, (255, 14, 4), -1)
        cv2.circle(right, (210, 55), 3, (8, 250, 20), -1)
        lamps = signal_candidates(left)
        self.assertGreater(lamps[0], .8)
        self.assertAlmostEqual(lamps[1], .4, places=2)
        self.assertAlmostEqual(lamps[2], 70/180, places=2)
        self.assertAlmostEqual(lamps[8], 0.)
        x = features(right, EMPTY, EMPTY, 0, navigation=NAV, side_views=(left, right))
        swapped = features(right, EMPTY, EMPTY, 0, navigation=NAV, side_views=(right, left))
        indices = [FEATURE_NAMES.index(k) for k in SIDE_NAMES]
        np.testing.assert_equal(x[indices[:6]], swapped[indices[6:]])
        self.assertGreater(x[FEATURE_NAMES.index("rgb_right_green_strength")], .8)
        self.assertGreater(x[FEATURE_NAMES.index("rgb_signal_green_strength")], .8)
        self.assertEqual(x[FEATURE_NAMES.index("rgb_left_green_strength")], 0.)
        with self.assertRaisesRegex(ValueError, "left and right"):
            features(right, EMPTY, EMPTY, 0, navigation=NAV, side_views=())

    def test_uniform_sky_and_large_signs_do_not_register_as_lamps(self):
        scene = np.full((180, 320, 3), (243, 215, 12), dtype=np.uint8)
        scene[10:70, 20:70] = (240, 0, 0)
        np.testing.assert_equal(signal_candidates(scene), np.zeros(len(SIGNAL_NAMES)))

    def test_spatial_grid_preserves_lateral_side_forward_band_and_doppler(self):
        # Distances are referenced to the bumper and the lidar sits at ego x=0.
        lidar = np.array([[BUMPER_X + 2, -2, -1, 1], [BUMPER_X + 10, 5, -1, 1],
                          [BUMPER_X + 20, 0, -2.3, 1], [BUMPER_X + 60, 0, -1, 1]])
        # Two returns in one sector with opposite radial velocities.
        xf, yr = 9 + BUMPER_X - RADAR_X, -4
        azimuth, depth = np.arctan2(yr, xf), np.hypot(xf, yr)
        radar = np.array([[6, azimuth, 0, depth], [-3, azimuth, 0, depth]])
        x = features(IMAGE, lidar, radar, 4., navigation=NAV, side_views=SIDES)
        grid = dict(zip(GRID_NAMES, x[-len(GRID_NAMES):]))
        self.assertEqual({k for k, v in grid.items() if v != 0}, {
            "bev_lidar_density_0_4m_left", "bev_lidar_density_8_14m_far_right",
            "bev_radar_density_8_14m_far_left", "bev_radar_radial_velocity_8_14m_far_left"})
        self.assertAlmostEqual(grid["bev_radar_radial_velocity_8_14m_far_left"], .1)
        self.assertGreater(grid["bev_radar_density_8_14m_far_left"], grid["bev_lidar_density_0_4m_left"])

    def test_blank_scans_grid_bounds_and_modality_masks(self):
        x = features(IMAGE, EMPTY, EMPTY, 0., navigation=NAV, side_views=SIDES)
        indices = [FEATURE_NAMES.index(k) for k in GRID_NAMES]
        self.assertEqual(len(indices), 90)
        np.testing.assert_array_equal(x[indices], np.zeros(90))
        close = np.tile([BUMPER_X + 3, 0, -1, 1], (300, 1))
        speed = np.array([[100., 0., 0., 3.]])
        filled = features(IMAGE, close, speed, 0., navigation=NAV, side_views=SIDES)
        self.assertEqual(filled[FEATURE_NAMES.index("bev_lidar_density_0_4m_center")], 1.)
        self.assertEqual(filled[FEATURE_NAMES.index("bev_radar_radial_velocity_0_4m_center")], 1.)
        camera_steer, camera_drive = head_columns("camera")
        fusion_steer, fusion_drive = head_columns("fusion")
        self.assertTrue(set(indices).isdisjoint(camera_drive))
        self.assertTrue(set(indices).isdisjoint(fusion_steer))
        self.assertTrue(set(indices).issubset(fusion_drive))
        np.testing.assert_equal(camera_steer, fusion_steer)

    def test_packed_range_matches_full_observation_without_using_cameras(self):
        rng = np.random.default_rng(3)
        rgb = rng.integers(0, 256, size=IMAGE.shape, dtype=np.uint8)
        lidar = rng.uniform([0., -5., -2., 0.], [30., 5., 0., 1.], (176, 4))
        radar = rng.uniform([-8., -.3, -.16, 1.], [8., .3, .16, 22.], (32, 4))
        route = [3, -.25, 6.8, -1.4, 11.6, -3.0]
        packed = range_features(lidar, radar, 3., route)
        mask = input_mask("range")
        self.assertEqual(RANGE_FEATURE_NAMES, tuple(name for name, used in zip(FEATURE_NAMES, mask) if used))
        self.assertEqual(len(packed), 108)
        for image in (rgb, IMAGE):
            full = features(image, lidar, radar, 3., navigation=route, side_views=(image, image))
            np.testing.assert_equal(full[mask], packed)
        self.assertTrue(np.isnan(expand_range(packed[None])[:, ~mask]).all())


class LearningTest(unittest.TestCase):
    def test_packed_diagnostic_fits_real_range_columns_and_rejects_reordered_schema(self):
        close = np.array([[4., 0., -1., 1.]])
        variants = np.stack([range_features(points, EMPTY, 3., NAV) for points in (EMPTY, close)])
        x = np.tile(variants, (64, 1))
        y = np.tile([[0., .6, 0.], [0., 0., .8]], (64, 1))
        route = np.column_stack((np.arange(65.), np.zeros(65)))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data, dest = folder / "range.npz", folder / "weights.npz"
            records = dict(X=x, y=y, seed=np.repeat(11, len(y)), route_xy=route, route_lengths=[len(route)],
                           route_seeds=[11])
            np.savez(data, feature_names=RANGE_FEATURE_NAMES, **records)
            with patch("crosswalk.steering_search.GENERATIONS", 2):
                report = train(data, dest, inputs="range")
            model = Model.load(dest)
            self.assertEqual(report["packed_feature_names"], list(RANGE_FEATURE_NAMES))
            self.assertIn("steering_search_final_fit", report)
            packed_prediction = np.array([model.predict_range(x) for x in variants])
            np.testing.assert_allclose(packed_prediction[:, 1:], y[:2, 1:], atol=.04)
            expanded = expand_range(variants)
            expanded[:, ~model.mask] = 1.e5
            np.testing.assert_allclose(model.predict_batch(expanded), packed_prediction)
            np.savez(data, feature_names=RANGE_FEATURE_NAMES[::-1], **records)
            with self.assertRaisesRegex(ValueError, "incompatible demonstration schema"):
                train(data, dest, inputs="range")

    def test_actions_learn_from_demonstrations_and_roundtrip(self):
        rng = np.random.default_rng(13)
        n = 480
        X = np.tile(features(IMAGE, EMPTY, EMPTY, 2, navigation=NAV, side_views=SIDES), (n, 1))
        offset = rng.uniform(-1, 1, n)
        X[:, NAV_COLS[1::2]] = offset[:, None]
        visual = FEATURE_NAMES.index(VISION_NAMES[0])
        X[:, visual] = np.arange(n) % 2
        y = np.column_stack((.55 * offset, .7 * (1 - X[:, visual]), .8 * X[:, visual]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savez(path / "records.npz", X=X, y=y, seed=np.repeat(np.arange(6), n // 6))
            report = train(path / "records.npz", path / "policy.npz", inputs="camera", route_refinement=False)
            model = Model.load(path / "policy.npz")
            self.assertEqual(model.inputs, "camera")
            held = report["development_holdout_before_final_refit"]
            self.assertEqual(held["frames"], n // 6)
            sensitivity = held["mean_replacement_pedal_sensitivity"]
            self.assertGreater(sensitivity["front_object_detections"]["brake_mean_absolute_change"], .1)
            self.assertEqual(sensitivity["side_camera_statistics_and_candidates"]["brake_mean_absolute_change"], 0)
            for metric in held["actuators"].values():
                self.assertLess(metric["mae"], .04)
            np.testing.assert_allclose(model.predict_batch(X[:12]), y[:12], atol=.05)
            changed = X[:12].copy()
            changed[:, ~model.mask] = np.nan
            np.testing.assert_equal(model.predict_batch(changed), model.predict_batch(X[:12]))
            np.testing.assert_allclose(model.predict(changed[1]), y[1], atol=.05)
            self.assertTrue(set(NAV_NAMES).issubset(report["head_features"]["steer"]))
            self.assertTrue(set(VISION_NAMES).issubset(report["head_features"]["drive"]))

    def test_imitation_receives_exact_original_and_reflected_controls(self):
        X = np.tile(features(IMAGE, EMPTY, EMPTY, 3., navigation=NAV, side_views=SIDES), (8, 1))
        # Deliberately inconsistent with the straight centerline.
        steer = np.array([-.9, .8, -.2, .75, -.6, .05, .3, -.09], dtype=np.float32)
        y = np.column_stack((steer, np.ones(8), np.zeros(8))).astype(np.float32)
        original = y.copy()
        with patch("crosswalk.train._fit_head", wraps=_fit_head) as heads:
            fit(X, y, epochs=1)
        np.testing.assert_array_equal(heads.call_args_list[0].args[1], np.r_[steer, -steer])
        np.testing.assert_array_equal(y, original)

    def test_search_changes_only_neural_steering_and_scores_withheld_geometry(self):
        X = np.tile(features(IMAGE, EMPTY, EMPTY, 3., navigation=NAV, side_views=SIDES), (8, 1))
        X[:, NAV_COLS[1::2]] = np.arange(8)[:, None] / 10.
        y = np.tile([.45, .6, 0.], (8, 1))
        model = fit(X, y, epochs=2)
        pedal = [v.copy() for v in model.heads[1]]
        hidden = model.heads[0][0].copy()
        t = np.arange(0., 75.)
        paths = [np.column_stack((t, .007*t**2)), np.column_stack((t, -.007*t**2))]
        dev = [np.column_stack((t, .01*t**2))]
        with patch("crosswalk.steering_search.POPULATION", 12), patch("crosswalk.steering_search.GENERATIONS", 3):
            report = refine(model, paths, evaluation_routes=dev)
        self.assertEqual(report["demonstration_route_count"], 2)
        training = report["training_kinematic_rollouts"]
        self.assertLessEqual(training["after_search"]["objective"], training["imitation_only"]["objective"])
        self.assertGreater(report["withheld_kinematic_rollouts"]["windows"], 0)
        np.testing.assert_equal(hidden, model.heads[0][0])
        for original, actual in zip(pedal, model.heads[1]):
            np.testing.assert_equal(original, actual)

    def test_route_split_keeps_validation_seeds_out_of_search_and_saves_baseline(self):
        X = np.tile(features(IMAGE, EMPTY, EMPTY, 3., navigation=NAV, side_views=SIDES), (16, 1))
        y = np.tile([.04, .6, 0.], (16, 1))
        seeds = np.repeat([11, 12], 8)
        t = np.arange(70.)
        routes = [np.column_stack((t, t**2*k)) for k in (-.007, .007)]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = folder / "data.npz"
            np.savez(data, X=X, y=y, seed=seeds, route_xy=np.concatenate(routes),
                     route_seeds=[11, 12], route_lengths=[70, 70])
            with patch("crosswalk.steering_search.POPULATION", 12), patch("crosswalk.steering_search.GENERATIONS", 2):
                report = train(data, folder / "final.npz", imitation_path=folder / "baseline.npz")
            self.assertEqual(report["steering_search_seed_holdout"]["demonstration_route_count"], 1)
            self.assertEqual(report["steering_search_final_fit"]["demonstration_route_count"], 2)
            final = Model.load(folder / "final.npz")
            baseline = Model.load(folder / "baseline.npz")
            np.testing.assert_equal(final.predict_batch(X)[:, 1:], baseline.predict_batch(X)[:, 1:])

    def test_legacy_observations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for old_width in (45, 78):
                np.savez(path / "old.npz", X=np.ones((20, old_width)), y=np.zeros((20, 3)))
                with self.assertRaisesRegex(ValueError, "incompatible demonstration schema"):
                    train(path / "old.npz", path / "model.npz")

    def test_range_signal_and_steering_labels_control_independent_heads(self):
        rng = np.random.default_rng(12)
        X = np.tile(features(IMAGE, EMPTY, EMPTY, 2, navigation=NAV, side_views=SIDES), (256, 1))
        cell = FEATURE_NAMES.index("bev_lidar_density_4_8m_left")
        X[:, cell] = rng.choice([.0, 1.], len(X))
        X[:, NAV_COLS[1]] = rng.choice([-1., 1.], len(X))
        brake = X[:, cell] == 1.
        y = np.column_stack((-.6 * X[:, NAV_COLS[1]], .7 * ~brake, .8 * brake))
        model = fit(X, y)
        np.testing.assert_allclose(model.predict_batch(X), y, atol=.05)
        self.assertEqual(model.inputs, "fusion")
        with self.assertRaisesRegex(ValueError, "N x 3"):
            fit(X, y[:, 1])


if __name__ == "__main__":
    unittest.main()
