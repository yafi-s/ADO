import tempfile
import unittest
from pathlib import Path

import numpy as np

from crosswalk.policy import CRUISE_SPEED, FEATURE_NAMES, Model, expert_speed, features, rule_speed
from crosswalk.train import fit, train


IMAGE = np.zeros((40, 60, 3), dtype=np.uint8)
EMPTY = np.empty((0, 4), dtype=np.float32)


class PerceptionTest(unittest.TestCase):
    def test_empty_scans_and_rgb_change(self):
        a = features(IMAGE, EMPTY, EMPTY, 0)
        b = features(np.full_like(IMAGE, 255), EMPTY, EMPTY, 0)
        np.testing.assert_equal(a[:9], b[:9])
        self.assertEqual(len(a), len(FEATURE_NAMES))
        self.assertGreater(np.linalg.norm(a - b), 1)
        self.assertEqual(rule_speed(a), CRUISE_SPEED)

    def test_ground_and_shoulder_excluded_from_travel_corridor(self):
        lidar = np.array([[3, 0, -2.28, 1], [4, -2.0, -1.1, .9], [10, .5, -1.5, .7]])
        radar = np.array([[3, 0, -0.15, 6], [1.5, 0, 0, 6]])
        x = features(IMAGE, lidar, radar, 4)
        self.assertAlmostEqual(x[1], 7.75)
        self.assertAlmostEqual(x[2], 5.75)
        self.assertAlmostEqual(x[0], 5.75)
        self.assertAlmostEqual(x[3], 1.75)
        self.assertAlmostEqual(x[5], 1.5)
        self.assertAlmostEqual(x[8], 4.0)
        self.assertLess(rule_speed(x), CRUISE_SPEED)

    def test_braking_target_and_nonfinite_hits(self):
        self.assertEqual(expert_speed(2), 0)
        self.assertEqual(expert_speed(60), CRUISE_SPEED)
        self.assertLess(expert_speed(6), expert_speed(10))
        self.assertEqual(features(IMAGE, [[np.nan, 0, -1, .5]], EMPTY, 3)[0], 60)


class LearningTest(unittest.TestCase):
    def test_camera_ablation_excludes_radar_and_lidar_at_fit_and_inference(self):
        rng = np.random.default_rng(17)
        X = rng.uniform(0, 1, size=(180, len(FEATURE_NAMES))).astype(np.float32)
        y = 5 * X[:, 9]
        X[:, 0] = y  # A tempting direct label channel excluded from this model.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            np.savez(folder / "data.npz", X=X, y=y)
            report = train(folder / "data.npz", folder / "camera.npz", inputs="camera")
            self.assertNotIn("corridor_gap_m", report["final_model_split_counts"])
            self.assertIn(FEATURE_NAMES[9], report["final_model_split_counts"])
            model = Model.load(folder / "camera.npz")
            self.assertEqual(model.inputs, "camera")
            columns, counts = np.unique(model.splits[model.splits >= 0], return_counts=True)
            self.assertEqual(report["final_model_split_counts"],
                             {FEATURE_NAMES[i]: int(n) for i, n in zip(columns, counts)})
            self.assertTrue(all(name == "ego_speed_mps" or name.startswith("rgb_")
                                for name in report["final_model_split_counts"]))
            perturbed = X.copy()
            perturbed[:, :8] = rng.uniform(0, 60, size=(len(y), 8))
            refit = fit(perturbed, y, inputs="camera")
            np.testing.assert_array_equal(model.splits, refit.splits)
            np.testing.assert_array_equal(model.thresholds, refit.thresholds)
            np.testing.assert_array_equal(model.values, refit.values)
            changed = X[0].copy()
            changed[:8] = 10_000.0
            self.assertEqual(model.predict(X[0]), model.predict(changed))
            changed[:8] = np.nan
            self.assertEqual(model.predict(X[0]), model.predict(changed))

    def test_learns_from_labels_and_reloads(self):
        # Labels deliberately disagree with the rules for identical sensor gaps.
        # The only distinguishing feature is a visual observation.
        count = 150
        X = np.tile(features(IMAGE, EMPTY, EMPTY, 1), (count, 1))
        X[:, 9] = np.arange(count) % 2
        y = np.where(X[:, 9] > 0, 1.0, 5.0)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            np.savez(folder / "data.npz", X=X, y=y, episode=np.repeat(np.arange(5), 30))
            report = train(folder / "data.npz", folder / "model.npz", folder / "report.json")
            model = Model.load(folder / "model.npz")
            self.assertAlmostEqual(model.predict(X[0]), 5.0, delta=.08)
            self.assertAlmostEqual(model.predict(X[1]), 1.0, delta=.08)
            dev = report["development_holdout_before_final_refit"]
            self.assertLess(dev["learned"]["mae_mps"], .08)
            self.assertGreater(dev["rules"]["mae_mps"], 1.0)
            self.assertEqual(dev["frames"], 30)

    def test_loads_original_fusion_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez(path, version=1, feature_names=FEATURE_NAMES, bias=2.0,
                     splits=[[0, -1, -1]], thresholds=[[4, 0, 0]], values=[[0, -1, 1]])
            model = Model.load(path)
            x = features(IMAGE, EMPTY, EMPTY, 0)
            self.assertEqual(model.inputs, "fusion")
            self.assertEqual(model.predict(x), 3.0)
            x[0] = 1.0
            self.assertEqual(model.predict(x), 1.0)

    def test_uninformative_data_is_rejected(self):
        X = np.zeros((20, len(FEATURE_NAMES)), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "insufficient variation"):
            fit(X, np.full(20, CRUISE_SPEED))


if __name__ == "__main__":
    unittest.main()
