import tempfile
import unittest
from pathlib import Path

import numpy as np

from crosswalk.policy import FEATURE_NAMES, NAV_NAMES, Model, features
from crosswalk.train import fit, train
from crosswalk.vision import VISION_NAMES


IMAGE = np.zeros((40, 60, 3), dtype=np.uint8)
EMPTY = np.empty((0, 4), dtype=np.float32)
NAV = (3, 0, 7, 0, 13, 0)


class PerceptionTest(unittest.TestCase):
    def test_rgb_range_navigation_and_detection_encoding(self):
        lidar = np.array([[3, 0, -2.28, 1], [4, -2.0, -1.1, .9], [10, .5, -1.5, .7]])
        radar = np.array([[3, 0, -0.15, 6], [1.5, 0, 0, 6]])
        item = {"label": "person", "score": 0.8, "box": [0.45, 0.1, 0.55, 0.6]}
        x = features(IMAGE, lidar, radar, 4, [item], NAV)
        self.assertEqual(len(x), len(FEATURE_NAMES))
        np.testing.assert_allclose(x[[0, 1, 2, 3, 5, 8]], [5.75, 7.75, 5.75, 1.75, 1.5, 4])
        np.testing.assert_allclose(x[[FEATURE_NAMES.index(k) for k in VISION_NAMES]], [.8, .5, 0, 0, 0, 0])
        np.testing.assert_equal(x[-6:], NAV)
        b = features(np.full_like(IMAGE, 255), EMPTY, EMPTY, 0, navigation=NAV)
        self.assertEqual(b[0], 60)
        self.assertGreater(np.linalg.norm(x[9:33] - b[9:33]), 1)
        with self.assertRaisesRegex(ValueError, "route points"):
            features(IMAGE, EMPTY, EMPTY, 0, navigation=[3, 0])


class LearningTest(unittest.TestCase):
    def test_actions_learn_from_demonstrations_and_roundtrip(self):
        rng = np.random.default_rng(13)
        n = 480
        X = np.tile(features(IMAGE, EMPTY, EMPTY, 2, navigation=NAV), (n, 1))
        offset = rng.uniform(-1, 1, n)
        X[:, -5::2] = offset[:, None]
        visual = FEATURE_NAMES.index(VISION_NAMES[0])
        X[:, visual] = np.arange(n) % 2
        y = np.column_stack((.55 * offset, .7 * (1 - X[:, visual]), .8 * X[:, visual]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savez(path / "records.npz", X=X, y=y, seed=np.repeat(np.arange(6), n // 6))
            report = train(path / "records.npz", path / "policy.npz", inputs="camera")
            model = Model.load(path / "policy.npz")
            self.assertEqual(model.inputs, "camera")
            held = report["development_holdout_before_final_refit"]
            self.assertEqual(held["frames"], n // 6)
            for metric in held["actuators"].values():
                self.assertLess(metric["mae"], .04)
            np.testing.assert_allclose(model.predict_batch(X[:12]), y[:12], atol=.05)
            changed = X[:12].copy()
            changed[:, :8] = np.nan
            np.testing.assert_equal(model.predict_batch(changed), model.predict_batch(X[:12]))
            np.testing.assert_allclose(model.predict(changed[1]), y[1], atol=.05)
            self.assertTrue(set(NAV_NAMES).issubset(report["head_features"]["steer"]))
            self.assertTrue(set(VISION_NAMES).issubset(report["head_features"]["drive"]))

    def test_range_signal_and_steering_labels_control_independent_heads(self):
        rng = np.random.default_rng(12)
        X = np.tile(features(IMAGE, EMPTY, EMPTY, 2, navigation=NAV), (256, 1))
        X[:, 0] = rng.choice([3.0, 50.0], len(X))
        X[:, -5] = rng.choice([-1., 1.], len(X))
        brake = X[:, 0] == 3
        y = np.column_stack((-.6 * X[:, -5], .7 * ~brake, .8 * brake))
        model = fit(X, y)
        np.testing.assert_allclose(model.predict_batch(X), y, atol=.05)
        self.assertEqual(model.inputs, "fusion")
        with self.assertRaisesRegex(ValueError, "N x 3"):
            fit(X, y[:, 1])


if __name__ == "__main__":
    unittest.main()
