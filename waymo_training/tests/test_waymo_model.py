"""Synthetic fixtures only. Local smoke tests do not start a Waymo training run."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

if importlib.util.find_spec("tensorflow"):
    from waymo_training.evaluate import different_group_images, evaluate, scores
    from waymo_training.model import (acceleration_basis, configure_device, constant_velocity,
                                      fit_scales, load_data, validate_data)
    from waymo_training.train import fit_model


def fixture(directory):
    rng = np.random.default_rng(25)
    count = 30
    past = rng.normal(size=(count, 16, 6)).astype(np.float32)
    t = np.arange(1, 21, dtype=np.float32) / 4
    future = past[:, -1:, 2:4] * t[:, None] + (0.05 * t ** 2)[:, None]
    arrays = {"past": past, "future": future, "intent": np.arange(count) % 4,
              "group": np.array([f"segment-{i // 2:02d}" for i in range(count)]),
              "frame_index": np.arange(count) % 2, "timestamps": np.zeros(count, dtype=np.int64),
              "images": rng.integers(0, 256, (count, 3, 64, 96, 3), dtype=np.uint8),
              "split": np.array(["train"] * 18 + ["dev"] * 6 + ["test"] * 6)}
    for prefix, selected in (("train", arrays["split"] == "train"), ("val", arrays["split"] != "train")):
        np.savez(directory / f"{prefix}-fixture.npz", **{k: v[selected] for k, v in arrays.items()})


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "Waymo training requires optional TensorFlow")
class WaymoModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_device("cpu")

    def test_splits_units_normalization_and_swapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(Path(tmp))
            data = load_data(tmp)
        train = np.flatnonzero(data["split"] == "train")
        scales = fit_scales(data, train)
        np.testing.assert_allclose(scales["history_mean"], data["past"][train].mean((0, 1)), atol=1e-6)
        np.testing.assert_allclose(constant_velocity(data["past"])[:, -1], 5 * data["past"][:, -1, 2:4])
        np.testing.assert_allclose(acceleration_basis().sum(1), 0.5 * (np.arange(1, 21) / 4) ** 2)
        self.assertEqual(scores(np.arange(20)[None, :])["displacement_3s_m"], 11)
        rows = np.flatnonzero(data["split"] == "test")
        swapped = different_group_images(rows, data["group"])
        self.assertEqual(set(rows), set(swapped))
        self.assertTrue(np.all(data["group"][rows] != data["group"][swapped]))
        with self.assertRaisesRegex(ValueError, "only on train"):
            fit_scales(data, rows)
        data["split"][0] = "dev"
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            validate_data(data)
        data["split"][0] = "train"
        data["frame_index"][1] = 0
        with self.assertRaisesRegex(ValueError, "Repeated"):
            validate_data(data)

    def test_fit_save_restore_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            fixture(run)
            data = load_data(run)
            config = {"scales": fit_scales(data, np.flatnonzero(data["split"] == "train")),
                      "max_epochs": 2, "patience": 2, "learning_rate": 0.001,
                      "batch_size": 6, "seeds": [41], "dataset_source_sha256": "mock"}
            curve = {}
            for mode in ("history", "vision"):
                fit_model(data, config, mode, 41, run, curve)
                self.assertTrue((run / "resume" / f"{mode}-41" / "checkpoint").is_file())
                self.assertEqual(len(curve[f"{mode}-41"]["epochs"]), 2)
                # Resume restores the completed epoch and skips completed models.
                fit_model(data, config, mode, 41, run, curve)
                self.assertEqual(len(curve[f"{mode}-41"]["epochs"]), 2)
            report = evaluate(data, run, config)
            test = report["splits"]["test"]
            self.assertEqual(test["segments"], 3)
            self.assertGreater(test["models"]["vision-41"]["unrelated_camera_triplets"]["mean_waypoint_change_m"], 0.0)
            self.assertEqual(len(report["weights"]["vision-41"]["sha256"]), 64)
            self.assertEqual(json.loads((run / "evaluation.json").read_text())["splits"]["dev"]["windows"], 6)


if __name__ == "__main__":
    unittest.main()
