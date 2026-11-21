"""Portable inference contract and independent Keras comparison on mock data."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from waymo_training.portable import FORMAT, TrajectoryPolicy, export, pool_rgb


class PortableTests(unittest.TestCase):
    def test_inference_module_without_tensorflow(self):
        result = subprocess.run([sys.executable, "-c",
                                 "import sys; from waymo_training.portable import TrajectoryPolicy; "
                                 "assert 'tensorflow' not in sys.modules and 'keras' not in sys.modules"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_resize_rounding_and_color(self):
        import cv2
        rng = np.random.default_rng(74)
        imgs = [rng.integers(0, 256, (75 + i, 102 + i, 3), dtype=np.uint8) for i in range(3)]
        resized = np.stack([cv2.resize(x, (96, 64), interpolation=cv2.INTER_AREA) for x in imgs])
        expected = resized.reshape(3, 32, 2, 48, 2, 3).mean((2, 4)).round().astype(np.uint8)
        np.testing.assert_array_equal(pool_rgb(imgs), expected)
        with self.assertRaisesRegex(ValueError, "uint8 RGB"):
            pool_rgb([x.astype(np.float32) for x in imgs])

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "optional export dependency")
    def test_nontrivial_export_and_metre_parity(self):
        from waymo_training.model import file_hash, fit_scales, load_data, make_model
        rng = np.random.default_rng(1231)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features"
            features.mkdir()
            for source, split in (("train", ["train"] * 7), ("val", ["dev"] * 2 + ["test"] * 5)):
                n = len(split)
                np.savez(features / f"{source}-fixture.npz",
                         images=rng.integers(0, 256, (n, 3, 64, 96, 3), dtype=np.uint8),
                         past=rng.normal(0, 2.5, (n, 16, 6)).astype(np.float32),
                         future=rng.normal(0, 7.0, (n, 20, 2)).astype(np.float32),
                         intent=np.arange(n) % 4, split=split, group=[f"{source}-{i}" for i in range(n)],
                         timestamps=np.zeros(n, dtype=np.int64), frame_index=np.zeros(n, dtype=np.int32))
            data = load_data(features)
            config = {"seeds": [41, 73], "sources": data["sources"], "dataset_source_sha256": "fixture",
                      "scales": fit_scales(data, np.flatnonzero(data["split"] == "train"))}
            (root / "config.json").write_text(json.dumps(config))
            curves = {f"vision-{seed}": {"best_epoch": 1, "epochs": [{"dev_ADE_m": ade}]}
                      for seed, ade in ((41, 2.1), (73, 1.9))}
            (root / "training_curve.json").write_text(json.dumps(curves))
            model = make_model("vision", 73)
            for name in ("history_acceleration", "fused_acceleration"):
                layer = model.get_layer(name)
                layer.set_weights([rng.normal(0, 0.15, w.shape).astype(np.float32) for w in layer.get_weights()])
            ckpt = root / "checkpoint-vision-73.weights.h5"
            model.save_weights(ckpt)
            (root / "evaluation.json").write_text(json.dumps({"weights": {"vision-73": {"sha256": file_hash(ckpt)}}}))
            target = root / "private-policy.npz"
            result = export(root, features, target, sample_size=5)
            self.assertEqual(result["seed"], 73)
            self.assertEqual(result["format"], FORMAT)
            self.assertEqual(result["parity"]["rows"], 5)
            self.assertLess(result["parity"]["max_abs_coordinate_delta_m"], 0.0001)
            self.assertEqual(result["bytes"], target.stat().st_size)
            policy = TrajectoryPolicy.load(target)
            with np.load(features / "val-fixture.npz") as z:
                rgb, history = z["images"][0], z["past"][0]
            route = policy.predict(rgb, history, 1)
            different = policy.predict(rgb[[2, 1, 0]], history, 1)
            self.assertEqual(route.shape, (20, 2))
            self.assertEqual(route.dtype, np.float32)
            self.assertGreater(np.abs(different - route).max(), 0.0001)
            self.assertGreater(np.abs(policy.predict(rgb, history, 2) - route).max(), 0.0001)
            with self.assertRaises(ValueError):
                policy.predict(rgb, history[:15], 1)
            with self.assertRaises(ValueError):
                policy.predict(rgb, history, 4)
            with self.assertRaisesRegex(ValueError, "missing"):
                export(root, features, target, seed=1)
            print("Synthetic export:", result["bytes"], "bytes;", result["parity"])


if __name__ == "__main__":
    unittest.main()
