import hashlib
import unittest

import cv2
import numpy as np

from crosswalk.vision import (
    ANCHORS, MODEL_PATH, VISION_NAMES, Detector, _decode, _letterbox, summarize,
)


class VisionTests(unittest.TestCase):
    def test_pinned_model_runs_on_black_rgb(self):
        expected = "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186"
        self.assertEqual(hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(), expected)
        threads = cv2.getNumThreads()
        self.addCleanup(cv2.setNumThreads, threads)
        cv2.setNumThreads(1)
        detector = Detector()
        for shape in [(90, 160), (160, 90)]:
            self.assertEqual(detector.detect(np.zeros((*shape, 3), np.uint8)), [])

    def test_letterbox_preserves_rgb_and_aspect(self):
        rgb = np.zeros((100, 200, 3), np.uint8)
        rgb[:, :, 0] = 255
        blob, content = _letterbox(rgb)
        self.assertEqual(content, (0, 104, 416, 208))
        np.testing.assert_allclose(
            blob[0, :, 104, 0], [(255 - 103.53) / 57.375, -116.28 / 57.12, -123.675 / 58.395],
            atol=1e-6,
        )
        self.assertLess(blob[0, 0, 103, 0], 0)
        self.assertEqual(_letterbox(rgb.transpose(1, 0, 2))[1], (104, 0, 208, 416))

    def test_decode_clips_padding_and_suppresses_only_within_class(self):
        outputs = []
        for level in ANCHORS:
            scores = np.zeros((1, len(level), 80), np.float32)
            logits = np.full((1, len(level), 4, 8), -100.0, np.float32)
            logits[:, :, :, 3] = 0  # 3 bins x stride 8 = 24 px each side.
            outputs.extend((scores, logits.reshape(1, len(level), 32)))
        scores = outputs[0][0]
        target = 13 * 52 + 25  # center (203.5, 107.5)
        scores[target, 0] = 0.90
        scores[target - 1, 0] = 0.81  # overlapping person
        scores[target + 1, 2] = 0.82  # overlapping car
        scores[0, 0] = 0.99  # inside black top padding
        scores[target + 8, [0, 11]] = [0.80, 0.95]  # stop sign wins
        results = _decode(outputs, (0, 104, 416, 208), 0.35)
        self.assertEqual([r["label"] for r in results], ["person", "car"])
        np.testing.assert_allclose(results[0]["box"], [179.5 / 416, 0, 227.5 / 416, 27.5 / 208])
        self.assertAlmostEqual(results[0]["score"], 0.9)

    def test_summary_chooses_central_tallest_per_group(self):
        items = [
            {"label": "person", "score": 0.9, "box": [0, 0, 0.1, 1]},
            {"label": "bus", "score": 0.9, "box": [0.3, 0.3, 0.5, 0.5]},
            {"label": "car", "score": 0.7, "box": [0.4, 0.0, 0.8, 0.5]},
        ]
        self.assertEqual(len(VISION_NAMES), 6)
        np.testing.assert_allclose(summarize(items), [0, 0, 0, 0.7, 0.5, 0.2])
        self.assertEqual(summarize(items[::-1]), summarize(items))
