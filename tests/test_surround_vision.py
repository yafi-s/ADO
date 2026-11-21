"""The paired observer must preserve physical camera identity and per-view box scales."""

import unittest

import numpy as np

from crosswalk.surround_vision import GRID, SIZE, STRIDE, SurroundDetector


class PairGeometry(unittest.TestCase):
    def test_small_front_crop_remaps_bounds_and_preserves_the_full_scale_reading(self):
        observer = SurroundDetector.__new__(SurroundDetector)
        observed = []

        def detect(rgb):
            observed.append(rgb.shape)
            return [dict(label="person", score=.8, box=[.2, .3, .25, .5]),
                    dict(label="car", score=.9, box=[.0, .3, .02, .5]),
                    dict(label="car", score=.9, box=[.2, .2, .8, .8])]

        observer.detect = detect
        zoom, = observer.horizon_boxes(np.zeros((360, 640, 3), dtype=np.uint8))
        self.assertEqual(observed, [(144, 320, 3)])
        self.assertEqual(zoom["inference_layout"], "front-horizon-2x")
        np.testing.assert_allclose(zoom["box"], [.35, (112+43.2)/360, .375, 184/360])
        full = dict(zoom, inference_layout="single")
        separate = dict(zoom, box=[.6, .4, .62, .44])
        self.assertEqual(observer.merge_horizon([full], [zoom, separate]), [full, separate])

    def test_splits_views_and_ignores_gutter(self):
        observer = SurroundDetector.__new__(SurroundDetector)
        observer.threshold = .35
        blue = np.full((180, 320, 3), [0, 0, 255], dtype=np.uint8)
        red = np.full_like(blue, [255, 0, 0])

        def forward(canvas):
            self.assertEqual(canvas[100, 100].tolist(), [255, 0, 0])
            self.assertEqual(canvas[500, 100].tolist(), [0, 0, 255])
            self.assertTrue((canvas[315:325] == 114).all())
            output = np.zeros((len(GRID), 85), np.float32)
            # Same relative location in the two views, plus a fictitious object in the gap.
            for index, (x, y, class_id) in enumerate(((280, 157.5, 0), (280, 482.5, 1),
                                                       (280, SIZE / 2, 2))):
                output[index, :2] = np.array([x, y]) / STRIDE[index] - GRID[index]
                output[index, 2:4] = np.log(np.array([56, 63]) / STRIDE[index])
                output[index, 4] = output[index, 5 + class_id] = .95
            return output

        observer._forward = forward
        top, bottom = observer.detect_pair(red, blue)
        self.assertEqual([b["label"] for b in top], ["person"])
        self.assertEqual([b["label"] for b in bottom], ["bicycle"])
        np.testing.assert_allclose(top[0]["box"], [.45, .4, .55, .6], atol=1e-6)
        np.testing.assert_allclose(bottom[0]["box"], top[0]["box"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
