"""Physics, CARLA label mapping and display behavior of the thermal illustration."""
import unittest

import carla
import numpy as np

from crosswalk.thermal import _band_radiance, thermal_rgb


class ThermalTests(unittest.TestCase):
    def test_band_power_at_room_temperature(self):
        # A 300 K blackbody emits about 55 W/(m² sr) across this band.
        self.assertAlmostEqual(float(_band_radiance(26.85)), 54.93, delta=0.08)
        self.assertTrue(np.all(np.diff(_band_radiance([0, 10, 20, 30])) > 0))

    def test_white_hot_order_for_carla_classes_and_valid_rgb(self):
        labels = carla.CityObjectLabel
        tags = np.array([[int(tag) for tag in (labels.Sky, labels.Vegetation, labels.Roads,
                                               labels.Car, labels.Pedestrians)]], np.uint8)
        image = thermal_rgb(tags, np.full(tags.shape, 4.0, np.float32))
        self.assertEqual(image.shape, (1, 5, 3))
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image[..., 0], image[..., 1])
        np.testing.assert_array_equal(image[..., 0], image[..., 2])
        self.assertTrue(np.all(np.diff(image[0, :, 0].astype(int)) > 0))

    def test_path_radiance_reduces_contrast_toward_air(self):
        labels = carla.CityObjectLabel
        tags = np.broadcast_to(np.array([[int(labels.Pedestrians)], [int(labels.Vegetation)],
                                         [int(labels.Sky)]], np.uint8), (3, 4))
        distance = np.broadcast_to(np.array([0, 1000, 10000, np.inf], np.float32), tags.shape)
        result = thermal_rgb(tags, distance)[:, :, 0].astype(int)
        self.assertGreater(result[0, 0], result[0, 1])
        self.assertGreater(result[0, 1], result[0, 2])
        self.assertLess(result[1, 0], result[1, 1])
        self.assertLessEqual(abs(result[0, 2] - result[1, 2]), 2)
        self.assertLessEqual(abs(result[0, 2] - result[0, 3]), 2)
        self.assertLessEqual(np.ptp(result[2]), 2)

    def test_seeds_reproduce_display_dither(self):
        tags = np.full((16, 12), int(carla.CityObjectLabel.Car), np.uint8)
        depth = np.full(tags.shape, 30, np.float32)
        first = thermal_rgb(tags, depth, seed=22)
        np.testing.assert_array_equal(first, thermal_rgb(tags, depth, seed=22))
        second = thermal_rgb(tags, depth, seed=23)
        self.assertFalse(np.array_equal(first, second))
        self.assertLessEqual(np.max(np.abs(first.astype(int) - second)), 2)

    def test_missing_depth_returns_air_and_unknown_labels_fall_back(self):
        depth = np.array([[np.inf, np.nan, -1.0, -np.inf]], np.float32)
        far = thermal_rgb(np.array([[12, 9, 28, 254]], np.uint8), depth)
        self.assertLessEqual(np.ptp(far), 2)
        tags = np.array([[0, 255, 254]], np.uint8)
        self.assertLessEqual(np.ptp(thermal_rgb(tags, np.zeros(tags.shape, np.float32))), 2)

    def test_requires_aligned_two_dimensional_tags_and_depth(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            thermal_rgb(np.zeros((2, 3), np.uint8), np.ones((3, 2)))
        with self.assertRaisesRegex(ValueError, "aligned"):
            thermal_rgb(np.zeros((2, 3, 3), np.uint8), np.ones((2, 3, 3)))
        with self.assertRaisesRegex(ValueError, "semantic IDs"):
            thermal_rgb(np.ones((3, 2), np.float32), np.ones((3, 2)))
