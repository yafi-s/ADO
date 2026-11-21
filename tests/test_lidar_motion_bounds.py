"""Exercise LiDAR grid cells at both strict square boundaries."""

import unittest
from unittest.mock import patch

import numpy as np

from crosswalk.lidar_motion import CELL, RADIUS, Sweep, cv2


class LidarMotionBoundsTests(unittest.TestCase):
    def test_float32_interior_returns_land_in_the_last_cell_on_either_axis(self):
        radius = np.float32(RADIUS)
        inner = np.nextafter(radius, np.float32(0))
        outer = np.nextafter(radius, np.float32(np.inf))
        cell_count = int(2 * RADIUS / CELL)
        center_cell = cell_count // 2

        for axis in (0, 1):
            with self.subTest(axis=axis):
                # Points on either strict boundary and just outside must be
                # excluded before indexing, even when the upper bin is clamped.
                raw = np.zeros((11, 4), dtype=np.float32)
                raw[:, axis] = [inner, inner, inner, inner, inner, inner,
                                -inner, radius, outer, -radius, -outer]
                raw[:, 1 - axis] = [0, 0.125, 0.25, 0.375, 0.5, 0.625,
                                    1, 2, 3, 4, 5]
                raw[:6, 2] = np.linspace(-1, 0, 6)

                with patch.object(cv2, "dilate", wraps=cv2.dilate) as dilate:
                    sweep = Sweep(raw, np.zeros(2), np.eye(2), 0)

                expected = {(cell_count - 1, center_cell + offset) for offset in range(3)}
                expected.add((0, center_cell + 4))
                if axis == 1:
                    expected = {(y, x) for x, y in expected}
                occupied = np.argwhere(dilate.call_args.args[0])
                self.assertEqual(set(map(tuple, occupied)), expected)
                self.assertEqual(len(sweep.clusters), 1)
                np.testing.assert_array_equal(np.sort(sweep.clusters[0].ids), np.arange(6))

    def test_interior_cells_keep_their_original_bins(self):
        raw = np.array([
            [35.75, 0.25, 0, 0], [0.125, 35.875, 0, 0],
            [-35.75, -0.25, 0, 0], [-0.125, -35.875, 0, 0],
            [3.25, -4.25, 0, 0], [-3.25, 4.25, 0, 0],
        ], dtype=np.float32)
        expected = np.floor((raw[:, :2].astype(float) + RADIUS) / CELL).astype(int)

        with patch.object(cv2, "dilate", wraps=cv2.dilate) as dilate:
            Sweep(raw, np.zeros(2), np.eye(2), 0)

        occupied = np.argwhere(dilate.call_args.args[0])
        self.assertEqual(set(map(tuple, occupied)), set(map(tuple, expected)))
