"""Regression checks using unchanged native images and synchronized laser packets."""

from io import BytesIO
from pathlib import Path
import unittest
from zipfile import ZipFile

import cv2
import numpy as np

from crosswalk.traffic_cones import detect_cones, detect_views

ARCHIVE = Path(__file__).resolve().parents[1] / "data/barrier-recognition/town03-native.zip"


def sample(name):
    with ZipFile(ARCHIVE) as archive:
        image = cv2.imdecode(np.frombuffer(archive.read(name + ".jpg"), np.uint8), cv2.IMREAD_COLOR)
        with np.load(BytesIO(archive.read(name + ".npz"))) as data:
            xyz = data["lidar"][:, :3].copy()
    xyz[:, 2] += 2.30
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), xyz


class TrafficConeTests(unittest.TestCase):
    def test_actual_three_band_body_requires_measured_range(self):
        rgb, xyz = sample("holdout_2_cone_7")
        [measured] = detect_views({"front": rgb}, xyz)
        assert measured["sources"] == ["rgb", "lidar"]
        assert measured["label"] == "traffic_cone"
        assert measured["range_source"] == "projected_xyz"
        assert measured["lidar_xyz_hits"] == measured["lidar_hits"] == len(measured["surface_xyz"])
        assert np.hypot(measured["x"] - 7, measured["y"] - 2.2) < 0.15
        assert 0.9 < measured["height"] < 1.7 and 0.4 < measured["width"] < 0.95
        assert detect_cones(rgb, "front", []) == []
        assert detect_cones(np.zeros_like(rgb), "front", xyz) == []
        assert detect_cones(rgb, "front", xyz + np.array([0, 0, 20.0])) == []

    def test_heightless_range_requires_distinct_ground_compatible_returns(self):
        rgb, xyz = sample("fit_0_cone_5")
        [ranged] = detect_cones(rgb, "front", xyz)
        hits = ranged["surface_xyz"]
        unique = np.unique(np.round(np.asarray(hits)[:, :2], 1), axis=0)
        assert detect_cones(rgb, "front", [], [unique[0]] * 4) == []
        [measured] = detect_cones(rgb, "front", [], unique)
        assert measured["range_source"] == "ground_gated_xy" and measured["lidar_xyz_hits"] == 0
        assert measured["lidar_hits"] >= 2 and measured["surface_xyz"] == []
        assert detect_cones(rgb, "front", [], unique * 2) == []

    def test_actual_distinct_roadside_objects_are_rejected(self):
        for name in ("holdout_3_barricade_10", "holdout_2_parked_car_5", "holdout_3_planter_13"):
            with self.subTest(name=name):
                rgb, xyz = sample(name)
                self.assertEqual(detect_cones(rgb, "front", xyz), [])

    def test_camera_registration_dimensions_are_checked(self):
        rgb, xyz = sample("fit_0_cone_5")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            detect_cones(rgb, "left", xyz)
