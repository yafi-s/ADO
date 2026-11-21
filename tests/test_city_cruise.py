import io
import unittest
import zipfile
from pathlib import Path

import numpy as np

from crosswalk.city_cruise import UrbanCruiseModel
from crosswalk.city_policy import INDEX


ROOT = Path(__file__).resolve().parents[1]


class CityCruiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT/"models/city-cruise/candidate.npz"
        with zipfile.ZipFile(ROOT/"data/city-cruise/lessons.zip") as archive:
            member = next(name for name in archive.namelist() if name.endswith("/lesson.npz"))
            with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as data:
                cls.fast = data["X"][data["teacher_speed"] > 10.9][::8]

    def test_measured_open_road_and_posted_limit(self):
        model = UrbanCruiseModel.load(self.path)
        speed = [model._plan(x) for x in self.fast]
        self.assertGreater(np.median(speed)*3.6, 35.)
        model.posted_limit_kmh = 30.
        self.assertLessEqual(max(model._plan(x) for x in self.fast), 30./3.6)
        model.posted_limit_kmh = float("nan")
        self.assertLessEqual(max(model._plan(x) for x in self.fast), 30./3.6)

    def test_close_obstacle_cannot_receive_cruise_increment(self):
        model = UrbanCruiseModel.load(self.path)
        x = self.fast[0].copy()
        x[INDEX["path_body_lidar"]] = x[INDEX["path_body_radar"]] = .5
        x[INDEX["ego_speed_mps"]] = 9.
        _, throttle, brake = model.predict(x)
        self.assertLess(model.last_speed_request, .15)
        self.assertEqual(throttle, 0.)
        self.assertGreater(brake, .4)
