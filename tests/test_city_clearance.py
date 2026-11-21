"""Offline checks for geometry-based roadside pace, including temporal negative controls."""
import unittest
from unittest.mock import patch

import numpy as np

from crosswalk.city_clearance import EdgeObserver, _edge_profile
from crosswalk.city_drive import load_driver
from crosswalk.city_policy import INDEX
from crosswalk.experiment import PROJECT


PATH = np.array([[x, 0.] for x in [0., 3., 7., 13., 18., 25., 32., 40.]])
ROUTE = PATH[1:4].ravel()


def cloud(width=.4, tall=False, blocked=False):
    points = []
    for x in np.arange(1., 22., .3):
        for y in np.arange(-7., 7., .55):
            points.append((x, y, -2.3, .8))
    for center in [5., 10., 15.]:
        for dx in np.linspace(-width/2, width/2, 10):
            for h in [.4, .72, 1.2 if not tall else 1.9]:
                points.append((center+dx, .8 if blocked else 1.7, h-2.3, .8))
    return np.asarray(points, dtype=np.float32)


class ClearanceTests(unittest.TestCase):
    def test_profile_requires_visible_pavement_and_narrow_external_surfaces(self):
        self.assertEqual(len(_edge_profile(cloud(), PATH)), 3)
        for scene in (cloud(width=2.), cloud(tall=True), cloud(blocked=True)):
            self.assertEqual(_edge_profile(scene, PATH), [])
        no_background = cloud()
        self.assertEqual(_edge_profile(no_background[no_background[:, 1]<3.], PATH), [])

    def test_temporal_confidence_and_person_veto(self):
        scan = dict(lidar=cloud(), radar=np.empty((0, 4), np.float32))
        counter = 0
        def read(dy=0., detections=()):
            nonlocal counter
            counter += 1
            cells = [(s, 1.6+dy*counter, .4, 14.) for s in (3., 7., 11.)]
            with patch("crosswalk.city_clearance._edge_profile", return_value=cells):
                return observer.observe(scan, detections, ROUTE, 0.)[-2]
        observer = EdgeObserver()
        self.assertLess(read(), 0)
        for _ in range(8):
            final = read()
        self.assertGreater(final, 0)
        self.assertLess(read(detections=[dict(label="person", box=(.5,.3,.55,.5), score=.7)]), 0)
        observer = EdgeObserver()
        for _ in range(12):
            self.assertLess(read(dy=.07), 0)

    def test_blocked_corridor_dominates_confirmed_edge(self):
        base, _, _ = load_driver(PROJECT/"models/city-pace/candidate.npz")
        new, _, _ = load_driver(PROJECT/"models/city-clearance/candidate.npz")
        with np.load(PROJECT/"results/city-pace/seed-211/observations.npz") as archive:
            row = archive["features"][1000].copy()
        for name in ("path_body_lidar", "path_body_lidar_recent", "path_body_radar"):
            row[INDEX[name]] = 1.
        row[INDEX["ego_speed_mps"]] = 5.
        expected = base.predict(row)
        result = new.predict(np.insert(row, len(row)-2, [.5, 14.]))
        np.testing.assert_allclose(result, expected)
        self.assertGreater(result[2], .9)


if __name__ == "__main__":
    unittest.main()
