import unittest

import carla
import numpy as np

from crosswalk.overtake_transfer import EGO_STATION, SOURCE, second_corridor
from tests.test_urban_map import MAP


@unittest.skipUnless(MAP.is_file(), 'Town03 OpenDRIVE unavailable')
class TransferCorridorTests(unittest.TestCase):
    def test_real_alternative_has_clearance_and_road_for_the_complete_fixture(self):
        native = carla.Map('Town03_Opt', MAP.read_text(encoding='utf8'))
        corridor, geometry = second_corridor(native)
        self.assertEqual((geometry['road_id'], geometry['source_lane'], geometry['target_lane']), (67, -2, -1))
        self.assertLess(geometry['elevation_change_m'], -2.)
        self.assertLess(geometry['maximum_lateral_residual_m'], .01)
        self.assertAlmostEqual(geometry['minimum_lane_width_m'], 3.5)
        for ahead in (-24., 0., 2., 24., 85.):
            center = native.get_waypoint_xodr(*SOURCE, EGO_STATION + ahead).transform.location
            arc, lateral = corridor.coordinates(np.array([center.x, center.y]))
            self.assertGreaterEqual(arc, 0.)
            self.assertLessEqual(arc, corridor.arc[-1])
            self.assertAlmostEqual(arc, EGO_STATION + ahead - geometry['station_interval_m'][0], places=3)
            self.assertAlmostEqual(lateral, 0., places=3)
