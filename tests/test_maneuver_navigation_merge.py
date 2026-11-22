"""Station-limited visibility and navigation handoff on the native Market route."""
import unittest
from unittest.mock import Mock

import carla
import numpy as np

from crosswalk.maneuver_map import discover, merge_approach_ceiling
from crosswalk.route_maneuvers import RouteManeuvers
from crosswalk.urban_map import urban_route
from tests.test_urban_map import MAP
from tests import test_route_maneuvers as fixtures


@unittest.skipUnless(MAP.exists(), "Town03 OpenDRIVE unavailable")
class NavigationMergeTests(unittest.TestCase):
    at = fixtures.RouteManeuverTests.at
    observe = fixtures.RouteManeuverTests.observe

    @classmethod
    def setUpClass(cls):
        road_map = carla.Map("Town03_Opt", MAP.read_text())
        course = urban_route(road_map, 212)
        cls.corridors = discover(road_map, [p.transform.location for p in course.points])
        cls.c = next(c for c in cls.corridors if c.merge_at is not None)

    def setUp(self):
        self.driver = RouteManeuvers(self.corridors)
        self.ego = Mock()
        self.ego.bounding_box = carla.BoundingBox(carla.Location(z=.76), carla.Vector3D(2.45,.92,.76))

    def test_archived_entrance_geometry_permits_only_source_approach(self):
        # full-rehearsal02, seed 212, terminal trace at route station 821.935 m.
        station = 22.046
        features = (42.471, .026, 60., 60., 0., 0., 0.)
        sight = [27.25, 22.05, 41.01, 22.05]
        self.assertEqual(self.c.source_key, (4,3))
        self.assertEqual(self.c.target_key, (4,2))
        self.assertEqual(self.c.arc[0], 0.)
        self.assertGreater(self.c.navigation_target_from, self.c.merge_at+8.)
        self.assertGreater(self.c.end-self.c.navigation_target_from, 30.)
        ceiling = merge_approach_ceiling(self.c,station,0.,features,[True]*4,sight,True)
        self.assertGreater(ceiling,0.)
        self.assertLessEqual(ceiling,2.)
        for _ in range(9):
            info,goals = self.observe(station,lead=station+42.471)
        self.assertFalse(info["target_gap_accepted"])
        self.assertFalse(info["committed"])
        self.assertEqual(info["turn_signal"],"off")
        self.assertGreater(self.driver.speed_limit,0.)
        self.assertLess(np.max(abs(goals[1::2])),.05)
        for dx in np.arange(.2,11.,.2):
            info,_ = self.observe(station+dx,lead=station+42.471)
            if info["committed"]:
                break
        self.assertTrue(info["target_gap_accepted"])
        self.assertGreater(info["pavement_reach_m"][3],24.)
        self.assertTrue(info["committed"])

    def test_obstruction_or_missing_sight_keeps_source_stopped(self):
        for changes in ({"rear":19.}, {"pavement":False}, {"lead":32.}):
            self.driver = RouteManeuvers(self.corridors)
            for _ in range(9):
                info,_ = self.observe(22.,**({"lead":64.5}|changes))
            self.assertFalse(info["committed"])
            self.assertEqual(self.driver.speed_limit,0.)
        ceiling = merge_approach_ceiling(self.c,34.,0.,(38.,0.,60.,60.,0.,0.,1.),
                                         [True]*4,[40.,20.,40.,20.],True)
        self.assertEqual(ceiling,0.)

    def test_handoff_requires_target_body_heading_and_route(self):
        def settled(s, alpha):
            pose = self.at(s,alpha)
            forward = pose.get_forward_vector()
            heading = np.array((forward.x,forward.y))
            xy = np.array((pose.location.x,pose.location.y))
            station,lateral = self.c.locate(xy,heading)
            return self.c.navigation_target_settled(station,lateral,heading,
                                                    self.ego.bounding_box.get_world_vertices(pose))
        self.assertFalse(settled(self.c.merge_at+2.,1.))
        station = self.c.navigation_target_from+5.
        self.assertFalse(settled(station,.82))
        self.assertTrue(settled(station,1.))
        self.ego.bounding_box = carla.BoundingBox(carla.Location(y=.8,z=.76), carla.Vector3D(2.45,.92,.76))
        self.assertFalse(settled(station,1.))
        self.ego.bounding_box = carla.BoundingBox(carla.Location(z=.76), carla.Vector3D(2.45,.92,.76))
        pose = self.at(station,1.)
        yaw = np.deg2rad(pose.rotation.yaw+10.)
        self.assertFalse(self.c.navigation_target_settled(station,-3.5,np.array((np.cos(yaw),np.sin(yaw))),
                                                         self.ego.bounding_box.get_world_vertices(pose)))
        self.driver.corridor = self.c
        self.driver.start = station-24.
        for _ in range(3):
            info,goals = self.observe(station,1.,rear=station-4.)
        self.assertFalse(info["committed"])
        self.assertTrue(info["navigation_merge"])
        self.assertEqual(info["turn_signal"],"off")
        self.assertIsNone(self.driver.speed_limit)
        np.testing.assert_array_equal(goals,np.zeros(14))
        # The former source lane can now contain a nearby bus. The normal
        # planner takes the navigation route well before the junction.
        for offset in (2.,10.,20.):
            info,goals = self.observe(station+offset,1.,lead=station+offset+5.)
            self.assertFalse(info["committed"])
            np.testing.assert_array_equal(goals,np.zeros(14))

    def test_first_observation_already_on_navigation_target(self):
        info,goals = self.observe(self.c.navigation_target_from+6.,1.)
        self.assertFalse(info["committed"])
        self.assertEqual(info["maneuver_phase"],"route_merge_complete")
        np.testing.assert_array_equal(goals,np.zeros(14))


if __name__ == "__main__":
    unittest.main()
