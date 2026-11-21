"""Counterfactual tests for the production tactical planner and true native map."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import carla
import numpy as np

from crosswalk.overtake import GapChoice, PassingCorridor, TacticalPass
from tests.test_urban_map import MAP


class TacticalChoiceTests(unittest.TestCase):
    def test_fitted_traffic_counterfactuals(self):
        policy = GapChoice()
        for row, expected in (((60,0,60,60),'keep'), ((32,12,60,60),'keep'),
                              ((32,1,60,60),'pass'), ((32,1,4,60),'yield'),
                              ((32,1,60,8),'yield')):
            self.assertEqual(policy.predict(row)[0], expected)


@unittest.skipUnless(MAP.is_file(),'Town03 OpenDRIVE unavailable')
class TacticalPassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map=carla.Map('Town03_Opt',MAP.read_text(encoding='utf8'))
        cls.corridor=PassingCorridor.from_map(cls.map)
        cls.ego=Mock()
        cls.ego.get_transform.return_value=cls.map.get_waypoint_xodr(4,3,78.).transform
        cls.defaults=np.array((3.,0.,7.,0.,13.,0.,18.,0.,25.,0.,32.,0.,40.,0.))

    def test_geometry_only_configuration_ignores_actor_data(self):
        scene=SimpleNamespace(_map=self.map,_urban=SimpleNamespace(lane_change=SimpleNamespace(source_lane=(4,3))))
        self.assertIsNotNone(TacticalPass.from_scene_map(scene))
        self.assertTrue(scene.tactical_lane_scoring)

    def test_with_sensor_obstacle_waits_and_commits_after_adjacent_traffic_departs(self):
        rng=np.random.default_rng(8)
        def surface(x,y):
            return rng.normal((x,y,-1.1,.5),(.045,.27,.15,.015),size=(80,4))
        pavement=np.array([(x,-3.5,-2.30,.4) for x in np.r_[np.arange(-23.,-4.,.4),np.arange(5.,24.,.4)]])
        lead=np.r_[surface(27.,0.),pavement]
        neighbor=surface(-4.,-3.5)
        planner=TacticalPass(self.corridor)
        for _ in range(12):
            kept=planner.preview(self.ego,np.r_[lead,neighbor],self.defaults,4.)
        self.assertEqual(planner.info['maneuver'],'yield')
        np.testing.assert_allclose(kept[1::2],0.,atol=.01)
        self.assertGreater(planner.apply((0.,.3,0.),4.)[2],.2)
        for _ in range(10):
            passed=planner.preview(self.ego,lead,self.defaults,4.)
        self.assertEqual(planner.info['maneuver'],'passing')
        self.assertLess(passed[-1],-3.49)
        self.assertEqual(planner.apply((0.,.3,0.),4.),(0.,.3,0.))

    def test_unobserved_target_lane_cannot_authorize_a_pass(self):
        planner=TacticalPass(self.corridor)
        source_only=np.tile((26.,0.,-1.1,.5),(20,1))
        for _ in range(12):
            planner.preview(self.ego,source_only,self.defaults,4.)
        self.assertEqual(planner.info['maneuver'],'yield')
        self.assertEqual(planner.info['adjacent_pavement_seen'],[False,False])

    def test_empty_street_and_same_speed_traffic_stay_in_lane(self):
        planner=TacticalPass(self.corridor)
        for _ in range(10):
            kept=planner.preview(self.ego,np.empty((0,4)),self.defaults,4.)
        self.assertFalse(planner.info['committed'])
        np.testing.assert_allclose(kept[1::2],0.,atol=.01)
        for step in range(14):
            xyz=np.tile((16.+step*.9,0.,-1.,.6),(20,1))
            planner.preview(self.ego,xyz,self.defaults,4.)
        self.assertEqual(planner.info['choice'],'keep')
        self.assertGreater(planner.info['source_speed_mps'],8.5)


if __name__=='__main__':
    unittest.main()
