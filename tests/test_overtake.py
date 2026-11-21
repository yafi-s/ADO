"""Counterfactual tests for the production tactical planner and true native map."""
import gzip
import json
import unittest
from pathlib import Path
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

    def test_measured_wait_survives_scan_loss_and_releases_into_a_verified_gap(self):
        planner = TacticalPass(self.corridor, measured_hold=True)
        pavement = np.array([(x,-3.5,-2.30,.4) for x in np.r_[np.arange(-23.,-4.,.4),np.arange(5.,24.,.4)]])
        lead = np.tile((26.,0.,-1.1,.5), (20,1))
        neighbor = np.tile((-4.,-3.5,-1.1,.5), (20,1))
        def tick(*surfaces):
            planner.preview(self.ego, np.vstack((pavement,*surfaces)), self.defaults, 0.)
        for _ in range(8):
            tick(lead, neighbor)
        for _ in range(4):
            tick(neighbor)
            self.assertEqual(planner.speed_limit, 0.)
            self.assertEqual(planner.info['raw_source_gap_m'], 60.)
            self.assertAlmostEqual(planner.info['source_speed_mps'], 0.)
        tick(lead)
        tick(lead)
        self.assertEqual(planner.speed_limit, 0.)
        tick(lead, neighbor)  # late rear vehicle cancels the opening
        self.assertFalse(planner.info['committed'])
        for _ in range(3):
            tick(lead)
        self.assertTrue(planner.info['committed'])
        self.assertIsNone(planner.speed_limit)
        tick(lead, neighbor)  # still centered, so a commitment is reversible
        self.assertFalse(planner.info['committed'])
        self.assertEqual(planner.speed_limit, 0.)

        # If the lead turns away, require observable road at its old position.
        for _ in range(12):
            tick(neighbor)
        self.assertEqual(planner.speed_limit, 0.)
        clear_road = np.array([(x,0.,-2.30,.4) for x in np.arange(27.,40.,.25)])
        tick(neighbor, clear_road)
        self.assertTrue(planner.info['source_pavement_clear'])
        json.dumps(planner.info, allow_nan=False)
        self.assertIsNone(planner.speed_limit)

    def test_accelerating_lead_releases_wait_even_with_occupied_adjacent_lane(self):
        planner = TacticalPass(self.corridor, measured_hold=True)
        def tick(distance):
            lead = np.tile((distance,0.,-1.1,.5), (20,1))
            neighbor = np.tile((-4.,-3.5,-1.1,.5), (20,1))
            planner.preview(self.ego, np.vstack((lead, neighbor)), self.defaults, 0.)
        for _ in range(10):
            tick(21.)
        self.assertEqual(planner.speed_limit, 0.)
        for i in range(1, 16):
            tick(21.+1.1*i)
        self.assertEqual(planner.info['choice'], 'keep')
        self.assertIsNone(planner.speed_limit)
        self.assertFalse(planner.info['committed'])

    def test_recorded_gap_streams_hold_until_commit_and_preserve_clear_controls(self):
        root = Path(__file__).resolve().parents[1]
        archives = root / 'results/tactical-pass/paired-4fppmp4k'
        cases = [(archives / 'occupied.jsonl.gz', .3, 8.0),
                 (archives / 'clear.jsonl.gz', None, .5),
                 (archives / 'no-lead.jsonl.gz', None, None),
                 (root / 'web/media/overtake-yield/telemetry.json', .3, 8.1),
                 (root / 'web/media/overtake-clear/telemetry.json', None, .5)]
        for path, first_hold, first_commit in cases:
            with self.subTest(path=path.name, parent=path.parent.name):
                if path.suffix == '.gz':
                    with gzip.open(path, 'rt') as file:
                        rows = [json.loads(line) for line in file]
                else:
                    rows = json.loads(path.read_bytes())
                planner = TacticalPass(self.corridor, measured_hold=True)
                held = []
                committed = None
                # Positions, raw four lane measurements and pavement coverage
                # are available before control. Omit unrecorded source pavement.
                for row in rows:
                    sample = row.get('tactical', row.get('tactical_pass'))
                    if not sample:
                        continue
                    gap, speed = planner._source_motion(sample['corridor_station_m'],
                                                        sample['source_gap_m'], np.empty(0))
                    planner.coverage = sample['adjacent_pavement_seen']
                    planner.measure = Mock(return_value=(gap, speed, sample['target_front_m'], sample['target_rear_m']))
                    planner.preview(self.ego, np.empty((0,4)), self.defaults, row['speed_mps'])
                    t = row.get('time_s', row.get('sim_time'))
                    if planner.commit is not None:
                        committed = t
                        break
                    if planner.speed_limit is not None:
                        held.append(t)
                self.assertEqual(committed, first_commit)
                if first_hold is None:
                    self.assertFalse(held)
                else:
                    np.testing.assert_allclose(held, np.arange(first_hold, first_commit, .1))


class MeasuredActuatorTests(unittest.TestCase):
    def test_yield_uses_the_fitted_pace_actuator_and_release_slew(self):
        from crosswalk.city_pace import CityPaceModel
        from crosswalk.city_policy import INDEX
        root = Path(__file__).resolve().parents[1]
        model = CityPaceModel.load(root / 'models/city-pace/candidate.npz')
        with np.load(root / 'results/city-cruise/seed-212/observations.npz') as data:
            x = data['features'][1100].copy()
        planner = TacticalPass(None, measured_hold=True)
        planner.yielding = True
        model.tactical_speed_limit = planner.speed_limit
        for speed in (4., 1., 0.):
            x[INDEX['ego_speed_mps']] = speed
            predicted = model.predict(x)
            self.assertEqual(model.last_speed_request, 0.)
            self.assertEqual(planner.apply(predicted, speed), predicted)
            self.assertEqual(predicted[1], 0.)
            self.assertGreater(predicted[2], 0.)
        planner.yielding = False
        model.tactical_speed_limit = planner.speed_limit
        model.predict(x)
        self.assertGreater(model.last_speed_request, 0.)
        self.assertLessEqual(model.last_speed_request, .18501)


if __name__=='__main__':
    unittest.main()
