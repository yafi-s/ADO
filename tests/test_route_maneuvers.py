"""Native-map, sensor-counterfactual checks for the route maneuver candidate."""
import unittest
from unittest.mock import Mock

import carla
import numpy as np

from crosswalk.maneuver_choice import ManeuverChoice
from crosswalk.route_maneuvers import RouteManeuvers, compare_progress
from tests.test_urban_map import MAP


class FittedManeuvers(unittest.TestCase):
    def test_matched_queues_have_no_projected_progress(self):
        equal,gain,_ = compare_progress("pass",32.46,.047,32.51,60.,0.,34.)
        self.assertEqual(equal,"follow")
        self.assertLess(abs(gain),.25)
        clear,gain,_ = compare_progress("pass",22.,1.5,60.,60.,0.,51.)
        self.assertEqual(clear,"pass")
        self.assertGreater(gain,15.)
        nearby,_,_=compare_progress("pass",22.,1.5,3.,5.,0.,51.)
        self.assertEqual(nearby,"yield")

    def test_independent_traffic_choices(self):
        model = ManeuverChoice()
        for x, expected in [
            ((60,0,60,60,0,0,4), "keep"),
            ((22,8,60,60,0,0,4), "follow"),
            ((24,1.4,60,60,0,0,4), "pass"),
            ((22,0,60,60,0,0,4), "pass"),
            ((22,1,60,21,0,9,4), "yield"),
            ((22,1,6,60,4,0,4), "yield"),
        ]:
            self.assertEqual(model.predict(x)[0], expected)


@unittest.skipUnless(MAP.exists(),"Town03 OpenDRIVE unavailable")
class RouteManeuverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.road_map = carla.Map("Town03_Opt", MAP.read_text())
        cls.route = [cls.road_map.get_waypoint_xodr(67,-2,float(s)).transform.location for s in range(30,225)]
        cls.corridors = RouteManeuvers.from_route(cls.road_map, cls.route).corridors

    def setUp(self):
        self.driver = RouteManeuvers(self.corridors)
        self.c = self.corridors[0]
        self.ego = Mock()
        self.ego.bounding_box = carla.BoundingBox(carla.Location(z=.76),carla.Vector3D(2.45,.92,.76))
        self.ego.get_light_state.return_value = carla.VehicleLightState.LowBeam

    def at(self, s, alpha=0.):
        xy = self.c.goals(np.array([s,s+.2]),np.array([alpha,alpha]))
        z = np.interp(s,self.c.arc,self.c.source[:,2])
        yaw = np.rad2deg(np.arctan2(*(xy[1]-xy[0])[::-1]))
        pose = carla.Transform(carla.Location(*xy[0],z),carla.Rotation(yaw=yaw))
        self.ego.get_transform.return_value = pose
        return pose

    def observe(self, s, alpha=0., lead=None, rear=None, pavement=True, source_behind=None):
        pose = self.at(s,alpha)
        patches = []
        def patch(stations,lane,height):
            xy = self.c.goals(stations,np.full(len(stations),lane))
            z = np.interp(stations,self.c.arc,self.c.source[:,2])+height
            patches.append(np.column_stack((xy,z)))
        if pavement:
            for lane in (0.,1.):
                patch(np.arange(s-45.,s+45.,.25),lane,0.)
        if lead is not None:
            patch(lead+np.linspace(-.02,.02,16),0.,.65)
        if source_behind is not None:
            patch(source_behind+np.linspace(-.02,.02,16),0.,.7)
        if rear is not None:
            patch(rear+np.linspace(-.02,.02,16),1.,.7)
        points = np.vstack(patches) if patches else np.empty((0,3))
        transform = np.asarray(pose.get_matrix())
        local = (points-transform[:3,3])@transform[:3,:3]
        local[:,2]-=2.30
        cloud = np.c_[local,np.ones(len(local))]
        directions = self.driver.preview(self.ego,cloud,np.zeros(14),4.)
        return self.driver.info, directions

    def test_curve_geometry_and_route_permission(self):
        self.assertEqual(self.c.source_key,(67,-2))
        s = self.c.begin + 32.
        for alpha in (0.,.6,1.):
            world = self.c.goals(np.array([s,s+32.]),np.full(2,alpha))
            native = self.c.interpolate(np.array([s,s+32.]),self.c.source)
            rel,side,_,width,inside = self.c.project_scan(np.c_[world,native[:,2]],s)
            np.testing.assert_allclose(rel,[0.,32.],atol=.15)
            np.testing.assert_allclose(side,-alpha*width,atol=.04)
            self.assertTrue(all(inside))
        route = [self.road_map.get_waypoint_xodr(24,-1,float(k)).transform.location for k in range(5,40)]
        self.assertFalse(RouteManeuvers.from_route(self.road_map,route).corridors)

    def test_stationary_then_clear_and_measured_return(self):
        start = self.c.begin+25.
        obstacle = start+24.
        phases = []
        for _ in range(16):
            info,_ = self.observe(start,lead=obstacle)
            phases.append(info["maneuver_phase"])
        self.assertIn("signal_left",phases)
        self.assertEqual(info["lead_kind"],"static")
        self.assertTrue(info["committed"])
        self.driver.update_lights(self.ego)
        self.assertEqual(info["signal_command_bits"],int(carla.VehicleLightState.LowBeam|carla.VehicleLightState.LeftBlinker))
        for delta in np.arange(.5,79.,.5):
            # Feed independent localized poses along both native lanes. The
            # production planner pulls steering toward the selected trajectory.
            alpha = min(1., delta/23.)
            if self.driver.return_start is not None:
                alpha = max(0.,1.-(start+delta-self.driver.return_start)/20.)
            info,_ = self.observe(start+delta,alpha,lead=obstacle)
            phases.append(info["maneuver_phase"])
            if phases[-1]=="rejoined":
                break
        self.assertIn("passing",phases)
        self.assertIn("signal_right",phases)
        self.assertEqual(phases[-1],"rejoined")

    def test_rejoin_remembers_confirmed_order_but_needs_current_gap(self):
        start = self.c.begin+27.
        obstacle = start+23.
        for _ in range(15):
            self.observe(start,lead=obstacle)
        for d in np.arange(.5,42.,.5):
            self.observe(start+d,min(1.,d/22.),lead=obstacle)
            if self.driver.passed_observed is not None:
                break
        self.assertIsNotNone(self.driver.passed_observed)
        for d in np.arange(d+.5,60.,.5):
            info,_ = self.observe(start+d,1.,pavement=False)
            self.assertNotEqual(info["maneuver_phase"],"change_right")
        for d in np.arange(60.,71.,.5):
            info,_ = self.observe(start+d,1.)
            if info["maneuver_phase"]=="change_right":
                break
        self.assertEqual(info["maneuver_phase"],"change_right")
        self.assertGreaterEqual(info["return_clear_scans"],5)
        self.assertEqual(info["source_rear_m"],60.)

    def test_new_source_rear_intrusion_cancels_early_rejoin(self):
        start = self.c.begin+25.
        lead = start+24.
        for _ in range(15):
            self.observe(start,lead=lead)
        for d in np.arange(.5,70.,.5):
            info,_ = self.observe(start+d,min(1.,d/22.),lead=lead)
            if info["maneuver_phase"]=="change_right":
                break
        self.assertEqual(info["maneuver_phase"],"change_right")
        # A new bumper closes into the source lane while the car is still
        # centred on the passing lane. A previously clear sweep cannot win.
        info,_ = self.observe(start+d+.25,1.,lead=lead,source_behind=start+d-5.)
        self.assertEqual(info["maneuver_phase"],"wait_rejoin")
        self.assertFalse(info["return_gap_accepted"])
        self.assertIsNone(self.driver.return_start)

    def test_vacated_target_cannot_brake_for_parallel_car_but_intrusion_still_can(self):
        s = self.c.begin+88.
        def returning():
            self.driver = RouteManeuvers(self.corridors)
            self.driver.corridor = self.c
            self.driver.start, self.driver.return_start = s-62., s-16.
            self.driver.return_length = 20.
        returning()
        for _ in range(3):
            info,_ = self.observe(s,.07,rear=s+.30)
            self.assertTrue(info["target_lane_vacated"])
            self.assertIsNone(self.driver.speed_limit)
        self.assertEqual(info["maneuver_phase"],"rejoined")
        self.assertEqual(info["turn_signal"],"off")
        returning()
        info,_ = self.observe(s,.48,rear=s+.30)
        self.assertFalse(info["target_lane_vacated"])
        self.assertEqual(info["turn_signal"],"right")
        self.assertEqual(self.driver.speed_limit,0.)
        returning()
        info,_ = self.observe(s,.07,rear=s+.30,lead=s+3.8)
        self.assertTrue(info["target_lane_vacated"])
        self.assertTrue(info["return_swept_intrusion"])
        self.assertEqual(self.driver.speed_limit,0.)

    def test_elevated_body_corners_use_planar_native_lane_bounds(self):
        self.driver.corridor = self.c
        self.driver.info = {"committed":True}
        station = self.c.begin+30.
        self.at(station,.46)
        self.ego.bounding_box = carla.BoundingBox(carla.Location(z=.85),carla.Vector3D(2.5,1.,.8))
        self.assertTrue(self.driver.legal_footprint(self.ego,self.road_map))
        self.at(station,-.5)
        self.assertFalse(self.driver.legal_footprint(self.ego,self.road_map))

    def test_closing_rear_blocks_and_late_intrusion_aborts(self):
        s = self.c.begin+31.
        for i in range(16):
            info,_ = self.observe(s,lead=s+24.,rear=s-28.+i*.95)
        self.assertEqual(info["maneuver_phase"],"yield")
        self.assertGreater(info["target_rear_closing_mps"],5.)
        self.assertEqual(self.driver.speed_limit,0.)
        for _ in range(28):
            info,_ = self.observe(s,lead=s+24.)
        self.assertTrue(info["committed"])
        info,_ = self.observe(s,lead=s+24.,rear=s-3.)
        self.assertFalse(info["committed"])
        self.assertEqual(self.driver.speed_limit,0.)

    def test_wait_survives_scan_loss_and_releases_for_departing_lead(self):
        s = self.c.begin+35.
        for _ in range(14):
            self.observe(s, lead=s+23., rear=s-5.)
        self.assertEqual(self.driver.speed_limit,0.)
        for _ in range(5):
            info,_ = self.observe(s, rear=s-5., pavement=False)
            self.assertEqual(info["maneuver_phase"],"yield")
        for i in range(18):
            info,_ = self.observe(s, lead=s+23.+i*.85, rear=s-5.)
        self.assertFalse(info["committed"])
        self.assertEqual(info["maneuver_phase"],"follow")
        self.assertEqual(info["lead_kind"],"departing")
        self.assertIsNone(self.driver.speed_limit)

    def test_confirmed_rolling_follow_and_rebrake_with_occupied_target(self):
        s = self.c.begin+35.
        for i in range(14):
            info,_ = self.observe(s, lead=s+21.+.16*i, rear=s-5.)
        self.assertEqual(info["maneuver_phase"],"yield")
        self.assertGreater(self.driver.speed_limit,1.)
        self.assertLessEqual(self.driver.speed_limit,1.7)
        for i in range(8):
            info,_ = self.observe(s, lead=s+21.+.16*13, rear=s-5.)
        self.assertFalse(info["committed"])
        self.assertEqual(self.driver.speed_limit,0.)
        self.observe(s, lead=s+21.+.16*13, rear=s-5.,pavement=False)
        self.assertEqual(self.driver.speed_limit,0.)

    def test_absent_pavement_or_insufficient_remaining_road(self):
        for station, pavement in [(self.c.begin+30.,False),(self.c.end-31.,True)]:
            self.driver = RouteManeuvers(self.corridors)
            for _ in range(14):
                info,_ = self.observe(station,lead=station+14.,pavement=pavement)
            self.assertFalse(info["committed"])
        for _ in range(15):
            info,_ = self.observe(self.c.begin+30.)
        self.assertFalse(info["committed"])
        self.assertEqual(info["turn_signal"],"off")


if __name__ == "__main__":
    unittest.main()
