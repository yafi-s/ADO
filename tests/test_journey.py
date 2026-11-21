"""Offline CARLA map and actor-observation tests for the continuous journey."""
import os
import unittest
from pathlib import Path
from unittest.mock import Mock

import carla

from crosswalk.journey import JourneyScenario, _junctions, _signal_site, journey_points, required_event_kinds

TOWN03 = Path(os.environ.get("CARLA_TOWN03_XODR",
    ".local/carla-1.0/CarlaUnreal/Packs/AdditionalMaps/Content/Maps/OpenDrive/Town03_Opt.xodr"))
MAP = Path(os.environ.get("CARLA_TOWN01_XODR", ".local/Town01_Opt-1.0.xodr"))


class Actor:
    def __init__(self, blueprint, pose):
        self.type_id = blueprint.id
        self.pose = pose
        self.velocity = carla.Vector3D()
        self.is_alive = True
        self.apply_control = Mock()
        extent = carla.Vector3D(.18,.18,.4) if blueprint.id.startswith("static.prop.") else carla.Vector3D(2.,1.,.8)
        self.bounding_box = carla.BoundingBox(carla.Location(),extent)

    def get_location(self):
        return self.pose.location

    def get_transform(self):
        return self.pose

    def get_velocity(self):
        return self.velocity

    def destroy(self):
        self.is_alive = False


@unittest.skipUnless(MAP.is_file(), "Set CARLA_TOWN01_XODR to Town01_Opt for offline geometry checks")
class JourneyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.road_map = carla.Map("Town01_Opt",MAP.read_text(encoding="utf8"))

    def scene(self, seed=1, blocked_van_attempts=0, road_map=None):
        road_map = road_map or self.road_map
        if road_map.name.startswith("Town03"):
            from crosswalk.urban_map import urban_route
            points = urban_route(road_map, seed).points
            start = next(i for i,wp in enumerate(points) if wp.is_junction and wp.junction_id==1352)
        else:
            points = journey_points(road_map, seed)
            start, _, _ = _junctions(points)[6]
        signal = Mock()
        signal.id = 4
        signal.get_stop_waypoints.return_value = [points[start-5]]
        signal.get_affected_lane_waypoints.return_value = [points[start+1]]
        signal.is_frozen.return_value = False
        signal.state = carla.TrafficLightState.Yellow
        signal.get_state.side_effect = lambda:signal.state
        signal.set_state.side_effect = lambda state:setattr(signal,"state",state)
        world = Mock()
        world.get_settings.return_value.fixed_delta_seconds = .25
        world.get_map.return_value = road_map
        world.get_actors.return_value.filter.return_value = [signal]
        catalog = world.get_blueprint_library.return_value
        def blueprint(name):
            bp = Mock()
            bp.id = name
            bp.has_attribute.return_value = False
            return bp
        catalog.find.side_effect = blueprint
        known = {"vehicle.*":"vehicle.gazelle.omafiets", "static.prop.*cone*":"static.prop.trafficcone01"}
        catalog.filter.side_effect = lambda name:[blueprint(known.get(name, name))]
        def spawn(bp, pose):
            nonlocal blocked_van_attempts
            if bp.id == "vehicle.sprinter.mercedes" and blocked_van_attempts > 0:
                blocked_van_attempts -= 1
                return None
            return Actor(bp, pose)

        world.try_spawn_actor.side_effect = spawn
        scene = JourneyScenario(world,"journey",seed)
        self.addCleanup(scene.close)
        return scene

    def test_continuous_unique_streets_and_route_choices(self):
        scenes = [self.scene(seed) for seed in (1,2)]
        for scene in scenes:
            points = scene._points
            self.assertTrue(1300 < scene.length < 1500)
            self.assertEqual(len(scene.junction_ids),len(set(scene.junction_ids)))
            self.assertEqual(len(scene.junction_ids),9)
            self.assertEqual(scene.route, [w.transform.location for w in journey_points(self.road_map,scene.seed)])
            self.assertTrue(all(.35 < a.distance(b)<1.6 for a,b in zip(scene.route,scene.route[1:])))
            self.assertEqual(set(d for _,_,d in scene.turns) & {"left","right"},{"left","right"})
            self.assertEqual(len(scene.events),9)
            self.assertEqual(sum(a.type_id=="vehicle.lincoln.mkz" for a in scene._actors),1)
            for e in scene.events:
                self.assertAlmostEqual(e.s,scene._coordinates(e.pose.location)[0],delta=.01)
            self.assertGreater(len({(w.road_id,w.lane_id) for w in points}),15)
        self.assertNotEqual(scenes[0].road_ids[-1],scenes[1].road_ids[-1])
        self.assertGreater(scenes[0].route[-1].distance(scenes[1].route[-1]),15)

    def test_unreached_events_never_count_and_signal_requires_matching_movement(self):
        scene = self.scene()
        for _ in range(40):
            scene.before_tick()
            scene.metrics()
        self.assertTrue(all(not e["handled"] and not e["exposed"] and not e["triggered"] and not e["cleared"]
                            for e in scene.metrics()["events"]))
        scene.ego.apply_control.assert_not_called()
        self.assertEqual(scene.signal.state,carla.TrafficLightState.Green)
        scene.signal.get_affected_lane_waypoints.return_value = []
        with self.assertRaisesRegex(RuntimeError,"physical traffic light"):
            _signal_site(scene._points,scene.turns,[scene.signal])
        scene.close()
        self.assertEqual(scene.signal.state,carla.TrafficLightState.Yellow)
        scene.world.freeze_all_traffic_lights.assert_called_with(False)

    def test_walker_requires_in_path_exposure_and_physical_clearance(self):
        scene = self.scene()
        e = next(e for e in scene.events if e.kind=="pedestrian-delayed")
        scene.ego.pose = scene._points[e.index-15].transform
        for _ in range(4):
            scene.before_tick()
        self.assertFalse(e.record()["triggered"])
        for _ in range(20):
            scene.before_tick()
        self.assertTrue(e.record()["triggered"])
        self.assertFalse(e.record()["exposed"])
        e.actor.pose = e.pose
        scene.before_tick()
        self.assertTrue(e.record()["exposed"])
        for _ in range(20):
            scene.before_tick()
        self.assertFalse(e.record()["cleared"])
        e.actor.pose = carla.Transform(e.pose.location+e.direction*7,e.pose.rotation)
        scene.before_tick()
        self.assertTrue(e.record()["cleared"])
        self.assertFalse(e.record()["handled"])
        scene.ego.pose = scene._points[e.index+10].transform
        scene.before_tick()
        self.assertTrue(e.record()["handled"])

    @unittest.skipUnless(TOWN03.is_file(), "Set CARLA_TOWN03_XODR for the urban course")
    def test_urban_bus_requires_actual_lane_transition_and_stopped_bus_pass(self):
        road_map = carla.Map("Town03_Opt",TOWN03.read_text(encoding="utf8"))
        scene = self.scene(road_map=road_map)
        self.assertEqual(set(e.kind for e in scene.events),set(required_event_kinds(road_map.name)))
        self.assertEqual(scene.destination,"market")
        self.assertGreaterEqual(scene.ambient.counts["walkers"],2)
        e = next(e for e in scene.events if e.kind=="bus")
        e.actor.bounding_box = carla.BoundingBox(carla.Location(),carla.Vector3D(5.7,1.25,1.7))
        c = e.lane_change
        for i in range(c.start-12,c.bus+18):
            scene.ego.pose = scene._points[i].transform
            scene.before_tick()
        self.assertTrue(e.record()["handled"], e.record())
        self.assertGreaterEqual(e.record()["measured_lane_change_m"],2.1)
        self.assertGreater(e.record()["minimum_bus_clearance_m"],.25)
        self.assertEqual(e.record()["source_lane"],[4,3])
        self.assertEqual(e.record()["target_lane"],[4,2])
        scene.ego.apply_control.assert_not_called()

    @unittest.skipUnless(TOWN03.is_file(), "Set CARLA_TOWN03_XODR for the urban course")
    def test_background_vehicles_need_measured_motion_near_ego(self):
        road_map = carla.Map("Town03_Opt",TOWN03.read_text(encoding="utf8"))
        scene = self.scene(road_map=road_map)
        cars = scene.ambient._cars
        self.assertEqual(len({c.road for c in cars}),3)
        for c in cars:
            self.assertTrue(scene.ambient._isolated(scene,c.path))
            for pos in c.path[::6]:
                lane = road_map.get_waypoint(pos,project_to_road=False,lane_type=carla.LaneType.Driving)
                self.assertEqual((lane.road_id,lane.lane_id),(c.road,c.lane))
        self.assertEqual(scene.metrics()["background_traffic_activity"]["moving_near_ego"],0)
        parallel = next(c for c in cars if c.movement=="parallel")
        route_i = min(range(len(scene.route)),
                      key=lambda i:scene.route[i].distance(parallel.actor.get_location()))
        scene.ego.pose = scene._points[route_i].transform
        scene.before_tick()
        scene.before_tick()
        self.assertTrue(parallel.active)
        self.assertEqual(scene.metrics()["background_traffic_activity"]["moving_near_ego"],0)
        parallel.actor.pose = carla.Transform(parallel.path[6],parallel.actor.pose.rotation)
        parallel.actor.velocity = parallel.actor.pose.get_forward_vector()*4.
        scene.before_tick()
        self.assertEqual(scene.metrics()["background_traffic_activity"]["moving_near_ego"],1)
        self.assertLess(parallel.closest,16.)
        self.assertGreater(parallel.traveled,3.)

    @unittest.skipUnless(TOWN03.is_file(), "Set CARLA_TOWN03_XODR for the urban course")
    def test_urban_queue_requires_observed_red_stop_and_moving_green_departure(self):
        road_map = carla.Map("Town03_Opt",TOWN03.read_text(encoding="utf8"))
        scene = self.scene(road_map=road_map)
        e = next(e for e in scene.events if e.kind=="signal")
        scene.ego.pose = scene._points[e.index-18].transform
        for _ in range(40):
            scene.before_tick()
        self.assertTrue(e.record()["exposed"])
        self.assertIn("queue_exposed_s",e.observed)
        self.assertNotIn("queue_departed_green_s",e.observed)
        self.assertFalse(e.record()["handled"])
        e.queue.pose = scene._points[e.index+1].transform
        e.queue.velocity = e.pose.get_forward_vector()*2.
        scene.before_tick()
        self.assertIn("queue_departed_green_s",e.observed)
        scene.ego.pose = scene._points[e.index+10].transform
        scene.before_tick()
        self.assertTrue(e.record()["handled"], e.record())

    def test_cones_require_measured_passage_of_every_physical_body(self):
        for lateral, expected in ((0.,True),(.7,False)):
            scene = self.scene()
            event = next(e for e in scene.events if e.kind=="roadworks")
            self.assertEqual(len(event.cones),6)
            for i in range(event.index-22,event.index+28):
                pose = scene._points[i].transform
                pose.location = pose.location + pose.get_right_vector()*lateral
                scene.ego.pose = pose
                scene.before_tick()
            record = event.record()
            self.assertEqual(record["cones_alongside"],6)
            self.assertEqual(record["handled"],expected)
            self.assertEqual(record["violation"],not expected)
            self.assertEqual(record["minimum_cone_clearance_m"] >= .10, expected)
            self.assertTrue(all(c.is_alive for c in event.cones))
            self.assertEqual(scene.ambient.counts,{"walkers":5,"vehicles":0})

    def test_cyclist_needs_motion_ahead_and_actual_exit(self):
        scene = self.scene()
        e = next(e for e in scene.events if e.kind=="cyclist")
        scene.ego.pose = scene._points[e.index-21].transform
        for _ in range(10):
            scene.before_tick()
        self.assertFalse(e.record()["triggered"])
        e.actor.velocity = e.pose.get_forward_vector()*2.
        scene.before_tick()
        self.assertTrue(e.record()["exposed"])
        self.assertFalse(e.record()["cleared"])
        e.actor.pose = carla.Transform(e.exit_path[-1],e.pose.rotation)
        scene.before_tick()
        self.assertTrue(e.record()["cleared"])
        self.assertFalse(e.record()["passed"])
        passage = next(i for i,s in enumerate(scene._station) if s>e.pass_station+.5)
        scene.ego.pose = scene._points[passage].transform
        scene.before_tick()
        self.assertTrue(e.record()["handled"])

    def test_red_passage_and_genuine_lead_restart(self):
        scene = self.scene()
        lead = next(e for e in scene.events if e.kind=="lead")
        scene.ego.pose = scene._points[lead.index-15].transform
        for _ in range(40):
            scene.before_tick()
        self.assertTrue(lead.record()["exposed"])
        self.assertFalse(lead.record()["cleared"])
        lead.actor.pose = scene._points[lead.index+9].transform
        lead.actor.velocity = carla.Vector3D(4,0,0)
        scene.before_tick()
        self.assertTrue(lead.record()["cleared"])
        signal = next(e for e in scene.events if e.kind=="signal")
        scene.ego.pose = scene._points[signal.index-20].transform
        scene.before_tick()
        scene.before_tick()
        self.assertTrue(signal.record()["exposed"])
        self.assertEqual(scene.signal.state,carla.TrafficLightState.Red)
        scene.ego.pose = scene._points[signal.index+1].transform
        scene.before_tick()
        self.assertEqual(scene.metrics()["red_light_violations"],1)
        self.assertFalse(signal.record()["handled"])


    def test_obstructed_curb_moves_van_and_walker_to_same_next_site(self):
        original, alternate = self.scene(), self.scene(blocked_van_attempts=2)
        initial = next(e for e in original.events if e.kind == "occluded")
        moved = next(e for e in alternate.events if e.kind == "occluded")
        self.assertEqual(moved.index, initial.index + 9)
        self.assertAlmostEqual(moved.s - initial.s, 9., delta=.1)
        self.assertAlmostEqual(moved.actor.get_location().distance(initial.actor.get_location()), 9., delta=.1)
        alternate.ego.pose = alternate._points[moved.index - 17].transform
        alternate.before_tick()
        self.assertIn("blocked_view_s", moved.observed)
        self.assertFalse(moved.record()["handled"])

    def test_occluded_walker_really_emerges_past_the_van(self):
        scene = self.scene()
        e = next(e for e in scene.events if e.kind=="occluded")
        scene.ego.pose = scene._points[e.index-17].transform
        scene.before_tick()
        self.assertIn("blocked_view_s",e.observed)
        self.assertNotIn("emerged_s",e.observed)
        e.actor.pose = e.pose
        scene.before_tick()
        self.assertIn("emerged_s",e.observed)
        self.assertIn("exposed_s",e.observed)

    def test_braking_needs_motion_and_left_needs_oncoming_clearance(self):
        scene = self.scene()
        e = next(e for e in scene.events if e.kind=="braking")
        scene.ego.pose = scene._points[e.index-15].transform
        for _ in range(4):
            scene.before_tick()
        self.assertFalse(e.record()["triggered"])
        e.actor.velocity = carla.Vector3D(2.4,0,0)
        scene.before_tick()
        self.assertTrue(e.record()["triggered"])
        e.actor.velocity = carla.Vector3D()
        scene.before_tick()
        self.assertIn("lead_stopped_s",e.observed)
        self.assertGreater(e.record()["peak_lead_deceleration_mps2"],0.)
        self.assertFalse(e.record()["cleared"])
        e = next(e for e in scene.events if e.kind=="left")
        scene.ego.pose = scene._points[e.index-15].transform
        for _ in range(12):
            scene.before_tick()
        self.assertTrue(e.record()["triggered"])
        self.assertFalse(e.record()["exposed"])
        e.actor.velocity = e.direction * 3.
        scene.before_tick()
        self.assertTrue(e.record()["exposed"])
        self.assertFalse(e.record()["cleared"])
        e.actor.pose = carla.Transform(e.conflict+e.direction*5.,e.actor.pose.rotation)
        scene.before_tick()
        self.assertTrue(e.record()["cleared"])
        self.assertFalse(e.record()["handled"])
        a,b,_ = scene.turns[0]
        scene.ego.pose = scene._points[a+2].transform
        scene.before_tick()
        self.assertEqual(scene.metrics()["turns_completed"]["left"],0)
        scene.ego.pose = scene._points[b+3].transform
        scene.before_tick()
        self.assertEqual(scene.metrics()["turns_completed"]["left"],1)
        self.assertTrue(e.record()["handled"])


if __name__ == "__main__":
    unittest.main()
