"""Serverless regressions against CARLA's actual OpenDRIVE parser and types."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import os
from pathlib import Path

import carla

from crosswalk.district import DESTINATIONS, DISTRICTS
from crosswalk.world import ROUTE_METERS, Scenario, _angle, _routes, _scene_routes, _vehicle_blueprint


TOWN01_MAP = Path(os.environ.get("CARLA_TOWN01_XODR",
    ".local/Town01_Opt-1.0.xodr"))


def two_way_street():
    lane = """<lane id="{}" type="driving" level="false">
        <link/><width sOffset="0" a="4" b="0" c="0" d="0"/>
    </lane>"""
    xodr = f"""<OpenDRIVE>
    <header revMajor="1" revMinor="4" name="example" version="1" north="10" south="-10" east="180" west="0"/>
    <road name="street" length="180" id="1" junction="-1"><link/>
    <planView><geometry s="0" x="0" y="0" hdg="0" length="180"><line/></geometry></planView>
    <lanes><laneSection s="0"><left>{lane.format(1)}</left>
    <center><lane id="0" type="none" level="false"/></center><right>{lane.format(-1)}</right>
    </laneSection></lanes></road></OpenDRIVE>"""
    return carla.Map("two-way-test", xodr)


def vehicle(x, y, length=4, width=2):
    return SimpleNamespace(
        get_location=lambda: carla.Location(x, y, 0),
        bounding_box=carla.BoundingBox(carla.Location(), carla.Vector3D(length / 2, width / 2, 1)),
    )


def fake_scene(seed, kind="braking", perturb=False, road_map=None, signals=(), **kwargs):
    world = Mock()
    world.get_settings.return_value.fixed_delta_seconds = 0.25
    world.get_map.return_value = road_map or two_way_street()
    world.get_actors.return_value.filter.return_value = signals
    person = Mock()
    person.id = "walker.pedestrian.0001"
    world.get_blueprint_library.return_value.filter.return_value = [person]

    def spawn(blueprint, transform):
        actor = Mock()
        actor.bounding_box = carla.BoundingBox(carla.Location(), carla.Vector3D(2, 1, 1))
        actor.get_location.return_value = transform.location
        actor.get_transform.return_value = transform
        actor.get_velocity.return_value = carla.Vector3D()
        return actor

    world.try_spawn_actor.side_effect = spawn
    return Scenario(world, kind, seed, perturb=perturb, **kwargs)


class ScenarioTests(unittest.TestCase):
    def test_vehicle_blueprints_resolve_released_catalog_names(self):
        for available in ("vehicle.lincoln.mkz", "vehicle.lincoln.mkz_2020"):
            catalog = Mock()
            catalog.filter.side_effect = lambda name: [available] if name == available else []
            catalog.find.side_effect = lambda name: name
            self.assertEqual(_vehicle_blueprint(catalog, "vehicle.lincoln.mkz", "vehicle.lincoln.mkz_2020"), available)

    def test_route_follows_lane_and_oncoming_geometry_is_reversed(self):
        road_map = two_way_street()
        points, near, far = next(_routes(road_map, 23))
        same_points, _, _ = next(_routes(road_map, 23))
        self.assertEqual(points[0].id, same_points[0].id)
        self.assertAlmostEqual(points[0].transform.location.distance(points[ROUTE_METERS].transform.location),
                               ROUTE_METERS, places=3)
        self.assertEqual(-points[0].lane_id, near.lane_id)
        self.assertEqual(near.lane_id, far.lane_id)
        self.assertGreater(len(points), ROUTE_METERS + 1)

    def setUp(self):
        self.scene = Scenario.__new__(Scenario)
        self.scene._set_route([carla.Location(0, 0, 0), carla.Location(64, 0, 0)])
        self.scene.ego = vehicle(0, 0)

    def test_lead_gap_measures_bumpers_and_excludes_oncoming_lane(self):
        self.scene.kind = "lead"
        self.scene.hazard = vehicle(16, 0)
        self.assertEqual(self.scene.expert_gap(), 12)
        self.scene.hazard = vehicle(16, -4)
        self.assertEqual(self.scene.expert_gap(), 60)
        self.scene.hazard = vehicle(3, 0)
        self.assertEqual(self.scene.expert_gap(), 0)
        self.scene.ego = vehicle(64, 0)
        self.scene.hazard = vehicle(70, 0)
        self.assertEqual(self.scene.expert_gap(), 2)

    def test_pedestrian_traversing_lane_changes_expert_label(self):
        self.scene.kind = "pedestrian"
        for lateral, expected in ((2.35, 60.0), (0.0, 7.7), (-2.35, 60.0)):
            self.scene.hazard = vehicle(10, lateral, length=0.6, width=0.5)
            self.assertAlmostEqual(self.scene.expert_gap(), expected, places=5)

    def test_braking_spawn_is_seeded_and_shares_the_ego_lane(self):
        first, repeat = fake_scene(23), fake_scene(23)
        self.assertEqual(first.weather, repeat.weather)
        self.assertEqual(first.route[0], repeat.route[0])
        self.assertEqual(first.hazard.get_location(), repeat.hazard.get_location())
        s, lateral = first._coordinates(first.hazard.get_location())
        self.assertGreaterEqual(round(s, 3), 18)
        self.assertLessEqual(round(s, 3), 24)
        self.assertAlmostEqual(lateral, 0, places=3)
        self.assertAlmostEqual(first.expert_gap(), s - 4, places=3)

    def test_braking_lead_cruises_brakes_holds_then_departs(self):
        scene = fake_scene(23)
        for _ in range(10):
            scene.before_tick()
        self.assertEqual(scene.phase, "lead cruising")
        self.assertIsNone(scene.metrics()["event_start_s"])
        self.assertGreater(scene.hazard.apply_control.call_args.args[0].throttle, 0)

        scene.ego.get_location.return_value = scene.route[scene._hazard_index - 14]
        scene.hazard.get_velocity.return_value = carla.Vector3D(scene._fx * 2.5, scene._fy * 2.5, 0)
        scene.before_tick()
        self.assertEqual(scene.phase, "lead braking")
        self.assertAlmostEqual(scene.expert_gap(), 10, places=3)
        started = scene.metrics()["event_start_s"]
        self.assertEqual(started, scene._time)
        # If the simulated car still rolls, keep braking even after the usual dwell.
        for _ in range(14):
            scene.before_tick()
            self.assertEqual(scene.phase, "lead braking")
            self.assertEqual(scene.hazard.apply_control.call_args.args[0].brake, 1)

        scene.hazard.get_velocity.return_value = carla.Vector3D()
        for _ in range(12):
            scene.before_tick()
            self.assertEqual(scene.phase, "lead stopped")
            self.assertEqual(scene.hazard.apply_control.call_args.args[0].brake, 1)
        for _ in range(15):
            scene.before_tick()
            self.assertEqual(scene.phase, "lead departing")
            self.assertGreater(scene.hazard.apply_control.call_args.args[0].throttle, 0)
        self.assertEqual(scene.metrics()["event_start_s"], started)

    def test_braking_fallback_for_distant_ego(self):
        scene = fake_scene(91)
        scene.hazard.get_location.return_value = scene.route[48]
        scene.hazard.get_velocity.return_value = carla.Vector3D(2.5, 0, 0)
        for _ in range(8):
            scene.before_tick()
        self.assertEqual(scene.phase, "lead cruising")
        scene.before_tick()
        self.assertGreater(scene.expert_gap(), 13)
        self.assertEqual(scene.phase, "lead braking")

    def test_pose_perturbation_is_seeded_and_preserves_reference_route(self):
        original = fake_scene(65, "clear")
        perturbed = fake_scene(65, "clear", perturb=True)
        repeated = fake_scene(65, "clear", perturb=True)
        self.assertEqual(original.route, perturbed.route)
        self.assertEqual(perturbed.ego.get_transform(), repeated.ego.get_transform())
        start, shifted = original.ego.get_transform(), perturbed.ego.get_transform()
        self.assertAlmostEqual(_angle(shifted.rotation.yaw - start.rotation.yaw), perturbed.spawn_yaw_deg, places=4)
        self.assertGreaterEqual(abs(perturbed.spawn_yaw_deg), 2)
        self.assertLessEqual(abs(perturbed.spawn_yaw_deg), 5)
        self.assertAlmostEqual(perturbed._coordinates(shifted.location)[1], perturbed.spawn_lateral_m, places=4)
        self.assertGreaterEqual(abs(perturbed.spawn_lateral_m), 0.15)
        self.assertLessEqual(abs(perturbed.spawn_lateral_m), 0.45)
        self.assertEqual(original.spawn_yaw_deg, 0)
        self.assertEqual(original.spawn_lateral_m, 0)

    def test_metrics_follow_polyline_through_bend(self):
        scene = fake_scene(23, "clear")
        scene._set_route([carla.Location(x, y, 0) for x, y in ((0, 0), (10, 0), (10, 10))])
        self.assertEqual(scene.length, 20)
        for x, y, s, lateral in ((2, 1, 2, 1), (9, 6, 16, 1), (11, 8, 18, -1)):
            location = carla.Location(x, y, 0)
            scene.ego.get_location.return_value = location
            self.assertEqual(scene._coordinates(location), (s, lateral))
            self.assertEqual(scene.metrics()["lateral_error_m"], abs(lateral))
            self.assertEqual(scene.metrics()["progress_m"], s)

    @unittest.skipUnless(TOWN01_MAP.is_file(), "Install CARLA or set CARLA_TOWN01_XODR to the Town01 OpenDRIVE map")
    def test_town01_turns_use_one_connector_and_restore_intersection_lights(self):
        road_map = carla.Map("Town01", TOWN01_MAP.read_text(encoding="utf-8"))
        for seed, direction, angle in ((42, "left", -90), (43, "right", 90)):
            signals = [Mock(), Mock()]
            states = [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]
            frozen = seed == 43
            for signal, state in zip(signals, states):
                signal.get_state.return_value = state
                signal.is_frozen.return_value = frozen
            scene = fake_scene(seed, "turn", True, road_map, signals)
            repeat = fake_scene(seed, "turn", True, road_map)
            self.assertEqual(repeat.route, scene.route)
            self.assertEqual(scene.turn_direction, direction)
            self.assertTrue(all(a.distance(b) < 2 for a, b in zip(scene.route, scene.route[1:])))
            first, last = (road_map.get_waypoint(p) for p in (scene.route[0], scene.route[-1]))
            self.assertAlmostEqual(_angle(last.transform.rotation.yaw - first.transform.rotation.yaw), angle, delta=3)
            self.assertGreater(scene.length, scene.route[0].distance(scene.route[-1]) + 5)
            scene.ego.get_location.return_value = scene.route[-2]
            self.assertAlmostEqual(scene.metrics()["progress_m"], scene.length - 1, delta=0.1)
            self.assertAlmostEqual(scene.metrics()["lateral_error_m"], 0, places=3)
            self.assertEqual(scene.expert_gap(), 60)
            scene.before_tick()
            scene.ego.apply_control.assert_not_called()
            scene.traffic.apply_control.assert_called_once()
            scene.world.get_actors.return_value.filter.assert_called_once_with("traffic.traffic_light")
            scene.world.freeze_all_traffic_lights.assert_called_once_with(True)
            for signal in signals:
                signal.set_state.assert_called_once_with(carla.TrafficLightState.Green)
            scene.close()
            scene.world.freeze_all_traffic_lights.assert_called_with(frozen)
            for signal, state in zip(signals, states):
                signal.set_state.assert_called_with(state)


@unittest.skipUnless(TOWN01_MAP.is_file(), "Town01 OpenDRIVE is required")
class DistrictScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = carla.Map("Town01", TOWN01_MAP.read_text(encoding="utf-8"))

    def scene(self, kind, seed=11, **kwargs):
        scene = fake_scene(seed, kind, road_map=self.map, **kwargs)
        self.addCleanup(scene.close)
        return scene

    def test_districts_exclude_entire_corridors_and_intersections(self):
        train, heldout = DISTRICTS.values()
        self.assertTrue(train[0].isdisjoint(heldout[0]))
        self.assertTrue(train[1].isdisjoint(heldout[1]))
        bounds = {}
        for district, (roads, junctions) in DISTRICTS.items():
            coords = []
            for kind, direction in (("trip", None), ("left", "left"), ("signal", "straight"),
                                    ("turn", "right"), ("clear", None)):
                for seed in (11, 12):
                    destination = tuple(DESTINATIONS[district])[seed % 2]
                    points, near, far = next(_scene_routes(self.map, seed, kind, district, destination, direction))
                    for w in [*points, near, far]:
                        self.assertTrue(w.junction_id in junctions if w.is_junction else w.road_id in roads)
                        coords.append(w.transform.location.y)
            bounds[district] = min(coords), max(coords)
        self.assertGreater(bounds["heldout"][0] - bounds["train"][1], 100)

    def test_named_destinations_span_two_intersections_and_have_fixed_endpoints(self):
        for district in DISTRICTS:
            ends = []
            for destination in DESTINATIONS[district]:
                scene = self.scene("trip", district=district, destination=destination)
                repeat = self.scene("trip", seed=99, district=district, destination=destination)
                self.assertEqual(scene.route, repeat.route)
                self.assertEqual(scene.metrics()["destination"], destination)
                self.assertEqual(len(scene.junction_ids), 2)
                self.assertEqual(len(scene.road_ids), 3)
                self.assertTrue(150 < scene.length < 300)
                self.assertFalse(self.map.get_waypoint(scene.route[scene._hazard_index]).is_junction)
                for a, b in zip(scene.route, scene.route[1:]):
                    self.assertLess(a.distance(b), 1.6)
                ends.append(scene.route[-1])
            self.assertGreater(ends[0].distance(ends[1]), 100)

    def test_physical_occluder_and_adjustable_walker_trigger(self):
        early = self.scene("occluded", district="heldout")
        late = self.scene("occluded", district="heldout", crossing_delay=1)
        self.assertTrue(early.occluder.apply_control.call_args.args[0].hand_brake)
        van_s, van_l = early._coordinates(early.occluder.get_location())
        walk_s, walk_l = early._coordinates(early.hazard.get_location())
        self.assertAlmostEqual(walk_s - van_s, 5, delta=.1)
        self.assertGreater(walk_l, 2)
        self.assertGreater(van_l, 2)
        for scene in [early, late]:
            scene.ego.get_location.return_value = scene.route[scene._hazard_index - 12]
            scene.before_tick()
        self.assertTrue(early.metrics()["event_triggered"])
        self.assertFalse(late.metrics()["event_triggered"])
        for _ in range(4):
            late.before_tick()
        self.assertEqual(late.metrics()["event_start_s"] - early.metrics()["event_start_s"], 1)
        center = late.route[late._hazard_index]
        late.hazard.get_location.return_value = center
        late.before_tick()
        self.assertEqual(late.phase, "paused in lane")
        for _ in range(10):
            late.before_tick()
        self.assertEqual(late.phase, "leaving lane")
        late.hazard.get_location.return_value = center + late._walk_direction * 7.0
        late.before_tick()
        self.assertTrue(late.metrics()["event_cleared"])
        late.ego.apply_control.assert_not_called()

    def test_signal_scores_bumper_crossing_once_and_restores_light(self):
        points, _, _ = next(_scene_routes(self.map, 11, "signal", "heldout", None, "straight"))
        stop = points[17]
        signal = Mock()
        signal.id = 15
        signal.get_stop_waypoints.return_value = [stop]
        signal.get_affected_lane_waypoints.return_value = [next(w for w in points if w.is_junction)]
        signal.get_state.return_value = carla.TrafficLightState.Yellow
        signal.is_frozen.return_value = False
        other = Mock()
        other.id = 3
        # Overlapping trigger box on the same stop line, controlling another arm.
        other.get_stop_waypoints.return_value = [stop]
        other.get_affected_lane_waypoints.return_value = [SimpleNamespace(road_id=999, lane_id=1)]
        for cross_on_red in (True, False):
            scene = self.scene("signal", district="heldout", signals=[signal, other])
            self.assertEqual(scene.signal, signal)
            signal.set_state.assert_called_with(carla.TrafficLightState.Red)
            scene.before_tick()
            self.assertTrue(scene.metrics()["event_triggered"])
            if not cross_on_red:
                for _ in range(24):
                    scene.before_tick()
                signal.set_state.assert_called_with(carla.TrafficLightState.Green)
            pose = stop.transform
            # Nose is 2 m ahead of the actor origin.
            pose.location = pose.location - pose.get_forward_vector() * 1.5
            scene.ego.get_transform.return_value = pose
            for _ in range(3):
                self.assertEqual(scene.metrics()["red_light_violations"], int(cross_on_red))
            scene.close()
            signal.set_state.assert_called_with(carla.TrafficLightState.Yellow)

    def test_unprotected_left_has_conflicting_through_traffic_and_releases(self):
        scene = self.scene("left", district="heldout")
        ego_yaw = scene.ego.get_transform().rotation.yaw
        other_yaw = scene.traffic.get_transform().rotation.yaw
        self.assertGreater(abs(_angle(ego_yaw - other_yaw)), 174)
        conflict = scene._conflict
        self.assertLess(abs(scene._coordinates(conflict)[1]), .05)
        pose = scene._line
        scene.ego.get_transform.return_value = carla.Transform(pose.location - pose.get_forward_vector()*6, pose.rotation)
        scene.before_tick()
        self.assertEqual(scene.phase, "yield to oncoming")
        self.assertGreater(scene.traffic.apply_control.call_args.args[0].throttle, 0)
        self.assertAlmostEqual(scene.expert_gap(), 4, delta=.05)
        scene.traffic.get_location.return_value = conflict + scene._oncoming_direction * 5
        scene.before_tick()
        self.assertTrue(scene.metrics()["event_cleared"])
        self.assertEqual(scene.expert_gap(), 60)
        scene.ego.apply_control.assert_not_called()

    def test_wide_arc_needs_intersection_driving_surface(self):
        scene = self.scene("turn", seed=11, district="heldout")
        def displace(index, lateral):
            p = scene.route[index]
            w = self.map.get_waypoint(p)
            scene.ego.get_location.return_value = p + w.transform.get_right_vector() * lateral
            return scene.metrics()
        wide = displace(26, -2.3)
        self.assertGreater(wide["lateral_error_m"], 1.85)
        self.assertFalse(wide["route_departure"])
        self.assertTrue(displace(5, -2.3)["route_departure"])


if __name__ == "__main__":
    unittest.main()
