"""Serverless regressions against CARLA's actual OpenDRIVE parser and types."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import carla

from crosswalk.world import ROUTE_METERS, Scenario, _routes


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


def braking_scene(seed):
    road_map = two_way_street()
    world = Mock()
    world.get_settings.return_value.fixed_delta_seconds = 0.25
    world.get_map.return_value = road_map

    def spawn(blueprint, transform):
        actor = Mock()
        actor.bounding_box = carla.BoundingBox(carla.Location(), carla.Vector3D(2, 1, 1))
        actor.get_location.return_value = transform.location
        actor.get_transform.return_value = transform
        actor.get_velocity.return_value = carla.Vector3D()
        return actor

    world.try_spawn_actor.side_effect = spawn
    return Scenario(world, "braking", seed)


class ScenarioTests(unittest.TestCase):
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
        self.scene.route = [carla.Location(0, 0, 0), carla.Location(64, 0, 0)]
        self.scene._fx, self.scene._fy = 1.0, 0.0
        self.scene.ego = vehicle(0, 0)

    def test_lead_gap_measures_bumpers_and_excludes_oncoming_lane(self):
        self.scene.kind = "lead"
        self.scene.hazard = vehicle(16, 0)
        self.assertEqual(self.scene.expert_gap(), 12)
        self.scene.hazard = vehicle(16, -4)
        self.assertEqual(self.scene.expert_gap(), 60)
        self.scene.hazard = vehicle(3, 0)
        self.assertEqual(self.scene.expert_gap(), 0)

    def test_pedestrian_traversing_lane_changes_expert_label(self):
        self.scene.kind = "pedestrian"
        for lateral, expected in ((2.35, 60.0), (0.0, 7.7), (-2.35, 60.0)):
            self.scene.hazard = vehicle(10, lateral, length=0.6, width=0.5)
            self.assertAlmostEqual(self.scene.expert_gap(), expected, places=5)

    def test_braking_spawn_is_seeded_and_shares_the_ego_lane(self):
        first, repeat = braking_scene(23), braking_scene(23)
        self.assertEqual(first.weather, repeat.weather)
        self.assertEqual(first.route[0], repeat.route[0])
        self.assertEqual(first.hazard.get_location(), repeat.hazard.get_location())
        s, lateral = first._coordinates(first.hazard.get_location())
        self.assertGreaterEqual(round(s, 3), 18)
        self.assertLessEqual(round(s, 3), 24)
        self.assertAlmostEqual(lateral, 0, places=3)
        self.assertAlmostEqual(first.expert_gap(), s - 4, places=3)

    def test_braking_lead_cruises_brakes_holds_then_departs(self):
        scene = braking_scene(23)
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
        scene = braking_scene(91)
        scene.hazard.get_location.return_value = scene.route[48]
        scene.hazard.get_velocity.return_value = carla.Vector3D(2.5, 0, 0)
        for _ in range(8):
            scene.before_tick()
        self.assertEqual(scene.phase, "lead cruising")
        scene.before_tick()
        self.assertGreater(scene.expert_gap(), 13)
        self.assertEqual(scene.phase, "lead braking")


if __name__ == "__main__":
    unittest.main()
