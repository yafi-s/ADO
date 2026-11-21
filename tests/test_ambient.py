"""Serverless checks of the background population and its Town01 placement."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import carla

from crosswalk.ambient import Ambient, _Walker
from crosswalk.district import DESTINATIONS, trip_points


TOWN01_MAP = Path(os.environ.get("CARLA_TOWN01_XODR", ".local/Town01_Opt-1.0.xodr"))


class WalkerControlTests(unittest.TestCase):
    def test_waits_walks_then_stops(self):
        actor = Mock()
        actor.get_location.return_value = carla.Location()
        walker = _Walker(actor, carla.Location(), carla.Vector3D(1, 0, 0), 5, 1, 1.5)
        walker.update(1.0, 0.05)
        actor.apply_control.assert_not_called()
        walker.update(2.0, 0.05)
        self.assertAlmostEqual(actor.apply_control.call_args.args[0].speed, 1.0)
        actor.get_location.return_value = carla.Location(4.9, 0, 0)
        walker.update(7.0, 0.05)
        self.assertAlmostEqual(actor.apply_control.call_args.args[0].speed, 0.0)
        walker.update(7.05, 0.05)
        self.assertEqual(actor.apply_control.call_count, 2)

    def test_non_trip_never_inspects_world(self):
        world = Mock()
        ambient = Ambient(world, SimpleNamespace(kind="pedestrian"))
        ambient.before_tick()
        ambient.close()
        self.assertEqual(ambient.counts, {"walkers": 0, "vehicles": 0})
        self.assertFalse(world.mock_calls)


@unittest.skipUnless(TOWN01_MAP.is_file(), "Town01 OpenDRIVE is required")
class Town01AmbientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.road_map = carla.Map("Town01", TOWN01_MAP.read_text(encoding="utf-8"))

    def setup_trip(self, district="heldout", destination="market", seed=101):
        points = trip_points(self.road_map, district, destination)
        scene = SimpleNamespace(kind="trip", district=district, destination=destination,
                                seed=seed, route=[point.transform.location for point in points],
                                _hazard_index=76, ego=Mock(), hazard=Mock(), traffic=Mock())
        models = {
            "walker.pedestrian.*": "walker.pedestrian.0001",
            "vehicle.tesla.model3": "vehicle.tesla.model3",
        }
        library = Mock()
        library.filter.side_effect = lambda pattern: (
            [Mock(id=models[pattern], has_attribute=Mock(return_value=False))] if pattern in models else []
        )
        world = Mock(get_map=Mock(return_value=self.road_map), get_blueprint_library=Mock(return_value=library))
        world.get_settings.return_value.fixed_delta_seconds = 0.05
        spawned = []

        def spawn(blueprint, pose):
            actor = Mock()
            actor.get_location.return_value = pose.location
            spawned.append((blueprint.id, pose, actor))
            return actor

        world.try_spawn_actor.side_effect = spawn
        return world, scene, {point.road_id for point in points}, spawned

    def test_walkers_keep_to_sidewalk_and_parked_cars_are_on_other_streets(self):
        for district, destinations in DESTINATIONS.items():
            for destination in destinations:
                with self.subTest(district=district, destination=destination):
                    world, scene, route_roads, spawned = self.setup_trip(district, destination)
                    ambient = Ambient(world, scene)
                    self.addCleanup(ambient.close)
                    self.assertEqual(ambient.counts["walkers"], 3)
                    self.assertIn(ambient.counts["vehicles"], (1, 2))
                    parked_streets = set()
                    for model, pose, actor in spawned:
                        if model.startswith("vehicle."):
                            waypoint = self.road_map.get_waypoint(pose.location, project_to_road=False)
                            self.assertFalse(waypoint.is_junction)
                            self.assertNotIn(waypoint.road_id, route_roads | parked_streets)
                            parked_streets.add(waypoint.road_id)
                            self.assertTrue(actor.apply_control.call_args.args[0].hand_brake)
                            self.assertGreater(min(pose.location.distance(point) for point in scene.route), 12.0)
                    for walker in ambient._walkers:
                        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                            spot = walker.origin + walker.direction * (walker.length * fraction)
                            waypoint = self.road_map.get_waypoint(spot, project_to_road=False,
                                                                 lane_type=carla.LaneType.Sidewalk)
                            self.assertIsNotNone(waypoint)
                            self.assertFalse(waypoint.is_junction)
                            self.assertGreater(min(spot.distance(point) for point in scene.route), 3.4)
                            self.assertGreater(spot.distance(scene.route[76]), 14.0)
                    ambient.before_tick()
                    world.tick.assert_not_called()
                    for actor in (scene.ego, scene.hazard, scene.traffic):
                        actor.apply_control.assert_not_called()
                    ambient.close()
                    ambient.close()
                    for _, _, actor in spawned:
                        actor.destroy.assert_called_once()

    def test_layout_repeats_for_seed_and_spawn_failure_is_optional(self):
        def layout(seed):
            world, scene, _, spawned = self.setup_trip(seed=seed)
            ambient = Ambient(world, scene)
            self.addCleanup(ambient.close)
            return [(model, pose.location.x, pose.location.y) for model, pose, _ in spawned]

        self.assertEqual(layout(101), layout(101))
        self.assertNotEqual(layout(101), layout(102))
        world, scene, _, _ = self.setup_trip()
        world.try_spawn_actor.side_effect = None
        world.try_spawn_actor.return_value = None
        ambient = Ambient(world, scene)
        self.assertEqual(ambient.counts, {"walkers": 0, "vehicles": 0})
        self.assertLessEqual(world.try_spawn_actor.call_count, 25)
        ambient.close()

    def test_partial_population_is_destroyed_on_error(self):
        world, scene, _, _ = self.setup_trip()
        actor = Mock()
        world.try_spawn_actor.side_effect = [actor, RuntimeError("lost connection")]
        with self.assertRaisesRegex(RuntimeError, "lost connection"):
            Ambient(world, scene)
        actor.destroy.assert_called_once()
