"""Serverless transfer geometry, time alignment, tracker and scoring regressions."""
from contextlib import ExitStack
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from crosswalk.control import Navigation
from waymo_training.simulator_transfer import (
    CalibratedTracker, DT, REAR_X, WHEELBASE, history_input, outcome, rear_xy, require_private, route_intent, run_episode, track_path,
)


class TransferTests(unittest.TestCase):
    def test_world_to_waymo_history_in_current_rear_frame(self):
        # Heading east in CARLA maps +world Y to ego right. Allow a fixed
        # orientation while accelerating laterally, so a missing sign is visible.
        t = np.arange(-42, 1) * DT
        positions = np.column_stack((100 + 3*t, 50 + t*t))
        hist = history_input(np.column_stack((12 + t, positions)), 0.)
        query = np.arange(-15, 1) / 4
        self.assertEqual((16, 6), hist.shape)
        np.testing.assert_allclose(hist[:, 0], 3*query, atol=1e-5)
        np.testing.assert_allclose(hist[:, 1], -query**2, atol=.00251)
        np.testing.assert_allclose(hist[:, 2], 3, atol=1e-5)
        np.testing.assert_allclose(hist[:, 3], -2*query, atol=1e-5)
        np.testing.assert_allclose(hist[:, 4], 0, atol=1e-5)
        np.testing.assert_allclose(hist[:, 5], -2, atol=1e-5)
        np.testing.assert_array_equal(hist[-1, :2], [0, 0])

    def test_turning_axle_positions_and_world_vectors_rotate_once(self):
        t = np.arange(-42, 1) * DT
        yaw = math.pi/2 + .1*t
        pts = np.array([rear_xy(35 + a, 12 + 4*a, b) for a, b in zip(t, yaw)])
        hist = history_input(np.column_stack((t, pts)), yaw[-1])
        # Current axes point +world Y forward and +world X left.
        np.testing.assert_allclose(hist[-1, 2:4], [4, 1 - .1*REAR_X], atol=2e-5)
        np.testing.assert_allclose(hist[-1, 4:6], [-.01*REAR_X, 0], atol=.0002)
        np.testing.assert_allclose(rear_xy(0, 0, math.pi/2), [0, REAR_X], atol=1e-12)

    def test_stationary_and_discontinuous_history(self):
        t = np.arange(43)*DT
        samples = np.column_stack((t, np.full_like(t, 22), np.full_like(t, -41)))
        np.testing.assert_allclose(history_input(samples, 1.0), 0, atol=1e-8)
        for invalid in (samples[:28], np.delete(samples, 16, axis=0)):
            with self.assertRaises(ValueError):
                history_input(invalid, 0)

    def test_route_command_uses_map_and_carla_handedness(self):
        def nav(end_y):
            return Navigation([SimpleNamespace(x=x, y=y) for x, y in ((0, 0), (12, 0), (12, end_y))])
        self.assertEqual(route_intent(nav(15), (0, 0), 0), 1)
        self.assertEqual(route_intent(nav(15), (0, 0), 3), 3)
        self.assertEqual(route_intent(nav(-15), (0, 0), 3), 2)
        self.assertEqual(route_intent(nav(-15), (12, -14.9), 3), 1)

    def test_forward_track_has_no_minimum_cruise_or_hidden_hazard_brake(self):
        t = np.arange(1, 21)/4
        path = np.column_stack((5*t, t*0))
        match = track_path(path, 5)
        self.assertEqual((match["target_speed_mps"], match["steer"], match["brake"], match["throttle"]),
                         (5., 0., 0., 0.))
        self.assertAlmostEqual(track_path(path, 4)["throttle"], .4)
        self.assertAlmostEqual(track_path(path, 6)["brake"], .25)
        self.assertEqual(track_path(np.zeros_like(path), 0)["throttle"], 0.)
        self.assertEqual(track_path(np.zeros_like(path), 4)["brake"], 1.)
        self.assertEqual(track_path(-path, 0)["target_speed_mps"], 0.)

    def test_left_bend_yields_negative_carla_steer_for_measured_wheelbase(self):
        t = np.arange(1, 21)/4
        path = np.column_stack((10*np.sin(.4*t), 10*(1 - np.cos(.4*t))))
        left = track_path(path, 4)
        right = track_path(path * [1, -1], 4)
        self.assertAlmostEqual(left["curvature_left_per_m"], .1)
        self.assertAlmostEqual(WHEELBASE, 2.86727)
        self.assertLess(left["steer"], 0)
        self.assertAlmostEqual(left["steer"], -right["steer"])
        self.assertAlmostEqual(left["target_speed_mps"], right["target_speed_mps"])
        with self.assertRaises(ValueError):
            track_path(np.full((20, 2), np.nan), 4)

    def test_collision_has_priority_and_incomplete_episode_is_not_a_pass(self):
        state = dict(progress_m=12., route_length_m=64., route_departure=False,
                     red_light_violations=0, yield_violations=0)
        self.assertIsNone(outcome(state, []))
        self.assertEqual(outcome(dict(state, progress_m=64.), []), "complete")
        self.assertEqual(outcome(dict(state, route_departure=True), []), "lane_departure")
        self.assertEqual(outcome(dict(state, red_light_violations=1), []), "red_light")
        self.assertEqual(outcome(dict(state, yield_violations=1), []), "yield_violation")
        self.assertEqual(outcome(dict(state, progress_m=64.), [(1, "walker")]), "collision")

    def test_private_output_excludes_nested_git_worktrees(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp) / "workspace"
            repo.mkdir()
            (repo / ".git").write_text("gitdir: linked", encoding="ascii")
            (repo / "crosswalk").mkdir()
            (repo / "crosswalk/runtime.py").touch()
            with self.assertRaises(ValueError):
                require_private(repo / "nested" / "weights.npz")
            outside = Path(tmp) / "private" / "research"
            self.assertEqual(require_private(outside), outside.resolve())

    def test_serverless_episode_applies_only_predicted_controls_and_catches_late_contact(self):
        import carla
        world, scene, rig, model = Mock(), Mock(), Mock(), Mock()
        scene.ego.type_id = "vehicle.lincoln.mkz"
        scene.route = [carla.Location(), carla.Location(x=60)]
        scene.weather = "ClearNoon"
        state = dict(route_id="stub", progress_m=0, route_length_m=60, lateral_error_m=0,
                     phase="crossing", hazard_gap_m=4, event_triggered=True, event_cleared=False,
                     red_light_violations=0, yield_violations=0, route_departure=False)
        scene.metrics.return_value = state
        images = [np.full((16, 16, 3), k, np.uint8) for k in range(3)]
        rig.read.return_value = dict(zip(("front", "left", "right"), images))
        rig.rendering = {}
        clock = iter(range(1, 53))
        snapshot = Mock()
        def tick():
            snapshot.frame = next(clock)
            snapshot.timestamp.elapsed_seconds = DT*snapshot.frame
            return snapshot.frame
        world.tick.side_effect = tick
        world.get_snapshot.return_value = snapshot
        snapshot.find.return_value.get_transform.return_value = carla.Transform()
        # Only delivered at the barrier after the last scored camera frame.
        rig.collisions.side_effect = lambda: [(50, "walker")] if snapshot.frame == 51 else []
        t = np.arange(1, 21)/4
        model.predict.return_value = np.column_stack((t, t*0))
        with ExitStack() as stack:
            folder = Path(stack.enter_context(TemporaryDirectory())) / "trial"
            stack.enter_context(patch("crosswalk.runtime.producer_info", return_value={"map": "Town01_Opt"}))
            stack.enter_context(patch("crosswalk.sensors.Rig", return_value=rig))
            stack.enter_context(patch("crosswalk.world.Scenario", return_value=scene))
            summary = run_episode(world, model, "pedestrian", 41, 0., 2, folder, False)
            rows = json.loads((folder / "trace.json").read_text())
        self.assertEqual(summary["outcome"], "collision")
        self.assertEqual([r["collision_count"] for r in rows], [0, 0, 1])
        self.assertEqual([r["applied_for_next_tick"] for r in rows], [True, True, False])
        self.assertEqual(len(scene.ego.apply_control.call_args_list), 3)
        for call in scene.ego.apply_control.call_args_list[1:]:
            command = call.args[0]
            self.assertAlmostEqual(command.throttle, .4)
            self.assertEqual(command.brake, 0.)
        for call in model.predict.call_args_list:
            observed, history, intent = call.args
            self.assertTrue(all(a is b for a, b in zip(observed, images, strict=True)))
            np.testing.assert_allclose(history, 0, atol=1e-8)
            self.assertEqual(intent, 1)
        scene.ego.set_autopilot.assert_not_called()
        scene.ego.set_target_velocity.assert_not_called()
        scene.close.assert_called_once()
        rig.close.assert_called_once()

    def test_calibrated_servo_tracks_low_speeds_and_applies_learned_stop(self):
        t = np.arange(1, 21) / 4
        servo = CalibratedTracker()
        speed = 0.
        # Simple plant with positive static resistance. The original 4% pedal
        # never breaks away here. This exercises controller dynamics only.
        for _ in range(250):
            cmd = servo.request(np.column_stack((.1*t, 0*t)), speed)
            accel = 7.5 * (cmd["throttle"]-.09) - 4*cmd["brake"]
            speed = max(0., speed + DT*accel)
        self.assertAlmostEqual(speed, .1, delta=.01)
        for _ in range(100):
            cmd = servo.request(np.zeros((20, 2)), speed)
            speed = max(0., speed + DT*(7.5*(cmd["throttle"]-.09)-4*cmd["brake"]))
        self.assertEqual((speed, cmd["throttle"]), (0., 0.))
        jump = CalibratedTracker().request(np.column_stack((20*t, -t)), 3.)
        self.assertEqual(jump["requested_accel_mps2"], 2.)
        self.assertLess(jump["throttle"], .4)
        self.assertGreater(jump["steer"], 0)

    def test_steady_setup_releases_constraint_before_policy_and_excludes_warmup_progress(self):
        import carla
        world, scene, rig, model = Mock(), Mock(), Mock(), Mock()
        scene.ego.type_id = "vehicle.lincoln.mkz"
        scene.route = [carla.Location(), carla.Location(x=60)]
        scene.weather, scene.hazard = "ClearNoon", None
        rig.collisions.return_value = []
        rig.read.return_value = dict.fromkeys(("front", "left", "right"), np.zeros((4,4,3), np.uint8))
        rig.rendering = {}
        snapshot = Mock()
        snapshot.frame = 0
        world.get_snapshot.return_value = snapshot
        def tick():
            snapshot.frame += 1
            snapshot.timestamp.elapsed_seconds = DT*snapshot.frame
            x = max(0, snapshot.frame-48) * .3
            snapshot.find.return_value.get_transform.return_value = carla.Transform(carla.Location(x=x))
            return snapshot.frame
        world.tick.side_effect = tick
        def metrics():
            return dict(route_id="stub", progress_m=(snapshot.frame-48)*.3, route_length_m=60,
                        lateral_error_m=0, phase="clear", hazard_gap_m=60., event_triggered=False,
                        event_cleared=False, red_light_violations=0, yield_violations=0, route_departure=False)
        scene.metrics.side_effect = metrics
        def predict(rgb, hist, intent):
            scene.ego.disable_constant_velocity.assert_called_once()
            np.testing.assert_allclose(hist[:,2], 3., atol=.0001)
            np.testing.assert_allclose(hist[:,4], 0., atol=.0003)
            return np.column_stack((3*np.arange(1,21)/4, np.zeros(20)))
        model.predict.side_effect = predict
        with ExitStack() as stack:
            dest = Path(stack.enter_context(TemporaryDirectory())) / "trial"
            stack.enter_context(patch("crosswalk.runtime.producer_info", return_value={"map":"Town01_Opt"}))
            stack.enter_context(patch("crosswalk.sensors.Rig", return_value=rig))
            stack.enter_context(patch("crosswalk.world.Scenario", return_value=scene))
            summary = run_episode(world, model, "clear", 41, 3., 2, dest, False, "calibrated-v2", "steady")
        scene.ego.enable_constant_velocity.assert_called_once()
        scene.ego.set_target_velocity.assert_not_called()
        self.assertEqual(scene.before_tick.call_count, 2)
        self.assertAlmostEqual(summary["handoff_progress_m"], 14.4)
        self.assertAlmostEqual(summary["model_progress_m"], .6)


if __name__ == "__main__":
    unittest.main()
