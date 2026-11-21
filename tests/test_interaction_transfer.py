import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from crosswalk import behavior_v2, policy
from waymo_training import interaction
from waymo_training.interaction_transfer import MotionTransfer, SensorTracks, SpeedServo, route_bend


EMPTY = np.empty((0, 4))
STRAIGHT = np.array([[x, 0.] for x in (3., 7., 13., 18., 25., 32., 40.)])
ROOT = Path(__file__).resolve().parents[1]


def person(x, y):
    # CARLA frame, RGB box and independent roof-lidar returns from one person.
    depth = x - 1.3
    det = dict(label="person", score=.9,
               box=[.5+(y-.28)/(2*depth), .5-320*.1/(360*depth),
                    .5+(y+.28)/(2*depth), .5+320*1.65/(360*depth)])
    cloud = np.array([[x, y+dy, z-2.3, .9] for dy in (-.19, 0., .19) for z in (.4, .8, 1.3, 1.65)])
    return dict(lidar=cloud, radar=EMPTY), [det]


class Forecast:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, x):
        self.calls.append(x.copy())
        return np.asarray(self.result)


class Pilot:
    def new_observer(self):
        return behavior_v2.Observer()

    def predict(self, x):
        return .27, .41, 0.


class TransferTests(unittest.TestCase):
    def test_route_uses_lane_tangents_and_waymo_handedness(self):
        points = STRAIGHT.copy()
        points[:, 1] += 1.5
        self.assertEqual(route_bend(points[:3], points[3:]), (0., True))
        points[3:, 1] += 5.
        bend, valid = route_bend(points[:3], points[3:])
        self.assertTrue(valid)
        self.assertAlmostEqual(bend, -math.pi/4)
        points[3] = points[2]
        self.assertFalse(route_bend(points[:3], points[3:])[1])

    def test_tracking_uses_two_fresh_world_positions_and_correct_velocity_sign(self):
        tracker = SensorTracks()
        # Ego drives forward at 5 m/s facing CARLA yaw 90°. Ped moves left at 2 m/s.
        obs, boxes = person(11.3, .4)
        first, counts = tracker.read(obs, boxes, (0., 0., 90.), 0.)
        self.assertEqual((first, counts["sensor_tracks"]), ([], 1))
        obs, boxes = person(10.3, 0.)
        updated, _ = tracker.read(obs, boxes, (0., 1., 90.), .2)
        self.assertEqual(len(updated), 1)
        actor = updated[0]
        self.assertEqual(actor.kind, "pedestrian")
        self.assertAlmostEqual(actor.vx, 0., delta=.08)
        self.assertAlmostEqual(actor.vy, 2., delta=.14)
        coasting, counts = tracker.read(obs, (), (0., 1.5, 90.), .3)
        self.assertEqual((len(coasting), counts["coasting_velocity_tracks"]), (1, 1))
        self.assertAlmostEqual(coasting[0].vy, actor.vy)
        expired, counts = tracker.read(obs, (), (0., 2., 90.), .8)
        self.assertEqual((expired, counts["sensor_tracks"], tracker.samples), ([], 0, {}))

    def test_servo_combines_horizons_and_bounds_signs(self):
        out = SpeedServo().request([6., 7., 8.], 5., .1)
        self.assertGreater(out[0], 0.)
        self.assertEqual(out[1:], (0., 1.))
        self.assertLess(SpeedServo().request([5., 4., 2.], 5., .1)[2], 0.)
        self.assertEqual(SpeedServo().request([0., 0., 0.], 0., .1), (0., 0., 0.))
        for target in ([0., 0., 0.], [500., 500., 500.]):
            throttle, brake, accel = SpeedServo().request(target, 10., .1)
            self.assertTrue(0 <= throttle <= 1 and 0 <= brake <= 1)
            self.assertEqual(throttle * brake, 0.)
            self.assertTrue(-4. <= accel <= 2.)
        with self.assertRaises(ValueError):
            SpeedServo().request([0., np.nan, 1.], 2., .1)

    def test_missing_supported_tracks_do_not_assert_clear_road(self):
        motion, marginal = Forecast([1., 1., 1.]), Forecast([3., 4., 5.])
        for fallback in (None, marginal):
            model = MotionTransfer(motion, fallback, Pilot())
            observer = model.new_observer()
            obs, boxes = person(10., .5)
            obs["lidar"] = EMPTY  # Real RGB result, unsupported by range data.
            for step in range(11):
                out = observer.observe_and_predict(obs, boxes, STRAIGHT[:3], 1+step*.1,
                    route_preview=STRAIGHT[3:], ego_pose=(0., 0., 0.), sim_time=step*.1,
                    detection_fresh=step%2==0)
                if step < 10:
                    self.assertEqual(out, (.27, 0., 1.))
                    self.assertIsNone(observer.last_info["forecast_mps"])
            json.dumps(observer.last_info, allow_nan=False)
            if fallback is None:
                self.assertEqual(out, (.27, .41, 0.))
                self.assertEqual(observer.last_info["forecast_source"], "ado_sensor_fallback")
            else:
                self.assertEqual(observer.last_info["forecast_source"], "history_only")
                np.testing.assert_allclose(marginal.calls[-1][:3], [2., 1., 1.])
                self.assertGreater(out[1], 0.)
            self.assertFalse(motion.calls)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                observer.observe_and_predict(obs, boxes, STRAIGHT[:3], 2., route_preview=STRAIGHT[3:],
                    ego_pose=(0., 0., 0.), sim_time=1.3, detection_fresh=False)

    def test_measured_vru_activates_motion_and_no_pedals_are_taken_from_steering(self):
        motion = Forecast([1., .5, 0.])
        model = MotionTransfer(motion, None, Pilot())
        observer = model.new_observer()
        for step in range(12):
            obs, boxes = person(12.-.2*step, .8-.02*step)
            steer, throttle, brake = observer.observe_and_predict(obs, boxes, STRAIGHT[:3], 2.,
                route_preview=STRAIGHT[3:], ego_pose=(.2*step, 0., 0.), sim_time=step*.1,
                detection_fresh=step%2==0)
        self.assertEqual(steer, .27)
        self.assertEqual(throttle, 0.)
        self.assertGreater(brake, 0.)
        self.assertEqual(len(motion.calls), 2)
        self.assertEqual(observer.last_info["forecast_source"], "interaction")
        self.assertEqual(observer.last_info["coasting_velocity_tracks"], 1)
        self.assertEqual(observer.last_info["forecast_mps"], [1., .5, 0.])
        self.assertTrue(motion.calls[-1][7])
        self.assertLess(motion.calls[-1][9], 0.)
        self.assertGreater(motion.calls[-1][11], 0.)
        self.assertEqual(model.new_observer().last_info, {})

    def test_checkpoint_contract_and_legacy_steering(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "motion.npz"
            n = interaction._basis(np.zeros(len(interaction.FEATURE_NAMES)), "interaction").shape[1]
            interaction.MotionSpeedModel("interaction", np.zeros((n, 3))).save(p)
            model = MotionTransfer.load(p, ROOT/"models"/"fusion.npz")
            obs = dict(lidar=EMPTY, radar=EMPTY, **{k: np.zeros((64, 64, 3), np.uint8)
                                                   for k in ("front", "left", "right")})
            local = STRAIGHT[:3].copy()
            local[:, 1] += [0., 1., 3.]
            x = policy.features(obs["front"], EMPTY, EMPTY, 0., [], local.ravel(),
                                side_views=(obs["left"], obs["right"]))
            pred = model.steering.predict(x)
            observer = model.new_observer()
            for i in range(11):
                transfer = observer.observe_and_predict(obs, [], local.ravel(), 0., route_preview=STRAIGHT[3:],
                                        ego_pose=(0., 0., 0.), sim_time=i*.1, detection_fresh=i%2==0)
            np.testing.assert_allclose(transfer, pred)
            with self.assertRaisesRegex(ValueError, "history-only"):
                MotionTransfer.load(p, ROOT/"models"/"fusion.npz", history_path=p)


if __name__ == "__main__":
    unittest.main()
