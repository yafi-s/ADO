import numpy as np

from crosswalk.behavior_v2 import FEATURE_NAMES, HORIZON
from waymo_training.carla_speed_calibration import COL, CompactTracks, inputs


def measurement(speed, gap):
    x = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    x[COL["ego_speed_mps"]] = speed
    for name in ("path_body_lidar", "left_edge_lidar", "right_edge_lidar"):
        x[COL[name]] = HORIZON
    x[COL["path_body_lidar"]] = gap
    return x


def test_history_requires_a_full_second_and_keeps_current_speed():
    tracks = CompactTracks()
    assert all(tracks.observe(measurement(i * .1, 8.)) is None for i in range(10))
    motion = tracks.observe(measurement(1., 8.))
    np.testing.assert_allclose(motion[:3], (1., 1., 1.))
    assert motion[3] == motion[7] == 0.


def test_camera_confirmation_and_measured_relative_velocity():
    tracks = CompactTracks()
    for i in range(11):
        x = measurement(2., 10. - i * .2)
        x[COL["rgb_vehicle_height"]] = .14
        motion = tracks.observe(x)
    assert motion[3] == 1.
    assert abs(motion[6] + 2.) < 1e-4
    x[COL["path_body_lidar"]] = HORIZON
    assert tracks.observe(x)[3] == 0.


def test_waymo_arm_appends_only_the_frozen_forecasts():
    class Prior:
        def predict(self, motion):
            return np.array([2., 3., 4.])

    x = measurement(2., 8.)
    motion = np.zeros(15)
    without = inputs(x, motion)
    with_prior = inputs(x, motion, Prior())
    np.testing.assert_array_equal(with_prior[0, :-3], without[0])
    np.testing.assert_array_equal(with_prior[0, -3:], [2., 3., 4.])
