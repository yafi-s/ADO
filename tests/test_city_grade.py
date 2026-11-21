"""Raw-sensor controls for clearing the native plane's false uphill returns."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from crosswalk.behavior_v2 import _range_cues
from crosswalk.city_grade import GradeModel, GradeObserver, supported_grade
from crosswalk.city_progress import ProgressModel
from crosswalk.city_progress_fit import fit
from crosswalk.policy import BUMPER_X, LIDAR_Z, RADAR_X, RADAR_Z


PATH = np.column_stack(([0., 3., 7., 13., 18., 25., 32., 40.], np.zeros(8)))
SPEED = 7.2


def uphill(x):
    """A five-metre vertical curve into a steady 9% uphill grade."""
    approach = np.maximum(0., np.asarray(x) - 11.)
    return .009 * np.minimum(approach, 5.)**2 + .09 * np.maximum(approach - 5., 0.)


def level(x):
    return np.zeros_like(x, dtype=float)


def lidar_returns(x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel() - LIDAR_Z,
                            np.full(x.size, .8))).astype(np.float32)


def radar_returns(x, y, z, *, motion=0.):
    x, y, z = np.broadcast_arrays(x, y, z)
    dx, dy, dz = x - RADAR_X, y, z - RADAR_Z
    azimuth = np.arctan2(dy, dx)
    altitude = np.arctan2(dz, np.hypot(dx, dy))
    depth = np.sqrt(dx**2 + dy**2 + dz**2)
    radial = motion - SPEED * dx / depth
    # Use the production front radar's field of view and a level mounting pose.
    visible = (np.abs(azimuth) <= np.deg2rad(18.)) & (np.abs(altitude) <= np.deg2rad(4.))
    return np.column_stack((radial[visible], azimuth[visible], altitude[visible],
                            depth[visible])).astype(np.float32)


def road_packets(surface=uphill):
    # Dense pavement samples in the path and both shoulders. Small deterministic
    # variation in lidar height keeps the fixture from relying on exact agreement.
    x, y = np.meshgrid(np.arange(4.4, 32., .27), np.arange(-2.9, 3., .22))
    lidar = lidar_returns(x, y, surface(x) + .008 * np.sin(2.7 * x + 3.1 * y))
    x, y = np.meshgrid(np.arange(12.5, 31., .63), [-1.25, -.6, .15, .85, 1.5])
    radar = radar_returns(x, y, surface(x))
    return lidar, radar


class GradeTests(unittest.TestCase):
    def setUp(self):
        self.lidar, self.radar = road_packets()

    def bounded_by(self, lidar, radar, x):
        clearance = supported_grade(lidar, radar, PATH, SPEED)
        self.assertGreaterEqual(clearance, 0.)
        self.assertLessEqual(clearance, x - BUMPER_X)

    def test_smooth_clear_grade_recovers_lidar_and_radar_false_hits(self):
        _, lidar_gap, radar_gap, *_ = _range_cues(self.lidar, self.radar, SPEED, PATH)
        for gap in (lidar_gap, radar_gap):
            self.assertGreater(gap, 17.)
            self.assertLess(gap, 25.)
        clearance = supported_grade(self.lidar, self.radar, PATH, SPEED)
        self.assertGreater(clearance, max(lidar_gap, radar_gap))
        self.assertLessEqual(clearance, float(self.lidar[:, 0].max() - BUMPER_X))

    def test_level_road_does_not_issue_grade_permission(self):
        self.assertEqual(supported_grade(*road_packets(level), PATH, SPEED), 0.)

    def test_missing_lidar_or_radar_does_not_issue_grade_permission(self):
        empty = np.empty((0, 4), dtype=np.float32)
        self.assertEqual(supported_grade(empty, self.radar, PATH, SPEED), 0.)
        self.assertEqual(supported_grade(self.lidar, empty, PATH, SPEED), 0.)

    def test_low_in_lane_cone_and_board_cannot_be_smoothed_into_the_road(self):
        start = 23.15
        cone_x = start + np.array([0., .06, .10, .12, .16])
        cone_y = np.array([.18, .08, .28, .16, .20])
        cone = lidar_returns(cone_x, cone_y, uphill(cone_x) + [.16, .20, .21, .27, .19])
        board_x = start + np.linspace(0., .12, 8)
        board = lidar_returns(board_x, np.linspace(-.75, .75, 8), uphill(board_x) + .23)
        for name, object_returns in (("cone", cone), ("board", board)):
            with self.subTest(object=name):
                self.bounded_by(np.vstack((self.lidar, object_returns)), self.radar, start)

    def test_lidar_vehicle_body_limits_clearance(self):
        nose = 17.1
        y, h = np.meshgrid(np.linspace(-.82, .82, 12), [.45, .82, 1.30])
        car = lidar_returns(nose, y, uphill(nose) + h)
        floor_visible = ((self.lidar[:, 0] < nose) | (self.lidar[:, 0] > nose + 4.)
                         | (abs(self.lidar[:, 1]) > .9))
        scene = np.vstack((self.lidar[floor_visible], car))
        self.bounded_by(scene, self.radar, nose)

    def test_moving_doppler_rejects_an_apparently_road_height_return(self):
        x = 23.7
        stationary = radar_returns(x, .25, uphill(x))
        self.assertGreater(supported_grade(self.lidar, np.vstack((self.radar, stationary)), PATH, SPEED),
                           x - BUMPER_X)
        for motion in (-3.1, 3.1):
            with self.subTest(motion=motion):
                moving = radar_returns(x, .25, uphill(x), motion=motion)
                self.assertEqual(len(moving), 1)
                self.bounded_by(self.lidar, np.vstack((self.radar, moving)), x)

    def test_both_shoulders_and_the_center_require_independent_support(self):
        y = self.lidar[:, 1]
        for name, keep in (("left", y >= -1.08), ("right", y <= 1.08), ("center", abs(y) >= 1.10)):
            with self.subTest(band=name):
                self.assertEqual(supported_grade(self.lidar[keep], self.radar, PATH, SPEED), 0.)

    def test_distant_returns_do_not_bridge_an_unseen_section(self):
        start, end = 14.2, 20.5
        x = self.lidar[:, 0]
        self.bounded_by(self.lidar[(x < start) | (x > end)], self.radar, start)

    def test_elevated_surface_after_an_abrupt_step_is_not_clear_grade(self):
        edge = 19.8

        def stepped(x):
            return uphill(x) + .54 * (x >= edge)

        lidar, radar = road_packets(stepped)
        self.bounded_by(lidar, radar, edge)

    def test_excessive_cross_slope_and_unseen_back_of_crest(self):
        banked = self.lidar.copy()
        banked[:, 2] += .14 * banked[:, 1]
        self.assertEqual(supported_grade(banked, self.radar, PATH, SPEED), 0.)
        # A rising ribbon terminating at a crest supplies no surface evidence
        # on the hidden side. The radar can still place a return on its near edge.
        crest = 25.
        self.bounded_by(self.lidar[self.lidar[:,0] <= crest],self.radar,crest)

    def test_near_in_path_hazard_inhibits_grade_permission(self):
        nose = 6.5
        y = np.linspace(-.6, .6, 8)
        lidar_obstacle = lidar_returns(nose, y, .8)
        radar_obstacle = radar_returns(nose, y, .8)
        self.assertGreater(len(radar_obstacle), 0)
        self.assertEqual(supported_grade(np.vstack((self.lidar, lidar_obstacle)),
                                         np.vstack((self.radar, radar_obstacle)), PATH, SPEED), 0.)

    def test_near_low_board_precedes_first_pavement_fit_center(self):
        # A close board remains visible to lidar but sits below the old plane's
        # obstacle threshold and outside the radar's downward field of view.
        board = lidar_returns(4.6, np.linspace(-.5, .5, 8), .12)
        scene = np.vstack((self.lidar, board))
        self.assertEqual(len(radar_returns(4.6, 0., .12)), 0)
        self.assertGreater(_range_cues(scene, self.radar, SPEED, PATH)[1], 17.)
        self.assertEqual(supported_grade(scene, self.radar, PATH, SPEED), 0.)

    def test_unexplained_recent_body_return_must_age_out(self):
        from tests.test_city_progress import packet
        obs = packet(100, self.lidar)
        obs["radar"] = self.radar
        observer = GradeObserver()
        def observe(frame, support):
            obs["progress_frame"] = frame
            with patch("crosswalk.city_grade.supported_grade", return_value=support):
                return observer.observe(obs,[],PATH[1:4].ravel(),SPEED,route_preview=PATH[4:].ravel(),
                                        stopline_gap=-2.)[-1]
        self.assertEqual(observe(100,0.),0.)
        for frame in range(101,109):
            self.assertEqual(observe(frame,29.),0.)
        self.assertEqual(observe(109,29.),29.)
        self.assertEqual(observe(111,29.),0.)

    def test_only_clearance_veto_is_reconsidered(self):
        from tests.test_city_progress import roadrow, NAMES
        root = Path(__file__).resolve().parents[1]
        coeff, _ = fit(root/"data/city-cruise/lessons.zip")
        parent = (root/"models/city-clearance/candidate.npz").read_bytes()
        row = roadrow().astype(np.float32)
        for key in ("path_body_radar","path_body_lidar","path_body_lidar_recent","path_body_radar_recent"):
            row[NAMES[key]] = 21.
        for label, change in (("clear",{}),("unknown",{}),("person",{"rgb_person_height":.14}),
                              ("signal",{"mapped_lamp_gap_m":12.}),("crossing",{"crossing_stop_gap_m":2.})):
            with self.subTest(label=label):
                prior, revised = ProgressModel(parent,coeff), GradeModel(parent,coeff)
                prior.new_observer()
                revised.new_observer()
                x = row.copy()
                for key,value in change.items():
                    x[NAMES[key]] = value
                extent = 0. if label=="unknown" else 28.
                a = prior.predict(x)
                b = revised.predict(np.r_[x[:-2],np.array([extent,extent],dtype=x.dtype),x[-2:]])
                if label=="clear":
                    self.assertGreater(revised.last_speed_request,prior.last_speed_request+2.)
                else:
                    np.testing.assert_allclose(a,b)


if __name__ == "__main__":
    unittest.main()
