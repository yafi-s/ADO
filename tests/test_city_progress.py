"""Positive and negative controls on the normal City Progress observation and inference path."""
import unittest

import numpy as np

from crosswalk.city_drive import load_driver
from crosswalk.city_progress import FEATURE_NAMES, PROFILE, FlankMotion, ProgressObserver, _profile
from crosswalk.experiment import PROJECT
from crosswalk.overtake import TacticalPass

PATH = np.column_stack(([0., 3., 7., 13., 18., 25., 32., 40.], np.zeros(8)))
NAMES = {name: i for i, name in enumerate(FEATURE_NAMES)}


def pavement():
    x, y = np.meshgrid(np.arange(.5, 31., .38), np.arange(-10., 10., .4))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -2.3), np.ones(x.size)))


def packet(frame, scan):
    return dict(progress_frame=frame, progress_camera_age_s=.1, lidar=scan,
                radar=np.array([[0., 0., -.3, 1.]]),
                front=np.repeat((np.arange(200, dtype=np.uint8)[None, :, None] + 20), 108, axis=0))


def roadrow():
    row = np.zeros(len(FEATURE_NAMES), dtype=float)
    for name in NAMES:
        if ("lidar" in name or "radar" in name or "oncoming" in name) and "speed" not in name:
            row[NAMES[name]] = 36.
    row[[NAMES[n] for n in ("route_3m_x_m", "route_7m_x_m", "route_13m_x_m")]] = [3., 7., 13.]
    row[NAMES["junction_gap"]] = -2.
    row[NAMES["mapped_lamp_gap_m"]] = 60.
    row[NAMES["ego_speed_mps"]] = 4.
    row[NAMES["open_edge_margin_m"]] = -1.
    row[[NAMES[n] for n in PROFILE]] = [24., 24., 0., 0., 3.]
    return row


class ProgressTests(unittest.TestCase):
    def model(self):
        model, cols, _ = load_driver(PROJECT / "models/city-progress/candidate.npz")
        self.assertEqual(cols, FEATURE_NAMES)
        model.new_observer()
        return model

    def test_three_consecutive_visible_packets_after_dark_stale_or_missing(self):
        observer = ProgressObserver()
        def observe(data, detections=()):
            return observer.observe(data, detections, PATH[1:4].ravel(), 4.,
                                    route_preview=PATH[4:].ravel(), stopline_gap=-2.)[-len(PROFILE):]
        scan = pavement()
        for i in range(3):
            result = observe(packet(i, scan))
        self.assertEqual(tuple(result[:2]), (24., 24.))
        self.assertEqual(observe(packet(3, scan), [dict(label="person", box=[.1,.2,.18,.4], score=.9)])[-1], 0)
        self.assertEqual(observe(packet(4, scan))[0], 0)
        self.assertEqual(observe(packet(6, scan))[-1], 1)
        dark = packet(7, scan)
        dark["front"][:] = 0
        self.assertEqual(observe(dark)[-1], 0)
        stale = packet(8, scan)
        stale["progress_camera_age_s"] = 1.
        self.assertEqual(observe(stale)[-1], 0)
        for i in range(9, 12):
            hidden = observe(packet(i, scan[scan[:,1] < 3.]))
        self.assertEqual((hidden[0], hidden[1]), (24., 0.))

    def test_static_flank_requires_fresh_adjacent_lane_and_still_has_blind_end(self):
        scan = pavement()
        scan = scan[scan[:,1]<3.]
        wall = np.array([[x, 2.2, h-2.3, 1.] for x in np.arange(2.3, 11., .2) for h in [.5, 1., 1.8]])
        scene = np.vstack([scan,wall])
        passing = dict(committed=True, source_age_scans=0, source_speed_mps=.1,
                       adjacent_pavement_seen=[True,True],target_front_m=45.,target_rear_m=45.,source_lateral_m=-3.5)
        _, crossing, end, margin = _profile(scene, PATH, passing)
        self.assertEqual(crossing, 0.)
        self.assertGreater(end, 8.)
        self.assertLess(end, 9.)
        self.assertGreater(margin, 1.)
        for override in ({"committed":False},{"source_age_scans":3},{"target_rear_m":4.},{"source_speed_mps":4.}):
            self.assertEqual(_profile(scene, PATH, passing | override)[2],0.)
        person = wall.copy()
        person[:,0] = 4.
        person[:,1] = 1.5
        self.assertLess(_profile(np.vstack([scene,person]), PATH, passing)[2], 2.)

    def test_clear_straight_and_turn_use_different_fitted_speed(self):
        model = self.model()
        x = roadrow()
        model.predict(x)
        straight = model.last_speed_request
        self.assertGreater(straight, 6.)
        self.assertGreater(straight, model.progress_evidence["base_request_mps"])
        model.new_observer()
        x[NAMES["bend_7_13"]] = .5
        model.predict(x)
        self.assertLess(model.last_speed_request, straight - 2.)

    def test_nearest_beam_saturation_requires_continuous_stationary_tip(self):
        track = FlankMotion()
        for frame in range(12):
            travel = frame * .3
            saturated = frame >= 6
            evidence = dict(raw_source_gap_m=2.8 if saturated else 6.,
                            source_speed_mps=3. if saturated else .05)
            self.assertTrue(track.stationary(frame, travel, 14.-travel, 1., evidence))
        # The formerly stationary object now moves with ego and the leading
        # end ceases to shift back relative to ego. Its authority expires.
        for frame in range(12,20):
            still = track.stationary(frame, frame*.3, 10.7, 1., evidence)
        self.assertFalse(still)
        self.assertFalse(track.stationary(30, 9., 5., 1., evidence))

    def test_one_positive_source_pavement_scan_cannot_release_a_wait(self):
        planner = TacticalPass(None, measured_hold=True, confirmed_release=True)
        empty = np.empty(0)
        visible = np.arange(29., 38., .3)
        for _ in range(6):
            planner._source_motion(0., 26., empty)
        for _ in range(7):
            planner._source_motion(0., 60., empty)
        planner._source_motion(0., 60., visible)
        self.assertFalse(planner.source_clear)
        planner._source_motion(0., 26., empty)
        for _ in range(7):
            planner._source_motion(0., 60., empty)
        for expected in (False,False,True):
            planner._source_motion(0., 60., visible)
            self.assertEqual(planner.source_clear, expected)

    def test_vetoes_signed_and_tactical_guards_preserve_full_action(self):
        x = roadrow()
        def compare(row, signed=None, tactical=None):
            candidate = self.model()
            base, _, _ = load_driver(PROJECT / "models/city-clearance/candidate.npz")
            base.new_observer()
            for m in (candidate,base):
                m.signed_speed_limit, m.tactical_speed_limit = signed, tactical
            reference = np.delete(row,[NAMES[n] for n in PROFILE])
            np.testing.assert_allclose(candidate.predict(row), base.predict(reference))
            self.assertAlmostEqual(candidate.last_speed_request, base.last_speed_request)
        for name, value in (("confirmed_scans",0), ("path_body_radar",3.), ("oncoming_gap",10.),
                            ("mapped_lamp_gap_m",10.),("rgb_person_height_recent",.12),("observed_conflict_m",0.)):
            row = x.copy()
            row[NAMES[name]] = value
            compare(row)
        compare(x, signed=2.)
        compare(x,tactical=0.)

if __name__ == "__main__":
    unittest.main()
