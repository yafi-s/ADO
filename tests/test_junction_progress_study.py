"""Reference geometry tests for the independently scored cross-traffic transfer."""
import unittest

import numpy as np

from crosswalk.junction_progress_study import grade, station_speeds
from crosswalk.grade_transfer_study import score as hill_score


def box(x, y, hx, hy):
    return [[x-hx, y-hy], [x+hx, y-hy], [x+hx, y+hy], [x-hx, y+hy]]


class JunctionGradeTests(unittest.TestCase):
    def paired(self):
        clear = []
        occupied = []
        for tick in range(151):
            t = tick / 10.
            free_x = 3. * (t - 5.)
            cross_y = 3. * (t - 5.)
            yield_x = 3. * (t - 10.)
            ego = box(yield_x, 0., 2., .9)
            traffic = box(0., cross_y, .9, 2.)
            clear.append(dict(sim_time=t, ego_xy=box(free_x, 0., 2., .9)))
            occupied.append(dict(sim_time=t, ego_xy=ego, traffic_xy=traffic,
                                 traffic_station_m=cross_y, gap_m=float(np.hypot(max(abs(yield_x)-2.9, 0.),
                                                                               max(abs(cross_y)-2.9, 0.)))))
        return clear, occupied, dict(axes=[[1., 0.], [0., 1.]], center_xy=[0., 0.],
                                     collision_events=0, outcome="complete")

    def test_positive_requires_unsafe_counterfactual_and_actual_separation(self):
        clear, traffic, summary = self.paired()
        score = grade(clear, traffic, summary)
        self.assertTrue(score["passed"])
        self.assertEqual(0., score["counterfactual_min_body_gap_m"])
        self.assertGreater(score["post_encroachment_s"], 1.)

    def test_unthreatened_clear_trajectory_is_inconclusive(self):
        clear, traffic, summary = self.paired()
        for a,b in zip(clear, traffic):
            a["ego_xy"] = b["ego_xy"]
        self.assertFalse(grade(clear, traffic, summary)["passed"])

    def test_near_miss_and_physical_collision_fail_independently(self):
        clear, traffic, summary = self.paired()
        traffic[40]["gap_m"] = .5
        self.assertTrue(grade(clear, traffic, summary)["near_miss"])
        self.assertFalse(grade(clear, traffic, summary)["passed"])
        traffic[40]["gap_m"] = 5.
        summary["collision_events"] = 1
        self.assertFalse(grade(clear, traffic, summary)["passed"])

    def test_offset_timestamps_fail(self):
        clear, traffic, summary = self.paired()
        traffic[10]["sim_time"] += .1
        with self.assertRaises(ValueError):
            grade(clear, traffic, summary)

    def test_station_speed_uses_applied_control_and_local_bracketing(self):
        trace = [dict(frame=100, ego_station_m=-1., sim_time=1., speed_mps=4., speed_request_mps=2.),
                 dict(frame=101, ego_station_m=1., sim_time=1.1, speed_mps=2., speed_request_mps=0.)]
        samples = station_speeds(trace)
        self.assertEqual(set(samples), {"0.0"})
        self.assertEqual(samples["0.0"]["time_s"], 1.05)
        self.assertEqual(samples["0.0"]["actual_kmh"], 10.8)
        self.assertEqual(samples["0.0"]["applied_target_kmh"], 7.2)

    def test_hill_obstacle_needs_sustained_close_stop_and_stationary_car(self):
        rows = [dict(frame=i, sim_time=.1*i, speed_mps=0., speed_request_mps=0.,
                     ego_station_m=15., parked_grade_station_m=25., parked_origin_xyz=[0.,0.,0.], gap_m=4.)
                for i in range(30)]
        summary = dict(outcome="restarted", collision_events=0)
        self.assertTrue(hill_score(rows, summary, True)["passed"])
        self.assertFalse(hill_score(rows[:-1], summary, True)["passed"])
        self.assertFalse(hill_score(rows, dict(summary, outcome="lane_departure"), True)["passed"])
        rows[-1]["parked_origin_xyz"] = [1.,0.,0.]
        self.assertFalse(hill_score(rows, summary, True)["passed"])


if __name__ == "__main__":
    unittest.main()
