import unittest
from crosswalk.workzone_study import longest_stop


class WorkzoneScoringTests(unittest.TestCase):
    def test_stop_requires_continuous_dwell_at_a_board_in_lane(self):
        stopped = dict(board_in_ego_path=True, bumper_gap_m=3., progress_m=51., speed_mps=.02)
        rows = [{**stopped, "progress_m":.1}]*8 + [stopped]*7 + [{**stopped, "board_in_ego_path":False}] + [stopped]*5
        self.assertEqual(longest_stop(rows), .7)
        self.assertEqual(longest_stop([{**stopped, "board_in_ego_path":False}]*12), 0.)
        self.assertEqual(longest_stop([{**stopped, "bumper_gap_m":35.}]*12), 0.)
