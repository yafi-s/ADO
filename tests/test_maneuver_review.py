import unittest

from tools.publish_maneuver_review import event_frame


class ManeuverReviewTests(unittest.TestCase):
    def test_link_uses_first_scored_camera_frame_even_when_event_clock_is_a_tick_ahead(self):
        event = {"kind": "slow-traffic", "returned_to_route_s": 4.6}
        rows = [
            {"frame": 119, "sim_time": 4.4, "events": [{"kind": "in-lane-obstruction", "returned_to_route_s": 4.6}]},
            {"frame": 120, "sim_time": 4.5, "events": [event]},
            {"frame": 121, "sim_time": 4.6, "events": [event]},
        ]
        self.assertEqual(event_frame(rows, event, "returned_to_route_s"), 120)
        with self.assertRaisesRegex(ValueError, "Missing native exposure"):
            event_frame(rows[:1], event, "returned_to_route_s")
