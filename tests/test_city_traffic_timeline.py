import unittest

from tools.city_traffic_timeline import compare_by_position, moving_nearby


class CityTrafficTests(unittest.TestCase):
    def test_same_road_link_uses_nearest_position_and_preserves_waiting_frames(self):
        former = {"trial": "city-life", "label": "Previous", "samples": [[10, 95, 8.0], [19, 185, 8.0]]}
        newer = {"trial": "city-progress", "label": "Revised", "samples": [[8, 67, 8.1], [9, 77, 15.0]]}
        route = [[0, 0], [20, 0]]
        logs = {"city-life": {"route": route, "frames": [(8.0, 95, 0), (8.0, 185, 1), (9.2, 188, 5)]},
                "city-progress": {"route": route, "frames": [(7.8, 66, 19), (8.1, 67, 20), (15.0, 77, 23)]}}
        compare_by_position([former, newer], logs)
        self.assertEqual(newer["comparison"]["frames"], [(95, 8.0, 0), None])
        self.assertEqual(former["comparison"]["frames"], [(67, 8.1, 20), (67, 8.1, 20)])
        logs["city-life"]["route"] = [[100, 0], [120, 0]]
        with self.assertRaises(ValueError):
            compare_by_position([former, newer], logs)

    def test_counts_only_continuously_moving_nearby_bodies(self):
        history = []
        for i in range(6):
            actors = {1: ("vehicle", i * 0.25, 0), 2: ("pedestrian", 20, i * 0.11),
                      3: ("pedestrian", 5, 4), 4: ("vehicle", 70, i * 0.3),
                      5: ("vehicle", 4 if i < 3 else 25, 8)}
            if i != 2:
                actors[6] = ("pedestrian", 12, i * 0.1)
            history.append((i / 10, (0, 0), actors))
        self.assertIsNone(moving_nearby(history[:4]))
        self.assertEqual(moving_nearby(history), [1, 1])
