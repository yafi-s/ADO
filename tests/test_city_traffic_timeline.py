import unittest

from tools.city_traffic_timeline import moving_nearby


class CityTrafficTests(unittest.TestCase):
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
