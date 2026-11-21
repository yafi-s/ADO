"""Serverless native Town03 route checks, including the vehicle footprint during the pass."""
import math
import os
from pathlib import Path
import unittest
from unittest.mock import Mock

import carla

from crosswalk.urban_map import urban_route


MAP = Path(os.environ.get(
    "CARLA_TOWN03_XODR",
    Path(os.environ.get("CARLA_ROOT", ".local/carla-1.0"))
    / "CarlaUnreal/Packs/AdditionalMaps/Content/Maps/OpenDrive/Town03_Opt.xodr",
))


def lane(wp):
    return wp.road_id, wp.lane_id


def angle(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


@unittest.skipUnless(MAP.is_file(), "Set CARLA_TOWN03_XODR or CARLA_ROOT for native Town03 checks")
class UrbanMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = carla.Map("Town03_Opt", MAP.read_text(encoding="utf8"))
        cls.course = urban_route(cls.map, 17)

    def test_continuous_repeatable_city_itinerary(self):
        for seed in (0, 17, 99):
            route = urban_route(self.map, seed)
            points = route.points
            distances = [p.transform.location.distance(q.transform.location)
                         for p, q in zip(points, points[1:])]
            self.assertGreater(sum(distances), 1400)
            self.assertLess(sum(distances), 1450)
            self.assertGreater(min(distances), 0.73)
            self.assertLess(max(distances), 1.24)
            self.assertEqual((points[0].road_id, points[-1].road_id), (24, 23))
            self.assertEqual(len({p.junction_id for p in points if p.is_junction}), 11)
        duplicate = urban_route(self.map, 17, destination="market")
        self.assertEqual([p.id for p in duplicate.points], [p.id for p in self.course.points])
        self.assertEqual(duplicate.sites, self.course.sites)
        with self.assertRaisesRegex(ValueError, "destination"):
            urban_route(self.map, destination="missing")

    def test_dashed_same_direction_pass_and_full_vehicle_envelope(self):
        c, points = self.course.lane_change, self.course.points
        self.assertEqual(lane(points[c.start - 1]), c.source_lane)
        self.assertEqual(lane(points[c.end]), c.target_lane)
        self.assertGreater(c.bus - c.end, 20)
        self.assertGreater(c.end - c.start, 32)
        lateral = []
        for w in points[c.start:c.end + 1]:
            source = self.map.get_waypoint_xodr(*c.source_lane, w.s)
            target = source.get_left_lane()
            self.assertEqual(lane(target), c.target_lane)
            self.assertEqual(source.left_lane_marking.type, carla.LaneMarkingType.Broken)
            self.assertIn(source.lane_change, (carla.LaneChange.Left, carla.LaneChange.Both))
            self.assertEqual(lane(target.get_right_lane()), c.source_lane)
            self.assertGreater(source.lane_id * target.lane_id, 0)
            self.assertLess(abs(angle(target.transform.rotation.yaw - source.transform.rotation.yaw)), 0.3)
            self.assertLess(abs(angle(w.transform.rotation.yaw - source.transform.rotation.yaw)), 12)
            p, center = w.transform.location, source.transform.location
            left = target.transform.location - center
            lateral.append(((p.x - center.x) * left.x + (p.y - center.y) * left.y)
                           / math.hypot(left.x, left.y))
            # Conservative body footprint, including yaw while crossing the marking.
            forward, right = w.transform.get_forward_vector(), w.transform.get_right_vector()
            for nose in (-2.65, 2.65):
                for edge in (-1.07, 1.07):
                    corner = p + forward * nose + right * edge
                    road = self.map.get_waypoint(corner, project_to_road=False)
                    self.assertIsNotNone(road)
                    self.assertIn(lane(road), (c.source_lane, c.target_lane))
        self.assertAlmostEqual(lateral[0], 0, places=4)
        self.assertAlmostEqual(lateral[-1], 3.5, places=3)
        self.assertTrue(all(-1e-4 <= b - a <= 0.20 for a, b in zip(lateral, lateral[1:])))
        # First and last samples have matched tangents, and the entire car clears
        # the 2.6 m wide stationary bus with both vehicles centered in their lanes.
        self.assertAlmostEqual(lateral[1] - lateral[0], 0, delta=0.005)
        self.assertAlmostEqual(lateral[-1] - lateral[-2], 0, delta=0.005)
        separation = c.bus_pose.location.distance(points[c.bus].transform.location)
        self.assertGreater(separation - 1.3 - 1.07, 1.0)
        self.assertTrue(all(lane(p) == c.target_lane for p in points[c.end:c.bus + 12]))

    def test_encounter_sites_and_crossing_traffic(self):
        r = self.course
        for kind in ("pedestrian", "pedestrian-delayed", "occluded"):
            w = r.points[r.sites[kind]]
            self.assertEqual(w.get_right_lane().lane_type, carla.LaneType.Parking)
        work = r.points[r.sites["roadworks"]:r.sites["roadworks"] + 21]
        self.assertTrue(all(lane(p) == lane(work[0]) and p.lane_width >= 3.4 for p in work))
        self.assertTrue(all(abs(angle(p.transform.rotation.yaw - work[0].transform.rotation.yaw)) < 1
                            for p in work))
        for kind, minimum, maximum in (("cyclist", 30, 37), ("lead", 50, 80), ("braking", 50, 80)):
            index = r.sites[kind]
            point = r.points[index]
            steps = 0
            while steps <= maximum:
                choices = point.next(1.0)
                if len(choices) > 1:
                    break
                self.assertEqual(len(choices), 1)
                point = choices[0]
                steps += 1
            self.assertTrue(minimum <= steps <= maximum)
            self.assertIn(lane(r.points[index + steps + 1]), {lane(p) for p in choices})
        start, end = r.left_turn
        curve = r.points[start:end]
        self.assertLess(angle(curve[-1].transform.rotation.yaw - curve[0].transform.rotation.yaw), -75)
        near, far = r.oncoming
        through = far.next(25)
        self.assertIn((743, -1), {lane(w) for w in through})
        direction = near.transform.location - far.transform.location
        length = math.hypot(direction.x, direction.y)
        projections = []
        for wp in curve:
            pos = wp.transform.location - far.transform.location
            projections.append((pos.y * direction.x - pos.x * direction.y) / length)
        self.assertLess(min(projections), -4)
        self.assertGreater(max(projections), 4)
        nearest = curve[min(range(len(curve)), key=lambda i: abs(projections[i]))]
        pos = nearest.transform.location - far.transform.location
        along = (pos.x * direction.x + pos.y * direction.y) / length
        self.assertTrue(10 < along < length - 10)

    def test_binds_physical_light_and_rejects_other_lane_group(self):
        r = self.course
        start = next(i for i, p in enumerate(r.points) if p.junction_id == 1352)
        stop = r.points[start - 5]
        valid = Mock(id=21)
        valid.get_stop_waypoints.return_value = [stop]
        valid.get_affected_lane_waypoints.return_value = [r.points[start + 2]]
        opposite = Mock(id=5)
        opposite.get_stop_waypoints.return_value = [stop]
        opposite.get_affected_lane_waypoints.return_value = [r.oncoming[0]]
        index, actor, waypoint = r.signal_site([opposite, valid])
        self.assertEqual((index, actor, waypoint), (start - 5, valid, stop))
        with self.assertRaisesRegex(RuntimeError, "No native signal"):
            r.signal_site([opposite])
