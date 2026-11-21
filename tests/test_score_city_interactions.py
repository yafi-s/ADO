"""Geometry regressions for the offline city-drive audit."""

import math
import unittest

from tools.score_city_interactions import Body, Clearance, box_for, front_observations, rotation, signed_gap


def body(x, y=0., z=0.):
    return Body((x, y, z), rotation(0., 0., 0.), (2., 2., 2.), (0., 0., 0.), (0., 0., 0.))


class InteractionGeometryTests(unittest.TestCase):
    def test_body_rotates_offset_and_composes_box_rotation(self):
        actor = {"x": 3., "y": 5., "z": 0., "yaw": 90., "dimensions_m": [4., 2., 2.],
                 "bbox_offset_m": [1., 0., 2.], "bbox_rotation_deg": [0., 90., 0.]}
        box, method = box_for(actor, {})
        self.assertEqual(method, "native_actor_box")
        for measured, expected in zip(box.center, (3., 6., 2.)):
            self.assertAlmostEqual(measured, expected)
        self.assertAlmostEqual(max(x for x, _ in box.polygon), 5.)
        self.assertAlmostEqual(max(y for _, y in box.polygon), 7.)

    def test_planar_distance_measures_corner_gap_and_penetration(self):
        origin = body(0.)
        self.assertAlmostEqual(signed_gap(origin.polygon, body(3., 3.).polygon), math.sqrt(2))
        self.assertAlmostEqual(signed_gap(origin.polygon, body(1.5).polygon), -.5)
        self.assertAlmostEqual(signed_gap(origin.polygon, body(2.).polygon), 0.)
        small = Body((0., 0., 0.), rotation(0., 0., 0.), (1., 1., 1.), (0., 0., 0.), (0., 0., 0.))
        self.assertAlmostEqual(signed_gap(origin.polygon, small.polygon), -1.5)

    def test_elevation_and_unknown_shape_do_not_count_as_planar_contact(self):
        audit = Clearance()
        audit.observe({"frame": 1}, 0., body(0.), body(0., z=3.), "known")
        audit.observe({"frame": 2}, .2, body(0.), None, "actor_origin_only")
        audit.observe({"frame": 3}, 2.3, body(0.), body(2.3), "known")
        report = audit.report()
        self.assertEqual(report["closest_actor_origin"]["frame"], 1)
        self.assertEqual(report["at_or_below_m"]["0.0"], 0)
        self.assertAlmostEqual(report["smallest_xy_with_disjoint_elevation"]["vertical_gap_m"], 1.)
        self.assertAlmostEqual(report["closest_overlapping_elevation"]["ground_projection_gap_m"], .3)
        self.assertIsNone(report["closest_moving_actor"])

    def test_paired_calibration_requires_matching_recorded_dimensions(self):
        actor = {"x": 0., "y": 0., "z": 0., "yaw": 0., "dimensions_m": [4., 2., 2.], "blueprint": "vehicle.test"}
        replacement = {"dimensions_m": [4.01, 2., 2.], "offset_xyz": [0., 0., 0.],
                       "rotation": [0., 0., 0.], "matched_capture": "test-capture"}
        self.assertEqual(box_for(actor, {"vehicle.test": replacement}), (None, "actor_origin_only"))
        replacement["dimensions_m"][0] = 4.
        _, method = box_for(actor, {"vehicle.test": replacement})
        self.assertEqual(method, "matched_capture_blueprint_yaw_only")

    def test_moving_minimum_is_independent_of_closest_stationary_sample(self):
        audit = Clearance()
        audit.observe({"frame": 1}, 2.1, body(0.), body(2.1), "known")
        audit.observe({"frame": 2, "actor_moving": True}, 17., body(0.), body(17.), "known")
        self.assertEqual(audit.report()["closest_moving_actor"]["frame"], 2)

    def test_front_requires_ranged_noncoasting_unique_actor_and_track(self):
        bounds = [.4, .4, .6, .6]
        box = {"label": "car", "score": .9, "track_id": 8, "range_supported": True, "box": bounds}
        track = {"label": "car", "x": 20.1, "y": .1, "id": 8, "cameras": ["front"],
                 "lidar_hits": 2, "coasting": False}
        actor = {"type": "car"}
        actor_info = {"actor_id": 1, "local_xy_m": (20., 0.), "center_error_gate_m": 2.}
        targets = [(actor, actor_info, bounds)]
        self.assertEqual(len(front_observations({"tracks": [track]}, targets, [box, box])), 1)
        second = (actor, {**actor_info, "actor_id": 2}, bounds)
        self.assertFalse(front_observations({"tracks": [track]}, targets + [second], [box]))
        self.assertFalse(front_observations({"tracks": [{**track, "coasting": True}]}, targets, [box]))
        self.assertFalse(front_observations({"tracks": [track]}, targets, [{**box, "range_supported": False}]))


if __name__ == "__main__":
    unittest.main()
