"""World-frame projection, matched color classification, and validation accounting."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from crosswalk.lamp import CAMERAS, LampClassifier, LampExpert, calibrate, color_features, project_box, select_crop
from crosswalk.lamp_study import summarize


class LampTest(unittest.TestCase):
    def test_world_vertices_and_camera_axes(self):
        from crosswalk.lamp import head_vertices
        import carla
        box = carla.BoundingBox(carla.Location(110, 3, 4), carla.Vector3D(.1, .1, .5))
        class StaticSignal:
            def get_light_boxes(self):
                return [box]
        world_to_camera = carla.Transform(carla.Location(100, 0, 1)).get_inverse_matrix()
        vertices, = head_vertices(StaticSignal())
        np.testing.assert_allclose(vertices.mean(axis=0), [110, 3, 4], atol=1e-4)
        rect = project_box(vertices, world_to_camera, (360, 640, 3))
        self.assertTrue(rect[0] < 416 < rect[2])
        self.assertTrue(rect[1] < 84 < rect[3])
        self.assertIsNone(project_box(vertices, carla.Transform(carla.Location(120, 0, 1)).get_inverse_matrix(),
                                      (360, 640, 3)))
        images = {name: np.zeros((360, 640, 3), np.uint8) for name in CAMERAS}
        self.assertIsNone(select_crop([], images, dict.fromkeys(CAMERAS, world_to_camera)))

    def test_learned_margins(self):
        red, green = [np.full((15, 11, 3), 10, np.uint8) for _ in range(2)]
        red[2:4, 4:7] = [205, 66, 50]
        green[11:13, 4:7] = [45, 205, 100]
        pair = np.array([color_features(red), color_features(green)])
        classifier = LampClassifier.fit([pair])
        self.assertGreater(classifier.score(pair[0] * .7), 0)
        self.assertLess(classifier.score(pair[1] * .7), 0)

    def test_missing_views_count_and_same_poles_excluded(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for channel, color in ((2, "red"), (1, "green")):
                image = np.zeros((8, 8, 3), np.uint8)
                image[2:4, 3:5, channel] = 255
                cv2.imwrite(str(root / f"{color}.png"), image)
            views = {c: dict(crop=f"{c}.png", camera="front", rect=[0, 0, 8, 8], head=0) for c in ("red", "green")}
            visible = dict(observations=views, actual_gap_m=5.)
            missing = dict(observations={c: dict(crop=None) for c in views}, actual_gap_m=1.)
            for seed, xyz, pairs in [(11, [0, 1, 2], [visible]), (31, [0, 1, 2], [visible]),
                                     (32, [4, 1, 2], [visible, missing])]:
                record = dict(seed=seed, district="train", signal_xyz=xyz, pairs=pairs, wall_seconds=.1)
                (root / f"seed-{seed}.json").write_text(json.dumps(record))
            summary = summarize(root)
            self.assertEqual(summary["excluded_shared_pole_seeds"], [31])
            self.assertEqual(summary["balanced_accuracy_all_pairs"], .5)
            self.assertFalse(summary["meets_probe_criterion"])

    def test_actuators_follow_demonstrated_controls_and_retain_base_when_unmapped(self):
        classifier = LampClassifier(np.zeros(4), np.array([1., 0., 0., 0.]))
        calibration = calibrate([-1., -.8, .8, 1.], [0, 0, 1, 1])
        # Deliberately supply reverse demonstrations to establish that red does not
        # invoke a hand-written brake action in the expert.
        expert = LampExpert(classifier, calibration, [[0., 5, 4.], [5., 5, 4.]], [-.7, .6])
        base = (.17, .23, 0.)
        self.assertEqual(expert.apply(None, 2., base), base)
        red, green = np.array([1., 0., 0., 0.]), np.array([-1., 0., 0., 0.])
        self.assertGreater(expert.apply((5., red), 2., base)[1], .45)
        self.assertGreater(expert.apply((5., green), 2., base)[2], .5)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "expert.npz"
            expert.save(path)
            np.testing.assert_allclose(LampExpert.load(path).apply((5., red), 2., base),
                                       expert.apply((5., red), 2., base))

    def test_mixture_history_is_local_to_current_approach_and_episode(self):
        classifier = LampClassifier(np.zeros(4), np.array([1., 0., 0., 0.]))
        expert = LampExpert(classifier, [7., 0.], [[0., 5., 4.], [5., 5., 4.]], [.8, .8])
        first, second = expert.for_episode(), expert.for_episode()
        red, green = (5., np.array([1., 0., 0., 0.])), (5., np.array([-1., 0., 0., 0.]))
        base = (.12, .1, 0.)
        self.assertGreater(first.apply(red, 2., base)[1], .78)
        self.assertGreater(first.apply(green, 2., base)[1], .78)
        self.assertLess(second.apply(green, 2., base)[1], .12)
        self.assertEqual(first.apply(None, 2., base), base)
        self.assertLess(first.apply(green, 2., base)[1], .12)
