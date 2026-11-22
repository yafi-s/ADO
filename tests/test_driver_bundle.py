"""The canonical stationary-vehicle driver must reproduce the natively qualified research harness."""
import hashlib
import inspect
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import Mock

import carla
import numpy as np

from crosswalk import drive
from crosswalk.city_drive import load_driver
from crosswalk.maneuver_choice import ManeuverChoice
from crosswalk.motor_queue import MotorQueue, VehicleQueuePlanner
from crosswalk.stationary_intent import StationaryIntent

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models/city-maneuvers/stationary-driver.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BundleTests(unittest.TestCase):
    def test_bundle_selects_the_qualified_components(self):
        driver, names, schema = load_driver(BUNDLE)
        self.assertEqual(schema, "ado-city-maneuvers-v1")
        self.assertEqual(driver.driver_bundle_sha256, digest(BUNDLE))
        self.assertEqual(driver.pace_sha256, "12257634d76f30bdcdfb7a61b34c1cddecbc5713ee88720631bc9bb7b8af1071")
        self.assertEqual(driver.maneuver_choice_sha256, "52e04b5ab64550e5f8116e51f8bd3e9614e9740bc8379e0b942b3f2378c6a624")
        self.assertEqual(driver.stationary_intent_sha256, "997f3f8774fff55790f163249dae6e6d50efb26a85e55712e8c5b965eee560b3")
        self.assertTrue(driver.uses_route_maneuvers)
        plain, plain_names, _ = load_driver(ROOT / "models/city-maneuvers/candidate.npz")
        self.assertEqual(list(names), list(plain_names))
        self.assertFalse(hasattr(plain, "stationary_intent_model"))
        reference = ManeuverChoice()
        rng = np.random.default_rng(4)
        for row in rng.uniform([0, -2, 0, 0, -2, 0, -80], [60, 12, 60, 60, 12, 20, 80], (400, 7)):
            self.assertEqual(driver.maneuver_choice_model.predict(row)[0], reference.predict(row)[0])

    def test_changed_or_escaping_component_is_refused(self):
        definition = json.loads(BUNDLE.read_bytes())
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "route-maneuvers").mkdir()
            (folder / "pin").mkdir()
            for name in ("candidate.npz", "stationary-intent.npz"):
                shutil.copyfile(BUNDLE.parent / name, folder / "pin" / name)
            shutil.copyfile(ROOT / "models/route-maneuvers/choice.json", folder / "route-maneuvers/choice.json")
            target = folder / "pin/stationary-driver.json"
            target.write_text(json.dumps(definition))
            load_driver(target)
            for change in (("pace", "sha256", "0" * 64), ("stationary_intent", "sha256", "0" * 64),
                           ("maneuver_choice", "path", str((folder / "route-maneuvers/choice.json").resolve()))):
                bad = json.loads(json.dumps(definition))
                bad[change[0]][change[1]] = change[2]
                target.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_driver(target)


def _research_queue():
    from tools.maneuver_motor_queue_candidate import MotorQueue as Research
    return Research()


def _sighting(frame, speed):
    return dict(box_frame=frame, label="car", radial_ground_speed_mps=speed, stationary_doppler=abs(speed) < .55)


class CanonicalQueueTests(unittest.TestCase):
    def test_learned_queue_matches_the_research_harness_step_for_step(self):
        head = StationaryIntent()
        for seed in range(12):
            rng = random.Random(seed)
            canonical, research = MotorQueue(), _research_queue()
            source, ego_station, source_v = 60. + 30. * rng.random(), 0., 0.
            for frame in range(260):
                phase = (frame // 40 + seed) % 4
                source_v = (0. if phase in (0, 2) else 1.6 if phase == 1 else min(4., source_v + .3))
                source += .1 * source_v + (rng.random() - .5) * .04
                ego_v = max(0., min(8., (source - ego_station - 8.) * .3))
                ego_station += .1 * ego_v
                gap = source - ego_station
                age = 0 if rng.random() > .15 else rng.choice((1, 2, 3))
                witness = _sighting(frame - frame % 2, source_v + (rng.random() - .5) * .3) if rng.random() > .25 else None
                args = (frame, ego_station, gap, gap if age == 0 else 70., source_v, age, ego_v, witness)
                canonical.update(*args)
                research.update(*args)
                gain = 30. * rng.random() - 10.
                features = canonical.preference_features(frame, gap, ego_v, source_v, gain)
                self.assertEqual(features, research.preference_features(frame, gap, ego_v, source_v, gain))
                canonical.learned_preference(head, features)
                research.learned_preference(head, features)
                for name in ("anchor", "since", "wait", "preference", "probability_follow", "intent_gain_m", "intent_leaf"):
                    self.assertEqual(getattr(canonical, name), getattr(research, name), (seed, frame, name))
                for fitted in ("keep", "follow", "yield", "pass"):
                    self.assertEqual(canonical.choose(fitted), research.choose(fitted))
                    self.assertEqual(canonical.choose(fitted, True), research.choose(fitted, True))

    def test_source_association_is_the_qualified_function(self):
        from crosswalk import source_vehicle
        from tools import maneuver_source_vehicle
        for name in ("_matrix", "source_surface_vehicle"):
            self.assertEqual(inspect.getsource(getattr(source_vehicle, name)),
                             inspect.getsource(getattr(maneuver_source_vehicle, name)))

    def test_planner_on_native_map_matches_the_research_planner(self):
        from tools.maneuver_motor_queue_candidate import candidate_planner
        from tests.test_maneuver_source_vehicle import surface
        from tests.test_urban_map import MAP
        if not MAP.is_file():
            self.skipTest("Native map unavailable")
        road = carla.Map("Town03_Opt", MAP.read_text())
        route = [road.get_waypoint_xodr(67, -2, k).transform.location for k in range(173, 286)]
        pose = road.get_waypoint_xodr(67, -2, 205.).transform
        ego = Mock()
        ego.get_transform.return_value = pose
        ego.bounding_box = carla.BoundingBox(carla.Location(z=.76), carla.Vector3D(2.45, .92, .76))
        head = StationaryIntent()
        research_type = candidate_planner(head)
        data = surface()
        ground = np.asarray([(v, -3.5, -2.30, .5) for v in np.arange(-37., 40., .30)])
        sweep = np.vstack((data["lidar"], ground))
        for boxes, expected in ((data["detections"], "follow"), ([], "pass")):
            for bound in (False, True):
                canonical = VehicleQueuePlanner.from_route(road, route, intent=head, choice=ManeuverChoice())
                research = research_type.from_route(road, route)
                for i in range(22):
                    camera = dict(frame=121 + i, boxes=boxes, box_frame=121 + i - i % 2, box_pose=pose, radar=data["radar"])
                    if bound:
                        canonical.bind_camera(camera)
                        a = canonical.preview(ego, sweep, np.zeros(14), 2.)
                    else:
                        a = canonical.preview(ego, sweep, np.zeros(14), 2., camera_inputs=camera)
                    b = research.preview(ego, sweep, np.zeros(14), 2., camera_inputs=camera)
                    np.testing.assert_array_equal(a, b)
                    self.assertEqual(canonical.info, research.info)
                    self.assertEqual(canonical.speed_limit, research.speed_limit)
                self.assertEqual(canonical.info["choice"], expected)
                self.assertEqual(canonical.stationary_intent_sha256, head.sha256)
                if bound:
                    # A bound exposure is consumed by exactly one preview.
                    canonical.preview(ego, sweep, np.zeros(14), 2.)
                    self.assertIsNone(canonical.camera_inputs)


class EpisodeBindingTests(unittest.TestCase):
    def test_exposure_detector_runs_once_per_image(self):
        inner = Mock()
        inner.detect.side_effect = lambda image: [dict(id=id(image))]
        cached = drive._ExposureDetector(inner)
        first, second = np.zeros((2, 2, 3)), np.zeros((2, 2, 3))
        self.assertIs(cached.detect(first), cached.detect(first))
        cached.detect(second)
        self.assertEqual(inner.detect.call_count, 2)

    def test_research_harness_anchors_remain_unique(self):
        text = inspect.getsource(drive.episode)
        for anchor in ('                if step % 2 == 0:\n                    detections = detector.detect(obs["front"])\n',
                       '            speed = vehicle_speed(scene.ego)\n',
                       '                    points = tactical_pass.preview(scene.ego, obs["lidar"], points, speed)'):
            self.assertEqual(text.count(anchor), 1)


if __name__ == "__main__":
    unittest.main()
