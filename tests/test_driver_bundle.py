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
from crosswalk.motor_queue import (STATIONARY_RADAR_FACE, FollowHold, MotorQueue, VehicleQueuePlanner,
                                   radar_face, small_radar_face)
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
                canonical = VehicleQueuePlanner.from_route(road, route, intent=head, choice=ManeuverChoice(),
                                                           defer_confirm_frames=1, follow_hold_frames=0,
                                                           stationary_witness=False, radar_face_min=0)
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


class _AlwaysDefer:
    """The fitted head's identity with every decision forced to a deferral."""

    def __init__(self, head):
        self.checkpoint_path, self.sha256 = head.checkpoint_path, head.sha256

    def details(self, features):
        return "defer_to_existing_choice", .1, 5., 0


def _replay(hold, runs):
    """Step a hold through run-length (associated, preference, frames) decisions."""
    frame, kept = 1000, []
    for associated, preference, count in runs:
        for _ in range(count):
            kept.append(hold.step(frame, associated, preference, False))
            frame += 1
    return kept


FOLLOW, DEFER = "follow", "defer_to_existing_choice"


class FollowHoldTests(unittest.TestCase):
    def test_disabled_hold_is_the_research_rule(self):
        rng = random.Random(3)
        hold = FollowHold(1, 0, False)
        self.assertFalse(hold.enabled)
        for frame in range(2000):
            associated = rng.random() > .3
            preference = rng.choice((FOLLOW, DEFER, None))
            self.assertEqual(hold.step(frame, associated, preference, rng.random() > .8, rng.random() > .5),
                             associated and preference == FOLLOW)

    def test_recorded_flips_never_reach_the_planner(self):
        # Head decisions on the evaluated frames of the three native runs that
        # met the held sedan, from their live.jsonl telemetry.
        recorded = {
            "queue-intent-native02": [(True, FOLLOW, 29), (True, DEFER, 4), (True, FOLLOW, 160)],
            "queue-intent-native03": [(True, FOLLOW, 20), (True, DEFER, 8), (True, FOLLOW, 5),
                                      (True, DEFER, 5), (True, FOLLOW, 125)],
            "integrated-loop-01": [(True, FOLLOW, 18), (False, None, 6), (True, FOLLOW, 5), (True, DEFER, 2),
                                   (True, FOLLOW, 8), (True, DEFER, 2), (True, FOLLOW, 6), (True, DEFER, 1),
                                   (True, FOLLOW, 170)],
        }
        for name, runs in recorded.items():
            self.assertTrue(all(_replay(FollowHold(), runs)), name)
            self.assertFalse(all(_replay(FollowHold(1, 0, False), runs)), name)

    def test_sustained_deferral_is_confirmed(self):
        kept = _replay(FollowHold(), [(True, FOLLOW, 10), (True, DEFER, 40)])
        self.assertTrue(all(kept[:39]))
        self.assertFalse(any(kept[39:]))

    def test_lost_track_hold_expires(self):
        kept = _replay(FollowHold(), [(True, FOLLOW, 5), (False, None, 40)])
        self.assertTrue(all(kept[:35]))
        self.assertFalse(any(kept[35:]))
        # A confirmed deferral is not turned back into following when the track drops.
        kept = _replay(FollowHold(), [(True, DEFER, 35), (False, None, 5)])
        self.assertFalse(any(kept[30:]))

    def test_hold_never_cancels_a_lane_change_under_way(self):
        hold = FollowHold()
        self.assertTrue(hold.step(10, True, FOLLOW, False))
        self.assertFalse(hold.step(11, False, None, True))
        self.assertFalse(hold.step(12, False, None, False))

    def test_a_skipped_frame_restarts_the_confirmation(self):
        hold = FollowHold(3, 0)
        self.assertTrue(hold.step(1, True, DEFER, False))
        self.assertTrue(hold.step(2, True, DEFER, False))
        self.assertTrue(hold.step(5, True, DEFER, False))
        self.assertTrue(hold.step(6, True, DEFER, False))
        self.assertFalse(hold.step(7, True, DEFER, False))
        with self.assertRaises(ValueError):
            FollowHold(0, 0)
        with self.assertRaises(ValueError):
            FollowHold(1, -1)

    def test_stationary_witness_holds_until_the_track_confirms(self):
        # prefix-hold-01 frames 1223-1232: camera and radar saw the stopped sedan on
        # exposures 1225, 1227 and 1229 while lidar speed noise delayed the track.
        sightings = {1225: 1225, 1226: 1225, 1227: 1227, 1228: 1227, 1229: 1229, 1230: 1229, 1231: 1231, 1232: 1231}
        hold, kept = FollowHold(), {}
        for frame in range(1223, 1233):
            sighting = sightings.get(frame) and dict(box_frame=sightings[frame], stationary_doppler=True)
            witnessed = hold.witness(frame, sighting)
            associated = frame >= 1231
            kept[frame] = hold.step(frame, associated, FOLLOW if associated else None, False, witnessed)
        self.assertEqual([f for f, k in kept.items() if not k], [1223, 1224, 1225, 1226])
        # One exposure (native02 frames 1080-1081 at the cones) does not hold, nor does a moving vehicle.
        hold = FollowHold()
        for frame, sighting in ((1080, dict(box_frame=1080, stationary_doppler=True)),
                                (1081, dict(box_frame=1080, stationary_doppler=True)),
                                (1082, dict(box_frame=1082, stationary_doppler=False)),
                                (1083, None)):
            self.assertFalse(hold.step(frame, False, None, False, hold.witness(frame, sighting)))
        # Sightings older than the window lapse, and a witness never cancels a lane change.
        hold = FollowHold()
        hold.witness(10, dict(box_frame=10, stationary_doppler=True))
        self.assertTrue(hold.step(12, False, None, False, hold.witness(12, dict(box_frame=12, stationary_doppler=True))))
        self.assertFalse(hold.step(13, False, None, True, hold.witness(13, None)))
        self.assertFalse(hold.step(22, False, None, False, hold.witness(22, None)))

    def test_planner_keeps_following_through_an_unconfirmed_deferral(self):
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
        head = _AlwaysDefer(StationaryIntent())
        data = surface()
        ground = np.asarray([(v, -3.5, -2.30, .5) for v in np.arange(-37., 40., .30)])
        sweep = np.vstack((data["lidar"], ground))
        choices = {}
        for hold in ((1, 0), (30, 30)):
            # The synthetic surface's sparse radar grid is below the radar face limit,
            # which RadarFaceTests covers; this test isolates the hold.
            planner = VehicleQueuePlanner.from_route(road, route, intent=head, choice=ManeuverChoice(),
                                                     defer_confirm_frames=hold[0], follow_hold_frames=hold[1],
                                                     stationary_witness=hold != (1, 0), radar_face_min=0)
            rows = []
            for i in range(22):
                planner.bind_camera(dict(frame=121 + i, boxes=data["detections"], box_frame=121 + i - i % 2,
                                         box_pose=pose, radar=data["radar"]))
                planner.preview(ego, sweep, np.zeros(14), 2.)
                rows.append((planner.queue.anchor is not None, planner.info["choice"], planner.pass_votes))
            choices[hold] = rows
        associated = [i for i, row in enumerate(choices[30, 30]) if row[0]]
        self.assertTrue(associated)
        self.assertEqual(choices[1, 0][-1][1], "pass")
        for i in associated:
            self.assertEqual(choices[30, 30][i][1:], ("follow", 0))
        self.assertEqual(planner.info["source_motor_vehicle"]["consecutive_head_deferrals"], len(associated))


# High02 frames 6213-6226 (capture at ddcde43): the planner's measured source-lane
# surface and the camera sightings NanoDet matched there as a "car" over the in-lane
# cone group: frame, ego station, gap, raw gap, source speed, source age, ego speed
# and (box frame, in-box radar returns, range, radial ground speed).
_CONE_FRAMES = (
    (6213, 36.576, 31.370, 31.370, 20.0, 0, 3.505, None),
    (6214, 36.934, 30.203, 30.203, 20.0, 0, 3.643, None),
    (6215, 37.307, 29.830, 60.000, 20.0, 1, 3.795, None),
    (6216, 37.672, 29.442, 29.442, 0.0, 0, 3.550, None),
    (6217, 38.012, 28.824, 60.000, 0.0, 1, 3.305, (6217, 80, 28.824, -.125)),
    (6218, 38.329, 28.836, 28.836, 0.0, 0, 3.060, (6217, 49, 28.836, -.114)),
    (6219, 38.623, 28.386, 60.000, 0.0, 1, 2.831, (6219, 76, 28.386, -.101)),
    (6220, 38.896, 28.477, 28.477, 0.0, 0, 2.655, (6219, 74, 28.477, -.074)),
    (6221, 39.155, 28.212, 28.212, .2, 0, 2.527, (6221, 95, 28.212, -.053)),
    (6222, 39.403, 27.981, 27.981, .319, 0, 2.446, (6221, 85, 27.981, -.032)),
    (6223, 39.646, 27.493, 27.493, .037, 0, 2.416, (6223, 89, 27.493, -.010)),
    (6224, 39.888, 27.463, 27.463, .037, 0, 2.407, (6223, 95, 27.463, -.002)),
    (6225, 40.128, 27.203, 27.203, .02, 0, 2.409, None),
    (6226, 40.370, 26.884, 26.884, .141, 0, 2.428, None),
)


def _radar_sighting(box_frame, hits, distance, radial, stationary=True):
    return dict(box_frame=box_frame, label="car", confidence=.4, radar_hits=hits, source_range_m=distance,
                radial_ground_speed_mps=radial, stationary_doppler=stationary)


def _replay_cones(minimum):
    queue, hold, witnessed, anchored, small = MotorQueue(), FollowHold(), [], [], []
    for frame, station, gap, raw, source_v, age, ego_v, seen in _CONE_FRAMES:
        sighting = seen and _radar_sighting(*seen)
        if small_radar_face(sighting, minimum):
            small.append(frame)
            sighting = None
        queue.update(frame, station, gap, raw, source_v, age, ego_v, sighting)
        if hold.witness(frame, sighting):
            witnessed.append(frame)
        if queue.anchor is not None:
            anchored.append(frame)
    return witnessed, anchored, small


class RadarFaceTests(unittest.TestCase):
    def test_cone_group_box_never_confirms_or_holds(self):
        # Without the limit the logged inputs hold from 6219, as High02 did, and confirm
        # the cones as a stopped car (High02 confirmed at 6221; its log clips the
        # surface speed at zero, so this replay can confirm a little earlier).
        witnessed, anchored, small = _replay_cones(0.)
        self.assertEqual(witnessed[0], 6219)
        self.assertLessEqual(anchored[0], 6221)
        self.assertFalse(small)
        witnessed, anchored, small = _replay_cones(STATIONARY_RADAR_FACE)
        self.assertEqual((witnessed, anchored), ([], []))
        self.assertEqual(small, list(range(6217, 6225)))

    def test_logged_sedan_faces_are_kept(self):
        # The smallest logged faces of the held sedan (native03 frame 6249 between
        # exposures, native02 frame 1304 on an exposure) and the largest cone face.
        self.assertFalse(small_radar_face(_radar_sighting(6248, 117, 40.21, -.05)))
        self.assertFalse(small_radar_face(_radar_sighting(1304, 153, 33.54, -.02)))
        self.assertTrue(small_radar_face(_radar_sighting(6221, 95, 28.21, -.05)))
        self.assertAlmostEqual(radar_face(_radar_sighting(6221, 95, 28.21, -.05)), 2680., delta=1.)
        # A moving vehicle and an empty frame are left as they are.
        self.assertFalse(small_radar_face(_radar_sighting(9, 10, 20., 6., stationary=False)))
        self.assertFalse(small_radar_face(None))
        with self.assertRaises(ValueError):
            VehicleQueuePlanner([], intent=StationaryIntent(), choice=ManeuverChoice(), radar_face_min=-1.)

    def test_planner_leaves_a_small_radar_face_to_the_route_planner(self):
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
        data = surface()
        ground = np.asarray([(v, -3.5, -2.30, .5) for v in np.arange(-37., 40., .30)])
        sweep = np.vstack((data["lidar"], ground))
        outcome = {}
        # 88 returns at 30 m is a cone-group face; three times as many is a car's.
        for copies in (1, 3):
            planner = VehicleQueuePlanner.from_route(road, route, intent=StationaryIntent(), choice=ManeuverChoice())
            held = []
            for i in range(8):
                planner.bind_camera(dict(frame=121 + i, boxes=data["detections"], box_frame=121 + i - i % 2,
                                         box_pose=pose, radar=np.vstack([data["radar"]] * copies)))
                planner.preview(ego, sweep, np.zeros(14), 2.)
                held.append(planner.defer_source)
            outcome[copies] = (planner.queue.anchor is not None, any(held),
                               planner.info["source_motor_vehicle"]["small_radar_face"])
        anchored, held, small = outcome[1]
        self.assertFalse(anchored or held)
        self.assertAlmostEqual(small["radar_face"], 2640., delta=1.)
        self.assertEqual(outcome[3], (True, True, None))


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
