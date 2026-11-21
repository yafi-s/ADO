"""Exercise journey media, aligned control readback and resumable installation offline."""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import carla
import cv2
import numpy as np

from crosswalk import journey_capture as capture, showcase
from crosswalk.journey import required_event_kinds
from crosswalk.sensors import pose


def scored_events(map_name=capture.URBAN_MAP):
    return [dict(kind=kind, triggered=True, exposed=True, cleared=True, passed=True, handled=True, violation=False)
            for kind in required_event_kinds(map_name)]


def summary(**changes):
    return dict(dict(outcome="complete", frames=3, duration_s=191.3, route_length_m=812.2, progress_m=812.2,
                     collision_events=0, yield_violations=0, red_light_violations=0, events=scored_events(),
                     district="townwide", destination="east"), **changes)


def fake_scene(world, kind, seed, **kwargs):
    return SimpleNamespace(ego=world.try_spawn_actor(), close=Mock())


def simulated_episode(world, kind, seed, policy, model, *, recording, scene_factory, **kwargs):
    if "teacher" in kwargs:
        raise AssertionError("The capture supplied a teacher")
    scene = scene_factory(world, kind, seed)
    world.tick()
    trace = []
    for _ in range(3):
        frame = world.tick()
        score = dict(frame=frame, sim_time=len(trace) / 10, steer=0.1, throttle=0.4, brake=0.0,
                     collision_count=0, scenario=kind, map=capture.URBAN_MAP)
        perception = dict(score, observer_map={"ego": [frame, 0, 0]}, lidar=[[frame, 0]],
                          radar=[[frame, 0, 0]], tracks=[{"x": frame}], detections=[])
        front = np.full((360, 640, 3), 220 - frame, dtype=np.uint8)
        recording.append(perception, {"front": front})
        trace.append(score)
    world.tick()
    scene.close()
    return summary(duration_s=.3), trace, None, None


def fake_world():
    world = Mock()
    current_attributes = {}
    blueprint = world.get_blueprint_library.return_value.find.return_value
    blueprint.has_attribute.return_value = True
    blueprint.set_attribute.side_effect = current_attributes.__setitem__
    camera = Mock(type_id="sensor.camera.rgb", is_alive=True, attributes=current_attributes)
    world.spawn_actor.return_value = camera
    hero = Mock(id=111, attributes={"role_name": "hero"})
    world.try_spawn_actor.return_value = hero
    world.get_settings.return_value = SimpleNamespace(synchronous_mode=True, no_rendering_mode=False,
                                                       fixed_delta_seconds=.10000000149, substepping=True,
                                                       max_substep_delta_time=.01, max_substeps=10)
    world.get_map.return_value.generate_waypoints.return_value = [
        SimpleNamespace(lane_type=carla.LaneType.Driving, road_id=9, section_id=0, lane_id=-1,
                        s=index, transform=pose(x=index, y=20, z=4)) for index in (0, 2)
    ]
    world.get_environment_objects.return_value = []
    frames = iter(range(21, 26))

    def tick():
        frame = next(frames)
        transform = pose(x=frame, z=3)
        stamp = frame / 10
        snapshot = world.get_snapshot.return_value
        snapshot.frame = frame
        snapshot.timestamp = SimpleNamespace(elapsed_seconds=stamp)
        snapshot.find.return_value.get_transform.return_value = transform
        data = np.full((18, 32, 4), frame * 4, dtype=np.uint8)
        camera.listen.call_args.args[0](SimpleNamespace(frame=frame, transform=transform, timestamp=stamp,
                                                        width=32, height=18, raw_data=data.tobytes()))
        hero.get_control.return_value = SimpleNamespace(steer=frame * .01, throttle=0.2, brake=0.1)
        return frame

    world.tick.side_effect = tick
    return world


class JourneyCaptureTests(unittest.TestCase):
    def test_outcome_requires_full_planned_encounters_and_clean_termination(self):
        required = required_event_kinds(capture.URBAN_MAP)
        self.assertTrue(capture._result(summary(duration_s=590), required)["passed"])
        events = scored_events()
        events[-1]["handled"] = False
        score = capture._result(summary(outcome="lane_departure", progress_m=412, duration_s=94,
                                        red_light_violations=1, events=events), required)
        self.assertFalse(score["passed"])
        self.assertEqual(score["failure_reasons"], ["route_incomplete", "below_minimum_duration",
                                                    "red_light_violation", "planned_encounters_incomplete"])
        self.assertFalse(capture._result(summary(events=[*events[:-1], events[0]]), required)["passed"])
        town01 = summary(events=scored_events(capture.MAP_NAME))
        self.assertTrue(capture._result(town01, required_event_kinds(capture.MAP_NAME))["passed"])
        self.assertFalse(capture._result(town01, required)["passed"])

    def test_single_ego_and_unbroken_ticks_are_mandatory(self):
        world = Mock()
        first, second = (Mock(id=n, attributes={"role_name": "hero"}) for n in (11, 12))
        world.try_spawn_actor.return_value = first
        world.spawn_actor.return_value = second
        observed = capture.JourneyWorld(world, Mock())
        self.assertIs(observed.try_spawn_actor(), first)
        with self.assertRaisesRegex(RuntimeError, "second hero"):
            observed.spawn_actor()
        second.destroy.assert_called_once()
        self.assertIs(observed.ego, first)
        camera = SimpleNamespace(rows=[{"world_elapsed_s": 1.0, "camera_frame": 10, "camera_timestamp_s": 1.1}])
        perception = SimpleNamespace(front_offsets=[(0, 1)])
        with self.assertRaisesRegex(ValueError, "world frame 10"):
            capture._camera_interval({"frames": 1}, [{"frame": 10}], camera, perception)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg suite is required")
    def test_real_video_join_and_recovery_without_a_simulator(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            output = root / "web/media/long-city-drive"
            world = fake_world()
            policy = dict(path="frozen.npz", sha256="c" * 64,
                          camera_profile={"rgb": "native", "rgb_screen_percentage": 100.0,
                                          "camera_ray_tracing": False, "ground_truth": "native"})
            stack.enter_context(patch.multiple(capture, PROJECT=root, CHASE_SIZE=(32, 18), MIN_DURATION=0))
            stack.enter_context(patch.object(capture, "_load_inputs", return_value=(Mock(), fake_scene, Mock(), policy)))
            producer = dict(server_version="1.0", client_version="1.0", map=capture.URBAN_MAP)
            stack.enter_context(patch.object(capture, "producer_info", return_value=producer))
            stack.enter_context(patch.object(capture.city_map, "producer_info", return_value=producer))
            stack.enter_context(patch.object(capture, "episode", side_effect=simulated_episode))
            stack.enter_context(patch.object(capture, "simulator", side_effect=AssertionError("Simulator was launched")))
            stack.enter_context(patch.object(showcase, "sleep"))
            original_commit = showcase._commit_file

            def lock_front(source, target):
                if target.name == "front.mp4":
                    raise PermissionError("sharing violation")
                original_commit(source, target)

            with patch.object(showcase, "_commit_file", side_effect=lock_front):
                with self.assertRaisesRegex(RuntimeError, "--install-staged"):
                    capture.run(world, output, model_path=root / "frozen.npz")
            self.assertTrue((output / "chase.mp4").is_file())
            self.assertFalse((output / "manifest.json").exists())
            install, = (root / "out/journey-captures").glob("*/delivery-*/install.json")
            with patch.object(sys, "argv", ["journey_capture", "--install-staged", str(install.parent)]):
                self.assertEqual(capture.main(), 0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["catalog_entry"]["path"], "long-city-drive")
            self.assertEqual(manifest["catalog_entry"]["trial_id"], "long-city-drive")
            self.assertEqual(manifest["catalog_entry"]["map_url"], "./media/long-city-drive/town.json")
            self.assertEqual(manifest["catalog_entry"]["map"], capture.URBAN_MAP)
            self.assertEqual(manifest["map_url"], manifest["catalog_entry"]["map_url"])
            town = json.loads((output / "town.json").read_text(encoding="utf-8"))
            self.assertEqual(town["producer"], producer)
            self.assertEqual(town["roads"], [[[0., 20., 4.], [2., 20., 4.]]])
            self.assertIn("bus", manifest["required_event_kinds"])
            self.assertFalse(manifest["benchmark_member"])
            self.assertFalse(manifest["teacher"])
            self.assertEqual(manifest["ego_actor_id"], 111)
            self.assertEqual([manifest[key] for key in ("first_frame", "last_frame", "poster_frame")], [22, 24, 24])
            self.assertEqual(manifest["warmup_frames"], 1)
            self.assertEqual(manifest["chase_source_frames"], 5)
            self.assertEqual(manifest["front_source_frames"], 3)
            self.assertEqual(manifest["master"]["chase"]["nb_read_frames"], "3")
            for name, digest in manifest["sha256"].items():
                self.assertEqual(capture.fingerprint(output / name), digest)
            rows = json.loads((output / "replay.json").read_text(encoding="utf-8"))
            for index, row in enumerate(rows):
                frame = 22 + index
                self.assertEqual(row["frame"], frame)
                self.assertEqual(row["camera_frame"], frame)
                self.assertEqual(row["video_time"], index / 10)
                self.assertEqual(row["tracks"][0]["x"], frame)
                self.assertAlmostEqual(row["steer"], frame * .01)
                self.assertEqual(row["commanded_control"]["steer"], 0.1)
            for camera, shades in (("chase", (88, 92, 96)), ("front", (198, 197, 196))):
                reader = cv2.VideoCapture(str(output / f"{camera}.mp4"))
                try:
                    for shade in shades:
                        exists, pixels = reader.read()
                        self.assertTrue(exists)
                        self.assertAlmostEqual(float(pixels.mean()), shade, delta=3)
                    self.assertFalse(reader.read()[0])
                finally:
                    reader.release()
            install_root = install.parent.parent
            (install_root / "raw/replay.json").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint failed verification"):
                capture.finalize_staged(install_root)

            def partial_episode(*args, **kwargs):
                result, trace, data, targets = simulated_episode(*args, **kwargs)
                result.update(outcome="collision", collision_events=1)
                return result, trace, data, targets

            review_output = root / "web/media/full-route"
            with patch.object(capture, "episode", side_effect=partial_episode):
                review = capture.run(fake_world(), review_output, model_path=root / "frozen.npz")
            self.assertFalse(review_output.exists())
            self.assertFalse(review["passed"])
            self.assertIn("collision", review["failure_reasons"])
            self.assertIsNone(review["catalog_entry"])
            failed, = (root / "out/journey-captures" / review["capture_id"]).glob("delivery-*/manifest.json")
            self.assertTrue((failed.parent / "front.mp4").is_file())
            self.assertTrue((failed.parent / "master/chase.mp4").is_file())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg suite is required")
    def test_capped_web_transcode_keeps_every_source_frame(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, delivery = root / "original.mp4", root / "web.mp4"
            size, frame_count = (128, 72), 30
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=128x72:r=10",
                            "-frames:v", str(frame_count), "-c:v", "libx264", "-threads", "1", "-crf", "6",
                            str(source)], check=True)
            original = capture._verify_video(source, size, frame_count)
            limit = 20 * 1024
            self.assertGreater(source.stat().st_size, limit)
            before = capture.fingerprint(source)
            with patch.object(capture, "WEB_LIMIT", limit):
                web = capture._web_video(source, original, delivery, frame_count)
            self.assertLess(delivery.stat().st_size, limit)
            self.assertEqual(web["nb_read_frames"], str(frame_count))
            self.assertEqual(web["r_frame_rate"], "10/1")
            self.assertEqual(capture.fingerprint(source), before)


if __name__ == "__main__":
    unittest.main()
