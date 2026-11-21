"""Verify observer alignment, final scoring, provenance and publication without a CARLA host."""
from contextlib import ExitStack, nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from crosswalk import showcase
from crosswalk.sensors import pose


def camera_world():
    world = Mock()
    world.get_blueprint_library.return_value.find.return_value.has_attribute.return_value = True
    world.spawn_actor.return_value = Mock(attributes={}, type_id="sensor.camera.rgb", is_alive=True)
    world.get_settings.return_value = SimpleNamespace(
        synchronous_mode=True, no_rendering_mode=False, fixed_delta_seconds=.1,
        substepping=True, max_substep_delta_time=.01, max_substeps=10)
    return world


def emit(world, frame, shade=0):
    transform = pose(x=frame, z=1.2)
    image = SimpleNamespace(frame=frame, transform=transform, timestamp=frame * .1, width=32, height=18,
                            raw_data=np.full((18, 32, 4), shade, dtype=np.uint8).tobytes())
    world.spawn_actor.return_value.listen.call_args.args[0](image)
    snapshot = world.get_snapshot.return_value
    snapshot.frame = frame
    snapshot.timestamp.elapsed_seconds = frame * .1
    snapshot.find.return_value.get_transform.return_value = transform


class ShowcaseTests(unittest.TestCase):
    def test_only_exterior_camera_and_matching_pose_are_collected(self):
        with TemporaryDirectory() as tmp:
            observer = showcase.ChaseCapture(32, 18, 75, io.BytesIO())
            world = camera_world()
            observer.attach(world, Mock(id=4))
            self.addCleanup(observer.close)
            for frame in (1, 2, 3):
                emit(world, frame)
            self.assertEqual(len(observer.frames), 2)
            with self.assertRaisesRegex(RuntimeError, "Missing chase frame"):
                observer.append(1)
            with self.assertRaisesRegex(RuntimeError, "world snapshot"):
                observer.append(2)
            observer.append(3)
            self.assertEqual(observer.rows[0]["ego_pose"]["x"], 3.)
            self.assertEqual(observer.rows[0]["camera_frame"], 3)
            world.get_blueprint_library.return_value.find.assert_called_once_with("sensor.camera.rgb")
            mount = world.spawn_actor.call_args.args[1]
            self.assertLess(mount.location.x, -5)
            self.assertGreater(mount.location.z, 3)
            self.assertEqual(observer.rendering["requested"]["dlss_screen_percentage"], "75")
            self.assertIsNone(observer.rendering["dlss_gpu_active"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
    def test_completed_capture_matches_pixels_timing_model_and_actual_final_scoring(self):
        with TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            models = root / "models"
            models.mkdir()
            trained = b"stub model"
            (models / "fusion.npz").write_bytes(trained)
            model_sha = hashlib.sha256(trained).hexdigest()
            producer = dict(client_version="1.0", server_version="1.0", map="Town01_Opt")
            fit = dict(model_sha256={"fusion": model_sha}, fit_id="test-fit", training_producer=producer,
                       render_profile=dict(rgb="mixed", rgb_screen_percentage=50.0,
                                           camera_ray_tracing=False, ground_truth="native"))
            (models / "fit.json").write_text(json.dumps(fit))
            world = camera_world()
            stack.enter_context(patch.object(showcase, "PROJECT", root))
            stack.enter_context(patch.object(showcase.Model, "load", return_value=Mock(inputs="fusion")))
            launch = stack.enter_context(patch.object(showcase, "simulator", return_value=nullcontext(world)))
            stack.enter_context(patch.object(showcase, "producer_info", return_value=producer))
            stack.enter_context(patch.dict(showcase.RESOLUTIONS, tiny=(32, 18)))
            stack.enter_context(patch.dict(os.environ, CROSSWALK_RGB_RENDER="native"))

            def episode(observed, kind, seed, policy, model, **settings):
                self.assertEqual((kind, seed, policy), ("trip", showcase.SEED, "fusion"))
                self.assertEqual(settings["destination"], "market")
                self.assertEqual(showcase.render_profile(), fit["render_profile"])
                world.try_spawn_actor.return_value = Mock(id=7, is_alive=True, attributes={"role_name": "hero"})
                observed.try_spawn_actor("blueprint", "transform")
                self.assertIs(observed.get_settings(), world.get_settings())
                recording = settings["recording"]
                self.assertTrue(recording.primary_rig_only)
                self.assertEqual(settings["run_id"], "showcase-211-trip-fusion")
                trace = []
                for i, (frame, shade) in enumerate(zip(range(28, 33), (0, 40, 120, 185, 255))):
                    def tick():
                        emit(world, frame, shade)
                        return frame
                    world.tick.side_effect = tick
                    self.assertEqual(observed.tick(), frame)
                    if 0 < i < 4:
                        row = dict(frame=frame, steer=.1, speed_mps=2.5, collision_count=0)
                        trace.append(row)
                        perception = dict(row, lidar=[[frame, 2.]], radar=[[frame, 1., -1.]],
                                          tracks=[{"id": 1, "x": frame}], detections=[],
                                          observer_map={"ego": [frame, 0., 0.]})
                        rgb = np.full((360, 640, 3), 255 - shade, np.uint8)
                        recording.append(perception, {"front": rgb})
                trace[-1]["collision_count"] = 1  # collision barrier delivers after capture
                return dict(outcome="collision", frames=3), trace, None, None

            stack.enter_context(patch.object(showcase, "episode", side_effect=episode))
            output = root / "web/media/showcase/trip"
            manifest = showcase.capture("trip", "High", "tiny", 75, 2668, output)
            launch.assert_called_once_with(2668, attach=False, quality="High")
            self.assertEqual(os.environ["CROSSWALK_RGB_RENDER"], "native")
            self.assertFalse(manifest["benchmark_member"])
            self.assertEqual(manifest["outcome"], "collision")
            self.assertEqual(manifest["model_sha256"], model_sha)
            self.assertEqual(manifest["producer"], producer)
            self.assertEqual(int(manifest["video"]["nb_read_frames"]), 3)
            self.assertEqual(manifest["video"]["r_frame_rate"], "10/1")
            self.assertEqual(manifest["sha256"].keys(), {
                "chase.mp4", "chase.jpg", "poster.jpg", "telemetry.json", "front.mp4", "front.jpg", "replay.json"})
            self.assertEqual(manifest["catalog_entry"]["path"], "showcase/trip")
            self.assertEqual(manifest["catalog_entry"]["cameras"], ["chase", "front"])
            self.assertEqual(manifest["catalog_entry"]["manifest"], "./media/showcase/trip/manifest.json")
            self.assertEqual(manifest["front_video"]["width"], 640)
            for name, expected in manifest["sha256"].items():
                self.assertEqual(showcase.fingerprint(output / name), expected)
            log = json.loads((output / "telemetry.json").read_text())
            self.assertEqual([(r["camera_frame"], r["frame"]) for r in log], [(n, n) for n in (29, 30, 31)])
            self.assertEqual(log[-1]["collision_count"], 1)
            self.assertEqual(log[-1]["camera_pose"]["x"], 31.)
            replay = json.loads((output / "replay.json").read_text())
            self.assertEqual(len(replay), len(log))
            for index, (score, perception) in enumerate(zip(log, replay, strict=True)):
                frame = perception["frame"]
                self.assertEqual(frame, score["camera_frame"])
                self.assertEqual(perception["video_time"], index / 10)
                self.assertEqual(perception["lidar"][0][0], frame)
                self.assertEqual(perception["radar"][0][0], frame)
                self.assertEqual(perception["tracks"][0]["x"], frame)
                self.assertEqual(perception["observer_map"]["ego"][0], frame)
                self.assertEqual(perception["collision_count"], score["collision_count"])
            self.assertEqual(manifest["poster_frame"], 31)
            for camera, shades in [("chase", (40, 120, 185)), ("front", (215, 135, 70))]:
                video = cv2.VideoCapture(str(output / f"{camera}.mp4"))
                try:
                    for shade in shades:
                        exists, pixels = video.read()
                        self.assertTrue(exists)
                        self.assertAlmostEqual(float(pixels.mean()), shade, delta=3)
                    self.assertFalse(video.read()[0])
                finally:
                    video.release()
                poster = cv2.imread(str(output / f"{camera}.jpg"))
                self.assertAlmostEqual(float(poster.mean()), shades[-1], delta=1)
            with self.assertRaises(FileExistsError):
                showcase.capture("trip", "High", "tiny", 75, 2668, output)
            launch.assert_called_once()
            with patch.object(showcase, "encode_and_check", side_effect=ValueError("decode mismatch")):
                with self.assertRaisesRegex(ValueError, "decode mismatch"):
                    showcase.capture("trip", "High", "tiny", 75, 2668, root / "failed")
                self.assertFalse((root / "failed").exists())

    def test_perception_sink_refuses_a_different_world_tick(self):
        observer = SimpleNamespace(rows=[{"camera_frame": 330}])
        sink = showcase.PerceptionCapture(observer, io.BytesIO(), io.StringIO())
        with self.assertRaisesRegex(ValueError, "disagree at frame 329"):
            sink.append({"frame": 329}, {})
        self.assertEqual(sink.front_offsets, [])


if __name__ == "__main__":
    unittest.main()
