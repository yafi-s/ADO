"""Check side-camera mounts and tick alignment without a simulator server."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from crosswalk.sensors import Rig, render_profile


def stub_rig(test, *, visuals=False, ue5=False):
    world, ego = Mock(), Mock()
    actors = []

    def blueprint(name):
        render_keys = set()
        if ue5 and name.startswith("sensor.camera."):
            render_keys.update(("use_ray_tracing", "enable_dlss", "dlss_screen_percentage"))
        if name == "sensor.camera.rgb":
            render_keys.add("enable_postprocess_effects")
        bp = Mock(type_id=name)
        bp.has_attribute.side_effect = render_keys.__contains__
        return bp

    def spawn(bp, mount, *, attach_to):
        test.assertIs(attach_to, ego)
        attributes = dict(call.args for call in bp.set_attribute.call_args_list)
        actor = Mock(is_alive=True, attributes=attributes, type_id=bp.type_id)
        actors.append((mount, actor))
        return actor

    world.get_blueprint_library.return_value.find.side_effect = blueprint
    world.spawn_actor.side_effect = spawn
    rig = Rig(world, ego, visuals)
    test.addCleanup(rig.close)
    names = ("front", "left", "right", "lidar", "radar", "collision")
    if visuals:
        names += ("chase", "tags", "depth")
    return rig, dict(zip(names, actors, strict=True))


class RigTests(unittest.TestCase):
    def test_side_cameras_are_native_rgb_and_synchronized_without_recording(self):
        rig, streams = stub_rig(self)
        for name, yaw in (("left", -70), ("right", 70)):
            mount, actor = streams[name]
            self.assertEqual(actor.type_id, "sensor.camera.rgb")
            self.assertEqual(mount.rotation.yaw, yaw)
            self.assertEqual(mount.rotation.pitch, 0)
            self.assertGreater(mount.location.y * yaw, 0)
            self.assertEqual(actor.attributes, {
                "image_size_x": "320", "image_size_y": "180", "fov": "90", "sensor_tick": "0",
                "enable_postprocess_effects": "true",
            })
        radar_raw_order = [-3.25, 0.125, -0.25, 19.0]  # velocity, azimuth, altitude, depth
        for index, name in enumerate(("front", "left", "right", "lidar", "radar")):
            listen = streams[name][1].listen.call_args.args[0]
            if index < 3:
                raw = bytes((30, 40, 50 + index, 255))
            else:
                raw = np.array(radar_raw_order if name == "radar" else [0, 0, 0, 0], dtype="<f4").tobytes()
            listen(SimpleNamespace(frame=30, width=1, height=1, raw_data=b""))
            listen(SimpleNamespace(frame=31, width=1, height=1, raw_data=raw))
        observation = rig.read(31, timeout=0)
        np.testing.assert_array_equal(observation["left"], [[[51, 40, 30]]])
        np.testing.assert_array_equal(observation["right"], [[[52, 40, 30]]])
        np.testing.assert_array_equal(observation["radar"], [radar_raw_order])
        rig.close()
        for _, actor in streams.values():
            actor.stop.assert_called_once()
            actor.destroy.assert_called_once()

    def test_ue5_render_profiles_preserve_camera_geometry_and_native_ground_truth(self):
        for env, rgb_enabled, rgb_scale in (
            ({}, "true", "58.0"),
            ({"CROSSWALK_RGB_RENDER": "native"}, "false", "100.0"),
            ({"CROSSWALK_RGB_RENDER": "dlss", "CROSSWALK_DLSS_PERCENT": "50"}, "true", "50.0"),
        ):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                rig, streams = stub_rig(self, visuals=True, ue5=True)
                self.assertEqual(rig.rendering["requested"], render_profile())
                for name in ("front", "left", "right", "chase", "tags", "depth"):
                    _, actor = streams[name]
                    self.assertEqual(actor.attributes["use_ray_tracing"], "false")
                    reported = rig.rendering["cameras"][name]
                    self.assertTrue(reported["attributes_match_request"])
                    self.assertIsNone(reported["dlss_gpu_active"])
                    if name in ("tags", "depth"):
                        self.assertEqual(actor.attributes["enable_dlss"], "false")
                        self.assertEqual(actor.attributes["dlss_screen_percentage"], "100.0")
                        self.assertNotIn("enable_postprocess_effects", actor.attributes)
                        self.assertEqual(str(streams[name][0]), str(streams["front"][0]))
                        for key in ("image_size_x", "image_size_y", "fov"):
                            self.assertEqual(actor.attributes[key], streams["front"][1].attributes[key])
                    else:
                        self.assertEqual(actor.attributes["enable_dlss"], rgb_enabled)
                        self.assertEqual(actor.attributes["dlss_screen_percentage"], rgb_scale)
                        self.assertEqual(actor.attributes["enable_postprocess_effects"], "true")

    def test_legacy_blueprint_reports_unavailable_dlss(self):
        with patch.dict(os.environ, {}, clear=True):
            rig, _ = stub_rig(self)
        reported = rig.rendering["cameras"]["front"]
        self.assertEqual(reported["requested_attributes"]["enable_dlss"], "true")
        self.assertIsNone(reported["actor_attributes"]["enable_dlss"])
        self.assertFalse(reported["attributes_match_request"])

    def test_rejects_invalid_render_settings(self):
        for env in (
            {"CROSSWALK_RGB_RENDER": "fast"},
            {"CROSSWALK_DLSS_PERCENT": "24"},
            {"CROSSWALK_DLSS_PERCENT": "NaN"},
            {"CROSSWALK_DLSS_PERCENT": "inf"},
            {"CROSSWALK_DLSS_PERCENT": "fast"},
        ):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError):
                    render_profile()


if __name__ == "__main__":
    unittest.main()
