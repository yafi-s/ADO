"""Check side-camera mounts and tick alignment without a simulator server."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from crosswalk.sensors import Rig, render_profile


def stub_rig(test, *, visuals=False, ue5=False, range_only=False):
    world, ego = Mock(), Mock()
    actors = []

    def blueprint(name):
        if range_only:
            test.assertIn(name, {"sensor.lidar.ray_cast", "sensor.other.radar", "sensor.other.collision"})
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
    rig = Rig(world, ego, visuals, range_only=range_only)
    test.addCleanup(rig.close)
    names = ("lidar", "radar", "collision")
    if not range_only:
        names = ("front", "left", "right") + names
    if visuals:
        names += ("chase", "tags", "depth")
    return rig, dict(zip(names, actors, strict=True))


class RigTests(unittest.TestCase):
    def test_range_only_rig_skips_all_camera_blueprints(self):
        with patch("crosswalk.sensors.render_profile", side_effect=AssertionError("Accessed camera settings")):
            rig, streams = stub_rig(self, range_only=True)
        point = np.array([3, 0.1, 0.2, 5], dtype="<f4")
        for name in ("lidar", "radar"):
            listen = streams[name][1].listen.call_args.args[0]
            listen(SimpleNamespace(frame=31, raw_data=point.tobytes()))
        contact = streams["collision"][1].listen.call_args.args[0]
        contact(SimpleNamespace(frame=31, other_actor=SimpleNamespace(type_id="vehicle.example")))

        self.assertEqual(rig.rendering, {"requested": None, "cameras": {}})
        self.assertEqual(rig.collisions(), [(31, "vehicle.example")])
        observation = rig.read(31, timeout=0)
        self.assertEqual(observation.keys(), {"lidar", "radar"})
        for values in observation.values():
            np.testing.assert_array_equal(values, [point])

    def test_range_only_rig_rejects_visuals_before_contacting_world(self):
        world = Mock()
        with self.assertRaisesRegex(ValueError, "Visuals require camera sensors"):
            Rig(world, Mock(), visuals=True, range_only=True)
        world.get_blueprint_library.assert_not_called()

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
        for env, enabled_cameras, rgb_scale in (
            ({}, {"left", "right", "chase"}, "50.0"),
            ({"CROSSWALK_RGB_RENDER": "mixed", "CROSSWALK_DLSS_PERCENT": "67"}, {"left", "right", "chase"}, "67.0"),
            ({"CROSSWALK_RGB_RENDER": "native"}, set(), "100.0"),
            ({"CROSSWALK_RGB_RENDER": "dlss", "CROSSWALK_DLSS_PERCENT": "58"}, {"front", "left", "right", "chase"}, "58.0"),
        ):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                rig, streams = stub_rig(self, visuals=True, ue5=True)
                self.assertEqual(rig.rendering["requested"], render_profile())
                dimensions = {
                    "front": ("640", "360"), "chase": ("512", "288"),
                    "left": ("320", "180"), "right": ("320", "180"),
                    "tags": ("320", "180"), "depth": ("320", "180"),
                }
                for name in ("front", "left", "right", "chase", "tags", "depth"):
                    _, actor = streams[name]
                    self.assertEqual(actor.attributes["use_ray_tracing"], "false")
                    self.assertEqual((actor.attributes["image_size_x"], actor.attributes["image_size_y"]),
                                     dimensions[name])
                    reported = rig.rendering["cameras"][name]
                    self.assertTrue(reported["attributes_match_request"])
                    self.assertIsNone(reported["dlss_gpu_active"])
                    self.assertEqual(actor.attributes["enable_dlss"], str(name in enabled_cameras).lower())
                    self.assertEqual(actor.attributes["dlss_screen_percentage"],
                                     rgb_scale if name in enabled_cameras else "100.0")
                    if name in ("tags", "depth"):
                        self.assertNotIn("enable_postprocess_effects", actor.attributes)
                        self.assertEqual(str(streams[name][0]), str(streams["front"][0]))
                        self.assertEqual(actor.attributes["fov"], streams["front"][1].attributes["fov"])
                    else:
                        self.assertEqual(actor.attributes["enable_postprocess_effects"], "true")

    def test_legacy_blueprint_reports_unavailable_dlss(self):
        with patch.dict(os.environ, {}, clear=True):
            rig, _ = stub_rig(self)
        reported = rig.rendering["cameras"]["left"]
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
