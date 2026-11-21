"""Mount registration, rigid parenting, visual-only physics and GLB landmarks."""
import json
from math import atan2, degrees
from pathlib import Path
import struct
import unittest
from unittest.mock import Mock

import carla
from crosswalk.robotaxi import BodyKit, MOUNTS, PARTS
from crosswalk.sensors import Rig


class RobotaxiTests(unittest.TestCase):
    def test_all_parts_are_rigid_and_collision_free_and_cleanup_is_idempotent(self):
        world, ego = Mock(), Mock()
        world.get_settings.return_value.no_rendering_mode = False
        children = [Mock(is_alive=True) for _ in PARTS]
        blueprint = world.get_blueprint_library.return_value.find.return_value
        attributes = {}
        snapshots = []
        blueprint.set_attribute.side_effect = attributes.__setitem__

        def spawn(bp, transform, **kwargs):
            snapshots.append(attributes.copy())
            return children[len(snapshots) - 1]

        world.spawn_actor.side_effect = spawn
        kit = BodyKit(world, ego)
        world.get_blueprint_library.return_value.find.assert_called_once_with("static.prop.briefcase")
        for call, actor, attrs, part in zip(world.spawn_actor.call_args_list, children, snapshots, PARTS, strict=True):
            self.assertIs(call.kwargs["attach_to"], ego)
            self.assertEqual(call.kwargs["attachment_type"], carla.AttachmentType.Rigid)
            self.assertNotIn("mass", attrs)
            path = part.mesh.path
            self.assertEqual(attrs["mesh_path"], path + "." + path.rsplit("/", 1)[1])
            # An off-center asset pivot should still put the center of its
            # visible geometry at the designed location in the ego frame.
            pivot_offset = carla.Location(*part.mesh.center)
            center = call.args[1].transform(pivot_offset)
            for actual, expected in zip((center.x, center.y, center.z), part.center, strict=True):
                self.assertAlmostEqual(actual, expected, places=5)
            actor.set_simulate_physics.assert_called_once_with(False)
            actor.set_collisions.assert_called_once_with(False)
            actor.apply_texture.assert_called_once()
        kit.close()
        kit.close()
        for actor in children:
            actor.destroy.assert_called_once()

    def test_failed_part_spawn_releases_prior_children(self):
        world = Mock()
        actor = Mock(is_alive=True)
        world.spawn_actor.side_effect = [actor, RuntimeError("spawn failed")]
        with self.assertRaisesRegex(RuntimeError, "spawn failed"):
            BodyKit(world, Mock())
        actor.destroy.assert_called_once()

    def test_glb_landmarks_match_the_native_sensor_actors(self):
        path = Path(__file__).resolve().parents[1] / "assets/robotaxi/roof-carrier.glb"
        data = path.read_bytes()
        magic, version, size, chunk_size, kind = struct.unpack_from("<4sIII4s", data)
        self.assertEqual((magic, version, size, kind), (b"glTF", 2, len(data), b"JSON"))
        body = json.loads(data[20:20+chunk_size])
        nodes = {node["name"]: node for node in body["nodes"]}

        world = Mock()
        rig = Rig(world, Mock())
        try:
            calls = world.spawn_actor.call_args_list
            for (name, (x, y, z, yaw)), spawn in zip(MOUNTS.items(), calls[:5], strict=True):
                mount = spawn.args[1]
                self.assertAlmostEqual(x, mount.location.x, places=5)
                self.assertAlmostEqual(y, mount.location.y, places=5)
                self.assertAlmostEqual(z, mount.location.z, places=5)
                self.assertAlmostEqual(yaw, mount.rotation.yaw, places=5)
                marker = nodes["sensor_origin_" + name]
                for expected, actual in zip((x, z, y), marker["translation"], strict=True):
                    self.assertAlmostEqual(expected, actual, places=5)
                _, qy, _, qw = marker.get("rotation", [0,0,0,1])
                self.assertAlmostEqual(-degrees(2 * atan2(qy,qw)), yaw, places=4)
        finally:
            rig.close()


if __name__ == "__main__":
    unittest.main()
