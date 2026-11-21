from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from crosswalk import runtime


def fake_world(map_path):
    world = Mock()
    world.get_map.return_value.name = map_path
    world.get_settings.side_effect = lambda: SimpleNamespace(synchronous_mode=False)
    return world


class RuntimeTests(unittest.TestCase):
    def test_shipping_binary_and_sensor_capable_low_profile(self):
        name = "CarlaUnreal-Win64-Shipping.exe" if os.name == "nt" else "CarlaUnreal-Linux-Shipping"
        platform = "Win64" if os.name == "nt" else "Linux"
        with TemporaryDirectory() as root, patch.dict(os.environ, CARLA_ROOT=root):
            executable = Path(root) / "CARLA" / "CarlaUnreal" / "Binaries" / platform / name
            executable.parent.mkdir(parents=True)
            executable.touch()
            self.assertTrue(runtime._server_executable().samefile(executable))
            args = runtime._launch_args(executable, 2468, runtime.DEFAULT_QUALITY)
        self.assertEqual(args[:2], [str(executable), "/Game/Carla/Maps/Town01_Opt"])
        self.assertIn("-RenderOffScreen", args)
        self.assertIn("-quality-level=Low", args)
        self.assertIn("-noraytracing", args)
        self.assertIn("-carla-streaming-port=2469", args)
        self.assertNotIn("-nullrhi", args)
        self.assertNotIn("-USEALLAVAILABLECORES", args)
        if os.name == "nt":
            self.assertIn("-d3d12", args)

    def test_checks_both_release_versions(self):
        client = Mock()
        with patch.object(runtime.carla, "Client", return_value=client), \
                patch.object(runtime.socket, "create_connection"):
            for server_version, client_version in (("0.9.16", "1.0"), ("1.0", "1.01"),
                                                   ("1.0.1", "1.0")):
                with self.subTest(server=server_version, client=client_version):
                    client.get_server_version.return_value = server_version
                    client.get_client_version.return_value = client_version
                    with self.assertRaisesRegex(RuntimeError, "Expected CARLA 1.0"):
                        runtime._connect(2468, None, "log")
            client.get_server_version.return_value = "1.0-fe0178f89"
            client.get_client_version.return_value = "1.0"
            self.assertEqual(runtime._connect(2468, None, "log"), (client, "1.0-fe0178f89", "1.0"))

    def test_attached_world_and_manager_are_restored_on_interrupt(self):
        world = fake_world(r"\Game\Carla\Maps\Town01_Opt.Town01_Opt")
        before = SimpleNamespace(synchronous_mode=False, no_rendering_mode=True)
        during = SimpleNamespace()
        world.get_settings.side_effect = [before, during]
        client = Mock()
        client.get_world.return_value = world
        manager = client.get_trafficmanager.return_value
        with patch.object(runtime, "_connect", return_value=(client, "1.0-hash", "1.0")), \
                patch.object(runtime, "_check_port_free") as preflight, \
                patch.object(runtime.subprocess, "Popen") as create:
            with self.assertRaises(KeyboardInterrupt), runtime.simulator(attach=True, with_manager=True) as (w, tm):
                self.assertIs(tm, manager)
                self.assertEqual(runtime.producer_info(w),
                                 {"map": "Town01_Opt", "server_version": "1.0-hash", "client_version": "1.0"})
                self.assertTrue(during.synchronous_mode)
                self.assertFalse(during.no_rendering_mode)
                raise KeyboardInterrupt
            create.assert_not_called()
            preflight.assert_called_once_with(2488)
        self.assertEqual([c.args for c in manager.set_synchronous_mode.call_args_list], [(True,), (False,)])
        self.assertEqual([c.args for c in world.apply_settings.call_args_list], [(during,), (before,)])
        with self.assertRaises(ValueError):
            runtime.producer_info(world)

    def test_wrong_attached_world_stays_untouched(self):
        world = fake_world("Town10HD_Opt")
        client = Mock()
        client.get_world.return_value = world
        with patch.object(runtime, "_connect", return_value=(client, "1.0", "1.0")), \
                self.assertRaisesRegex(RuntimeError, "Town10HD_Opt"), runtime.simulator(attach=True):
            self.fail("wrong map was accepted")
        client.load_world.assert_not_called()
        world.apply_settings.assert_not_called()

    def test_owned_process_fallback_map_and_cleanup(self):
        target = "/Game/Carla/Maps/Town01_Opt"
        client = Mock()
        client.get_world.return_value = fake_world("Town10HD_Opt")
        client.get_available_maps.return_value = ["/Game/Carla/Maps/Town02_Opt", target]
        client.load_world.return_value = fake_world(target)
        with TemporaryDirectory() as root, ExitStack() as patches:
            patches.enter_context(patch.object(runtime, "LOCAL", Path(root)))
            patches.enter_context(patch.object(runtime, "_server_executable",
                                               return_value=Path(root) / "CarlaUnreal/Binaries/Win64/server"))
            patches.enter_context(patch.object(runtime, "_check_port_free"))
            create = patches.enter_context(patch.object(runtime.subprocess, "Popen"))
            process = create.return_value
            process.poll.return_value = None
            process.wait.side_effect = [subprocess.TimeoutExpired("server", 10), None]
            patches.enter_context(patch.object(runtime, "_connect", return_value=(client, "1.0", "1.0")))
            with self.assertRaises(KeyboardInterrupt), runtime.simulator() as world:
                self.assertEqual(runtime.producer_info(world)["map"], "Town01_Opt")
                raise KeyboardInterrupt
            client.load_world.assert_called_once_with(target)
            process.terminate.assert_called_once()
            process.kill.assert_called_once()
            self.assertTrue(create.call_args.kwargs["stdout"].closed)
            client.get_available_maps.return_value = []
            process.poll.return_value = 1
            with self.assertRaisesRegex(RuntimeError, "AdditionalMaps"), runtime.simulator():
                self.fail("accepted a missing map")


if __name__ == "__main__":
    unittest.main()
