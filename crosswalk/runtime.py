"""Own one offscreen CARLA instance and restore an explicitly attached instance."""
from contextlib import contextmanager
from pathlib import Path
import os
import socket
import subprocess
import time

import carla


def _server_executable():
    root = Path(os.environ.get("CARLA_ROOT", ".local/carla")).resolve()
    names = ("CarlaUE4-Win64-Shipping.exe",) if os.name == "nt" else ("CarlaUE4-Linux-Shipping",)
    for name in names:
        executable = next(root.rglob(name), None) if root.is_dir() else None
        if executable:
            return executable
    raise FileNotFoundError("Install CARLA 0.9.16 using setup.ps1 or set CARLA_ROOT to the unpacked simulator.")


@contextmanager
def simulator(port=2468, attach=False, quality="Low"):
    process = None
    old_settings = None
    world = None
    log = None
    client = carla.Client("127.0.0.1", port, worker_threads=2)
    try:
        if not attach:
            for n in (port, port + 1, port + 2):
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", n))
            exe = _server_executable()
            Path(".local").mkdir(exist_ok=True)
            log = open(".local/carla-server.log", "ab")
            args = [str(exe), "/Game/Carla/Maps/Town01", "-RenderOffScreen", "-nosound",
                    "-unattended", "-nosplash", "-carla-rpc-port=" + str(port),
                    "-quality-level=" + quality, "-ResX=640", "-ResY=360",
                    "-USEALLAVAILABLECORES", "-corelimit=4", "-stdout"]
            flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0
            process = subprocess.Popen(args, cwd=exe.parent, stdout=log, stderr=log, creationflags=flags)

        deadline = time.monotonic() + 90
        client.set_timeout(2.0)
        while time.monotonic() < deadline:
            if process and process.poll() is not None:
                raise RuntimeError(f"CARLA exited with status {process.returncode}. See .local/carla-server.log.")
            try:
                version = client.get_server_version()
                break
            except RuntimeError:
                time.sleep(0.4)
        else:
            raise TimeoutError(f"CARLA did not answer on port {port}")
        if not version.startswith("0.9.16"):
            raise RuntimeError(f"Expected CARLA 0.9.16, received {version}")
        client.set_timeout(45.)
        world = client.get_world()
        if world.get_map().name.split("/")[-1] != "Town01":
            if attach:
                raise RuntimeError("An attached server must already be in Town01.")
            world = client.load_world("Town01")
        old_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = 0.1
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)
        yield world
    finally:
        try:
            if world is not None and old_settings is not None:
                world.apply_settings(old_settings)
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if log:
                log.close()
