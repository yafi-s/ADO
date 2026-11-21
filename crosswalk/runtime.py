"""Launch an offscreen CARLA 1.0 server or temporarily use an idle one."""
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import socket
import subprocess
import time

import carla


CARLA_VERSION = "1.0"
MAP_NAME = "Town01_Opt"
MAP_ASSET = f"/Game/Carla/Maps/{MAP_NAME}"
DEFAULT_QUALITY = "Low"
LOCAL = Path(__file__).resolve().parents[1] / ".local"
_ACTIVE_PRODUCERS = {}


def producer_info(world):
    """Return the server, client and map observed for this active world."""
    try:
        return _ACTIVE_PRODUCERS[id(world)].copy()
    except KeyError:
        raise ValueError("This world is outside its verified simulator session.") from None


def _server_executable():
    root = Path(os.environ.get("CARLA_ROOT") or LOCAL / "carla-1.0").resolve()
    name = "CarlaUnreal-Win64-Shipping.exe" if os.name == "nt" else "CarlaUnreal-Linux-Shipping"
    # Also accept the single enclosing folder some release archives contain.
    for relative in (f"CarlaUnreal/Binaries/*/{name}", f"*/CarlaUnreal/Binaries/*/{name}"):
        for executable in root.glob(relative):
            if executable.is_file():
                return executable
    raise FileNotFoundError(f"Install CARLA {CARLA_VERSION} using setup.ps1 or point CARLA_ROOT "
                            f"at the unpacked simulator. Could not find {name} under {root}.")


def _launch_args(exe, port, quality, *, physics_only=False):
    if quality not in {"Low", "Medium", "High", "Epic"}:
        raise ValueError(f"Unknown CARLA quality level: {quality}")
    # One combined config override follows UE's -ini:<category>: syntax.
    # Shipping builds strip -execcmds. The early -noraytracing flag also
    # disables the RT subsystem before any scene is created.
    map_section = "[/Script/EngineSettings.GameMapsSettings]"
    map_object = f"{MAP_ASSET}.{MAP_NAME}"
    overrides = [f"{map_section}:{key}={map_object}" for key in ("GameDefaultMap", "ServerDefaultMap")]
    args = [str(exe), MAP_ASSET, "-unattended", "-nosplash", "-nosound", "-notrace",
            f"-carla-rpc-port={port}", f"-carla-streaming-port={port + 1}", "-stdout"]
    if physics_only:
        return args + ["-nullrhi", "-corelimit=2", "-ini:Engine:" + ",".join(overrides)]

    args += ["-d3d12" if os.name == "nt" else "-vulkan", "-RenderOffScreen", "-ResX=320",
             "-ResY=180", "-corelimit=4", f"-quality-level={quality}"]
    if quality == "Low":
        args.append("-noraytracing")
        overrides += [f"[/Script/Engine.RendererSettings]:{key}=0" for key in (
            "r.DynamicGlobalIlluminationMethod", "r.ReflectionMethod", "r.Shadow.Virtual.Enable",
            "r.Lumen.HardwareRayTracing", "r.PathTracing", "r.RayTracing",
        )]
    return args + ["-ini:Engine:" + ",".join(overrides)]


def _map_name(path):
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].split(".", 1)[0]


def _check_port_free(port):
    with socket.socket() as sock:
        # Windows' default bind can succeed on an address shared by another process.
        if os.name == "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError as error:
            raise OSError(f"CARLA needs a free local TCP port at {port}: {error}") from error


def _stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _connect(port, process, log_path):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"CARLA exited with status {process.returncode}. See {log_path}.")
        try:
            # Recreate clients that attempted an RPC connection before UE was ready.
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
            client = carla.Client("127.0.0.1", port, worker_threads=2)
            client.set_timeout(4.)
            server_version = client.get_server_version()
            client_version = client.get_client_version()
            break
        except (OSError, RuntimeError):
            time.sleep(0.5)
    else:
        raise TimeoutError(f"CARLA did not answer on port {port} within 180 seconds.")
    for version in (server_version, client_version):
        if version.partition("-")[0] != CARLA_VERSION:
            raise RuntimeError(f"Expected CARLA {CARLA_VERSION}, got server {server_version} "
                               f"and Python client {client_version}.")
    client.set_timeout(90.)
    return client, server_version, client_version


@contextmanager
def simulator(port=2468, attach=False, quality=DEFAULT_QUALITY, with_manager=False, *, physics_only=False):
    """Use physics_only for camera-free vehicle work. Spawning a camera with null RHI crashes UE."""
    if attach and physics_only:
        raise ValueError("Physics-only operation requires a freshly launched CARLA server.")
    process = None
    log_path = LOCAL / f"carla-1.0-{port}.log"
    with ExitStack() as cleanup:
        if with_manager:
            # CARLA exposes no getter for TM sync mode. Create our own TM on a
            # free port, including when using somebody else's idle world.
            _check_port_free(port + 20)
        if not attach:
            for offset in (0, 1, 2):
                _check_port_free(port + offset)
            exe = _server_executable()
            args = _launch_args(exe, port, quality, physics_only=physics_only)
            LOCAL.mkdir(exist_ok=True)
            log = cleanup.enter_context(log_path.open("ab"))
            flags = 0
            startup = None
            if os.name == "nt":
                flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
                startup = subprocess.STARTUPINFO()
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup.wShowWindow = subprocess.SW_HIDE
            # Start the actual server, bypassing the wrapper that forks a child.
            process = subprocess.Popen(args, cwd=exe.parents[3], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                                       startupinfo=startup)
            cleanup.callback(_stop_process, process)

        client, server_version, client_version = _connect(port, process, log_path)
        world = client.get_world()
        observed_map = _map_name(world.get_map().name)
        if observed_map != MAP_NAME:
            if attach:
                raise RuntimeError(f"An attached server must already be in {MAP_NAME}, found {observed_map}.")
            maps = client.get_available_maps()
            target = next((path for path in maps if _map_name(path) == MAP_NAME), None)
            if target is None:
                raise RuntimeError(f"{MAP_NAME} is missing from the server. Install the CARLA "
                                   f"{CARLA_VERSION} AdditionalMaps pack using setup.ps1.")
            world = client.load_world(target)
            observed_map = _map_name(world.get_map().name)
            if observed_map != MAP_NAME:
                raise RuntimeError(f"CARLA loaded {observed_map} when {MAP_NAME} was requested.")

        if attach:
            cleanup.callback(world.apply_settings, world.get_settings())
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.no_rendering_mode = physics_only
        settings.fixed_delta_seconds = 0.1
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)

        _ACTIVE_PRODUCERS[id(world)] = dict(server_version=server_version,
                                           client_version=client_version, map=observed_map)
        cleanup.callback(_ACTIVE_PRODUCERS.pop, id(world))
        if with_manager:
            tm = client.get_trafficmanager(port + 20)
            cleanup.callback(tm.set_synchronous_mode, False)
            tm.set_synchronous_mode(True)
            yield world, tm
        else:
            yield world
