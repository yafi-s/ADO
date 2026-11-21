"""Film an independent, closed-loop CARLA drive with one exterior RGB observer."""
import argparse
from collections import deque
from contextlib import contextmanager, nullcontext
from functools import partial
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory, mkdtemp
from threading import Condition
from time import sleep

import cv2
import numpy as np

from .drive import DT, episode
from .experiment import PROJECT, fingerprint, read_json, validate_producer, write_json
from .policy import Model
from .runtime import producer_info, simulator
from .sensors import pose, render_profile


RESOLUTIONS = {"960x540": (960, 540), "1280x720": (1280, 720), "1920x1080": (1920, 1080)}
SEED = 211
MOUNT = dict(x=-7.8, y=0.55, z=3.40, pitch=-10.5, yaw=-2.0)
FPS = round(1 / DT)
FRONT_SIZE = (640, 360)
CASES = ("trip", "pedestrian", "market-demo")
CURATED_NOTE = "The training-district Market route was selected because the learned driver completed it during development."
ARTIFACTS = ("chase.mp4", "front.mp4", "chase.jpg", "front.jpg", "poster.jpg", "telemetry.json", "replay.json")


def transform_fields(transform):
    location, rotation = transform.location, transform.rotation
    return dict(x=location.x, y=location.y, z=location.z,
                pitch=rotation.pitch, yaw=rotation.yaw, roll=rotation.roll)


class ChaseCapture:
    """Capture each tick, then select the scored interval from the episode's final trace."""

    def __init__(self, width, height, percentage, stream):
        self.width, self.height, self.percentage = width, height, percentage
        self.stream = stream
        self.frames = deque(maxlen=2)
        self.ready = Condition()
        self.rows = []
        self.camera = None

    def attach(self, world, ego):
        self.world, self.ego_id = world, ego.id
        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        requested = dict(image_size_x=str(self.width), image_size_y=str(self.height), fov="78",
                         sensor_tick="0", enable_postprocess_effects="true", use_ray_tracing="false",
                         enable_dlss="true", dlss_screen_percentage=str(self.percentage))
        for key, value in requested.items():
            if not blueprint.has_attribute(key):
                raise RuntimeError(f"The CARLA RGB camera is missing {key}")
            blueprint.set_attribute(key, value)
        self.camera = world.spawn_actor(blueprint, pose(**MOUNT), attach_to=ego)
        try:
            attributes = self.camera.attributes
            self.rendering = dict(requested=requested, actor_attributes={k: attributes.get(k) for k in requested},
                                  dlss_gpu_active=None, mount_vehicle_local=MOUNT,
                                  blueprint=self.camera.type_id)
            self.camera.listen(self._receive)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.camera is not None and self.camera.is_alive:
            try:
                self.camera.stop()
            finally:
                self.camera.destroy()
        self.camera = None

    def _receive(self, image):
        with self.ready:
            self.frames.append(image)
            self.ready.notify()

    def append(self, frame):
        with self.ready:
            if not self.ready.wait_for(lambda: self.frames and self.frames[-1].frame >= frame, timeout=30):
                raise RuntimeError(f"Timed out waiting for chase frame {frame}")
            image = next((item for item in self.frames if item.frame == frame), None)
            if image is None:
                raise RuntimeError(f"Missing chase frame {frame}, received {self.frames[-1].frame}")
        if self.rows and frame != self.rows[-1]["camera_frame"] + 1:
            raise RuntimeError(f"Discontinuous showcase frame {frame}")
        if (image.width, image.height) != (self.width, self.height):
            raise ValueError(f"Unexpected CARLA observer size {image.width} × {image.height}")
        snapshot = self.world.get_snapshot()
        if snapshot.frame != frame:
            raise RuntimeError(f"Expected world snapshot {frame}, received {snapshot.frame}")
        actor = snapshot.find(self.ego_id)
        if actor is None:
            raise RuntimeError(f"CARLA ego missing from world snapshot {frame}")
        bgra = np.frombuffer(image.raw_data, np.uint8).reshape(self.height, self.width, 4)
        success, jpeg = cv2.imencode(".jpg", bgra[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 96])
        if not success:
            raise RuntimeError(f"Could not encode observer frame {frame}")
        offset = self.stream.tell()
        self.stream.write(jpeg.tobytes())
        self.rows.append(dict(offset=offset, size=jpeg.nbytes, camera_frame=image.frame,
                              world_elapsed_s=snapshot.timestamp.elapsed_seconds,
                              camera_timestamp_s=image.timestamp,
                              ego_pose=transform_fields(actor.get_transform()),
                              camera_pose=transform_fields(image.transform)))
        if len(self.rows) % 100 == 0:
            print(f"Captured {len(self.rows)} ticks through CARLA frame {frame}", flush=True)


class ObservedWorld:
    """Forward the original episode's CARLA calls and sample the same tick after it advances."""

    def __init__(self, world, observer):
        self.world, self.observer, self.ego = world, observer, None

    def __getattr__(self, name):
        return getattr(self.world, name)

    def try_spawn_actor(self, *args, **kwargs):
        actor = self.world.try_spawn_actor(*args, **kwargs)
        if actor is not None and actor.attributes.get("role_name") == "hero":
            self.ego = actor
        return actor

    def tick(self, *args, **kwargs):
        if self.observer.camera is None:
            if self.ego is None:
                raise RuntimeError("The driving episode must spawn its ego before the first tick")
            self.observer.attach(self.world, self.ego)
        frame = self.world.tick(*args, **kwargs)
        self.observer.append(frame)
        return frame


class PerceptionCapture:
    """Collect scored observations from the policy rig without requesting any extra sensors."""

    primary_rig_only = True

    def __init__(self, observer, front_stream, row_stream):
        self.observer, self.front_stream, self.row_stream = observer, front_stream, row_stream
        self.front_offsets = []

    def append(self, telemetry, observations):
        frame = telemetry["frame"]
        if not self.observer.rows or self.observer.rows[-1]["camera_frame"] != frame:
            raise ValueError(f"Perception and chase camera disagree at frame {frame}")
        required = {"lidar", "radar", "tracks", "detections", "observer_map"}
        if not required.issubset(telemetry) or "ego" not in telemetry["observer_map"]:
            raise ValueError(f"Missing perception telemetry at frame {frame}")
        rgb = observations["front"]
        if rgb.dtype != np.uint8 or rgb.shape != (FRONT_SIZE[1], FRONT_SIZE[0], 3):
            raise ValueError(f"Unexpected policy front camera image at frame {frame}")
        success, jpeg = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not success:
            raise RuntimeError(f"Could not encode front camera frame {frame}")
        self.front_offsets.append((self.front_stream.tell(), jpeg.nbytes))
        self.front_stream.write(jpeg.tobytes())
        self.row_stream.write(json.dumps(telemetry, separators=(",", ":"), allow_nan=False) + "\n")


def write_replay(path, source, telemetry):
    """Join measured perception to final barrier-corrected scores by CARLA frame."""
    with source.open(encoding="utf-8") as rows, path.open("w", encoding="utf-8") as output:
        output.write("[")
        for line, score in zip(rows, telemetry, strict=True):
            row = json.loads(line)
            if row["frame"] != score["frame"]:
                raise ValueError("Replay observation differs from its final scored frame")
            row.update(score)
            if score["video_index"]:
                output.write(",")
            json.dump(row, output, separators=(",", ":"), allow_nan=False)
        output.write("]\n")


@contextmanager
def policy_profile(profile):
    """Match the policy-camera settings recorded by the shipped fit."""
    requested = {"CROSSWALK_RGB_RENDER": profile["rgb"],
                 "CROSSWALK_DLSS_PERCENT": str(profile["rgb_screen_percentage"])}
    previous = {key: os.environ.get(key) for key in requested}
    os.environ.update(requested)
    try:
        if profile != render_profile():
            raise ValueError("The shipped policy render profile cannot be reproduced")
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def encode_and_check(raw, output, size, start, count):
    common = {"capture_output": True, "text": True, "check": True}
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1",
                    "-f", "mjpeg", "-framerate", str(FPS), "-i", str(raw),
                    "-filter_threads", "1", "-vf",
                    f"trim=start_frame={start}:end_frame={start+count},setpts=PTS-STARTPTS,scale=out_range=tv",
                    "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p",
                    "-color_range", "tv", "-crf", "17", "-preset", "veryfast",
                    "-movflags", "+faststart", str(output)], **common)
    probe = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
                            "-count_frames", "-show_entries",
                            "stream=width,height,codec_name,r_frame_rate,nb_read_frames,pix_fmt",
                            "-of", "json", str(output)], **common)
    video, = json.loads(probe.stdout)["streams"]
    expected = (size[0], size[1], count, f"{FPS}/1", "h264", "yuv420p")
    actual = (video["width"], video["height"], int(video["nb_read_frames"]), video["r_frame_rate"],
              video["codec_name"], video["pix_fmt"])
    if actual != expected:
        raise ValueError(f"Encoded showcase video mismatch: {actual} vs {expected}")
    return video


def _retry_lock(operation, *args):
    for delay in (.15, .35, .75, 1.5, None):
        try:
            return operation(*args)
        except OSError as error:
            if delay is None or (not isinstance(error, PermissionError)
                                 and getattr(error, "winerror", None) not in (5, 32, 33)):
                raise
            sleep(delay)


def _commit_file(source, target):
    # Both operations expose a complete file and refuse to overwrite an existing name.
    # The temporary file and target are on the same volume.
    if os.name == "nt":
        os.rename(source, target)
    else:
        os.link(source, target)


def _install_file(source, target, digest):
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or fingerprint(target) != digest:
            raise FileExistsError(f"An existing file differs from the saved capture: {target}")
        return
    with NamedTemporaryFile(prefix=".showcase-copy-", suffix=".part", dir=target.parent.parent,
                            delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        if fingerprint(temporary) != digest:
            raise OSError(f"Copy verification failed for {source.name}")
        try:
            _retry_lock(_commit_file, temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or fingerprint(target) != digest:
                raise
    finally:
        temporary.unlink(missing_ok=True)


def install_staged(stage, *, artifacts=ARTIFACTS, retry_module="crosswalk.showcase"):
    """Resume a validated capture at its original destination, installing the manifest last."""
    stage = Path(stage).resolve(strict=True)
    checkpoint = read_json(stage / "install.json")
    output = Path(checkpoint["output"])
    manifest_path = stage / "manifest.json"
    if not output.is_absolute() or fingerprint(manifest_path) != checkpoint["manifest_sha256"]:
        raise ValueError(f"Invalid showcase installation checkpoint at {stage}")
    manifest = read_json(manifest_path)
    hashes = manifest["sha256"]
    if set(hashes) != set(artifacts) or any(fingerprint(stage / name) != hashes[name] for name in artifacts):
        raise ValueError(f"Showcase artifact hashes disagree with the saved manifest at {stage}")
    files = {**{name: hashes[name] for name in artifacts}, "manifest.json": checkpoint["manifest_sha256"]}
    try:
        _retry_lock(partial(output.mkdir, parents=True, exist_ok=True))
        if output.is_symlink():
            raise FileExistsError(f"The recording destination is a symbolic link: {output}")
        # Check for conflicts before adding any files. A partial installation can continue
        # only if every existing file belongs to precisely this recording.
        for path in output.iterdir():
            if (path.name not in files or path.is_symlink() or not path.is_file()
                    or _retry_lock(fingerprint, path) != files[path.name]):
                raise FileExistsError(f"The output contains unrelated or modified data: {path}")
        for name, digest in files.items():
            _install_file(stage / name, output / name, digest)
    except OSError as error:
        raise RuntimeError(f"Could not install the showcase in {output}. The verified recording is saved in {stage}. "
                           f'Retry without CARLA using python -m {retry_module} --install-staged "{stage}". '
                           f"{error}") from error
    return manifest


def capture(case, quality, resolution, percentage, port, output):
    """Record one viewing drive on a fresh, graphics-capable CARLA host."""
    return _record(partial(simulator, port, attach=False, quality=quality), case, quality, resolution, percentage, output)


def run(world, case, output, *, quality="High", resolution="1920x1080", percentage=67):
    """Record on a caller-owned graphics world launched with the specified quality level."""
    return _record(partial(nullcontext, world), case, quality, resolution, percentage, output)


def _record(world_session, case, quality, resolution, percentage, output):
    if case not in CASES:
        raise ValueError(f"Unknown showcase case {case!r}")
    curated = case == "market-demo"
    scenario = "trip" if curated else case
    seed, district = (31, "train") if curated else (SEED, "heldout")
    selection_note = CURATED_NOTE if curated else "This independent showcase case and seed were selected before capture."
    label = {"market-demo": "Curated Market drive · training district", "trip": "Showcase · Market trip",
             "pedestrian": "Showcase · Pedestrian crossing"}[case]
    disclosure = dict(cohort="showcase", district=district, selection_note=selection_note, benchmark_member=False)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory to preserve {output}")
    if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        raise RuntimeError("FFmpeg and FFprobe are required to capture and verify the video")
    fit = read_json(PROJECT / "models/fit.json")
    model_bytes = (PROJECT / "models/fusion.npz").read_bytes()
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    if model_sha != fit["model_sha256"]["fusion"]:
        raise ValueError("Shipped fusion weights disagree with the fit manifest")
    model = Model.load(BytesIO(model_bytes))
    if model.inputs != "fusion":
        raise ValueError("The showcase requires the shipped fusion policy")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root = PROJECT / "out/showcase-captures"
    checkpoint_root.mkdir(exist_ok=True, parents=True)
    stage = Path(mkdtemp(prefix=f"{case}-", dir=checkpoint_root))
    print(f"Showcase capture checkpoint: {stage}", flush=True)
    with TemporaryDirectory(prefix=".sensors-", dir=stage, ignore_cleanup_errors=True) as work:
        work = Path(work)
        raw, front_raw, rows = work / "chase.mjpeg", work / "front.mjpeg", work / "perception.jsonl"
        size = RESOLUTIONS[resolution]
        trial_id = f"showcase-{seed}-{case}-fusion"
        with (raw.open("wb") as stream, front_raw.open("wb") as front_stream,
              rows.open("w", encoding="utf-8") as row_stream, policy_profile(fit["render_profile"])):
            observer = ChaseCapture(*size, percentage, stream)
            perception = PerceptionCapture(observer, front_stream, row_stream)
            with world_session() as world:
                producer = validate_producer(producer_info(world))
                if producer != fit["training_producer"]:
                    raise ValueError("Showcase simulator differs from the shipped fit producer")
                settings = world.get_settings()
                observed_settings = {k: getattr(settings, k) for k in (
                    "synchronous_mode", "no_rendering_mode", "fixed_delta_seconds", "substepping",
                    "max_substep_delta_time", "max_substeps")}
                if (not settings.synchronous_mode or settings.no_rendering_mode
                        or abs(settings.fixed_delta_seconds - DT) > 1e-6):
                    raise ValueError("CARLA must run in synchronous graphics mode at the policy timestep")
                try:
                    result, trace, *_ = episode(ObservedWorld(world, observer), scenario, seed, "fusion", model,
                                                district=district, weather="ClearNoon",
                                                recording=perception, run_id=trial_id,
                                                destination="market" if scenario == "trip" else None)
                finally:
                    observer.close()
        if (not trace or not observer.rows or len(trace) != result["frames"]
                or len(perception.front_offsets) != len(trace)):
            raise ValueError("Missing scored frames or observer exposures")
        start = trace[0]["frame"] - observer.rows[0]["camera_frame"]
        if start < 0 or start + len(trace) > len(observer.rows):
            raise ValueError("Scored interval extends outside the observer recording")
        selected = observer.rows[start:start + len(trace)]
        telemetry = []
        for index, (row, exposure) in enumerate(zip(trace, selected, strict=True)):
            if row["frame"] != exposure["camera_frame"]:
                raise ValueError("Telemetry frame differs from its camera exposure")
            metadata = {k: v for k, v in exposure.items() if k not in ("offset", "size")}
            telemetry.append({**row, **metadata, "video_index": index, "video_time": index / FPS})
        poster_index = min(2 * FPS, len(selected) - 1)
        poster = selected[poster_index]
        with raw.open("rb") as stream:
            stream.seek(poster["offset"])
            (stage / "poster.jpg").write_bytes(stream.read(poster["size"]))
        shutil.copyfile(stage / "poster.jpg", stage / "chase.jpg")
        with front_raw.open("rb") as stream:
            offset, length = perception.front_offsets[poster_index]
            stream.seek(offset)
            (stage / "front.jpg").write_bytes(stream.read(length))
        write_json(stage / "telemetry.json", telemetry)
        write_replay(stage / "replay.json", rows, telemetry)
        video = encode_and_check(raw, stage / "chase.mp4", size, start, len(trace))
        front_video = encode_and_check(front_raw, stage / "front.mp4", FRONT_SIZE, 0, len(trace))
        files = {name: fingerprint(stage / name) for name in ARTIFACTS}
        media_root = (PROJECT / "web/media").resolve()
        entry = None
        if output.is_relative_to(media_root):
            directory = output.relative_to(media_root).as_posix()
            entry = dict(label=label, path=directory, cameras=["chase", "front"], trial_id=trial_id,
                         manifest=f"./media/{directory}/manifest.json", **disclosure)
        manifest = dict(purpose="ADO independent closed-loop showcase drive", **disclosure,
                        trial_id=trial_id, catalog_entry=entry, replay="replay.json",
                        producer=producer, quality_requested=quality, world_settings=observed_settings,
                        fit_id=fit["fit_id"],
                        model_path="models/fusion.npz", model_sha256=model_sha,
                        observer=observer.rendering, video=video, front_video=front_video, frame_period_s=DT,
                        pose_coordinates="CARLA world metres and Euler degrees",
                        control_timing="The policy observes frame t and commands the following tick.",
                        first_frame=trace[0]["frame"], last_frame=trace[-1]["frame"],
                        poster_frame=poster["camera_frame"], policy_render_profile=fit["render_profile"],
                        outcome=result["outcome"], summary=result, sha256=files)
        write_json(stage / "manifest.json", manifest)
        write_json(stage / "install.json", dict(output=str(output), manifest_sha256=fingerprint(stage / "manifest.json")))
    return install_staged(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scenario", choices=("trip", "pedestrian"), help="Prespecified seed-211 held-out drive")
    group.add_argument("--development-demo", action="store_true", help="Curated seed-31 Market drive in the training district")
    group.add_argument("--all", action="store_true", help="Record all three viewing drives sequentially on one fresh host")
    group.add_argument("--install-staged", type=Path, metavar="CHECKPOINT",
                       help="Verify and install a saved recording without starting CARLA")
    parser.add_argument("--quality", choices=("Medium", "High", "Epic"), default="High")
    parser.add_argument("--resolution", choices=RESOLUTIONS)
    parser.add_argument("--dlss-percent", "--dlss-percentage", choices=(67, 75), type=int)
    parser.add_argument("--port", type=int, default=2668)
    parser.add_argument("--output", type=Path, help="Fresh recording directory, or parent directory for --all")
    args = parser.parse_args()
    if args.install_staged is not None:
        if args.output is not None:
            parser.error("--install-staged uses the destination saved in the checkpoint")
        manifest = install_staged(args.install_staged)
        print(f"Installed {manifest['trial_id']} from {args.install_staged.resolve()}", flush=True)
        return
    if not 1 <= args.port <= 65533:
        parser.error("Choose a port between 1 and 65533")
    case = "market-demo" if args.development_demo else args.scenario or "trip"
    root = PROJECT / "web/media/showcase"
    resolution = args.resolution or ("1920x1080" if args.development_demo or args.all else "1280x720")
    percentage = args.dlss_percent or (67 if args.development_demo or args.all else 75)
    if args.all:
        root = args.output or root
        targets = [(name, root / name) for name in CASES]
    else:
        targets = [(case, args.output or root / case)]
    for _, target in targets:
        if target.exists():
            parser.error(f"A recording already exists at {target}; choose a fresh destination")
    if not args.all:
        _, output = targets[0]
        manifest = capture(case, args.quality, resolution, percentage, args.port, output)
        print(f"{manifest['outcome']}, {manifest['summary']['frames']} frames — {output.resolve()}", flush=True)
        return
    with simulator(args.port, attach=False, quality=args.quality) as world:
        for name, output in targets:
            manifest = run(world, name, output, quality=args.quality, resolution=resolution, percentage=percentage)
            print(f"{manifest['outcome']}, {manifest['summary']['frames']} frames — {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
