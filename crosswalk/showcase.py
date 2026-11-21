"""Film an independent, closed-loop CARLA drive with one exterior RGB observer."""
import argparse
from collections import deque
from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from threading import Condition

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


def capture(scenario, quality, resolution, percentage, port, output):
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
    with TemporaryDirectory(prefix=".showcase-", dir=output.parent) as work:
        work = Path(work)
        stage = work / "publish"
        stage.mkdir()
        raw = work / "chase.mjpeg"
        size = RESOLUTIONS[resolution]
        with raw.open("wb") as stream, policy_profile(fit["render_profile"]):
            observer = ChaseCapture(*size, percentage, stream)
            with simulator(port, attach=False, quality=quality) as world:
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
                    result, trace, *_ = episode(ObservedWorld(world, observer), scenario, SEED, "fusion", model,
                                                district="heldout", weather="ClearNoon",
                                                destination="market" if scenario == "trip" else None)
                finally:
                    observer.close()
        if not trace or not observer.rows or len(trace) != result["frames"]:
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
            telemetry.append({**row, **metadata, "video_index": index})
        poster = selected[min(2 * FPS, len(selected) - 1)]
        with raw.open("rb") as stream:
            stream.seek(poster["offset"])
            (stage / "poster.jpg").write_bytes(stream.read(poster["size"]))
        write_json(stage / "telemetry.json", telemetry)
        video = encode_and_check(raw, stage / "chase.mp4", size, start, len(trace))
        files = {p.name: fingerprint(p) for p in sorted(stage.iterdir())}
        manifest = dict(purpose="ADO independent closed-loop showcase drive", benchmark_member=False,
                        producer=producer, quality_requested=quality, world_settings=observed_settings,
                        fit_id=fit["fit_id"],
                        model_path="models/fusion.npz", model_sha256=model_sha,
                        observer=observer.rendering, video=video, frame_period_s=DT,
                        pose_coordinates="CARLA world metres and Euler degrees",
                        control_timing="The policy observes frame t and commands the following tick.",
                        first_frame=trace[0]["frame"], last_frame=trace[-1]["frame"],
                        poster_frame=poster["camera_frame"], policy_render_profile=fit["render_profile"],
                        outcome=result["outcome"], summary=result, sha256=files)
        write_json(stage / "manifest.json", manifest)
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("trip", "pedestrian"), default="trip")
    parser.add_argument("--quality", choices=("Medium", "High", "Epic"), default="High")
    parser.add_argument("--resolution", choices=RESOLUTIONS, default="1280x720")
    parser.add_argument("--dlss-percent", choices=(67, 75), type=int, default=75)
    parser.add_argument("--port", type=int, default=2668)
    parser.add_argument("--output", type=Path, help="Fresh directory for the complete recording")
    args = parser.parse_args()
    if not 1 <= args.port <= 65533:
        parser.error("Choose a port between 1 and 65533")
    output = args.output or PROJECT / "web/media/showcase" / args.scenario
    manifest = capture(args.scenario, args.quality, args.resolution, args.dlss_percent, args.port, output)
    print(f"{manifest['outcome']}, {manifest['summary']['frames']} frames — {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
