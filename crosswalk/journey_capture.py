"""Record and publish one continuous, closed-loop CARLA city drive."""
import argparse
from contextlib import ExitStack
from functools import partial
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import mkdtemp
from zipfile import ZIP_DEFLATED, ZipFile

from . import city_map
import cv2

from .drive import episode
from .experiment import PROJECT, fingerprint, read_json, save_arrays, write_json
from .runtime import MAP_NAME, producer_info, simulator
from .showcase import (ARTIFACTS, DT, FPS, FRONT_SIZE, ChaseCapture, ObservedWorld,
                       PerceptionCapture, encode_and_check, install_staged, policy_profile, write_replay)


NAME = "long-city-drive"
KIND = "journey"
QUALITY = "High"
CHASE_SIZE = (1920, 1080)
DLSS_PERCENTAGE = 67
SEED = 211
MIN_DURATION = 120
DEFAULT_MAX_STEPS = 9000
WEB_LIMIT = 92 * 1024**2
DEFAULT_OUTPUT = PROJECT / "web" / "media" / NAME
URBAN_MAP = "Town03_Opt"
CAPTURE_MAPS = (MAP_NAME, URBAN_MAP)
MAP_ASSET = "town.json"
CONTROLS = ("steer", "throttle", "brake")
MODULE = "crosswalk.journey_capture"
RAW_FILES = ("chase.mjpeg", "front.mjpeg", "perception.jsonl", "poster.jpg", "front.jpg",
             "telemetry.json", "replay.json")


class JourneyCamera(ChaseCapture):
    def attach(self, world, ego):
        self.ego = ego
        super().attach(world, ego)
        readback = self.rendering["actor_attributes"]
        requested = self.rendering["requested"]
        fixed = ("enable_dlss", "use_ray_tracing", "enable_postprocess_effects")
        numeric = ("image_size_x", "image_size_y", "dlss_screen_percentage", "sensor_tick", "fov")
        if (any(str(readback[key]).lower() != requested[key] for key in fixed)
                or any(readback[key] is None or float(readback[key]) != float(requested[key]) for key in numeric)):
            raise ValueError(f"CARLA did not accept the requested chase camera attributes: {readback!r}")

    def append(self, frame):
        super().append(frame)
        applied = self.ego.get_control()
        self.rows[-1]["applied_control"] = {key: float(getattr(applied, key)) for key in CONTROLS}


class JourneyWorld(ObservedWorld):
    def _remember_ego(self, actor):
        if actor is not None and actor.attributes.get("role_name") == "hero":
            if self.ego is not None and actor.id != self.ego.id:
                actor.destroy()
                raise RuntimeError("The journey attempted to spawn a second hero")
            self.ego = actor
        return actor

    def try_spawn_actor(self, *args, **kwargs):
        return self._remember_ego(self.world.try_spawn_actor(*args, **kwargs))

    def spawn_actor(self, *args, **kwargs):
        return self._remember_ego(self.world.spawn_actor(*args, **kwargs))


PERIMETER_VIEWS = ("left", "right", "rear_left", "rear_right")
PACE_SCHEMAS = ("ado-city-pace-v1", "ado-city-clearance-v1", "ado-city-progress-v1", "ado-city-maneuvers-v1")
SURROUND_SCHEMAS = ("ado-city-cruise-v1", *PACE_SCHEMAS)


class PerimeterCapture(PerceptionCapture):
    def __init__(self, camera, front, rows, streams, witness=None):
        super().__init__(camera, front, rows)
        self.streams = streams
        self.witness = witness
        self.witness_samples = {}
        if witness:
            witness.mkdir()

    def append(self, telemetry, observations):
        super().append(telemetry, observations)
        phase = telemetry.get("tactical_pass", {}).get("maneuver_phase")
        if self.witness is not None and 399. <= telemetry["progress_m"] < 548.:
            key = (int(telemetry["progress_m"] // 5), phase)
            if key not in self.witness_samples:
                # One complete control lidar/radar exposure per maneuver phase and five-metre bin.
                from .radar_geometry import RADARS
                filename = f"frame-{telemetry['frame']}.npz"
                pose = self.observer.ego.get_transform()
                save_arrays(self.witness / filename, frame=telemetry["frame"], lidar=observations["lidar"],
                            ego_xyz=[pose.location.x, pose.location.y, pose.location.z],
                            pitch_yaw_roll=[pose.rotation.pitch, pose.rotation.yaw, pose.rotation.roll],
                            **{r: observations[r] for r in RADARS})
                self.witness_samples[key] = filename
        for name, stream in self.streams.items():
            rgb = observations[name]
            if rgb.shape != (180, 320, 3) or rgb.dtype.name != "uint8":
                raise ValueError(f"Unexpected {name} perimeter camera at frame {telemetry['frame']}")
            ok, data = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 94])
            if not ok: raise RuntimeError(f"Failed to encode {name} camera")
            stream.write(data.tobytes())


def perimeter_artifacts(source):
    views = tuple(source.get("perimeter_views", ()))
    if views and views != PERIMETER_VIEWS:
        raise ValueError("Unexpected perimeter camera names")
    return tuple(name + ".mp4" for name in views)


def delivery_artifacts(manifest):
    replay = manifest.get("replay", "replay.json")
    if replay not in ("replay.json", "replay.json.gz"):
        raise ValueError("Unexpected replay location")
    items = tuple(replay if name == "replay.json" else name for name in ARTIFACTS)
    if manifest.get("map_asset") == MAP_ASSET:
        items += (MAP_ASSET,)
    return items + perimeter_artifacts(manifest)


def _snapshot(stage, policy):
    with ZipFile(stage / "capture-source.zip", "x", compression=ZIP_DEFLATED) as archive:
        for name, expected in policy["code_sha256"].items():
            path = Path(__file__).with_name(name)
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"Capture source changed during startup: {name}")
            archive.writestr(f"crosswalk/{name}", data)
    revision = None
    if (PROJECT / ".git").exists() and shutil.which("git"):
        commit = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], check=False, cwd=PROJECT,
                                capture_output=True, text=True)
        if commit.returncode == 0:
            revision = commit.stdout.strip()
    write_json(stage / "source-checkout.json", dict(head_at_capture_start=revision,
               source_archive="capture-source.zip", source_sha256=fingerprint(stage / "capture-source.zip"),
               note="Archive hashes identify the exact execution sources, including working tree edits."))


def _new_scene(scene_factory, observed_world, *args, **kwargs):
    scene = scene_factory(observed_world, *args, **kwargs)
    if observed_world.ego is None or scene.ego.id != observed_world.ego.id:
        scene.close()
        raise ValueError("The journey scene must own the captured hero")
    return scene


def _journal(stream):
    stream.flush()
    os.fsync(stream.fileno())


def _observed_settings(world):
    settings = world.get_settings()
    names = ("synchronous_mode", "no_rendering_mode", "fixed_delta_seconds", "substepping",
             "max_substep_delta_time", "max_substeps")
    if (not settings.synchronous_mode or settings.no_rendering_mode
            or settings.fixed_delta_seconds is None
            or not math.isclose(settings.fixed_delta_seconds, DT, abs_tol=1e-6)):
        raise ValueError("The journey requires synchronous CARLA graphics at the policy timestep")
    return {name: getattr(settings, name) for name in names}


def _camera_interval(summary, trace, camera, perception):
    if (not trace or len(trace) != summary["frames"] or len(trace) != len(perception.front_offsets)
            or not camera.rows):
        raise ValueError("The journey is missing scored observations or camera frames")
    start = trace[0]["frame"] - camera.rows[0]["camera_frame"]
    frames = camera.rows[start:start + len(trace)]
    if start < 0 or len(frames) != len(trace):
        raise ValueError("The scored journey extends outside the chase capture")
    first_elapsed = frames[0]["world_elapsed_s"]
    rows = []
    for index, (score, exposure) in enumerate(zip(trace, frames, strict=True)):
        frame = trace[0]["frame"] + index
        elapsed = exposure["world_elapsed_s"]
        if (score["frame"] != frame or exposure["camera_frame"] != frame
                or not math.isclose(elapsed, exposure["camera_timestamp_s"], abs_tol=1e-3)
                or not math.isclose(elapsed - first_elapsed, index * DT, abs_tol=1e-3)):
            raise ValueError(f"The recordings and score disagree at world frame {frame}")
        metadata = {key: value for key, value in exposure.items() if key not in ("offset", "size")}
        commanded = {key: score[key] for key in CONTROLS}
        rows.append({**score, **metadata, **exposure["applied_control"], "commanded_control": commanded,
                     "video_index": index, "video_time": index / FPS})
    return start, frames, rows


def _poster(path, offset, length, output):
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(length)
    if len(data) != length or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ValueError(f"Invalid poster exposure in {path}")
    output.write_bytes(data)


def _result(summary, required_event_kinds):
    events = summary["events"]
    failures = []
    if summary["outcome"] != "complete" or summary["progress_m"] < summary["route_length_m"] - 0.6:
        failures.append("route_incomplete")
    if summary["duration_s"] < MIN_DURATION:
        failures.append("below_minimum_duration")
    if summary["collision_events"]:
        failures.append("collision")
    if summary["red_light_violations"]:
        failures.append("red_light_violation")
    if summary["yield_violations"]:
        failures.append("yield_violation")
    milestones = ("triggered", "exposed", "cleared", "passed", "handled")
    if (len(events) != len(required_event_kinds) or {event["kind"] for event in events} != set(required_event_kinds)
            or any(event["violation"] or not all(event[key] for key in milestones) for event in events)):
        failures.append("planned_encounters_incomplete")
    return {"passed": not failures, "failure_reasons": failures, "events": events}


def _verify_video(path, size, count):
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,codec_name,r_frame_rate,nb_read_frames,pix_fmt",
         "-of", "json", str(path)], capture_output=True, text=True, check=True,
    )
    stream, = json.loads(completed.stdout)["streams"]
    if (stream["width"], stream["height"], int(stream["nb_read_frames"]), stream["r_frame_rate"],
            stream["codec_name"], stream["pix_fmt"]) != (*size, count, f"{FPS}/1", "h264", "yuv420p"):
        raise ValueError(f"Web video does not match the captured timeline: {stream!r}")
    return stream


def _web_video(master, master_info, output, count, *, keep_resolution=False):
    if master.stat().st_size < WEB_LIMIT:
        shutil.copyfile(master, output)
        return master_info

    kbps = int(WEB_LIMIT * 8 * FPS * 0.965 / (1000 * count))
    size = (master_info["width"], master_info["height"])
    if size == CHASE_SIZE and kbps < 2400 and not keep_resolution:
        size = (1280, 720)
    common = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", str(master),
              "-filter_threads", "1", "-vf", f"scale={size[0]}:{size[1]}", "-an", "-c:v", "libx264",
              "-threads", "2", "-pix_fmt", "yuv420p", "-preset", "slow", "-b:v", f"{kbps}k",
              "-passlogfile", str(output.with_suffix(".pass"))]
    subprocess.run([*common, "-pass", "1", "-f", "null", os.devnull], check=True)
    subprocess.run([*common, "-pass", "2", "-movflags", "+faststart", str(output)], check=True)
    if output.stat().st_size >= WEB_LIMIT:
        raise ValueError(f"Web video exceeds the {WEB_LIMIT // 1024**2} MiB size budget: {output}")
    return _verify_video(output, size, count)


def _need_video_tools():
    if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        raise RuntimeError("FFmpeg and FFprobe are required to encode and verify the city drive")


def _source_checksums(scene_factory, model_type):
    from inspect import getfile
    from . import behavior_v2, control, journey_ambient, lamp, runtime, sensors, showcase, urban_map

    paths = [Path(getfile(source)) for source in (episode, scene_factory, model_type, city_map, control, lamp,
                                                 runtime, sensors, showcase)]
    paths.extend(Path(getfile(source)) for source in (behavior_v2, journey_ambient, urban_map))
    paths.append(Path(__file__))
    return {path.name: fingerprint(path) for path in paths}


def _load_inputs(model_path):
    from .city_drive import load_driver
    from .journey import JourneyScenario

    path = Path(model_path).resolve(strict=True)
    saved_sha = fingerprint(path)
    model, feature_names, schema = load_driver(path)
    if fingerprint(path) != saved_sha:
        raise ValueError("The driving checkpoint changed while it was loading")
    fit_path = PROJECT / "models" / "fit.json"
    fit = read_json(fit_path)
    policy = dict(path=path.name, sha256=saved_sha, schema=schema,
                  camera_profile=fit["render_profile"], camera_profile_source_sha256=fingerprint(fit_path),
                  code_sha256=_source_checksums(JourneyScenario, type(model)))
    if schema in SURROUND_SCHEMAS:
        from . import camera_geometry, lidar_motion, overtake, radar_geometry, robotaxi, signal_observer, surround, surround_vision, tracking
        for module in (camera_geometry, lidar_motion, overtake, radar_geometry, robotaxi, signal_observer, surround, surround_vision, tracking):
            source = Path(module.__file__)
            policy["code_sha256"][source.name] = fingerprint(source)
        policy["tactical_sha256"] = fingerprint(overtake.MODEL)
        policy["surround_detector_sha256"] = fingerprint(surround_vision.MODEL_PATH)
        policy["signal_classifier_sha256"] = fingerprint(signal_observer.CALIBRATION)
        policy["control_inputs"] = "Navigation, NanoDet front RGB, camera-measured signal color, lidar and forward radar. The pass planner also measures both lanes in 360-degree lidar."
        policy["display_inputs"] = "Five registered RGB cameras, 360-degree lidar and five Doppler radars."
    if schema in PACE_SCHEMAS:
        from . import barrier, barrier_chain, barrier_posts, barrier_scene, city_cruise, city_policy, cycle_association, speed_signs
        for module in (barrier, barrier_chain, barrier_posts, barrier_scene, city_cruise, city_policy, cycle_association, speed_signs):
            source = Path(module.__file__)
            policy["code_sha256"][source.name] = fingerprint(source)
        policy["barrier_sha256"] = fingerprint(barrier.MODEL)
        policy["barrier_post_appearance_sha256"] = fingerprint(barrier_posts.MODEL)
    if schema in ("ado-city-clearance-v1", "ado-city-progress-v1"):
        from . import city_clearance, city_drive, city_pace
        for module in (city_clearance, city_drive, city_pace):
            source = Path(module.__file__)
            policy["code_sha256"][source.name] = fingerprint(source)
    if schema in ("ado-city-progress-v1", "ado-city-maneuvers-v1"):
        # Include transitive imports and evaluation code as well as the direct model and rig.
        policy["code_sha256"] = {p.name: fingerprint(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        policy["control_feature_names"] = list(feature_names)
        from .city_drive import LoggedDriver
        model = LoggedDriver(model)
    if schema == "ado-city-maneuvers-v1":
        from .maneuver_choice import MODEL as maneuver_model
        policy["maneuver_choice_sha256"] = fingerprint(maneuver_model)
        policy["control_inputs"] += " Discretionary lane changes use the surveyed adjacent lane and ego-motion-compensated lidar returns."
    return model, JourneyScenario, model.new_observer, policy


def _capture(world, stage, output, model, scene_factory, observer_factory, policy, seed, max_steps, destination,
             map_name, crossing_delay=0., population_profile="standard"):
    from .journey import required_event_kinds

    producer = producer_info(world)
    if producer["map"] not in CAPTURE_MAPS or (map_name and producer["map"] != map_name):
        raise ValueError(f"Unexpected CARLA map for the city drive: {producer['map']!r}")
    with_barriers = policy.get("schema") in PACE_SCHEMAS
    expected_events = required_event_kinds(producer["map"], barriers=with_barriers,
                                          maneuvers=population_profile == "city-maneuvers")
    if population_profile in ("city-life", "city-maneuvers"):
        from . import ambient, city_population, district, population_score, traffic_cones, vision
        from . import policy as base_policy, world as scene_world
        if not with_barriers or producer["map"] != URBAN_MAP:
            raise ValueError("City-life capture requires the Town03 pace or clearance policy")
        scene_factory = partial(scene_factory, population_profile=population_profile)
        for module in (ambient, city_population, district, base_policy, population_score, traffic_cones, vision, scene_world):
            path = Path(module.__file__)
            policy["code_sha256"][path.name] = fingerprint(path)
        _snapshot(stage, policy)
    if with_barriers:
        scene_factory = partial(scene_factory, barriers=True)
    settings = _observed_settings(world)
    raw = stage / "raw"
    raw.mkdir()
    city_map.export(world, raw / MAP_ASSET)
    town = read_json(raw / MAP_ASSET)
    if town["producer"] != producer or not town["roads"]:
        raise ValueError("Viewer road geometry must come from the recording world")
    trial_id = output.name if output.name in (NAME, "surround-city", "urban-pace", "city-life", "city-progress", "city-maneuvers") else f"city-{seed}-{policy['sha256'][:8]}"
    with ((raw / "chase.mjpeg").open("wb") as chase, (raw / "front.mjpeg").open("wb") as front,
          (raw / "perception.jsonl").open("w", encoding="utf-8", newline="\n") as rows,
          policy_profile(policy["camera_profile"]), ExitStack() as extras):
        camera = JourneyCamera(*CHASE_SIZE, DLSS_PERCENTAGE, chase)
        surround = policy.get("schema") in SURROUND_SCHEMAS
        views = PERIMETER_VIEWS if surround else ()
        streams = {name: extras.enter_context((raw / f"{name}.mjpeg").open("wb")) for name in views}
        witness = stage / "maneuver-sweeps" if population_profile == "city-maneuvers" else None
        perception = PerimeterCapture(camera, front, rows, streams, witness) if surround else PerceptionCapture(camera, front, rows)
        episode_options = dict(surround=True, tactical=True, evidence_dir=stage / "surround-evidence") if surround else {}
        observed = JourneyWorld(world, camera)
        try:
            summary, trace, *_ = episode(
                observed, KIND, seed, policy.get("schema", "ado-behavior-v2"), model, weather="ClearNoon", district="townwide",
                recording=perception, run_id=trial_id, max_steps=max_steps, destination=destination, crossing_delay=crossing_delay,
                scene_factory=partial(_new_scene, scene_factory), observation_factory=observer_factory, **episode_options,
            )
        finally:
            camera.close()
        for stream in (chase, front, rows, *streams.values()):
            _journal(stream)
    start, frames, telemetry = _camera_interval(summary, trace, camera, perception)
    if any(row["map"].split(".", 1)[0] != producer["map"] for row in trace):
        raise ValueError("Scored telemetry and road geometry identify different CARLA maps")
    poster_index = min(2 * FPS, len(frames) - 1)
    poster = frames[poster_index]
    _poster(raw / "chase.mjpeg", poster["offset"], poster["size"], raw / "poster.jpg")
    _poster(raw / "front.mjpeg", *perception.front_offsets[poster_index], raw / "front.jpg")
    write_json(raw / "telemetry.json", telemetry)
    write_replay(raw / "replay.json", raw / "perception.jsonl", telemetry)
    source = dict(capture_id=stage.name, original_output=str(output), trial_id=trial_id, producer=producer,
                  quality_requested=QUALITY, world_settings=settings, policy=policy, observer=camera.rendering,
                  ego_actor_id=observed.ego.id, first_frame=trace[0]["frame"], last_frame=trace[-1]["frame"],
                  poster_frame=poster["camera_frame"], warmup_frames=start,
                  chase_source_frames=len(camera.rows), front_source_frames=len(perception.front_offsets),
                  map_asset=MAP_ASSET, required_event_kinds=list(expected_events), summary=summary,
                  **_result(summary, expected_events))
    source["perimeter_views"] = list(views)
    if population_profile == "city-maneuvers":
        source["maneuver_sweeps"] = {name: fingerprint(stage / "maneuver-sweeps" / name)
                                     for name in perception.witness_samples.values()}
    if policy.get("schema") in ("ado-city-progress-v1", "ado-city-maneuvers-v1"):
        names = policy["control_feature_names"]
        if len(model.observations) != len(trace) or any(len(row) != len(names) for row in model.observations):
            raise ValueError("The replay and native policy feature tape have different lengths")
        tape = stage / "policy-observations.npz"
        save_arrays(tape, frame=[row["frame"] for row in trace], features=model.observations,
                    feature_names=names, speed_request_mps=model.requested_speed,
                    posted_limit_kmh=model.posted_limits)
        source["policy_observations_sha256"] = fingerprint(tape)
    if population_profile != "standard":
        source["population_profile"] = population_profile
        reference = stage / "population-reference.jsonl.gz"
        score = population_score.score_population(raw / "perception.jsonl", reference)
        write_json(stage / "population-score.json", score)
        source["population_density"] = score["radii_m"]
        source["population_reference_sha256"] = fingerprint(reference)
        source["population_score_sha256"] = fingerprint(stage / "population-score.json")
        source["capture_source_sha256"] = fingerprint(stage / "capture-source.zip")
        source["capture_checkout"] = read_json(stage / "source-checkout.json")
        if summary["background_traffic_activity"]["participant_contacts"]:
            source["failure_reasons"].append("background_participant_contact")
        if not score["radii_m"]["35"]["two_of_each_frames"]:
            source["failure_reasons"].append("no_close_simultaneous_population")
        source["passed"] = not source["failure_reasons"]
    source["raw_sha256"] = {name: fingerprint(raw / name) for name in
                            (*RAW_FILES, MAP_ASSET, *(v + ".mjpeg" for v in views))}
    write_json(stage / "capture.json", source)


def _install_delivery(stage):
    manifest = read_json(Path(stage) / "manifest.json")
    return install_staged(stage, artifacts=delivery_artifacts(manifest), retry_module=MODULE)


def finalize_staged(stage, *, include_attempt=False):
    """Encode and install a synchronized capture without opening a simulator."""
    _need_video_tools()
    stage = Path(stage).resolve(strict=True)
    source = read_json(stage / "capture.json")
    output = Path(source["original_output"])
    raw = stage / "raw"
    map_asset = source.get("map_asset")
    raw_files = (*RAW_FILES, MAP_ASSET) if map_asset == MAP_ASSET else RAW_FILES
    perimeter_artifacts(source)
    raw_files += tuple(v + ".mjpeg" for v in source.get("perimeter_views", []))
    if (map_asset not in (None, MAP_ASSET) or not output.is_absolute() or source["capture_id"] != stage.name
            or set(source["raw_sha256"]) != set(raw_files)
            or any(fingerprint(raw / name) != sha for name, sha in source["raw_sha256"].items())):
        raise ValueError(f"Capture checkpoint failed verification: {stage}")
    final = Path(mkdtemp(prefix="delivery-", dir=stage))
    master = final / "master"
    master.mkdir()
    count = source["summary"]["frames"]
    start = source["warmup_frames"]
    video = {}
    masters = {}
    for name, size, first in (("chase", CHASE_SIZE, start), ("front", FRONT_SIZE, 0)):
        path = master / f"{name}.mp4"
        info = encode_and_check(raw / f"{name}.mjpeg", path, size, first, count)
        masters[name] = dict(path=f"master/{name}.mp4", sha256=fingerprint(path), file_bytes=path.stat().st_size,
                             **info)
        video[name] = _web_video(path, info, final / f"{name}.mp4", count,
                                 keep_resolution=source.get("population_profile") in ("city-life", "city-maneuvers"))
    views = source.get("perimeter_views", [])
    for name in views:
        video[name] = encode_and_check(raw / f"{name}.mjpeg", final / f"{name}.mp4", (320, 180), 0, count)
    copied = ("poster.jpg", "front.jpg", "telemetry.json")
    replay = "replay.json"
    if source.get("population_profile") in ("city-life", "city-maneuvers") and (raw / replay).stat().st_size >= 100 * 1024**2:
        replay = "replay.json.gz"
        with (raw / "replay.json").open("rb") as original, (final / replay).open("xb") as target:
            with gzip.GzipFile(fileobj=target, mode="wb", compresslevel=5, mtime=0) as compressed:
                shutil.copyfileobj(original, compressed)
        with gzip.open(final / replay, "rb") as check:
            if hashlib.file_digest(check, "sha256").hexdigest() != source["raw_sha256"]["replay.json"]:
                raise ValueError("Compressed replay does not reproduce the native capture")
    else:
        copied += (replay,)
    if map_asset:
        copied += (MAP_ASSET,)
    for name in copied:
        shutil.copyfile(raw / name, final / name)
    shutil.copyfile(final / "poster.jpg", final / "chase.jpg")
    disclosure = dict(cohort="showcase", district=source["summary"]["district"], benchmark_member=False,
                      selection_note="One closed-loop CARLA development route with staged encounters.")
    media = (PROJECT / "web" / "media").resolve()
    catalog = None
    directory = output.relative_to(media).as_posix() if output.is_relative_to(media) else None
    map_name = source["producer"]["map"]
    map_url = None
    artifacts = delivery_artifacts({**source, "replay": replay})
    if map_asset:
        map_url = f"./media/{directory}/{map_asset}" if directory is not None else f"./{map_asset}"
    if (source["passed"] or include_attempt) and directory is not None:
        title = "Urban pace and barriers" if source["policy"].get("schema") in PACE_SCHEMAS else ("Surround city drive" if views else "Long city drive")
        if source.get("population_profile") == "city-life":
            title = "City progress" if source["trial_id"] == "city-progress" else "City life"
        if source.get("population_profile") == "city-maneuvers":
            title = "City maneuvers"
        label = title if source["passed"] else f'City development drive · {source["summary"]["outcome"]}'
        catalog = dict(label=label, path=directory, cameras=["chase", "front", *views], trial_id=source["trial_id"],
                       manifest=f"./media/{directory}/manifest.json", **disclosure)
        if map_url:
            catalog.update(map_url=map_url, map=map_name)
    manifest = {key: value for key, value in source.items() if key != "original_output"}
    manifest.update(camera_videos=video, purpose="Continuous CARLA city drive", **disclosure, catalog_entry=catalog,
                    frame_period_s=DT, video=video["chase"], front_video=video["front"], replay=replay,
                    telemetry="telemetry.json", outcome=source["summary"]["outcome"], master=masters,
                    source_mode="one closed-loop simulation with simultaneous camera capture", teacher=False,
                    control_timing="Scalar controls were read back from the ego on frame t. "
                                   "The policy's command from frame t is for frame t+1.",
                    pose_coordinates="CARLA world metres and Euler degrees", map_url=map_url, map=map_name,
                    sha256={name: fingerprint(final / name) for name in artifacts})
    write_json(final / "manifest.json", manifest)
    if source["passed"] or include_attempt:
        write_json(final / "install.json", dict(output=str(output), manifest_sha256=fingerprint(final / "manifest.json")))
        print(f"Installing verified city drive from {final}", flush=True)
        return _install_delivery(final)
    print(f"City drive did not pass verification. Review the recording in {final}", flush=True)
    return manifest


def _prepare(output, model_path, seed, max_steps, quality, resolution, percentage):
    if (quality, resolution, percentage) != (QUALITY, "1920x1080", DLSS_PERCENTAGE):
        raise ValueError("The city drive requires High quality, 1920x1080 chase and DLSS 67")
    if not 0 <= seed <= 2147483647 or max_steps < 1:
        raise ValueError("Use a supported seed and a positive step limit")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a fresh city drive destination to preserve {output}")
    output = output.resolve()
    _need_video_tools()
    return output, _load_inputs(model_path)


def _record(world, output, inputs, seed, max_steps, destination, map_name, crossing_delay=0., population_profile="standard"):
    if not math.isfinite(crossing_delay) or not 0. <= crossing_delay <= 5.:
        raise ValueError("Crossing delay must be between zero and five seconds")
    model, scene_factory, observer_factory, policy = inputs
    checkpoint_root = Path(os.environ.get("ADO_JOURNEY_CAPTURE_ROOT") or PROJECT / "out" / "journey-captures")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=f"{NAME}-", dir=checkpoint_root))
    print(f"City drive capture checkpoint: {stage}", flush=True)
    write_json(stage / "request.json", dict(output=str(output), policy=policy, seed=seed, max_steps=max_steps,
                                             destination=destination, crossing_delay_s=crossing_delay, population_profile=population_profile,
                                             quality=QUALITY, requested_map=map_name,
                                             chase_size=CHASE_SIZE, dlss_percentage=DLSS_PERCENTAGE))
    try:
        _capture(world, stage, output, model, scene_factory, observer_factory, policy, seed, max_steps, destination,
                 map_name, crossing_delay, population_profile)
    except Exception as error:
        write_json(stage / "failure.json", dict(error_type=type(error).__name__, message=str(error)))
        raise
    print(f'Re-encode without CARLA using python -m {MODULE} --finalize-staged "{stage}"', flush=True)
    return stage


def run(world, output=DEFAULT_OUTPUT, *, model_path, seed=SEED, max_steps=DEFAULT_MAX_STEPS,
        destination=None, map_name=None, quality=QUALITY, resolution="1920x1080", percentage=DLSS_PERCENTAGE,
        crossing_delay=0., population_profile="standard"):
    """Record once on a caller-owned CARLA world launched at High quality."""
    output, inputs = _prepare(output, model_path, seed, max_steps, quality, resolution, percentage)
    return finalize_staged(_record(world, output, inputs, seed, max_steps, destination, map_name, crossing_delay,
                                   population_profile))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=Path, help="CARLA-trained driving policy checkpoint")
    group.add_argument("--finalize-staged", type=Path, metavar="CAPTURE", help="Encode a saved source without CARLA")
    group.add_argument("--install-staged", type=Path, metavar="DELIVERY", help="Install verified files without CARLA")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-attempt", action="store_true", help="Also export an unsuccessful drive with its actual score")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--population-profile", choices=("standard", "city-life", "city-maneuvers"), default="standard")
    parser.add_argument("--crossing-delay", type=float, default=0., help="Additional pedestrian entry delay in simulated seconds")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--map", dest="map_name", choices=CAPTURE_MAPS, default=URBAN_MAP)
    parser.add_argument("--destination", choices=("east", "south", "market"))
    parser.add_argument("--port", type=int, default=2668)
    parser.add_argument("--attach", action="store_true", help="Record using an already commissioned idle High-quality Town03 server")
    parser.add_argument("--quality", choices=(QUALITY,), default=QUALITY)
    parser.add_argument("--resolution", choices=("1920x1080",), default="1920x1080")
    parser.add_argument("--dlss-percentage", type=int, choices=(DLSS_PERCENTAGE,), default=DLSS_PERCENTAGE)
    args = parser.parse_args()
    if args.install_staged:
        manifest = _install_delivery(args.install_staged)
    elif args.finalize_staged:
        manifest = finalize_staged(args.finalize_staged, include_attempt=args.include_attempt)
    else:
        if not 1 <= args.port <= 65533:
            parser.error("Choose a port between 1 and 65533")
        if not args.model.is_file():
            parser.error(f"No policy checkpoint at {args.model}")
        if args.output.exists() or args.output.is_symlink():
            parser.error(f"A recording already exists at {args.output}")
        output, inputs = _prepare(args.output, args.model, args.seed, args.max_steps, args.quality,
                                  args.resolution, args.dlss_percentage)
        options = {URBAN_MAP: (None, "east", "market"), MAP_NAME: (None, "east", "south")}
        if args.destination not in options[args.map_name]:
            parser.error(f"Destination {args.destination!r} is unavailable on {args.map_name}")
        with simulator(args.port, attach=args.attach, quality=args.quality, map_name=args.map_name) as world:
            stage = _record(world, output, inputs, args.seed, args.max_steps, args.destination, args.map_name,
                            args.crossing_delay, args.population_profile)
        manifest = finalize_staged(stage, include_attempt=args.include_attempt)
    status = "passed" if manifest["passed"] else "review required"
    print(f"{NAME}: {status}, {manifest['outcome']}, {manifest['summary']['frames']} frames", flush=True)
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
