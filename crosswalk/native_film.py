"""Film the recorded held-out Market trip in CARLA's native replayer."""
import argparse
from contextlib import ExitStack
from dataclasses import dataclass, field
import gzip
import json
import math
from pathlib import Path
import re
import shutil
import struct
from tempfile import TemporaryDirectory, mkdtemp

import carla
import cv2
import numpy as np

from .district import trip_points
from .drive import DT
from .experiment import PROJECT, fingerprint, read_json, valid_checkpoint, write_json
from .robotaxi import PARTS
from .runtime import producer_info, simulator
from .sensors import Rig
from .showcase import (ChaseCapture, FPS, FRONT_SIZE, MOUNT, encode_and_check, install_staged as install_recording,
                       policy_profile, transform_fields)
from .tracking import Tracker
from .vision import Detector


TRIAL = "103-trip-fusion"
OUTPUT = PROJECT / "web/media/recorded-trip"
SOURCE = PROJECT / "results/native-replays" / (TRIAL + ".rec")
SIZE = (1920, 1080)
PACKET = struct.Struct("<BI")
VECTOR_PAIR = struct.Struct("<I6d")
CONTROL = struct.Struct("<IfffBi")
FRAME = struct.Struct("<Qdd")
# The terminal native packet is a sentinel with duration -1. Sampling just before
# each tick boundary includes its preceding pose and avoids CARLA enabling autopilot at EOF.
PLAYBACK_START = DT - .00001


def _artifacts(perception):
    shared = ("chase.mp4", "chase.jpg", "poster.jpg", "replay.json", "original-summary.json", "original-trace.jsonl.gz")
    return shared + (("front.mp4", "front.jpg") if perception else ())


def install_staged(stage):
    manifest = read_json(Path(stage)/"manifest.json")
    if manifest.get("trial_id") != TRIAL or manifest.get("cameras") not in (["chase"], ["chase", "front"]):
        raise ValueError("The checkpoint is not a native Market playback capture")
    return install_recording(stage, artifacts=_artifacts("front" in manifest["cameras"]), retry_module="crosswalk.native_film")


@dataclass
class NativeFrame:
    number: int
    duration: float
    elapsed: float
    simulation_time: float | None = None
    poses: dict = field(default_factory=dict)
    velocity: tuple | None = None
    control: tuple | None = None


def participants(summary):
    actors = {}
    current = None
    for line in summary.splitlines():
        creation = re.match(r"^ Create (\d+): (\S+) ", line)
        if creation:
            current = int(creation[1]) if creation[2].startswith(("vehicle.", "walker.")) else None
            if current is not None:
                actors[current] = dict(blueprint=creation[2], role=None)
        elif current is not None and line.startswith("  role_name = "):
            actors[current]["role"] = line.split(" = ", 1)[1]
    return actors


def decode(record, meta):
    """CARLA UE5 recorder v1 uses little-endian FVectors of three doubles."""
    actors = participants(meta["recorder_summary"])
    hero, = [a for a, info in actors.items() if info["role"] == "hero"]
    data = memoryview(record.read_bytes())
    if bytes(data[:18]) != b"\x01\x00\x0e\x00CARLA_RECORDER":
        raise ValueError("Unexpected CARLA recorder version or signature")
    map_length, = struct.unpack_from("<H", data, 26)
    if bytes(data[28:28+map_length]).decode() != meta["producer"]["map"]:
        raise ValueError("Mismatch between recorder and observer map")
    offset = 28 + map_length
    frames, box = [], None
    while offset < len(data):
        kind, length = PACKET.unpack_from(data, offset)
        offset += PACKET.size
        payload = data[offset:offset+length]
        offset += length
        if len(payload) != length:
            raise ValueError("Incomplete CARLA recorder packet")
        if kind == 0:
            frames.append(NativeFrame(*FRAME.unpack(payload)))
        elif kind == 20:
            frames[-1].simulation_time, = struct.unpack("<d", payload)
        elif kind in (6, 8, 12, 13):
            layout = CONTROL if kind == 8 else VECTOR_PAIR
            count, = struct.unpack_from("<H", payload)
            if len(payload) != 2 + count * layout.size:
                raise ValueError(f"Unexpected size for native packet {kind}")
            for actor, *values in layout.iter_unpack(payload[2:]):
                if kind == 6 and actor in actors:
                    x, y, z, roll, pitch, yaw = values
                    frames[-1].poses[actor] = dict(x=x/100, y=y/100, z=z/100, roll=roll, pitch=pitch, yaw=yaw)
                elif actor == hero:
                    if kind == 8:
                        frames[-1].control = tuple(values[:3])
                    elif kind == 12:
                        frames[-1].velocity = tuple(values[:3])
                    elif kind == 13:
                        box = [v/100 for v in values]
    return frames, actors, hero, box


def load_source(record, checkpoint):
    meta = read_json(record.with_suffix(".json"))
    if meta["trial_id"] != TRIAL or meta["filename"] != record.name or fingerprint(record) != meta["sha256"]:
        raise ValueError("Native recording disagrees with original observer metadata")
    if not valid_checkpoint(checkpoint):
        raise ValueError("Locked trial checkpoint failed its file hash check")
    score = read_json(checkpoint / "summary.json")
    with gzip.open(checkpoint / "trace.jsonl.gz", "rt", encoding="utf-8") as stream:
        trace = [json.loads(line) for line in stream]
    if (score["trial_id"], score["cohort"], score["district"], score["destination"], score["outcome"]) != (
            TRIAL, "primary", "heldout", "market", "complete"):
        raise ValueError("A completed primary held-out Market checkpoint is required")
    if score["producer"] != meta["producer"] or len(trace) != score["frames"] or not trace:
        raise ValueError("Checkpoint and native recorder producer or trace mismatch")
    frames, actors, hero, box = decode(record, meta)
    start = meta["observed_start_frame"]
    if len(frames) != meta["observed_stop_frame"] - start or box is None:
        raise ValueError("Incomplete native recording interval or ego geometry")
    for number, frame in enumerate(frames, 1):
        expected_duration = -1. if number == len(frames) else DT
        if (frame.number != number or abs(frame.duration - expected_duration) > 1e-7 or
                abs(frame.elapsed - (number-1)*DT) > 1e-5 or
                abs(frame.simulation_time - meta["start_elapsed_s"] - number*DT) > 1e-5):
            raise ValueError(f"Native simulation clock mismatch at packet frame {number}")
    if abs(frames[-1].simulation_time - meta["stop_elapsed_s"]) > 1e-6:
        raise ValueError("Recorder stop clock differs from observer stop clock")
    first = trace[0]["frame"] - start - 1
    if first < 0 or first + len(trace) >= len(frames):
        raise ValueError("The recording must include the tick applying the final scored command")
    for index, row in enumerate(trace, first):
        frame, following = frames[index:index+2]
        if (row["trial_id"] != TRIAL or row["policy_version"] != score["policy_version"]
                or row["frame"] != start + frame.number or hero not in frame.poses
                or abs(math.hypot(*frame.velocity) - row["speed_mps"]) > .000501
                or any(abs(value-row[key]) > .0000051 for value, key in
                       zip(following.control, ("steer", "throttle", "brake"), strict=True))):
            raise ValueError(f"Original speed, controls or frame mismatch at {row['frame']}")
    return meta, score, trace, frames[first:first+len(trace)], actors, hero, box


def matrix(values):
    return np.array(carla.Transform(carla.Location(**{k: values[k] for k in ("x", "y", "z")}),
                                    carla.Rotation(**{k: values.get(k, 0.) for k in ("roll", "pitch", "yaw")})).get_matrix())


def align(exposures, target, hero):
    """Search contiguous windows. Validate the actual RGB exposure and world snapshot, including rotation."""
    mount = matrix(MOUNT)
    native = np.stack([matrix(frame.poses[hero]) for frame in target])
    expected_camera = native @ mount
    poses = np.stack([matrix(row["ego_pose"]) for row in exposures])
    cameras = np.stack([matrix(row["camera_pose"]) for row in exposures])
    if len(poses) < len(native):
        return None, {"status": "unverified", "reason": "Recorded playback shorter than scored interval"}
    candidates = []
    for offset in range(len(poses)-len(native)+1):
        end = offset + len(native)
        distance = [np.linalg.norm(actual[:, :3, 3]-original[:, :3, 3], axis=1).max()
                    for actual, original in ((poses[offset:end], native), (cameras[offset:end], expected_camera))]
        angle = []
        for actual, original in ((poses[offset:end], native), (cameras[offset:end], expected_camera)):
            cosine = (np.einsum("nij,nij->n", actual[:, :3, :3], original[:, :3, :3]) - 1.)/2.
            angle.append(np.degrees(np.arccos(np.clip(cosine, -1., 1.))).max())
        candidates.append((max(distance), max(angle), offset, [float(v) for v in distance]))
    candidates.sort()
    passing = [v for v in candidates if v[0] <= .035 and v[1] <= .15]
    error, angle, offset, (ego_error, camera_error) = passing[0] if passing else candidates[0]
    status = "pose_verified" if len(passing) == 1 else "unverified"
    return (offset if status == "pose_verified" else None), dict(status=status, position_tolerance_m=.035,
        rotation_tolerance_deg=.15, passing_windows=len(passing), best_window=offset, max_ego_error_m=ego_error,
        max_observer_error_m=camera_error, max_rotation_error_deg=float(angle))


def _map(frame, actors, hero, path):
    ego = frame.poses[hero]
    others = []
    traffic = hazard = None
    # The scenario creates the crossing walker before it creates any ambient walkers.
    walkers = [key for key, item in actors.items() if item["blueprint"].startswith("walker.")]
    subject = min(walkers)
    for key, item in actors.items():
        pose = frame.poses[key]
        xy = [round(pose["x"], 2), round(pose["y"], 2)]
        if key == subject:
            hazard = xy
        elif item["role"] == "oncoming":
            traffic = xy
        elif key != hero:
            others.append(dict(type="walker" if key in walkers else "vehicle", x=xy[0], y=xy[1], yaw=pose["yaw"]))
    return dict(path=path, ego=[ego["x"], ego["y"], ego["yaw"]], ego_z=ego["z"],
                others=others, traffic=traffic, hazard=hazard, hazard_type="pedestrian")


def actor_alignment(exposures, scored, actors):
    """Measure all replayed road-user locations against the original movement packets."""
    worst = {}
    for actor_id, actor in actors.items():
        errors = []
        for exposure, frame in zip(exposures, scored, strict=True):
            actual = exposure["road_users"].get(actor_id)
            if actual is None:
                break
            original = frame.poses[actor_id]
            errors.append(math.dist([actual[k] for k in ("x", "y", "z")],
                                    [original[k] for k in ("x", "y", "z")]))
        worst[str(actor_id)] = dict(blueprint=actor["blueprint"], role=actor["role"],
                                    frames_checked=len(errors), max_position_error_m=max(errors, default=None))
    return worst


def display_returns(lidar, radar):
    """Preserve measured height while retaining the dashboard's XY and signed Doppler columns."""
    points = lidar[(lidar[:, 2] > -2.05) & (lidar[:, 2] < 1.5), :3]
    points = points[::max(1, math.ceil(len(points)/300))].astype(float)
    points[:, 2] += 2.30
    velocity, azimuth, altitude, depth = radar[::max(1, math.ceil(len(radar)/85))].T
    flat = depth*np.cos(altitude)
    radar_points = np.column_stack((2+flat*np.cos(azimuth), flat*np.sin(azimuth), velocity,
                                   .9+depth*np.sin(altitude)))
    return points.round(2).tolist(), radar_points.astype(float).round(2).tolist()


def _cleanup(client, world, initial_ids):
    # The native replayer enables autopilot at EOF and may leave actors behind after Stop(false).
    try:
        client.stop_replayer(False)
    finally:
        remaining = [actor for actor in world.get_actors() if actor.id not in initial_ids
                     and (actor.type_id.startswith(("vehicle.", "walker.", "sensor."))
                          or actor.attributes.get("role_name", "").startswith("robotaxi_"))]
        for actor in sorted(remaining, key=lambda a: a.type_id.startswith(("vehicle.", "walker."))):
            if actor.is_alive:
                actor.destroy()


def _restore_body_materials(actors):
    # Actor creation and parenting are native packets. The material and collision RPCs
    # in BodyKit are runtime-only, so restore them on the replayed parts.
    colors = {"robotaxi_" + part.name.replace(" ", "_"): part.color for part in PARTS}
    textures, restored = {}, set()
    for actor in actors:
        role = actor.attributes.get("role_name")
        if role not in colors or actor.type_id != "static.prop.briefcase":
            continue
        color = colors[role]
        if color not in textures:
            texture = carla.TextureColor(2, 2)
            for x in range(2):
                for y in range(2):
                    texture.set(x, y, carla.Color(*color))
            textures[color] = texture
        actor.set_simulate_physics(False)
        actor.set_collisions(False)
        actor.apply_texture(carla.MaterialParameter.Diffuse, textures[color])
        restored.add(role)
    return dict(expected_parts=len(colors), restored_parts=len(restored), missing_roles=sorted(colors.keys()-restored))


def _record(world, client, source, score, frames, participants_, box, work, perception):
    settings = world.get_settings()
    if not settings.synchronous_mode or settings.no_rendering_mode or abs((settings.fixed_delta_seconds or 0)-DT) > 1e-7:
        raise ValueError("Playback requires a synchronous rendering world with a fixed 0.1-second step")
    initial = list(world.get_actors())
    if any(a.type_id.startswith(("vehicle.", "walker.", "sensor.")) or a.attributes.get("role_name", "").startswith("robotaxi_") for a in initial):
        raise ValueError("Native playback requires an idle dedicated world without live participants")
    cv2.setNumThreads(2)
    with ExitStack() as stack:
        stack.callback(world.set_weather, world.get_weather())
        stack.callback(world.get_spectator().set_transform, world.get_spectator().get_transform())
        signals = world.get_actors().filter("traffic.traffic_light")
        if signals:
            stack.callback(world.freeze_all_traffic_lights, signals[0].is_frozen())
            for signal in signals:
                stack.callback(signal.set_state, signal.get_state())
        world.set_weather(getattr(carla.WeatherParameters, score["weather"]))
        client.set_replayer_time_factor(1.0)
        client.set_replayer_ignore_hero(False)
        client.set_replayer_ignore_spectator(True)
        response = client.replay_file(str(source.resolve()), PLAYBACK_START, 0., 0, False, False)
        stack.callback(_cleanup, client, world, {a.id for a in initial})
        if any(word in response.lower() for word in ("not found", "could not", "loading map", "not a recorder")):
            raise RuntimeError(f"Native CARLA replay failed: {response}")
        # CARLA applies v1's first record during replay_file. Tick once to publish spawned actors to the client.
        world.tick()
        heroes = [a for a in world.get_actors().filter("vehicle.*") if a.attributes.get("role_name") == "hero"]
        if len(heroes) != 1 or heroes[0].type_id != score["ego_blueprint"]:
            raise RuntimeError(f"Native replayer did not spawn the recorded ego: {response.strip()}")
        ego = heroes[0]
        replayed = list(world.get_actors())
        body = _restore_body_materials(replayed)
        road_users = {}
        for original_id, original in participants_.items():
            matches = [a for a in replayed if a.type_id == original["blueprint"]
                       and a.attributes.get("role_name") == original["role"]]
            if len(matches) == 1:
                road_users[original_id] = matches[0].id
        observed_box = ego.bounding_box
        if max(abs(x-y) for x,y in zip(box, [observed_box.location.x, observed_box.location.y, observed_box.location.z,
                                         observed_box.extent.x, observed_box.extent.y, observed_box.extent.z])) > .01:
            raise RuntimeError("Playback ego geometry differs from the recorded car")
        chase = ChaseCapture(*SIZE, 67, stack.enter_context((work/"chase.mjpeg").open("wb")))
        chase.attach(world, ego)
        stack.callback(chase.close)
        rows, front_offsets = [], []
        if perception:
            stack.enter_context(policy_profile(score["render_profile"]))
            rig = Rig(world, ego)
            stack.callback(rig.close)
            detector, tracker = Detector(), Tracker()
            front_stream = stack.enter_context((work/"front.mjpeg").open("wb"))
        # At time t CARLA interpolates the preceding and current packet. The final scored pose
        # is the penultimate packet, available at the recording endpoint. Stop before advancing again.
        for tick in range(2, frames[-1].number):
            replay_frame = world.tick()
            chase.append(replay_frame)
            snapshot = world.get_snapshot()
            chase.rows[-1]["road_users"] = {k: transform_fields(item.get_transform()) for k, actor_id in road_users.items()
                                            if (item := snapshot.find(actor_id)) is not None}
            if perception:
                obs = rig.read(replay_frame)
                ok, jpeg = cv2.imencode(".jpg", obs["front"][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 94])
                if not ok:
                    raise RuntimeError("Could not encode playback front RGB")
                front_offsets.append((front_stream.tell(), jpeg.nbytes))
                front_stream.write(jpeg.tobytes())
                pose = chase.rows[-1]["ego_pose"]
                detections = detector.detect(obs["front"])
                tracks = tracker.update(obs["lidar"], obs["radar"], detections,
                                        (pose["x"], pose["y"], pose["yaw"]), tick*DT)
                lidar, radar = display_returns(obs["lidar"], obs["radar"])
                rows.append(dict(lidar=lidar, radar=radar, detections=detections, tracks=tracks,
                                 detection_age_s=0., lidar_return_count=len(obs["lidar"]), radar_return_count=len(obs["radar"])))
        return chase.rows, chase.rendering, rows, front_offsets, response, body


def default_checkpoint():
    matches = list((PROJECT/"out/city").glob(f"*/evaluate/*/{TRIAL}/checkpoint.json"))
    if len(matches) != 1:
        raise ValueError("Specify --checkpoint with the directory for the locked primary trial")
    return matches[0].parent


def run(world, *, client, output=OUTPUT, record=SOURCE, checkpoint=None, perception=True):
    """Use an exclusive High-quality host with the project's simulator() settings."""
    checkpoint = Path(checkpoint) if checkpoint else default_checkpoint()
    output, record = Path(output).resolve(), Path(record).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh destination to preserve {output}")
    meta, summary, trace, scored, participants_, hero, box = load_source(record, checkpoint)
    if producer_info(world) != meta["producer"] or client.get_world().id != world.id:
        raise ValueError("Playback host differs from the recorder producer or provided world")
    if not all(shutil.which(name) for name in ("ffmpeg", "ffprobe")):
        raise RuntimeError("FFmpeg and FFprobe are required")
    waypoints = trip_points(world.get_map(), "heldout", "market")
    if summary["route_id"] != f"heldout:{waypoints[0].id}:{waypoints[-1].id}":
        raise ValueError("Reconstructed route differs from the locked trip")
    path = [[round(w.transform.location.x, 2), round(w.transform.location.y, 2)] for w in [*waypoints[:-1:4], waypoints[-1]]]
    output.parent.mkdir(parents=True, exist_ok=True)
    capture_root = PROJECT/"out/native-captures"
    capture_root.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=TRIAL+"-", dir=capture_root))
    print(f"Native film capture checkpoint: {stage}", flush=True)
    with TemporaryDirectory(prefix=".sensors-", dir=stage, ignore_cleanup_errors=True) as temporary:
        work = Path(temporary)
        exposures, observer, perception_rows, front_offsets, response, body = _record(
            world, client, record, summary, scored, participants_, box, work, perception)
        offset, alignment = align(exposures, scored, hero)
        verified = offset is not None
        start, count = (offset, len(scored)) if verified else (0, len(exposures))
        if verified:
            alignment["road_users"] = actor_alignment(exposures[start:start+count], scored, participants_)
        rows = []
        for index in range(count):
            exposure = exposures[start+index]
            row = dict(video_index=index, video_time=index/FPS, replay_frame=exposure["camera_frame"],
                       **{k:v for k,v in exposure.items() if k not in ("offset", "size", "camera_frame", "road_users")})
            if perception:
                row.update(perception_rows[start+index])
            if verified:
                row.update(trace[index], original_frame=trace[index]["frame"], native_frame=scored[index].number,
                           observer_map=_map(scored[index], participants_, hero, path))
            rows.append(row)
        (stage/"replay.json").write_text(json.dumps(rows, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        shutil.copyfile(checkpoint/"trace.jsonl.gz", stage/"original-trace.jsonl.gz")
        shutil.copyfile(checkpoint/"summary.json", stage/"original-summary.json")
        videos = {}
        poster_index = min(2*FPS, count-1)
        for name in (("chase", "front") if perception else ("chase",)):
            raw = work/(name+".mjpeg")
            video_size = SIZE if name == "chase" else FRONT_SIZE
            videos[name] = encode_and_check(raw, stage/(name+".mp4"), video_size, start, count)
            row = exposures[start+poster_index]
            location, length = ((row["offset"], row["size"]) if name == "chase" else front_offsets[start+poster_index])
            with raw.open("rb") as stream:
                stream.seek(location)
                (stage/(name+".jpg")).write_bytes(stream.read(length))
        shutil.copyfile(stage/"chase.jpg", stage/"poster.jpg")
        manifest = dict(purpose="Native CARLA visualization rerender of the original locked trajectory",
                        trial_id=TRIAL, cohort="native_replay", original_cohort="primary", district="heldout",
                        additional_scored_drive=False, producer=meta["producer"], quality_requested="High",
                        weather=summary["weather"], original_weather=summary["weather"], frame_period_s=DT,
                        selection_note="Selected after evaluation because the original held-out Market trip completed.",
                        observer=observer, video=videos["chase"], videos=videos, cameras=list(videos),
                        first_frame=trace[0]["frame"] if verified else None, last_frame=trace[-1]["frame"] if verified else None,
                        frames=count, summary=summary, outcome=summary["outcome"], alignment=alignment,
                        replay="replay.json", original_trace="original-trace.jsonl.gz", original_summary="original-summary.json",
                        geometry=dict(ego_blueprint=summary["ego_blueprint"], ego_box_center_extent_m=box, body=body),
                        sensor_provenance="New observations from the CARLA playback. Tracks use playback RGB, lidar and radar.",
                        range_columns={"lidar":["ego_x_m", "ego_y_m", "ego_z_m"],
                                       "radar":["ego_x_m", "ego_y_m", "radial_velocity_mps", "ego_z_m"]},
                        observer_map_provenance="Original recorded road-user positions and Town01 route waypoints.",
                        controls="Original trace commands at frame t were applied on tick t+1; all commands and speeds checked against native packets.",
                        native_recorder=dict(filename=record.name, sha256=meta["sha256"], metadata_sha256=fingerprint(record.with_suffix(".json")),
                                             observed_start_frame=meta["observed_start_frame"], observed_stop_frame=meta["observed_stop_frame"],
                                             replay_start_s=PLAYBACK_START,
                                             first_scored_packet=scored[0].number, last_scored_packet=scored[-1].number,
                                             playback_response=response.strip()),
                        sha256={name:fingerprint(stage/name) for name in _artifacts(perception)})
        write_json(stage/"manifest.json", manifest)
        write_json(stage/"install.json", dict(output=str(output), manifest_sha256=fingerprint(stage/"manifest.json")))
    return install_staged(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=2668)
    parser.add_argument("--attach", action="store_true", help="Use an explicitly reserved idle High-quality CARLA host")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--install-staged", type=Path, metavar="CHECKPOINT",
                        help="Resume publication from a verified capture without starting CARLA")
    parser.add_argument("--chase-only", action="store_true", help="Record only the exterior observer")
    args = parser.parse_args()
    if args.install_staged is not None:
        install_staged(args.install_staged)
        print(f"Installed native playback from {args.install_staged.resolve()}", flush=True)
        return
    if args.output.exists():
        parser.error("A film already exists at the requested output directory")
    with simulator(args.port, attach=args.attach, quality="High") as world:
        client = carla.Client("127.0.0.1", args.port, worker_threads=2)
        client.set_timeout(90.)
        manifest = run(world, client=client, output=args.output, checkpoint=args.checkpoint, perception=not args.chase_only)
    print(f"{manifest['frames']} frames, alignment {manifest['alignment']['status']} — {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
