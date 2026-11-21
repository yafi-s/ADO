"""Collect lossless, synchronized front-camera signal transitions on an idle Town03 server."""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
from queue import Queue

import carla
import numpy as np

from crosswalk.camera_geometry import CAMERAS
from crosswalk.replay_pack import _array_items
from crosswalk.runtime import simulator
from crosswalk.sensors import rgb_image
from crosswalk.signal_observer import fixtures


def _ego_pose(row):
    return carla.Transform(carla.Location(**{axis: row[axis] for axis in ("x", "y", "z")}),
                           carla.Rotation(**{axis: row[axis] for axis in ("pitch", "yaw", "roll")}))


def _original_approach(replay):
    with replay.open(encoding="utf8") as stream:
        for row in _array_items(stream):
            if row["frame"] > 1527:
                break
            if row["frame"] >= 1427:
                reference = row["observer_map"]["signal_reference"]["2024"]
                yield _ego_pose(row["ego_pose"]), reference, row["frame"]


def _parked_cycle(signal, distance):
    for stop in signal.get_stop_waypoints():
        previous = stop.previous(distance)
        if previous:
            pose = previous[0].transform
            return [(pose, state, None) for state in ("red", "green", "yellow", "red", "off") for _ in range(10)]
    raise ValueError(f"No {distance} m approach to signal {signal.get_opendrive_id()}")


def _distance_series(survey):
    fit = json.loads(Path("models/surround-signals/evaluation.json").read_text(encoding="utf8"))
    used = set(fit["train_poles"]) | set(fit["heldout_poles"])
    for signal_id in sorted({head.map_id for head in survey} - used):
        heads = [head for head in survey if head.map_id == signal_id]
        if len(heads) < 2 or heads[0].yaw is None:
            continue
        center = np.asarray([head.center for head in heads]).mean(axis=0)
        if not 5.5 <= center[2] <= 6.8:
            continue
        yaw = math.radians(heads[0].yaw)
        front = CAMERAS["front"]
        vertical_gap = 4.4
        for distance in (12., 25., 50.):
            horizontal = math.sqrt(distance**2 - vertical_gap**2) + front.x
            pose = carla.Transform(carla.Location(x=center[0] - horizontal * math.cos(yaw),
                                                  y=center[1] - horizontal * math.sin(yaw),
                                                  z=center[2] - vertical_gap - front.z),
                                   carla.Rotation(yaw=heads[0].yaw))
            schedule = [(pose, state, None) for state in ("red", "green", "yellow", "off") for _ in range(6)]
            yield f"unused_pole_{signal_id}_{int(distance)}m", signal_id, schedule
        return
    raise ValueError("No unused multi-head pole fits the distance study")


def _restore_signal(signal, state, frozen):
    signal.set_state(state)
    signal.freeze(frozen)


def check(world, destination, replay=Path("web/media/surround-city/replay.json"), *, study="transitions"):
    """Collect raw camera inputs and score-reference states from the same tick, without a controller."""
    settings = world.get_settings()
    if (not settings.synchronous_mode or settings.no_rendering_mode
            or not math.isclose(settings.fixed_delta_seconds or 0., .1, abs_tol=1e-6)):
        raise ValueError("The capture requires the synchronous 10 Hz rendering profile")
    if Path(world.get_map().name).name.split(".")[0] != "Town03_Opt":
        raise ValueError("The signal sequences require Town03_Opt")
    if world.get_actors().filter("vehicle.*"):
        raise ValueError("Capture the signal transitions before starting a driving trial")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    signals = {str(actor.get_opendrive_id()): actor
               for actor in world.get_actors().filter("traffic.traffic_light")}
    survey = fixtures(world)
    if study == "ranges":
        sequences = list(_distance_series(survey))
    else:
        sequences = [("original_approach", "2024", list(_original_approach(Path(replay))))]
        # Static calibration withheld pole 2010 from fitting. These poses and transitions are new.
        for distance in (10., 22.):
            sequences.append((f"heldout_{int(distance)}m", "2010", _parked_cycle(signals["2010"], distance)))
    heads = [dict(id=h.key, map_id=h.map_id, vertices=h.vertices.tolist(), yaw=h.yaw) for h in survey]
    front = CAMERAS["front"]
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    attrs = dict(image_size_x=front.width, image_size_y=front.height, fov=front.hfov, sensor_tick=0,
                 enable_dlss="false", use_ray_tracing="false", enable_postprocess_effects="true")
    for key, value in attrs.items():
        if blueprint.has_attribute(key):
            blueprint.set_attribute(key, str(value))
    evidence = dict(study=study, cameras={"front": attrs}, heads=heads, sequences=[])
    with ExitStack() as cleanup:
        previous_weather = world.get_weather()
        cleanup.callback(world.set_weather, previous_weather)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        camera = world.spawn_actor(blueprint, carla.Transform(carla.Location(z=10.)))
        cleanup.callback(camera.destroy)
        queue = Queue()
        camera.listen(queue.put)
        cleanup.callback(camera.stop)
        evidence["camera_readback"] = camera.attributes

        def exposure(pose):
            location = carla.Location(x=front.x, y=front.y, z=front.z)
            pose.transform(location)
            requested = carla.Transform(location, pose.rotation)
            camera.set_transform(requested)
            tick = world.tick()
            rgb = queue.get(timeout=12)
            while rgb.frame < tick:
                rgb = queue.get(timeout=12)
            if rgb.frame != tick:
                raise RuntimeError(f"RGB exposure {rgb.frame} does not match native frame {tick}")
            if not np.allclose(rgb.transform.get_matrix(), requested.get_matrix(), atol=.005, rtol=0.):
                raise RuntimeError(f"RGB exposure {tick} has a different camera pose than requested")
            return rgb_image(rgb), tick

        for name, signal_id, schedule in sequences:
            signal = signals[signal_id]
            previous_state, previous_frozen = signal.get_state(), signal.is_frozen()
            signal.freeze(True)
            frames, rows = [], []
            try:
                initial_pose, initial_state, _ = schedule[0]
                signal.set_state(getattr(carla.TrafficLightState, initial_state.capitalize()))
                for _ in range(8):
                    exposure(initial_pose)
                for index, (ego, requested, source) in enumerate(schedule):
                    value = getattr(carla.TrafficLightState, requested.capitalize())
                    if signal.get_state() != value:
                        signal.set_state(value)
                    frame, native = exposure(ego)
                    state = str(signal.get_state()).rsplit(".", 1)[-1].lower()
                    if state != requested:
                        raise RuntimeError(f"Native state differed from the requested signal state at frame {native}")
                    rows.append(dict(frame=native, source_frame=source, sim_time=index * .1, state=state,
                                     ego_pose=dict(x=ego.location.x, y=ego.location.y, z=ego.location.z,
                                                   pitch=ego.rotation.pitch, yaw=ego.rotation.yaw,
                                                   roll=ego.rotation.roll)))
                    frames.append(frame)
            finally:
                _restore_signal(signal, previous_state, previous_frozen)
            np.savez_compressed(destination / f"{name}.npz", front=np.stack(frames))
            evidence["sequences"].append(dict(name=name, target=signal_id, rows=rows))
    with (destination / "capture.json").open("w", encoding="utf8") as stream:
        json.dump(evidence, stream, indent=2)
        stream.write("\n")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path, default=Path("web/media/surround-city/replay.json"))
    parser.add_argument("--study", choices=("transitions", "ranges"), default="transitions")
    args = parser.parse_args()
    with simulator(args.port, attach=True, map_name="Town03_Opt") as active_world:
        capture = check(active_world, args.output, args.replay, study=args.study)
    print({item["name"]: len(item["rows"]) for item in capture["sequences"]}, flush=True)
