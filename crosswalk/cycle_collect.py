"""Capture synchronized native RGB, lidar and front radar for mounted-cycle association.

Run only on an idle synchronous Town03 CARLA server. Truth is saved separately
and used by the replay scorer only. Controlled poses cover rear and side views.
"""

import argparse
import json
from pathlib import Path
from queue import Queue

import carla
import cv2
import numpy as np

from .camera_geometry import CAMERAS
from .sensors import rgb_image


def offset(base, x=0.0, y=0.0, z=0.0, yaw=0.0):
    loc = base.transform(carla.Location(x=x, y=y, z=z))
    return carla.Transform(loc, carla.Rotation(yaw=base.rotation.yaw + yaw))


def read(queue, frame):
    while True:
        item = queue.get(timeout=12)
        if item.frame == frame:
            return item
        if item.frame > frame:
            raise RuntimeError(f"Missed native exposure {frame}")


def collect(port, directory):
    directory.mkdir(parents=True, exist_ok=False)
    client = carla.Client("localhost", port, worker_threads=2)
    client.set_timeout(18.0)
    world = client.get_world()
    road_map, settings = world.get_map(), world.get_settings()
    if (
        "Town03" not in road_map.name
        or not settings.synchronous_mode
        or abs(settings.fixed_delta_seconds - 0.1) > 0.001
    ):
        raise RuntimeError("Collector requires a synchronous idle Town03 server at 10 Hz")
    lib = world.get_blueprint_library()
    cycle = lib.find("vehicle.bh.crossbike")
    props = [
        bp.id
        for bp in lib.filter("static.prop.*")
        if "helmet" not in bp.id and any(n in bp.id for n in ("bike", "bicycle"))
    ]
    catalog = dict(cycle=cycle.id, attributes={a.id: str(a) for a in cycle}, other_bicycle_props=props)
    (directory / "catalog.json").write_text(json.dumps(catalog, indent=2))
    print(catalog, flush=True)
    sites = [("development", 67, -2, 23.0), ("heldout", 20, -1, 60.0)]
    scenes = ["rear", "oblique", "side", "walk", "near_walker", "stationary_cycle"]
    if props:
        scenes += ["parked_prop", "walker_parked_prop"]
    sensors, actors = {}, []
    cam = CAMERAS["front"]
    definitions = [
        (
            "rgb",
            "sensor.camera.rgb",
            (cam.x, cam.y, cam.z),
            dict(
                image_size_x=cam.width,
                image_size_y=cam.height,
                fov=90,
                enable_dlss="false",
                use_ray_tracing="false",
            ),
        ),
        (
            "lidar",
            "sensor.lidar.ray_cast",
            (0, 0, 2.3),
            dict(
                channels=32,
                range=60,
                points_per_second=160000,
                rotation_frequency=10,
                upper_fov=10,
                lower_fov=-28,
                noise_stddev=0.015,
                noise_seed=31,
            ),
        ),
        (
            "radar",
            "sensor.other.radar",
            (2, 0, 0.9),
            dict(range=60, horizontal_fov=36, vertical_fov=8, points_per_second=12000, noise_seed=32),
        ),
    ]
    try:
        for name, kind, _, attrs in definitions:
            bp = lib.find(kind)
            for k, v in attrs.items():
                if bp.has_attribute(k):
                    bp.set_attribute(k, str(v))
            sensor = world.spawn_actor(bp, carla.Transform())
            q = Queue()
            sensors[name] = sensor, q
            sensor.listen(q.put)
        with (directory / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
            for split, road, lane, station in sites:
                base = road_map.get_waypoint_xodr(road, lane, station).transform
                for scene in scenes:
                    who = ["walker"] if scene == "walk" else ["cycle"]
                    if scene == "near_walker":
                        who.append("walker")
                    if scene in ("parked_prop", "walker_parked_prop"):
                        who = ["parked"] + (["walker"] if scene.startswith("walker") else [])
                    track_actors = {}
                    for i, kind in enumerate(who):
                        bp = (
                            cycle
                            if kind == "cycle"
                            else (
                                min(lib.filter("walker.pedestrian.*"), key=lambda b: b.id)
                                if kind == "walker"
                                else lib.find(props[0])
                            )
                        )
                        actor = world.spawn_actor(bp, offset(base, 0, 3 * i, 2.0))
                        actors.append(actor)
                        actor.set_simulate_physics(False)
                        track_actors[kind] = actor
                    yaw = dict(rear=0.0, oblique=35.0, side=80.0, near_walker=48.0).get(scene, 65.0)
                    for i in range(15):
                        # Approach while the target translates laterally. Parked controls stay fixed in world coordinates.
                        dt = 0.1 * i
                        ego = offset(base, -11.8 + 3.5 * dt, 0.0, 0.08)
                        for name, _, xyz, _ in definitions:
                            sensors[name][0].set_transform(offset(ego, *xyz))
                        truth = {}
                        for kind, actor in track_actors.items():
                            moving = kind != "parked" and scene != "stationary_cycle"
                            x = (0.8 * dt if moving else 0.0) + (
                                2.2 if scene == "near_walker" and kind == "walker" else 0.0
                            )
                            y = (-1.0 + 1.4 * dt if moving else 0.0) + (
                                0.9 if kind == "walker" and len(who) > 1 else 0.0
                            )
                            pose = offset(base, x, y, 0.94 if kind == "walker" else 0.08, yaw)
                            actor.set_transform(pose)
                            truth[kind] = (
                                np.asarray(ego.get_inverse_matrix())
                                @ [pose.location.x, pose.location.y, pose.location.z, 1]
                            )[:2].tolist()
                        for _ in range(2 if i == 0 else 1):
                            frame = world.tick()
                            rgb, scan, radar = [
                                read(sensors[name][1], frame) for name in ("rgb", "lidar", "radar")
                            ]
                        stem = f"{split}-{scene}-{i:02d}"
                        if not cv2.imwrite(
                            str(directory / f"{stem}.png"),
                            cv2.cvtColor(rgb_image(rgb), cv2.COLOR_RGB2BGR),
                        ):
                            raise OSError("Could not persist RGB")
                        arrays = {
                            k: np.frombuffer(raw.raw_data, dtype=np.float32).reshape(-1, 4)
                            for k, raw in [("lidar", scan), ("radar", radar)]
                        }
                        np.savez_compressed(directory / f"{stem}.npz", **arrays)
                        row = dict(
                            stem=stem,
                            split=split,
                            scene=scene,
                            index=i,
                            world_frame=frame,
                            truth=truth,
                            ego=[ego.location.x, ego.location.y, ego.rotation.yaw],
                        )
                        manifest.write(json.dumps(row) + "\n")
                    manifest.flush()
                    for actor in track_actors.values():
                        actor.destroy()
                        actors.remove(actor)
                    print(split, scene, flush=True)
    finally:
        for actor in actors:
            actor.destroy()
        for sensor, q in sensors.values():
            sensor.stop()
            sensor.destroy()
        world.tick()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=2968)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.port, args.output)
