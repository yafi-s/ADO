"""Capture native RGB and lidar with matched barrier-present and absent controls in Town03.

The camera and lidar move together through poses relative to each site. Actor identity
and projected truth are recorded in the manifest only, separately from sensor data.
"""
import argparse
import json
from pathlib import Path
from queue import Queue
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import carla
import cv2
import numpy as np

from .camera_geometry import CAMERAS
from .sensors import rgb_image

SITES = (("fit", 67, -2, 23.), ("fit", 76, -1, 115.), ("holdout", 20, -1, 60.),
         ("holdout", 67, -2, 95.))
CONDITIONS = {"clear": None, "barricade": "static.prop.streetbarrier",
              "parked_car": "vehicle.mini.cooper", "cone": "static.prop.trafficcone01",
              "planter": "static.prop.plantpot07"}
# The native chain section runs from its post toward +local X. The terminal
# bollard is a separate native asset. An identically spaced, unlinked pair
# tests whether a recognizer actually sees the connecting chain.
CHAIN_CONDITIONS = {
    "raw_chain": "static.prop.chainbarrier", "chain": "static.prop.chainbarrier",
    "unlinked": "static.prop.chainbarrierend", "lone_post": "static.prop.chainbarrierend",
    "warning": "static.prop.warningconstruction", "barrel": "static.prop.barrel",
    "cone": "static.prop.trafficcone01", "barricade": "static.prop.streetbarrier", "clear": None,
}
CHAIN_SPAN_M = 1.44
POSES = ((5., 1.9), (7., 2.2), (10., 2.4), (13., 2.1), (16., 2.4), (20., 2.3))


def moved(base, dx, dy, dz, yaw=None):
    loc = base.transform(carla.Location(x=dx, y=dy, z=dz))
    rotation = carla.Rotation(yaw=base.rotation.yaw if yaw is None else yaw)
    return carla.Transform(loc, rotation)


def read_frame(queue, frame):
    while True:
        sample = queue.get(timeout=12)
        if sample.frame == frame:
            return sample
        if sample.frame > frame:
            raise RuntimeError("Sensor skipped requested frame")


def collect(port, output, *, chains=False, chain_validation=False):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Capture requires a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    client = carla.Client("localhost", port)
    client.set_timeout(20.)
    world = client.get_world()
    road_map = world.get_map()
    settings = world.get_settings()
    if ("Town03" not in road_map.name or not settings.synchronous_mode or
            abs((settings.fixed_delta_seconds or 0.) - .1) > 1e-6):
        raise ValueError("Collector requires an idle synchronous Town03 server at 10 Hz")
    library = world.get_blueprint_library()
    camera = CAMERAS["front"]
    sensors, manifest = [], []
    target = terminal = None
    try:
        for name, kind, attrs in [
            ("rgb", "sensor.camera.rgb", dict(image_size_x=camera.width, image_size_y=camera.height, fov=90,
                                             enable_dlss="false", use_ray_tracing="false")),
            ("lidar", "sensor.lidar.ray_cast", dict(channels=32, range=60, points_per_second=160000,
                  rotation_frequency=10, upper_fov=10, lower_fov=-28, noise_stddev=.015, noise_seed=31))]:
            bp = library.find(kind)
            for attr, val in attrs.items():
                if bp.has_attribute(attr):
                    bp.set_attribute(attr, str(val))
            sensor = world.spawn_actor(bp, carla.Transform())
            queue = Queue()
            sensor.listen(queue.put)
            sensors.append((sensor, queue))
        sites = (("validation", 24, -1, 31.), ("validation", 4, -1, 58.)) if chain_validation else SITES
        poses = ((5., 1.85), (7., 1.9), (10., 2.1)) if chain_validation else POSES
        for n, (split, road, lane, station) in enumerate(sites):
            wp = road_map.get_waypoint_xodr(road, lane, station)
            base = wp.transform
            # Keep the native prop broadside or oblique to the approach.
            for condition, kind in (CHAIN_CONDITIONS if chains else CONDITIONS).items():
                target_pose = moved(base, 0, 0, .08, base.rotation.yaw + (90 if chains or condition == "barricade" else 0)
                                    + ((8., -21.) if chain_validation else (0., 17., -12., 27.))[n])
                center_pose = target_pose
                if condition in ("chain", "unlinked", "raw_chain"):
                    target_pose = moved(center_pose, -CHAIN_SPAN_M / 2, 0., 0.)
                if kind:
                    target = world.try_spawn_actor(library.find(kind), target_pose)
                    if target is None:
                        raise RuntimeError(f"Cannot spawn {kind} at road {road}")
                    target.set_simulate_physics(False)
                if condition in ("chain", "unlinked"):
                    terminal = world.try_spawn_actor(library.find("static.prop.chainbarrierend"),
                                                     moved(target_pose, CHAIN_SPAN_M, 0., 0.))
                    if terminal is None:
                        raise RuntimeError("Cannot place the native endpoint")
                    terminal.set_simulate_physics(False)
                for sample, (distance, offset) in enumerate(
                        p for p in poses for _ in range(3 if chain_validation else 1)):
                    ego = moved(base, -distance, -offset, .04)
                    camera_pose = moved(ego, camera.x, camera.y, camera.z)
                    sensors[0][0].set_transform(camera_pose)
                    sensors[1][0].set_transform(moved(ego, 0., 0., 2.3))
                    for _ in range(1 if chain_validation and sample % 3 else 3):
                        f = world.tick()
                        exposure, scan = [read_frame(q, f) for _, q in sensors]
                    name = f"{split}_{n}_{condition}_{distance:g}"
                    if chain_validation:
                        name += f"_s{sample % 3}"
                    rgb = rgb_image(exposure)
                    points = np.frombuffer(scan.raw_data, dtype=np.float32).reshape(-1, 4)
                    # Keep only the front-facing sweep for this front-camera experiment.
                    points = points[points[:, 0] > 0]
                    cv2.imwrite(str(output / f"{name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 96])
                    np.savez_compressed(output / f"{name}.npz", lidar=points.astype(np.float16))
                    truth = None
                    if target is not None:
                        local = np.array(ego.get_inverse_matrix()) @ np.array([center_pose.location.x,
                             center_pose.location.y, center_pose.location.z, 1.])
                        truth = local[:2].tolist()
                    manifest.append(dict(name=name, split=split, road=road, lane=lane, station=station,
                        condition=condition, blueprint=kind, target_xy_m=truth, world_frame=f,
                        camera="front", distance_m=distance))
                    if chains:
                        inverse = np.array(ego.get_inverse_matrix())
                        bodies = []
                        for body in (target, terminal):
                            if body is None:
                                continue
                            pose = body.get_transform()
                            p = pose.location
                            center = inverse @ [p.x, p.y, p.z, 1.]
                            extent = body.bounding_box.extent
                            bodies.append(dict(blueprint=body.type_id, origin_xyz_m=center[:3].tolist(),
                                yaw_in_ego_deg=(pose.rotation.yaw-ego.rotation.yaw+180)%360-180,
                                world_bounds_half_xyz_m=[extent.x, extent.y, extent.z]))
                        manifest[-1]["scorer_bodies"] = bodies
                if terminal:
                    terminal.destroy()
                    terminal = None
                if target:
                    target.destroy()
                    target = None
                print(f"{n} {condition}", flush=True)
    finally:
        if terminal is not None:
            terminal.destroy()
        if target is not None:
            target.destroy()
        for sensor, _ in sensors:
            sensor.stop()
            sensor.destroy()
        world.tick()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def pack(captures, archive):
    rows = json.loads((captures / "manifest.json").read_text(encoding="utf-8"))
    archive.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive, "x") as output:
        files = ["manifest.json", *(row["name"] + ext for row in rows for ext in (".jpg", ".npz"))]
        for name in files:
            output.writestr(ZipInfo(name), (captures / name).read_bytes(), compress_type=ZIP_STORED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=2968)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="also write a new portable study archive")
    parser.add_argument("--chains", action="store_true", help="capture chain fixtures and controls")
    parser.add_argument("--chain-validation", action="store_true", help="score frozen detector on two additional streets")
    args = parser.parse_args()
    if args.archive and args.archive.exists():
        parser.error("Choose a new archive name")
    collect(args.port, args.output, chains=args.chains or args.chain_validation, chain_validation=args.chain_validation)
    if args.archive:
        pack(args.output, args.archive)
