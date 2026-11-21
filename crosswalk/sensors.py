"""Time-aligned CARLA sensors for perception and the driving console."""
from __future__ import annotations

import os
from dataclasses import asdict
from queue import Empty, Queue
import threading
import time

import carla
import numpy as np

from .thermal import thermal_rgb
from .camera_geometry import CAMERAS
from .radar_geometry import RADARS, CORNER_RADARS


def render_profile():
    """Return the requested rig profile. Actor readback is available after spawning the rig."""
    rgb = os.environ.get("CROSSWALK_RGB_RENDER", "mixed").strip().lower()
    if rgb not in ("native", "dlss", "mixed"):
        raise ValueError("CROSSWALK_RGB_RENDER must be native, dlss, or mixed.")
    percentage = 100.0
    if rgb != "native":
        raw = os.environ.get("CROSSWALK_DLSS_PERCENT", "50")
        try:
            percentage = float(raw)
        except ValueError:
            raise ValueError("CROSSWALK_DLSS_PERCENT must be between 25 and 100.") from None
        if not 25.0 <= percentage <= 100.0:
            raise ValueError("CROSSWALK_DLSS_PERCENT must be between 25 and 100.")
    return {
        "rgb": rgb, "rgb_screen_percentage": percentage,
        "camera_ray_tracing": False, "ground_truth": "native",
    }


def pose(x=0, y=0, z=0, pitch=0, yaw=0):
    return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw))


def rgb_image(image):
    bgra = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
    return bgra[:, :, 2::-1].copy()


def depth_image(image):
    bgra = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
    depth = bgra[:, :, 0].astype(np.float32) * 65536.0
    depth += bgra[:, :, 1].astype(np.float32) * 256.0
    depth += bgra[:, :, 2]
    return depth * (1000.0 / 16777215.0)


class Rig:
    def __init__(self, world, ego, visuals=False, range_only=False, *, surround=False):
        if visuals and range_only:
            raise ValueError("Visuals require camera sensors.")
        if surround and range_only:
            raise ValueError("Surround perception requires camera sensors.")
        profile = None if range_only else render_profile()
        self.rendering = {"requested": profile, "cameras": {}}
        self.sensors = []
        self.queues = {}
        self.contacts = []
        self.lock = threading.Lock()
        self._calibration_ego = ego if surround else None
        self._calibration_mounts = {}
        blueprints = world.get_blueprint_library()

        def attach(name, blueprint, mount, attributes):
            bp = blueprints.find(blueprint)
            camera_settings = None
            available = {}
            if blueprint.startswith("sensor.camera."):
                rgb = blueprint == "sensor.camera.rgb"
                use_dlss = rgb and (
                    profile["rgb"] == "dlss" or (profile["rgb"] == "mixed" and name != "front")
                )
                camera_settings = {
                    "use_ray_tracing": "false",
                    "enable_dlss": str(use_dlss).lower(),
                    "dlss_screen_percentage": str(profile["rgb_screen_percentage"] if use_dlss else 100.0),
                }
                if rgb:
                    camera_settings["enable_postprocess_effects"] = "true"
                available = {key: value for key, value in camera_settings.items() if bp.has_attribute(key)}
                dlss_keys = {"enable_dlss", "dlss_screen_percentage", "enable_postprocess_effects"}
                if use_dlss and not dlss_keys <= available.keys() and "enable_dlss" in available:
                    available["enable_dlss"] = "false"
                attributes = {**attributes, **available}
            for key, value in attributes.items():
                bp.set_attribute(key, str(value))
            sensor = world.spawn_actor(bp, mount, attach_to=ego)
            self.sensors.append(sensor)
            if surround and name in (*CAMERAS, "lidar", *RADARS):
                self._calibration_mounts[name] = mount
            if camera_settings is not None:
                readback = sensor.attributes
                actor_settings = {key: readback.get(key) for key in camera_settings}
                self.rendering["cameras"][name] = {
                    "requested_attributes": camera_settings,
                    "available_attributes": sorted(available),
                    "actor_attributes": actor_settings,
                    "attributes_match_request": all(
                        _attribute_matches(key, value, actor_settings[key])
                        for key, value in camera_settings.items()
                    ),
                    # Actor attributes acknowledge the requested configuration.
                    # The GPU may use a fallback, which Python cannot query.
                    "dlss_gpu_active": None,
                }
            if name == "collision":
                sensor.listen(self._on_contact)
            else:
                queue = self.queues[name] = Queue()
                sensor.listen(queue.put)

        try:
            if not range_only:
                for name in (CAMERAS if surround else ("front", "left", "right")):
                    cam = CAMERAS[name]
                    attrs = dict(image_size_x=cam.width, image_size_y=cam.height, fov=int(cam.hfov))
                    if name != "front":
                        attrs["sensor_tick"] = 0
                    attach(name, "sensor.camera.rgb", pose(cam.x, cam.y, cam.z, yaw=cam.yaw), attrs)
                    if surround:
                        readback = self.sensors[-1].attributes
                        if any(float(readback[k]) != v for k, v in attrs.items()):
                            raise RuntimeError(f"CARLA {name} geometry differs from the tracker calibration")
            attach("lidar", "sensor.lidar.ray_cast", pose(z=2.3), dict(
                channels=32, range=60, points_per_second=160000, rotation_frequency=10,
                upper_fov=10, lower_fov=-28, noise_stddev=0.015, noise_seed=31,
                **({"horizontal_fov": 360} if surround else {})))
            if surround:
                lidar = self.sensors[-1].attributes
                self.rendering["surround"] = {
                    "coordinates": "ego metres, +x forward, +y right, +z up; level mounts, yaw in degrees",
                    "camera_mounts": {name: asdict(cam) for name, cam in CAMERAS.items()},
                    "lidar_mount_xyz": [0., 0., 2.3], "lidar_horizontal_fov_deg": float(lidar["horizontal_fov"]),
                    "lidar_rotation_hz": float(lidar["rotation_frequency"]), "lidar_range_m": float(lidar["range"]),
                    "radar_horizontal_fov_deg": 36, "radar_facing": "front",
                }
            attach("radar", "sensor.other.radar", pose(2, z=0.9), dict(
                range=60, horizontal_fov=36, vertical_fov=8, points_per_second=12000, noise_seed=32))
            if surround:
                corners = self.rendering["surround"]["corner_radars"] = {}
                for index, name in enumerate(CORNER_RADARS):
                    x, y, z, yaw = RADARS[name]
                    attrs = dict(range=40, horizontal_fov=100, vertical_fov=12,
                                 points_per_second=4500, sensor_tick=0, noise_seed=33 + index)
                    attach(name, "sensor.other.radar", pose(x, y, z, yaw=yaw), attrs)
                    accepted = self.sensors[-1].attributes
                    if any(float(accepted[k]) != v for k, v in attrs.items()):
                        raise RuntimeError(f"CARLA {name} geometry differs from the tracker calibration")
                    corners[name] = dict(x=x, y=y, z=z, yaw=yaw, **attrs)
            attach("collision", "sensor.other.collision", pose(), {})
            if visuals:
                chase = dict(image_size_x=512, image_size_y=288, fov=90)
                ground_truth = dict(image_size_x=320, image_size_y=180, fov=90)
                attach("chase", "sensor.camera.rgb", pose(-7.5, z=4.1, pitch=-15), chase)
                attach("tags", "sensor.camera.semantic_segmentation", pose(1.3, z=1.65), ground_truth)
                attach("depth", "sensor.camera.depth", pose(1.3, z=1.65), ground_truth)
        except BaseException:
            self.close()
            raise

    def _on_contact(self, event):
        with self.lock:
            self.contacts.append((event.frame, event.other_actor.type_id))

    def collisions(self):
        with self.lock:
            return list(self.contacts)

    def read(self, frame, timeout=12):
        result = {}
        measured_poses = {}
        deadline = time.monotonic() + timeout
        for name, queue in self.queues.items():
            while True:
                try:
                    item = queue.get(timeout=max(0, deadline-time.monotonic()))
                except Empty:
                    raise RuntimeError(f"Timed out waiting for {name} sensor frame {frame}") from None
                if item.frame >= frame:
                    break
            if item.frame != frame:
                raise RuntimeError(f"Expected {name} frame {frame}, received {item.frame}")
            if self._calibration_ego is not None and name in self._calibration_mounts and hasattr(item, "transform"):
                measured_poses[name] = item.transform
            if name in ("radar", "lidar", *CORNER_RADARS):
                # CARLA 1.0 RadarDetection is four floats: velocity, azimuth,
                # altitude, depth. Ray-cast lidar carries x, y, z, intensity.
                result[name] = np.frombuffer(item.raw_data, "<f4").reshape(-1, 4).copy()
            elif name == "depth":
                result[name] = depth_image(item)
            elif name == "tags":
                bgra = np.frombuffer(item.raw_data, np.uint8).reshape(item.height, item.width, 4)
                result[name] = bgra[:, :, 2].copy()
            else:
                result[name] = rgb_image(item)
        if self._calibration_ego is not None and len(measured_poses) == len(self._calibration_mounts):
            # SensorData.transform is the world pose of this exposure, independent of
            # the requested relative attach transform. Verify the actual rigid chain.
            inverse = np.asarray(self._calibration_ego.get_transform().get_inverse_matrix())
            measured = {}
            for name, world_pose in measured_poses.items():
                local = inverse @ np.asarray(world_pose.get_matrix())
                expected = np.asarray(self._calibration_mounts[name].get_matrix())
                offset = float(np.linalg.norm(local[:3, 3] - expected[:3, 3]))
                cos_angle = (np.trace(local[:3, :3].T @ expected[:3, :3]) - 1.) / 2.
                angle = float(np.degrees(np.arccos(np.clip(cos_angle, -1., 1.))))
                measured[name] = dict(xyz_m=local[:3, 3].round(5).tolist(), translation_error_m=offset,
                                      rotation_error_deg=angle)
                if offset > .08 or angle > .5:
                    raise RuntimeError(f"{name} exposure pose differs from calibration by {offset:.3f} m, {angle:.2f} deg")
            self.rendering["surround"]["mount_readback"] = {"frame": int(frame), "sensors": measured}
            self._calibration_ego = None
        if "tags" in result:
            result["infrared"] = thermal_rgb(result.pop("tags"), result.pop("depth"), seed=frame)
        return result

    def close(self):
        for sensor in reversed(self.sensors):
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()


def _attribute_matches(key, requested, accepted):
    if accepted is None:
        return False
    if key == "dlss_screen_percentage":
        return float(requested) == float(accepted)
    return requested == accepted.lower()
