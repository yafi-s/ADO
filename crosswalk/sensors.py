"""Time-aligned CARLA sensors. Only the first three streams reach the policy."""
from __future__ import annotations

from queue import Empty, Queue
import threading
import time

import carla
import numpy as np

from .thermal import thermal_rgb


def pose(x=0, y=0, z=0, pitch=0):
    return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch))


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
    def __init__(self, world, ego, visuals=False):
        self.sensors = []
        self.queues = {}
        self.contacts = []
        self.lock = threading.Lock()
        blueprints = world.get_blueprint_library()

        def attach(name, blueprint, mount, attributes):
            bp = blueprints.find(blueprint)
            for key, value in attributes.items():
                bp.set_attribute(key, str(value))
            sensor = world.spawn_actor(bp, mount, attach_to=ego)
            self.sensors.append(sensor)
            if name == "collision":
                sensor.listen(self._on_contact)
            else:
                queue = self.queues[name] = Queue()
                sensor.listen(queue.put)

        try:
            camera = dict(image_size_x=640, image_size_y=360, fov=90)
            attach("front", "sensor.camera.rgb", pose(1.3, z=1.65), camera)
            attach("lidar", "sensor.lidar.ray_cast", pose(z=2.3), dict(
                channels=32, range=60, points_per_second=160000, rotation_frequency=10,
                upper_fov=10, lower_fov=-28, noise_stddev=0.015, noise_seed=31))
            attach("radar", "sensor.other.radar", pose(2, z=0.9), dict(
                range=60, horizontal_fov=36, vertical_fov=8, points_per_second=12000, noise_seed=32))
            attach("collision", "sensor.other.collision", pose(), {})
            if visuals:
                attach("chase", "sensor.camera.rgb", pose(-7.5, z=4.1, pitch=-15), camera)
                attach("tags", "sensor.camera.semantic_segmentation", pose(1.3, z=1.65), camera)
                attach("depth", "sensor.camera.depth", pose(1.3, z=1.65), camera)
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
            if name in ("radar", "lidar"):
                result[name] = np.frombuffer(item.raw_data, np.float32).reshape(-1, 4).copy()
            elif name == "depth":
                result[name] = depth_image(item)
            elif name == "tags":
                bgra = np.frombuffer(item.raw_data, np.uint8).reshape(item.height, item.width, 4)
                result[name] = bgra[:, :, 2].copy()
            else:
                result[name] = rgb_image(item)
        if "tags" in result:
            result["infrared"] = thermal_rgb(result.pop("tags"), result.pop("depth"), seed=frame)
        return result

    def close(self):
        for sensor in reversed(self.sensors):
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
