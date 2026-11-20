"""Time-aligned CARLA sensors. Only the first three streams reach the policy."""
from __future__ import annotations

from queue import Empty, Queue
import threading
import time

import carla
import numpy as np


def pose(x=0, y=0, z=0, pitch=0):
    return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch))


def rgb_image(image):
    bgra = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
    return bgra[:, :, 2::-1].copy()


def infrared_proxy(image):
    """Class-based white-hot illustration; a semantic render has no radiometry."""
    tags = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)[:, :, 2]
    intensity = np.full(256, 75, dtype=np.uint8)
    for name, level in {
        "Sky": 20, "Roads": 95, "Sidewalks": 105, "Vegetation": 55, "Buildings": 88,
        "Pedestrians": 245, "Rider": 235, "Car": 170, "Truck": 175, "Bus": 175,
        "Bicycle": 130, "Motorcycle": 145, "RoadLines": 100,
    }.items():
        intensity[int(getattr(carla.CityObjectLabel, name))] = level
    return np.repeat(intensity[tags][:, :, None], 3, axis=2)


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
                attach("infrared", "sensor.camera.semantic_segmentation", pose(1.3, z=1.65), camera)
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
            else:
                result[name] = infrared_proxy(item) if name == "infrared" else rgb_image(item)
        return result

    def close(self):
        for sensor in reversed(self.sensors):
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
