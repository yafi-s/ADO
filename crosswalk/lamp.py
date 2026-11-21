"""Project mapped signal heads into RGB views and learn their color from labelled crops."""

from dataclasses import dataclass

import numpy as np


CAMERAS = ("front", "left", "right")


def head_vertices(signal):
    """CARLA 1.0's get_light_boxes() already expresses each box in world coordinates."""
    import carla
    identity = carla.Transform()
    return [np.array([(v.x, v.y, v.z) for v in box.get_world_vertices(identity)])
            for box in signal.get_light_boxes()]


def project_box(vertices, world_to_camera, image_shape, fov_degrees=90.):
    """Return a padded pixel slice for a fully visible signal housing, or None."""
    height, width = image_shape[:2]
    vertices = np.asarray(vertices)
    transform = np.asarray(world_to_camera)
    forward, right, up = (vertices @ transform[:3, :3].T + transform[:3, 3]).T
    if np.any(forward <= .2):
        return None
    focal = width / (2 * np.tan(np.deg2rad(fov_degrees) * .5))
    u, v = width / 2 + focal * right / forward, height / 2 - focal * up / forward
    if min(u) < 0 or max(u) >= width or min(v) < 0 or max(v) >= height:
        return None
    pad = 2
    return (max(0, int(np.floor(min(u))) - pad), max(0, int(np.floor(min(v))) - pad),
            min(width, int(np.ceil(max(u))) + pad + 1), min(height, int(np.ceil(max(v))) + pad + 1))


def select_crop(heads, images, transforms):
    """Select the largest fully visible housing using static geometry alone."""
    candidates = []
    for name in CAMERAS:
        image = images[name]
        for head_index, vertices in enumerate(heads):
            rect = project_box(vertices, transforms[name], image.shape)
            if rect is not None:
                left, top, right, bottom = rect
                candidates.append(((bottom-top)*(right-left), name, head_index, rect))
    if not candidates:
        return None
    _, name, head_index, rect = max(candidates, key=lambda candidate: candidate[0])
    left, top, right, bottom = rect
    return name, head_index, rect, images[name][top:bottom, left:right]


def color_features(crop):
    """Measure upper and lower lamp chromaticity relative to the central housing."""
    h, w = crop.shape[:2]
    rgb = np.asarray(crop[:, int(.30*w):max(int(.30*w)+1, round(.70*w))], dtype=np.float64) / 255.
    r, g, b = rgb.transpose(2, 0, 1)
    opponent = (r-g) / (.12+r+g+b)
    upper, middle, lower = [opponent[round(lo*h):max(round(lo*h)+1, round(hi*h))]
                            for lo, hi in ((.16, .42), (.42, .58), (.58, .84))]
    neutral = np.median(middle)
    return 10 * np.asarray([upper.mean()-neutral, np.quantile(upper, .8)-neutral,
                            neutral-lower.mean(), neutral-np.quantile(lower, .2)])


@dataclass
class LampClassifier:
    center: np.ndarray
    weights: np.ndarray

    @classmethod
    def fit(cls, pairs):
        """Learn a signed ridge score, fitting independent red/green pairs equally."""
        x = np.asarray(pairs, dtype=np.float64)
        if x.ndim != 3 or x.shape[1] != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Expected one or more finite red/green feature pairs.")
        center = x.mean(axis=(0, 1))
        observations = (x - center).reshape(-1, x.shape[-1])
        labels = np.tile([1., -1.], len(x))
        weights = np.linalg.solve(observations.T @ observations + .03 * np.eye(len(center)),
                                  observations.T @ labels)
        return cls(center, weights)

    def score(self, features):
        """Positive scores indicate red. Magnitudes are uncalibrated."""
        return float((features - self.center) @ self.weights)


class RouteLamps:
    """Find controlling signals from planned route, static mapped heads and stop lines."""
    def __init__(self, world, route):
        import carla
        from .robotaxi import MOUNTS
        road_map = world.get_map()
        self.mount_inverses = {}
        for name in CAMERAS:
            x, y, z, yaw = MOUNTS[name]
            mount = carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw=yaw))
            self.mount_inverses[name] = np.asarray(mount.get_inverse_matrix())
        route_xyz = np.array([(p.x, p.y, p.z) for p in route])
        waypoints = [road_map.get_waypoint(p) for p in route]
        lanes = [(p.road_id, p.lane_id) if p else None for p in waypoints]
        self.map = []
        for signal in world.get_actors().filter("traffic.traffic_light"):
            for stop in signal.get_stop_waypoints():
                if not stop or (stop.road_id, stop.lane_id) not in lanes:
                    continue
                transform = stop.transform
                p, forward = transform.location, transform.get_forward_vector()
                position = np.array((p.x, p.y, p.z))
                dist = np.linalg.norm(route_xyz - position, axis=1)
                matches = [i for i in np.flatnonzero(dist <= 1.2) if lanes[i] == (stop.road_id, stop.lane_id)]
                if not matches:
                    continue
                controlled = {(w.road_id, w.lane_id) for w in signal.get_affected_lane_waypoints() if w}
                if not controlled.intersection(lanes):
                    continue
                heads = head_vertices(signal)
                if heads:
                    self.map.append((position[:2], np.array((forward.x, forward.y)), heads,
                                     float(min(dist[matches])), signal.id))

    def observe(self, ego, images):
        if not self.map:
            return None
        pose = ego.get_transform()
        heading = pose.get_forward_vector()
        xy = np.array((pose.location.x, pose.location.y))
        nose = ego.bounding_box.location.x + ego.bounding_box.extent.x
        heading = np.array((heading.x, heading.y))
        candidates = []
        for pos, arrow, heads, error, signal_id in self.map:
            delta = pos - (xy + nose * heading)
            gap = float(delta @ arrow)
            lateral = abs(float(delta[0]*arrow[1] - delta[1]*arrow[0]))
            if -1.0 <= gap <= 24. and lateral <= 2.25 and heading @ arrow > .90:
                candidates.append((gap, error, signal_id, heads))
        if not candidates:
            return None
        gap, _, _, heads = min(candidates, key=lambda x: (round(x[0], 1), x[1], x[2]))
        ego_inverse = np.asarray(pose.get_inverse_matrix())
        transforms = {name: mount @ ego_inverse for name, mount in self.mount_inverses.items()}
        selected = select_crop(heads, images, transforms)
        if selected is None:
            return None
        return gap, color_features(selected[-1])


def calibrate(scores, red_labels):
    """Regularized Platt scaling on scores from leave-one-fit-pole-out predictions."""
    scores = np.asarray(scores)
    design = np.column_stack([scores, np.ones(len(scores))])
    y = np.asarray(red_labels, dtype=float)
    weights = np.zeros(2)
    penalty = np.diag([.12, .03])
    for _ in range(16):
        probability = 1 / (1 + np.exp(-np.clip(design @ weights, -35., 35.)))
        curvature = probability * (1-probability)
        weights -= np.linalg.solve((design.T * curvature) @ design + penalty,
                                   design.T @ (probability-y) + penalty @ weights)
    return weights


class LampExpert:
    """Local actuator regression over real demonstrations on mapped signal approaches."""
    def __init__(self, classifier, calibration, points, pedals):
        self.classifier = classifier
        self.calibration = calibration
        self.points = np.asarray(points)
        self.pedals = np.asarray(pedals)

    def red_probability(self, color):
        slope, bias = self.calibration
        return float(1 / (1 + np.exp(-np.clip(slope*self.classifier.score(color)+bias, -35., 35.))))

    def signed_pedal(self, observation, speed):
        gap, color = observation
        # Distances in units of 0.2 probability, 1.0 m clearance and 0.5 m/s speed.
        query = [self.red_probability(color)*5., gap, speed*2.]
        squared = np.sum((self.points - query)**2, axis=1)
        count = min(12, len(squared))
        nearest = np.argpartition(squared, count-1)[:count]
        distances = squared[nearest]
        weights = 1 / (.15 + distances)
        return float(np.average(self.pedals[nearest], weights=weights))

    def apply(self, observation, speed, base_command):
        if observation is None:
            return base_command
        command = self.signed_pedal(observation, speed)
        return base_command[0], max(0., command), max(0., -command)

    def save(self, path):
        np.savez_compressed(path, **self.arrays())

    def arrays(self, prefix=""):
        fields = dict(version=1, center=self.classifier.center, weights=self.classifier.weights,
                      calibration=self.calibration, points=self.points, pedals=self.pedals)
        return {prefix+k: v for k, v in fields.items()}

    @classmethod
    def from_arrays(cls, data, prefix=""):
        center, weights, calibration, points, pedals = (
            np.asarray(data[prefix+name]) for name in ("center", "weights", "calibration", "points", "pedals"))
        if (int(data[prefix+"version"]) != 1 or center.shape != (4,) or weights.shape != (4,)
                or calibration.shape != (2,) or pedals.ndim != 1 or points.shape != (len(pedals), 3)
                or not len(pedals) or any(not np.isfinite(a).all() for a in (center, weights, calibration,
                                                                              points, pedals))
                or np.abs(pedals).max() > 1.):
            raise ValueError("Invalid signal expert model")
        return cls(LampClassifier(center, weights), calibration, points, pedals)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls.from_arrays(data)
