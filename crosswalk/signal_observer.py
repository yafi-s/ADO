"""Project surveyed CARLA signal heads and read the luminous pixels in the on-board cameras."""
from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np

from .camera_geometry import CAMERAS
from .lamp import head_vertices, project_box


@dataclass
class Head:
    key: str
    map_id: str
    vertices: np.ndarray
    center: np.ndarray
    yaw: float | None


def fixtures(world):
    heads = []
    for signal in world.get_actors().filter("traffic.traffic_light"):
        map_id = str(signal.get_opendrive_id())
        stops = [w.transform for w in signal.get_stop_waypoints() if w is not None]
        for vertices in head_vertices(signal):
            center = vertices.mean(axis=0)
            stop = min(stops, key=lambda s: math.hypot(s.location.x-center[0], s.location.y-center[1]),
                       default=None)
            key = "light/" + "/".join(f"{value:.2f}" for value in center)
            heads.append(Head(key, map_id, vertices, center, None if stop is None else stop.rotation.yaw))
    return heads


def illuminated_color(crop):
    """An unresolved or unlit housing supplies no color measurement."""
    height, width = crop.shape[:2]
    if height < 6 or width < 3:
        return None
    # project_box adds a two-pixel guard band. Exclude it before looking for a luminous bulb.
    rgb = crop[2:-2, 2:-2] if width > 5 else crop[2:-2, 1:-1]
    if rgb.size == 0:
        return None
    hue, sat, value = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    vivid = (sat >= 65) & (value >= 108)
    signal = np.maximum(0., value.astype(float) - np.percentile(value, 25)) * (sat / 255.)
    scores = []
    for color, hue_mask in (("red", (hue <= 10) | (hue >= 169)),
                            ("yellow", (hue > 10) & (hue <= 34)),
                            ("green", (hue > 34) & (hue <= 98))):
        mask = vivid & hue_mask & (signal >= 20)
        if not mask.any():
            continue
        intensity = signal[mask]
        score = float(np.sort(intensity)[-5:].sum())
        scores.append((score, color))
    scores.sort(reverse=True)
    if not scores or scores[0][0] < 46. or (len(scores) > 1 and scores[0][0] < 1.8 * scores[1][0]):
        return None
    return scores[0][1], scores[0][0]


COLOR_NAMES = ("off", "red", "yellow", "green")
CALIBRATION = Path(__file__).resolve().parents[1] / "models/surround-signals/colors.npz"
SUPPORT_THRESHOLD = .50
RESOLVED_CROP_HEIGHT = 13  # Three nominal pixel rows per bulb plus the four-pixel guard band.
CONFIRM_WINDOW_S = .25
CONSECUTIVE_WINDOW_S = .15
HOLD_LIMIT_S = .10


def housing_features(crop):
    # Opponent-color and achromatic vertical profiles, relative to the median inside the housing.
    # These bulbs can be only a few intensity levels brighter than their unlit lenses in CARLA.
    image = cv2.resize(crop, (12, 24), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    band = image[:, 4:8].mean(axis=1)
    side = image[:, [2, 9]].mean(axis=1)
    def profiles(colors):
        r, g, b = colors.T
        return np.column_stack((r-g, g-b, (r+g+b)/6.))
    center = profiles(band)
    relative = center - np.median(center[3:21], axis=0)
    lateral = center - profiles(side)
    features = [array[3:21].reshape(6, 3, 3).mean(axis=1).ravel() for array in (relative, lateral)]
    return np.concatenate(features)


class HousingClassifier:
    def __init__(self, path=CALIBRATION):
        with np.load(path,allow_pickle=False) as d:
            self.mean = d["mean"]
            self.scale = d["scale"]
            self.weights = d["weights"]
            self.intercept = d["intercept"]
            self.threshold = float(d["threshold"])

    def evidence(self, crop):
        x = (housing_features(crop)-self.mean) / self.scale
        logits = np.clip(x, -8, 8) @ self.weights + self.intercept
        prob = np.exp(logits-logits.max())
        prob /= prob.sum()
        label = int(np.argmax(prob))
        return COLOR_NAMES[label], float(prob[label])

    def read(self, crop):
        color, probability = self.evidence(crop)
        if color == "off" or probability < self.threshold:
            return None
        return color, probability


def bulb_contrast(crop, color, native_shape=None):
    """Measure a fitted color against nearby housing pixels in 8-bit RGB levels."""
    height = (native_shape if native_shape is not None else crop.shape[:2])[0]
    image = cv2.resize(crop, (12, 24), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    bulb = {"red": .17, "yellow": .50, "green": .83}[color]
    usable = 1 - 4 / height
    center = (2 / height + usable * bulb) * 23
    radius = max(1.5, .15 * usable * 24)
    rows = image[max(2, math.floor(center - radius)):min(22, math.ceil(center + radius) + 1)]
    center_rgb = rows[:, 4:8].mean(axis=(0, 1))
    housing_rgb = rows[:, [2, 3, 8, 9]].mean(axis=(0, 1))
    r, g, b = center_rgb - housing_rgb
    if color == "red":
        return r - g, r
    if color == "green":
        return g - (r + b) / 2, g
    chroma = min(center_rgb[:2]) - center_rgb[2] - (min(housing_rgb[:2]) - housing_rgb[2])
    return chroma, (r + g) / 2


def visible_bulb(crop, color, native_shape=None):
    height, width = native_shape if native_shape is not None else crop.shape[:2]
    if height > 14:
        return True
    if height < 10 or width < 7:
        return False
    chroma, brightness = bulb_contrast(crop, color, native_shape)
    return chroma >= .8 and brightness >= -3.


class SignalObserver:
    def __init__(self, world):
        self.classifier = HousingClassifier() if CALIBRATION.is_file() else None
        self.heads = fixtures(world)
        self.reference_actors = {str(a.get_opendrive_id()): a for a in world.get_actors().filter("traffic.traffic_light")}
        self.recent = {}
        self.candidates = {}
        self.mounts = {}
        import carla
        for name, cam in CAMERAS.items():
            pose = carla.Transform(carla.Location(cam.x, cam.y, cam.z), carla.Rotation(yaw=cam.yaw))
            self.mounts[name] = np.asarray(pose.get_inverse_matrix())

    def score_reference(self, readings):
        """Simulator labels for the evaluation log only, after the RGB observations are complete."""
        identities = {r["map_id"] for r in readings}
        return {key: str(self.reference_actors[key].get_state()).split(".")[-1].lower()
                for key in sorted(identities) if key in self.reference_actors}

    def _camera_votes(self, head, sensor_matrices, images):
        votes = []
        support = set()
        resolved = False
        for name, matrix in sensor_matrices.items():
            rgb = images[name]
            box = project_box(head.vertices, matrix, rgb.shape, CAMERAS[name].hfov)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            crop = rgb[y0:y1, x0:x1]
            if crop.shape[0] < 6 or crop.shape[1] < 4:
                continue
            resolved |= crop.shape[0] >= RESOLVED_CROP_HEIGHT
            if self.classifier is None:
                result = illuminated_color(crop)
                if result is not None:
                    color, strength = result
                    votes.append((strength, name, box, color))
                continue
            color, strength = self.classifier.evidence(crop)
            if color == "off":
                if strength >= self.classifier.threshold:
                    support.add(color)
            elif not visible_bulb(crop, color):
                support.add("unresolved")
            elif strength >= SUPPORT_THRESHOLD:
                support.add(color)
                if strength >= self.classifier.threshold:
                    votes.append((strength, name, box, color))
        return votes, support, resolved

    def read(self, pose, images, sim_time=None):
        inv = np.asarray(pose.get_inverse_matrix())
        sensor_matrices = {name: mount @ inv for name, mount in self.mounts.items() if name in images}
        if sim_time is not None:
            previous_candidates, previous_confirmed = self.candidates, self.recent
            self.candidates, self.recent = {}, {}
        readings = []
        for head in self.heads:
            dx = head.center[0] - pose.location.x
            dy = head.center[1] - pose.location.y
            distance = math.hypot(dx, dy)
            if distance > 60. or distance < .1:
                continue
            row = dict(id=head.key, map_id=head.map_id, x=round(float(head.center[0]), 3),
                       y=round(float(head.center[1]), 3), z=round(float(head.center[2]), 3),
                       governs_yaw=None if head.yaw is None else round(head.yaw, 2),
                       state="unknown", source="map")
            readings.append(row)
            if head.yaw is not None:
                yaw = math.radians(head.yaw)
                if (dx * math.cos(yaw) + dy * math.sin(yaw)) / distance < .45:
                    continue
            votes, support, resolved = self._camera_votes(head, sensor_matrices, images)
            if votes:
                strength, view, box, color = max(votes)
                if all(v[3] == color for v in votes):
                    if sim_time is not None:
                        prior = previous_candidates.get(head.key)
                        tiny = box[3] - box[1] < RESOLVED_CROP_HEIGHT
                        gap = sim_time - prior[1] if prior and prior[0] == color else math.inf
                        self.candidates[head.key] = (color, sim_time)
                        window = CONSECUTIVE_WINDOW_S if tiny else CONFIRM_WINDOW_S
                        if not 0 < gap <= window:
                            continue
                    row.update(state=color, view=view, source="camera", pixel_box=list(box),
                               pixel_size=[images[view].shape[1], images[view].shape[0]],
                               confidence=round(strength, 4) if self.classifier else None, age_s=0.)
                    if sim_time is not None:
                        self.recent[head.key] = (sim_time, color, view, not tiny)
            if sim_time is not None and not votes:
                prior = previous_candidates.get(head.key)
                if resolved and prior and support <= {prior[0]} and 0 < sim_time - prior[1] <= CONFIRM_WINDOW_S:
                    self.candidates[head.key] = prior
                if head.key in previous_confirmed:
                    last, color, view, previously_resolved = previous_confirmed[head.key]
                    age = sim_time - last
                    bounded = 0 < age <= HOLD_LIMIT_S or math.isclose(age, HOLD_LIMIT_S, abs_tol=1e-9)
                    if not previously_resolved or not bounded or not (support == {color} or (resolved and not support)):
                        continue
                    self.recent[head.key] = (last, color, view, previously_resolved)
                    row.update(state=color, view=view, source="camera", age_s=round(age, 2), inferred_from=(
                        "last camera observation corroborated by current pixels" if support else "last camera observation"))
        return readings
