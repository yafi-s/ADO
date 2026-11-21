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

    def read(self, crop):
        x = (housing_features(crop)-self.mean) / self.scale
        logits = np.clip(x, -8, 8) @ self.weights + self.intercept
        prob = np.exp(logits-logits.max())
        prob /= prob.sum()
        label = int(np.argmax(prob))
        if label == 0 or prob[label] < self.threshold:
            return None
        return COLOR_NAMES[label], float(prob[label])


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

    def read(self, pose, images, sim_time=None):
        inv = np.asarray(pose.get_inverse_matrix())
        sensor_matrices = {name: mount @ inv for name, mount in self.mounts.items() if name in images}
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
            votes = []
            for name, matrix in sensor_matrices.items():
                rgb = images[name]
                box = project_box(head.vertices, matrix, rgb.shape, CAMERAS[name].hfov)
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                crop = rgb[y0:y1, x0:x1]
                if crop.shape[0] < 6 or crop.shape[1] < 4:
                    continue
                result = self.classifier.read(crop) if self.classifier is not None else illuminated_color(crop)
                if result is not None:
                    color, strength = result
                    votes.append((strength, name, box, color))
            if votes:
                strength, view, box, color = max(votes)
                if all(v[3] == color for v in votes):
                    if sim_time is not None:
                        prior = self.candidates.get(head.key)
                        count = prior[2] + 1 if prior and prior[0]==color and 0 < sim_time-prior[1] <= .25 else 1
                        self.candidates[head.key] = (color, sim_time, count)
                        if count < 2:
                            # A conflicting isolated flash should not recolor a mapped head.
                            if head.key in self.recent and self.recent[head.key][1] != color:
                                del self.recent[head.key]
                            continue
                    row.update(state=color, view=view, source="camera", pixel_box=list(box),
                               pixel_size=[images[view].shape[1], images[view].shape[0]],
                               confidence=round(strength, 4) if self.classifier else None, age_s=0.)
                    if sim_time is not None:
                        self.recent[head.key] = (sim_time, color, view)
            if sim_time is not None and row["state"] == "unknown" and head.key in self.recent:
                last, color, view = self.recent[head.key]
                age = sim_time - last
                if 0 < age <= .30 and not votes:
                    row.update(state=color, view=view, source="camera", age_s=round(age, 2),
                               inferred_from="last camera observation")
        return readings
