"""Color and compactness measurements from traffic-signal-sized RGB regions."""

import cv2
import numpy as np


COLORS = ("red", "amber", "green")
SIGNAL_NAMES = tuple(f"rgb_signal_{color}_{measure}" for color in COLORS
                     for measure in ("strength", "x", "y", "diameter"))
SIDE_NAMES = tuple(f"rgb_{side}_{measure}" for side in ("left", "right")
                   for measure in ("upper_luma", "road_luma", "road_contrast",
                                   "red_strength", "amber_strength", "green_strength"))


def _rgb(image):
    image = np.asarray(image)
    if (image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 8
            or image.dtype != np.uint8):
        raise ValueError("expected a uint8 RGB camera image at least 8 x 8")
    return image


def signal_candidates(image):
    """Return one compact light candidate per color, including normalized image position.

    The housing around a lit lamp is usually darker than the lamp. Keeping the
    native pixel resolution matters for distant lights. Large walls, foliage,
    lane paint and sky artifacts should contribute little to these measurements.
    """
    image = _rgb(image)
    height, width = image.shape[:2]
    roi = image[:max(4, int(.57 * height))]
    hue, saturation, value = cv2.split(cv2.cvtColor(roi, cv2.COLOR_RGB2HSV))
    vivid = (saturation >= 65) & (value >= 100)
    masks = ((hue <= 10) | (hue >= 170), (hue > 10) & (hue <= 33),
             (hue > 33) & (hue <= 95))
    scale = width / 640
    max_extent = max(3, round(29 * scale))
    kernel = np.ones((2*max(2, round(6*scale))+1,)*2, np.uint8)
    local_dark = cv2.erode(value, kernel)
    peak_map = cv2.dilate(value, np.ones((3, 3), np.uint8))
    results = []
    for hue_mask in masks:
        _, _, stats, centers = cv2.connectedComponentsWithStats((hue_mask & vivid).astype(np.uint8))
        stats = stats[1:]
        centers = centers[1:]
        widths, heights, areas = stats[:, 2:].T
        keep = ((widths <= max_extent) & (heights <= max_extent)
                & (widths <= 2.5 * heights) & (heights <= 3 * widths)
                & (areas >= max(1, round(1.5 * scale**2)))
                & (areas >= .32 * widths * heights))
        centers = centers[keep]
        areas = areas[keep]
        best = (0., 0., 0., 0.)
        if len(centers):
            xs, ys = np.rint(centers.T).astype(np.intp)
            peak = peak_map[ys, xs].astype(np.float32)
            dark = local_dark[ys, xs]
            contrast = np.clip((peak-dark-14.) / 110., 0., 1.)
            strength = contrast * (peak/255.) * np.minimum(1., areas/max(2., 4.*scale**2))
            # Prefer a larger lamp when equally bright pixels saturate the camera.
            idx = np.lexsort((areas, strength))[-1]
            if strength[idx] > 0:
                cx, cy = centers[idx]
                best = (float(strength[idx]), float(2*cx/width - 1), float(cy/height),
                        float(np.sqrt(areas[idx])/width))
        results.extend(best)
    return results


def side_features(image):
    image = _rgb(image)
    h, w = image.shape[:2]
    # Side cameras are mounted diagonally; each sees approaching cross traffic
    # and lamps beyond the edge of the front view.
    reduced = image[::max(1, h//48), ::max(1, w//64)]
    gray = cv2.cvtColor(reduced, cv2.COLOR_RGB2GRAY)
    upper, road = np.array_split(gray, 2)
    return [float(upper.mean()/255.), float(road.mean()/255.), float(road.std()/255.),
            *signal_candidates(image)[::4]]
