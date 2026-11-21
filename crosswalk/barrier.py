"""Sensor-only CARLA striped barricade recognition using fitted RGB appearance and laser support."""
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np

from .camera_geometry import CAMERAS

MODEL = Path(__file__).resolve().parents[1] / "models" / "barrier-appearance.json"


def stripe_image(image):
    """Orange stripes have positive red/green contrast; white paint and grey asphalt have near-zero contrast."""
    r, g, b = cv2.split(image)
    return cv2.subtract(r, cv2.addWeighted(g, .65, b, .35, 0))


@lru_cache(maxsize=1)
def _model():
    data = json.loads(MODEL.read_text(encoding="utf-8"))
    return np.array(data["stripe_template"], np.uint8), data["threshold"]


def candidates(image, camera, threshold=None):
    template, fitted = _model()
    threshold = fitted if threshold is None else threshold
    rows, cols = image.shape[:2]
    cutoff = int(rows * .44)
    channel = stripe_image(image[cutoff:])
    # A board has adjacent orange and achromatic stripes. Blank grey regions contribute no proposals.
    if channel.max() < 12:
        return []
    bboxes, scores = [], []
    for width in (12, 15, 18, 22, 27, 33, 40, 49, 60, 73, 89, 109, 133, 162):
        if width > cols * .40:
            break
        height = round(width * template.shape[0] / template.shape[1])
        # The two rails sit around 0.6 m above ground. A broad pitch tolerance accepts road grades.
        expected_y = (camera.z - .60) / 1.19 * width + rows * .5 - cutoff
        y0 = max(0, int(expected_y - height * 1.3 - rows * .09))
        y1 = min(len(channel), int(expected_y + height * 1.3 + rows * .09))
        area = channel[y0:y1]
        if len(area) < height:
            continue
        patch = cv2.resize(template, (width, height), interpolation=cv2.INTER_LINEAR)
        corr = cv2.matchTemplate(area, patch, cv2.TM_CCOEFF_NORMED)
        peaks = (corr >= threshold) & (corr == cv2.dilate(corr, np.ones((7, 7), np.uint8)))
        for top, left in zip(*peaks.nonzero()):
            tile = area[top:top + height, left:left + width]
            # A dark shadow can copy a weak stripe pattern through JPEG noise.
            if float(tile.std()) < 4. or int(tile.max()) < 12:
                continue
            scores.append(float(corr[top, left]))
            bboxes.append([int(left), int(top) + cutoff + y0, width, height])
    found = []
    if scores:
        for i in cv2.dnn.NMSBoxes(bboxes, scores, threshold - 1e-6, .30):
            x, y, w, h = bboxes[i]
            # Include the posts above and below the fitted stripe panel.
            box = [max(0., (x-.03*w)/cols), max(0., (y-.18*h)/rows),
                   min(1., (x+1.03*w)/cols), min(1., (y+1.19*h)/rows)]
            found.append(dict(label="barrier", score=scores[i], box=box,
                              detector="ado-striped-barricade-v1", inference_layout="single"))
    return found


def detect(image, lidar, view, *, verify_posts=True):
    """Recognize supported striped boards and short connected post-and-chain barriers."""
    from .tracking import _project
    camera = CAMERAS[view]
    boxes = candidates(image, camera)
    from .barrier_chain import detect_chain
    points, ids, depths, u, v = _project(lidar, camera)
    result = []
    for item in detect_chain(image, camera, points):
        item["chain_ends_xy"] = [p.tolist() for p in item.pop("_endpoints")]
        item["chain_return_ids"] = [ids[group].tolist() for group in item.pop("_groups")]
        item["subtype_confidence"] = item["score"]
        if verify_posts:
            from .barrier_posts import review
            grade = review(image, camera, item)
            if not grade["confirmed"]:
                continue
            item["post_appearance"] = grade
        result.append(item)
    for candidate in boxes:
        left, top, right, bottom = candidate["box"]
        inside = (u >= left) & (u <= right) & (v >= top) & (v <= bottom)
        idx = np.flatnonzero(inside)
        ordered = idx[np.argsort(depths[idx])]
        for group in np.split(ordered, np.flatnonzero(np.diff(depths[ordered]) > .55) + 1):
            if len(group) < 3:
                continue
            depth = float(np.median(depths[group]))
            width = (right - left) * depth / camera.fx
            height = (bottom - top) * depth / camera.fy
            if (.65 <= width <= 1.95 and .55 <= height <= 1.68 and
                    .40 <= np.median(points[group, 2]) <= 1.40):
                result.append(dict(candidate, barrier_subtype="striped_board", subtype_confidence=candidate["score"]))
                break
    return result
