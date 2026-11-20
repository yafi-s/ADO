"""CPU NanoDet inference on the front RGB camera. Boxes use normalized xyxy."""

from pathlib import Path

import cv2
import numpy as np


MODEL_PATH = (Path(__file__).resolve().parents[1] / "vendor" / "nanodet" /
              "object_detection_nanodet_2022nov.onnx")
LABELS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VISION_NAMES = tuple(
    f"rgb_{group}_{measure}"
    for group in ("person", "vehicle")
    for measure in ("confidence", "height", "center_offset")
)
SIZE = 416
STRIDES = (8, 16, 32)
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
BINS = np.arange(8, dtype=np.float32)


def _anchors(stride):
    axis = np.arange(SIZE // stride, dtype=np.float32) * stride + (stride - 1) / 2
    x, y = np.meshgrid(axis, axis)
    return np.stack((x.ravel(), y.ravel()), axis=1)


ANCHORS = tuple(_anchors(stride) for stride in STRIDES)


def _letterbox(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("expected a nonempty uint8 RGB image")
    height, width = rgb.shape[:2]
    longest = max(height, width)
    w, h = max(1, width * SIZE // longest), max(1, height * SIZE // longest)
    left, top = (SIZE - w) // 2, (SIZE - h) // 2
    canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    canvas[top:top + h, left:left + w] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    # The upstream demo supplies RGB pixels in precisely this normalization order.
    blob = cv2.dnn.blobFromImage((canvas.astype(np.float32) - MEAN) / STD)
    return blob, (left, top, w, h)


def _decode(outputs, content, threshold):
    boxes, confidences, classes = [], [], []
    left, top, w, h = content
    for stride, anchors, scores, regression in zip(
        STRIDES, ANCHORS, outputs[::2], outputs[1::2], strict=True
    ):
        scores = scores[0]
        # Only decode high-confidence anchors. Require the winning COCO class to
        # be a road user so unrelated objects cannot become cars by elimination.
        selected = np.flatnonzero(scores.max(axis=1) > threshold)
        labels = scores[selected].argmax(axis=1)
        keep = np.isin(labels, list(LABELS))
        selected, labels = selected[keep], labels[keep]
        if not len(selected):
            continue
        logits = regression[0, selected].reshape(-1, 4, len(BINS))
        weights = np.exp(logits - logits.max(axis=2, keepdims=True))
        distances = (weights @ BINS) / weights.sum(axis=2) * stride
        points = anchors[selected]
        corners = np.column_stack((points - distances[:, :2], points + distances[:, 2:]))
        corners[:, ::2] = np.clip(corners[:, ::2] - left, 0, w)
        corners[:, 1::2] = np.clip(corners[:, 1::2] - top, 0, h)
        visible = (corners[:, 2] > corners[:, 0]) & (corners[:, 3] > corners[:, 1])
        boxes.extend(corners[visible])
        confidences.extend(scores[selected, labels][visible].tolist())
        classes.extend(labels[visible].tolist())
    if not boxes:
        return []
    corners = np.asarray(boxes)
    xywh = corners.copy()
    xywh[:, 2:] -= corners[:, :2]
    indices = cv2.dnn.NMSBoxesBatched(xywh.tolist(), confidences, classes, threshold, 0.6)
    normalized = corners / np.asarray([w, h, w, h], dtype=np.float32)
    return [
        {"box": normalized[i].tolist(), "label": LABELS[classes[i]], "score": confidences[i]}
        for i in np.asarray(indices).ravel()
    ]


class Detector:
    """One reusable OpenCV DNN network, called sequentially on RGB frames."""

    def __init__(self, threshold=0.35):
        self.threshold = threshold
        self.net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.outputs = self.net.getUnconnectedOutLayersNames()

    def detect(self, rgb):
        blob, content = _letterbox(rgb)
        self.net.setInput(blob)
        return _decode(self.net.forward(self.outputs), content, self.threshold)


def summarize(detections):
    """Per group, describe the tallest object centered in the middle half of view.

    Score breaks height ties. Horizontal offset spans -1 at the left edge to +1
    at the right. Missing groups contribute three zeros.
    """
    best = {"person": (0.0, 0.0, 0.0), "vehicle": (0.0, 0.0, 0.0)}
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        offset = x1 + x2 - 1.0
        if abs(offset) <= 0.5:
            group = "person" if item["label"] == "person" else "vehicle"
            candidate = (y2 - y1, item["score"], offset)
            best[group] = max(best[group], candidate)
    values = []
    for height, score, offset in best.values():
        values.extend([score, height, offset])
    return values
