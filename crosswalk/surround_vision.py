"""Camera-only YOLOX-S road-user detections for the surround observer."""

from pathlib import Path

import cv2
import numpy as np

from .vision import LABELS


MODEL_PATH = (Path(__file__).resolve().parents[1] / "vendor" / "yolox" /
              "object_detection_yolox_2022nov.onnx")
MODEL_ID = "opencv-zoo/yolox-s-2022nov-fp32"
SIZE = 640


def _anchors():
    grids, strides = [], []
    for stride in (8, 16, 32):
        side = SIZE // stride
        x, y = np.meshgrid(np.arange(side), np.arange(side))
        grids.append(np.stack([x.ravel(), y.ravel()], axis=1))
        strides.append(np.full((side * side, 1), stride))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


GRID, STRIDE = _anchors()


def _place(rgb, canvas, y, max_height):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("expected a nonempty uint8 RGB image")
    height, width = rgb.shape[:2]
    scale = min(SIZE / width, max_height / height)
    w, h = max(1, int(width * scale)), max(1, int(height * scale))
    canvas[y:y + h, :w] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return (0, y, w, y + h)


def _view_boxes(output, region, threshold, layout):
    classes = np.argmax(output[:, 5:], axis=1)
    scores = output[:, 4] * output[np.arange(len(output)), 5 + classes]
    keep = (scores > threshold) & np.isin(classes, list(LABELS))
    if not keep.any():
        return []
    center = (output[keep, :2] + GRID[keep]) * STRIDE[keep]
    size = np.exp(output[keep, 2:4]) * STRIDE[keep]
    origin = np.asarray(region[:2])
    limit = np.asarray(region[2:])
    # Assign centers to this view before clipping so another camera's detections
    # cannot acquire this camera's identity by straddling the padding.
    inside = np.all((center >= origin) & (center < limit), axis=1)
    if not inside.any():
        return []
    xy1 = np.maximum(center[inside] - size[inside] / 2, origin) - origin
    xy2 = np.minimum(center[inside] + size[inside] / 2, limit) - origin
    scores, classes = scores[keep][inside], classes[keep][inside]
    xywh = np.column_stack([xy1, xy2 - xy1])
    # Retain floating-point coordinates in OpenCV's class-wise NMS.
    indices = cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), classes.tolist(),
                                      threshold, 0.5)
    extent = limit - origin
    normalized = np.column_stack([xy1 / extent, xy2 / extent])
    return [{"box": normalized[i].tolist(), "score": float(scores[i]),
             "label": LABELS[classes[i]], "detector": MODEL_ID, "inference_layout": layout}
            for i in np.asarray(indices).ravel()]


class SurroundDetector:
    """Run one reusable local CPU network sequentially on uint8 RGB images."""

    def __init__(self, threshold=0.35, *, backend="auto"):
        if backend not in ("auto", "onnxruntime", "opencv"):
            raise ValueError("backend must be auto, onnxruntime or opencv")
        self.threshold = threshold
        self.session = None
        if backend != "opencv":
            try:
                import onnxruntime as ort
            except ImportError:
                if backend == "onnxruntime":
                    raise
            else:
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                options.inter_op_num_threads = 1
                options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                self.session = ort.InferenceSession(str(MODEL_PATH), sess_options=options,
                                                    providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
        if self.session is None:
            self.net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.backend = "onnxruntime" if self.session is not None else "opencv"

    def _forward(self, canvas):
        blob = cv2.dnn.blobFromImage(canvas)
        if self.session is not None:
            return self.session.run(None, {self.input_name: blob})[0][0]
        self.net.setInput(blob)
        return self.net.forward()[0]

    def detect(self, rgb):
        canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
        region = _place(rgb, canvas, 0, SIZE)
        return _view_boxes(self._forward(canvas), region, self.threshold, "single")

    def detect_pair(self, first_rgb, second_rgb):
        """Classify synchronized auxiliary images in one pass, with a 10-pixel gutter."""
        canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
        slot = (SIZE - 10) // 2
        top = _place(first_rgb, canvas, 0, slot)
        bottom = _place(second_rgb, canvas, SIZE - slot, slot)
        output = self._forward(canvas)
        return (_view_boxes(output, top, self.threshold, "pair-top"),
                _view_boxes(output, bottom, self.threshold, "pair-bottom"))
