"""Portable three-view Waymo predictor. Inference needs NumPy and OpenCV only."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np


FORMAT = "ado-waymo-vision-numpy-v1"
TIMES = np.arange(1, 21, dtype=np.float32) / 4
LAYERS = {
    "history_dense": (100, 64), "history_embedding": (64, 32), "history_acceleration": (32, 10),
    "conv1": (3, 3, 3, 8), "conv2": (3, 3, 8, 16), "conv3": (3, 3, 16, 24),
    "camera_embedding": (144, 20), "fusion": (92, 48), "fused_acceleration": (48, 10),
}
SHAPES = {"history_mean": (6,), "history_scale": (6,), "residual_speed_scale": (2,),
          "normalized_basis": (20, 5)}
for layer, shape in LAYERS.items():
    SHAPES[layer + "_kernel"] = shape
    SHAPES[layer + "_bias"] = (shape[-1],)


def pool_rgb(rgb_three):
    """Front, front-left, front-right, uint8 RGB HWC. Resize full frames by area."""
    if len(rgb_three) != 3:
        raise ValueError("Expected three RGB views in FRONT, FRONT_LEFT, FRONT_RIGHT order")
    frames = []
    for image in rgb_three:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
            raise ValueError("Expected each camera as uint8 RGB [height,width,3]")
        if image.shape[:2] != (64, 96):
            import cv2
            image = cv2.resize(image, (96, 64), interpolation=cv2.INTER_AREA)
        frames.append(image)
    pixels = np.stack(frames).reshape(3, 32, 2, 48, 2, 3)
    return pixels.mean((2, 4), dtype=np.float32).round().astype(np.uint8)


class TrajectoryPolicy:
    def __init__(self, weights, metadata):
        if metadata.get("format") != FORMAT:
            raise ValueError("Unsupported portable trajectory format")
        for name, shape in SHAPES.items():
            x = weights.get(name)
            if x is None or x.shape != shape or x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError(f"Invalid {name}: expected finite float32 {shape}")
        if np.any(weights["history_scale"] <= 0) or np.any(weights["residual_speed_scale"] <= 0):
            raise ValueError("Normalization scales must be positive")
        self.weights = weights
        self.metadata = metadata

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as source:
            weights = {key: source[key] for key in SHAPES}
            metadata = json.loads(source["metadata"].item())
        return cls(weights, metadata)

    def _layer(self, x, name):
        return x @ self.weights[name + "_kernel"] + self.weights[name + "_bias"]

    def _convolve(self, x, name):
        # All three spatial inputs are even. A stride-2 3x3 TensorFlow SAME
        # convolution consequently pads one row/column at the trailing edge.
        padded = np.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
        windows = windows[:, ::2, ::2].transpose(0, 1, 2, 4, 5, 3)
        kernel = self.weights[name + "_kernel"].reshape(-1, LAYERS[name][-1])
        output = windows.reshape(*windows.shape[:3], -1) @ kernel + self.weights[name + "_bias"]
        return np.maximum(output, 0)

    def predict(self, rgb_three, history, intent):
        """Return 20 float32 ego XY waypoints at .25-second intervals, in meters."""
        state = np.asarray(history, dtype=np.float32)
        if state.shape != (16, 6) or not np.isfinite(state).all():
            raise ValueError("Expected 16 finite history rows of x,y,vx,vy,ax,ay")
        if isinstance(intent, (bool, np.bool_)) or not isinstance(intent, (int, np.integer)) or not 0 <= intent < 4:
            raise ValueError("Expected intent 0=unknown, 1=straight, 2=left or 3=right")
        normalized = ((state - self.weights["history_mean"]) / self.weights["history_scale"]).clip(-20, 20)
        motion = np.concatenate((normalized.ravel(), np.eye(4, dtype=np.float32)[intent]))
        motion = np.maximum(self._layer(motion, "history_dense"), 0)
        motion = np.maximum(self._layer(motion, "history_embedding"), 0)
        camera = pool_rgb(rgb_three).astype(np.float32) * np.float32(1 / 127.5) - 1
        for name in ("conv1", "conv2", "conv3"):
            camera = self._convolve(camera, name)
        pooled = camera.reshape(3, 2, 2, 3, 2, 24).mean((2, 4)).reshape(3, 144)
        embedding = np.maximum(self._layer(pooled, "camera_embedding"), 0)
        fusion = np.maximum(self._layer(np.concatenate((motion, embedding.ravel())), "fusion"), 0)
        accel = self._layer(motion, "history_acceleration") + self._layer(fusion, "fused_acceleration")
        residual = self.weights["normalized_basis"] @ accel.reshape(5, 2)
        denom = np.maximum(TIMES[:, None], 1) * self.weights["residual_speed_scale"]
        return state[-1, 2:4] * TIMES[:, None] + residual * denom


def extract_weights(model, config, seed, checkpoint_hash):
    from .model import acceleration_basis, keras
    camera = next(layer.layer for layer in model.layers if isinstance(layer, keras.layers.TimeDistributed))
    cnn = [layer for layer in camera.layers if isinstance(layer, keras.layers.Conv2D)]
    projection = next(layer for layer in camera.layers if isinstance(layer, keras.layers.Dense))
    if len(cnn) != 3:
        raise ValueError("Expected three convolution layers")
    source = {"conv1": cnn[0], "conv2": cnn[1], "conv3": cnn[2], "camera_embedding": projection}
    weights = {k: np.asarray(config["scales"][k], dtype=np.float32)
               for k in ("history_mean", "history_scale", "residual_speed_scale")}
    weights["normalized_basis"] = acceleration_basis() / np.maximum(TIMES[:, None], 1)
    for name in LAYERS:
        layer = source[name] if name in source else model.get_layer(name)
        weights[name + "_kernel"], weights[name + "_bias"] = layer.get_weights()
    metadata = {
        "format": FORMAT, "seed": seed, "source_checkpoint_sha256": checkpoint_hash,
        "dataset_source_sha256": config["dataset_source_sha256"],
        "model": "ADO three-camera Waymo E2E trajectory predictor",
        "views": ["FRONT", "FRONT_LEFT", "FRONT_RIGHT"], "image_color": "RGB uint8",
        "image_preprocessing": "full frame area resize to 96x64, 2x2 mean round-to-even uint8, then pixel / 127.5 - 1",
        "camera_tensor_shape": [3, 32, 48, 3], "history_shape": [16, 6],
        "history_fields": ["x_m", "y_m", "vx_m_s", "vy_m_s", "ax_m_s2", "ay_m_s2"],
        "history_times_s": (np.arange(-15, 1) / 4).tolist(),
        "intent": ["unknown", "straight", "left", "right"],
        "output_shape": [20, 2], "output_times_s": TIMES.tolist(), "units": "meters",
        "frame": "current rear axle, x forward, y left", "parameters": model.count_params(),
    }
    return TrajectoryPolicy(weights, metadata)


def verify_heldout(policy, model, features, config, sample_size):
    from .model import load_data, predict, prepare_arrays
    data = load_data(features)
    if data["sources"] != config["sources"]:
        raise ValueError("Feature set differs from checkpoint training snapshot")
    denom = prepare_arrays(data, config["scales"])
    candidates = np.flatnonzero(data["split"] == "test")
    if not len(candidates):
        raise ValueError("No test rows in the feature snapshot")
    rng = np.random.default_rng(20261120)
    selection = rng.choice(candidates, min(sample_size, len(candidates)), replace=False)
    selection.sort()
    reference = predict(model, data, selection, "vision", denom)
    deltas = []
    start = 0
    for source in data["sources"]:
        with np.load(Path(features) / source["file"], allow_pickle=False) as rows:
            end = start + len(rows["split"])
            indices = selection[(selection >= start) & (selection < end)]
            if len(indices):
                frames = rows["images"][indices - start]
                for index, rgb in zip(indices, frames):
                    portable = policy.predict(rgb, data["past"][index], data["intent"][index])
                    deltas.append(portable - reference[len(deltas)])
            start = end
    errors = np.asarray(deltas)
    if errors.shape != reference.shape or not np.isfinite(errors).all():
        raise ValueError("Portable parity returned incomplete or nonfinite predictions")
    max_abs = float(np.abs(errors).max())
    if max_abs > 0.001:
        raise ValueError(f"Portable parity failed: {max_abs:.7f} m exceeds 0.001 m")
    return {"rows": len(selection), "segment_count": len(np.unique(data["group"][selection])),
            "split": "test", "sample_seed": 20261120, "max_abs_coordinate_delta_m": max_abs,
            "mean_waypoint_distance_m": float(np.linalg.norm(errors, axis=-1).mean()),
            "max_abs_tolerance_m": 0.001}


def export(run, features, output, seed=None, sample_size=32):
    # Local inference never imports TensorFlow; only this Colab conversion path does.
    from .model import file_hash, make_model, tf
    if sample_size < 1:
        raise ValueError("Positive verification sample size required")
    run, output = Path(run), Path(output)
    config = json.loads((run / "config.json").read_text("utf8"))
    curves = json.loads((run / "training_curve.json").read_text("utf8"))
    dev = {i: min(e["dev_ADE_m"] for e in curves[f"vision-{i}"]["epochs"]) for i in config["seeds"]}
    selected = min(dev, key=lambda i: (dev[i], i)) if seed is None else seed
    if selected not in dev:
        raise ValueError("Requested seed is missing from the run")
    ckpt = run / f"checkpoint-vision-{selected}.weights.h5"
    digest = file_hash(ckpt)
    evaluation = json.loads((run / "evaluation.json").read_text("utf8"))
    if digest != evaluation["weights"][f"vision-{selected}"]["sha256"]:
        raise ValueError("Checkpoint hash differs from evaluated model")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    model = make_model("vision", selected)
    model.load_weights(ckpt)
    policy = extract_weights(model, config, selected, digest)
    policy.metadata.update({"best_dev_ADE_m": dev[selected], "best_epoch": curves[f"vision-{selected}"]["best_epoch"],
                            "seed_selection": "lowest development ADE" if seed is None else "explicit seed"})
    policy.metadata["parity"] = verify_heldout(policy, model, features, config, sample_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as tmp:
        pending = Path(tmp.name)
    try:
        with pending.open("wb") as handle:
            np.savez_compressed(handle, metadata=np.array(json.dumps(policy.metadata, sort_keys=True)), **policy.weights)
        TrajectoryPolicy.load(pending)
        result = {"file": output.name, "bytes": pending.stat().st_size, "sha256": file_hash(pending), **policy.metadata}
        if output.exists():
            if result["sha256"] != file_hash(output):
                raise FileExistsError(f"Existing export differs: choose a fresh output path for {output}")
        else:
            pending.replace(output)
        return result
    finally:
        pending.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--parity-samples", type=int, default=32)
    args = parser.parse_args()
    result = export(args.run, args.features, args.output, args.seed, args.parity_samples)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
