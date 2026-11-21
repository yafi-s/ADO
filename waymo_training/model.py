"""CPU-sized three-view and motion-only waypoint predictors for Waymo E2E."""
import os
from pathlib import Path
from hashlib import sha256

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras

TIMES = np.arange(1, 21, dtype=np.float32) / 4
MODES = ("history", "vision")


def configure_device(device):
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        return "CPU"
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("GPU required. In Colab choose Runtime > Change runtime type > GPU.")
    tf.config.set_visible_devices(devices[0], "GPU")
    tf.config.experimental.set_memory_growth(devices[0], True)
    return tf.config.experimental.get_device_details(devices[0]).get("device_name", devices[0].name)


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(directory):
    """Snapshot available complete shards and keep the pooled RGB views in RAM."""
    files = sorted(Path(directory).glob("train-*.npz")) + sorted(Path(directory).glob("val-*.npz"))
    if not files:
        raise ValueError(f"No feature shards in {directory}")
    keys = ("past", "intent", "future", "group", "timestamps", "frame_index", "split", "images")
    chunks = {k: [] for k in keys}
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            for key in keys:
                value = z[key]
                if key == "images":
                    if value.shape[1:] != (3, 64, 96, 3) or value.dtype != np.uint8:
                        raise ValueError(f"Bad RGB schema in {path}")
                    n = len(value)
                    value = value.reshape(n, 3, 32, 2, 48, 2, 3).mean((3, 5), dtype=np.float32).round().astype("uint8")
                chunks[key].append(value)
    data = {k: np.concatenate(v) for k, v in chunks.items()}
    validate_data(data)
    data["sources"] = [{"file": p.name, "sha256": file_hash(p)} for p in files]
    return data


def validate_data(data):
    n = len(data["past"])
    if data["past"].shape != (n, 16, 6) or data["future"].shape != (n, 20, 2):
        raise ValueError("Expected 16 history samples and 20 future XY waypoints")
    for key in ("intent", "split", "group", "timestamps", "frame_index"):
        if data[key].shape != (n,):
            raise ValueError(f"Bad shape for {key}")
    if not np.isfinite(data["past"]).all() or not np.isfinite(data["future"]).all():
        raise ValueError("Non-finite state or future label")
    if not np.isin(data["intent"], np.arange(4)).all():
        raise ValueError("Unknown high-level command index")
    if not np.isin(data["split"], ("train", "dev", "test")).all():
        raise ValueError("Unexpected split")
    seen = {}
    unique = set()
    for g, t, s in zip(data["group"], data["frame_index"], data["split"]):
        if seen.setdefault(g, s) != s:
            raise ValueError(f"Source group appears in multiple splits: {g}")
        if (g, t) in unique:
            raise ValueError(f"Repeated segment and frame index: {g} {t}")
        unique.add((g, t))


def constant_velocity(past):
    # Waymo past state ends at t=0 in current ego frame.
    return past[:, -1:, 2:4] * TIMES[None, :, None]


def acceleration_basis():
    """Distance due to each 1s constant acceleration, integrated at label times."""
    t = TIMES[:, None] - np.arange(5, dtype=np.float32)[None, :]
    return (np.maximum(t, 0) ** 2 - np.maximum(t - 1, 0) ** 2) * 0.5


def fit_scales(data, train):
    if len(train) == 0 or not np.all(data["split"][train] == "train"):
        raise ValueError("Fit normalization only on train rows")
    history = data["past"][train]
    residual = (data["future"][train] - constant_velocity(history)) / np.maximum(TIMES[:, None], 1.0)
    return {
        "history_mean": history.mean((0, 1), dtype=np.float64).tolist(),
        "history_scale": history.std((0, 1), dtype=np.float64).clip(0.01).tolist(),
        "residual_speed_scale": np.sqrt(np.mean(residual ** 2, axis=(0, 1), dtype=np.float64)).clip(0.1).tolist(),
    }


def prepare_arrays(data, scales):
    mean = np.asarray(scales["history_mean"], np.float32)
    scale = np.asarray(scales["history_scale"], np.float32)
    data["history_input"] = ((data["past"] - mean) / scale).clip(-20, 20)
    data["intent_input"] = np.eye(4, dtype=np.float32)[data["intent"]]
    data["cv"] = constant_velocity(data["past"])
    denom = np.maximum(TIMES[:, None], 1.0) * np.asarray(scales["residual_speed_scale"], np.float32)
    data["target"] = (data["future"] - data["cv"]) / denom
    return denom


def make_model(mode, seed):
    if mode not in MODES:
        raise ValueError(mode)
    keras.utils.set_random_seed(seed)
    layers = keras.layers
    history = keras.Input((16, 6), name="history")
    intent = keras.Input((4,), name="intent")
    inputs = [history, intent]
    motion = layers.Concatenate()([layers.Flatten()(history), intent])
    motion = layers.Dense(64, activation="relu", name="history_dense")(motion)
    motion = layers.Dense(32, activation="relu", name="history_embedding")(motion)
    accel = layers.Dense(10, kernel_initializer="zeros", name="history_acceleration")(motion)
    if mode == "vision":
        rgb = keras.Input((3, 32, 48, 3), dtype="uint8", name="camera_triplet")
        inputs.append(rgb)
        cam = keras.Input((32, 48, 3))
        features = layers.Rescaling(1 / 127.5, offset=-1)(cam)
        for channels in (8, 16, 24):
            features = layers.Conv2D(channels, 3, strides=2, padding="same", activation="relu")(features)
        features = layers.AveragePooling2D((2, 2))(features)
        features = layers.Dense(20, activation="relu")(layers.Flatten()(features))
        encoder = keras.Model(cam, features, name="shared_camera_encoder")
        views = layers.TimeDistributed(encoder)(rgb)
        camera_embedding = layers.Flatten()(views)
    else:
        # Match the motion and fusion decoder, including every motion-to-fusion weight.
        camera_embedding = keras.ops.tile(keras.ops.zeros_like(intent[:, :1]), (1, 60))
    fused = layers.Concatenate()([motion, camera_embedding])
    fused = layers.Dense(48, activation="relu", name="fusion",
                         kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 700))(fused)
    fused_accel = layers.Dense(10, kernel_initializer="zeros", name="fused_acceleration")(fused)
    accel = layers.Add()([accel, fused_accel])
    accel = layers.Reshape((5, 2))(accel)
    # Each axis has been scaled by the train residual-speed RMS.
    basis = acceleration_basis() / np.maximum(TIMES[:, None], 1.0)
    trajectory = keras.ops.einsum("tk,bkd->btd", tf.constant(basis), accel)
    return keras.Model(inputs, trajectory, name=f"ado_waymo_{mode}")


def inputs_for(data, rows, mode, image_rows=None):
    values = [data["history_input"][rows], data["intent_input"][rows]]
    if mode == "vision":
        values.append(data["images"][rows if image_rows is None else image_rows])
    return values


def predict(model, data, rows, mode, denom, image_rows=None, batch_size=64):
    pieces = []
    for offset in range(0, len(rows), batch_size):
        subset = rows[offset:offset + batch_size]
        swapped = None if image_rows is None else image_rows[offset:offset + batch_size]
        scaled = model(inputs_for(data, subset, mode, swapped), training=False).numpy()
        pieces.append(data["cv"][subset] + scaled * denom)
    return np.concatenate(pieces)
