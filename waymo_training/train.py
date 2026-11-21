"""Train from scratch, select by dev ADE, and score held-out weights once."""
import argparse
from hashlib import sha256
import json
import platform
from pathlib import Path
import time

from .model import (MODES, configure_device, fit_scales, inputs_for, keras,
                    load_data, make_model, np, predict, prepare_arrays, tf)
from .evaluate import evaluate, write_json


def prefetched_batches(tensors, targets, order, batch_size):
    def lookup(indices):
        return tuple(tf.gather(t, indices) for t in tensors), tf.gather(targets, indices)

    ds = tf.data.Dataset.from_tensor_slices(order).batch(batch_size)
    opts = tf.data.Options()
    opts.threading.private_threadpool_size = 2
    opts.threading.max_intra_op_parallelism = 1
    return ds.with_options(opts).map(lookup, num_parallel_calls=1).prefetch(1)


def fit_model(data, config, mode, seed, run, curve):
    keras.backend.clear_session()
    model = make_model(mode, seed)
    optimizer = keras.optimizers.Adam(config["learning_rate"], clipnorm=5.0)
    optimizer.build(model.trainable_variables)
    model.compile(optimizer=optimizer, loss=keras.losses.Huber(delta=1.0), jit_compile=False)
    denom = prepare_arrays(data, config["scales"])
    train = np.flatnonzero(data["split"] == "train")
    dev = np.flatnonzero(data["split"] == "dev")
    name = f"{mode}-{seed}"
    history = curve.setdefault(name, {"epochs": [], "parameters": model.count_params()})
    epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False)
    best_var = tf.Variable(np.inf, dtype=tf.float64, trainable=False)
    best_epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False)
    state = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=epoch_var,
                                best=best_var, best_epoch=best_epoch_var)
    manager = tf.train.CheckpointManager(state, str(run / "resume" / name), max_to_keep=2)
    if manager.latest_checkpoint:
        state.restore(manager.latest_checkpoint).assert_existing_objects_matched()
    completed = int(epoch_var.numpy())
    history["epochs"] = [point for point in history["epochs"] if point["epoch"] <= completed]
    stale = completed - int(best_epoch_var.numpy())
    if completed == config["max_epochs"] or stale >= config["patience"]:
        return
    start = time.monotonic()
    elapsed = history.get("elapsed_seconds", 0.0)
    with tf.device("/CPU:0"):
        tensors = tuple(tf.convert_to_tensor(x) for x in inputs_for(data, train, mode))
        targets = tf.convert_to_tensor(data["target"][train])
    for epoch in range(completed + 1, config["max_epochs"] + 1):
        # Each epoch has its own seed. Interrupting does not change the minibatches.
        order = np.random.default_rng([seed, epoch]).permutation(len(train)).astype(np.int32)
        batches = prefetched_batches(tensors, targets, order, config["batch_size"])
        fit = model.fit(batches, epochs=1, verbose=0)
        loss = float(fit.history["loss"][0])
        predictions = predict(model, data, dev, mode, denom)
        dev_ade = float(np.linalg.norm(predictions - data["future"][dev], axis=-1).mean())
        if not np.isfinite(dev_ade) or not np.isfinite(loss):
            raise ValueError("Nonfinite loss or dev ADE")
        history["epochs"].append({"epoch": epoch, "train_huber": loss, "dev_ADE_m": dev_ade})
        if dev_ade < float(best_var.numpy()):
            best_var.assign(dev_ade)
            best_epoch_var.assign(epoch)
            temporary = run / f"pending-{name}.weights.h5"
            model.save_weights(temporary)
            temporary.replace(run / f"checkpoint-{name}.weights.h5")
        history["best_epoch"] = int(best_epoch_var.numpy())
        history["elapsed_seconds"] = round(elapsed + time.monotonic() - start, 2)
        write_json(run / "training_curve.json", curve)
        epoch_var.assign(epoch)
        manager.save(checkpoint_number=epoch)
        print(f"{mode} seed={seed} epoch={epoch} dev_ADE_m={dev_ade:.4f}", flush=True)
        if epoch - history["best_epoch"] >= config["patience"]:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 73])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--resume", action="store_true", help="resume completed epochs in the same output folder")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("--output must be empty or new; pass --resume to continue")
    if min(args.epochs, args.patience, args.batch_size) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Epoch count, patience, batch size must be positive and seeds distinct")
    device = configure_device(args.device)
    tf.config.experimental.enable_op_determinism()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_data(args.features)
    partitions = {s: np.flatnonzero(data["split"] == s) for s in ("train", "dev", "test")}
    if not all(len(v) > 0 for v in partitions.values()):
        raise ValueError("Nonempty train, dev and test splits required")
    config = {
        "seeds": args.seeds, "max_epochs": args.epochs, "patience": args.patience,
        "device": device, "batch_size": args.batch_size, "learning_rate": 0.001,
        "selection": "minimum window-weighted dev ADE separately per seed and mode",
        "loss": "coordinate Huber on CV residual divided by max(t,1)*train-axis-residual-speed RMS",
        "pixel_input": "2x2 area pooling of supplied 64x96 RGB producing 32x48 per camera",
        "normalization": "per-channel training history mean/std; train-only XY residual scaling",
        "segment_split_overlap": 0, "duplicate_segment_frame_indices": 0,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "tensorflow": tf.__version__, "keras": keras.__version__},
        "sources": data["sources"], "scales": fit_scales(data, partitions["train"]),
        "dataset_source_sha256": sha256(json.dumps(data["sources"], sort_keys=True).encode()).hexdigest(),
        "partitions": {k: {"windows": len(v), "segments": len(np.unique(data["group"][v]))} for k, v in partitions.items()},
    }
    config_path = args.output / "config.json"
    if args.resume:
        if not config_path.is_file():
            parser.error("--resume needs an existing run with config.json")
        saved = json.loads(config_path.read_text("utf8"))
        if any(config[k] != saved[k] for k in ("seeds", "max_epochs", "patience", "batch_size", "versions", "sources", "learning_rate")):
            raise ValueError("Resume requires the same feature snapshot, runtime versions and training options")
        config = saved
    else:
        write_json(config_path, config)
    print(config["partitions"], flush=True)
    curve_file = args.output / "training_curve.json"
    curve = json.loads(curve_file.read_text("utf8")) if args.resume and curve_file.exists() else {}
    for seed in args.seeds:
        for mode in MODES:
            fit_model(data, config, mode, seed, args.output, curve)
    result = evaluate(data, args.output, config)
    print(json.dumps(result["splits"]["test"]["seed_mean"], indent=2), flush=True)


if __name__ == "__main__":
    main()
