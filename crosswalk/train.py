"""Train actuator behavior cloning from native CARLA Traffic Manager demonstrations."""

import argparse
import json
from pathlib import Path

import numpy as np

from .policy import ACTION_NAMES, FEATURE_NAMES, NAV_NAMES, Model, forward, head_columns, input_mask


def _balanced_weights(bins):
    # Square-root frequency weighting retains frequent cruising and rare braking/turning.
    counts = np.bincount(bins)
    weights = 1.0 / np.sqrt(counts[bins])
    return (weights / weights.mean()).astype(np.float32)


def _fit_head(X, target, sample_weights, width, epochs, rng):
    params = [rng.normal(0, np.sqrt(1 / X.shape[1]), (X.shape[1], width)).astype(np.float32),
              np.zeros(width, np.float32),
              rng.normal(0, 0.05, (width, 1)).astype(np.float32),
              np.array([np.average(target, weights=sample_weights)], dtype=np.float32)]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    step = 0
    for epoch in range(epochs):
        shuffled = rng.permutation(len(target))
        learning_rate = 0.003 * (1 - 0.8 * epoch / epochs)
        for start in range(0, len(target), 128):
            rows = shuffled[start:start + 128]
            x = X[rows]
            hidden, prediction = forward(x, params)
            error = prediction[:, 0] - target[rows]
            delta = 2 * sample_weights[rows] * error / len(rows)
            gradient = delta[:, None] * params[2][:, 0] * (1 - hidden**2)
            grads = (np.einsum("bi,bj->ij", x, gradient), gradient.sum(axis=0),
                     np.einsum("bi,b->i", hidden, delta)[:, None], np.array([delta.sum()]))
            step += 1
            for p, m, v, g in zip(params, first, second, grads):
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g**2
                p -= learning_rate * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    return params


def fit(X, y, *, inputs="fusion", epochs=100, seed=2026):
    """Learn steering from local route and speed; learn pedal demand from perception and speed."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mask = input_mask(inputs)
    if (y.ndim != 2 or y.shape[1] != 3 or X.shape != (len(y), len(mask))
            or len(y) < 2 or epochs < 1):
        raise ValueError("expected feature rows and N x 3 demonstrated steering, throttle, brake labels")
    if (not np.isfinite(X[:, mask]).all() or not np.isfinite(y).all()
            or (y < [-1, 0, 0]).any() or (y > 1).any()):
        raise ValueError("invalid observations or actuator labels")

    steer_cols, drive_cols = head_columns(inputs)
    steer_x = X[:, steer_cols]
    # Reflection gives a paired example in the opposite direction. Only lateral
    # route coordinates and their steering labels change sign.
    mirror = np.asarray([-1 if FEATURE_NAMES[i] in NAV_NAMES and FEATURE_NAMES[i].endswith("_y_m")
                         else 1 for i in steer_cols], dtype=np.float32)
    steer_x = np.concatenate((steer_x, steer_x * mirror))
    steer_y = np.concatenate((y[:, 0], -y[:, 0]))
    drive_y = y[:, 1] - y[:, 2]
    batches = (
        (steer_cols, steer_x, steer_y, np.digitize(np.abs(steer_y), [.02, .08, .25, .55]), 24),
        (drive_cols, X[:, drive_cols], drive_y, np.digitize(drive_y, [-.5, -.05, .05, .5]), 40),
    )
    mean = np.zeros(len(mask), dtype=np.float32)
    scale = np.ones(len(mask), dtype=np.float32)
    heads = []
    for columns, observations, actions, bins, width in batches:
        mean[columns] = observations.mean(axis=0, dtype=np.float64)
        # A floor avoids magnifying sensor jitter in features nearly constant on a route.
        floor = [0.15 if FEATURE_NAMES[i] in NAV_NAMES else 0.03 for i in columns]
        scale[columns] = np.maximum(observations.std(axis=0, dtype=np.float64), floor)
        z = np.clip((observations - mean[columns]) / scale[columns], -8, 8)
        heads.append(_fit_head(z, actions, _balanced_weights(bins), width, epochs,
                               np.random.default_rng(seed)))
    return Model(inputs, mean, scale, heads)


def _metrics(y, predicted):
    error = predicted - y
    return {name: {"mae": float(np.abs(error[:, i]).mean()),
                   "rmse": float(np.sqrt((error[:, i]**2).mean()))}
            for i, name in enumerate(ACTION_NAMES)}


def train(data_path, out_path, report_path=None, inputs="fusion"):
    with np.load(data_path, allow_pickle=False) as records:
        X, y = records["X"], records["y"]
        group_key = next((k for k in ("seed", "seeds", "episode", "episodes", "episode_ids", "episode_id")
                          if k in records), None)
        groups = records[group_key].reshape(-1) if group_key else None
    columns = head_columns(inputs)
    report = {"frames": len(y), "inputs": inputs, "outputs": list(ACTION_NAMES),
              "features": int(input_mask(inputs).sum()),
              "head_features": {head: [FEATURE_NAMES[i] for i in indices]
                                for head, indices in zip(("steer", "drive"), columns)}}
    if groups is not None:
        if len(groups) != len(y):
            raise ValueError("group identifiers must match frame count")
        unique = np.unique(groups)
        report["group_key"] = group_key
        report["groups"] = len(unique)
        if len(unique) >= 2:
            held = np.random.default_rng(2026).permutation(unique)[:max(1, len(unique) // 5)]
            check = np.isin(groups, held)
            if (~check).sum() >= 2:
                validation_model = fit(X[~check], y[~check], inputs=inputs)
                report["development_holdout_before_final_refit"] = {
                    "held_groups": held.tolist(), "frames": int(check.sum()),
                    "actuators": _metrics(y[check], validation_model.predict_batch(X[check])),
                }
    model = fit(X, y, inputs=inputs)
    model.save(out_path)
    report["final_model_fit_frames"] = len(y)
    report["training_action_ranges"] = {name: [float(y[:, i].min()), float(y[:, i].max())]
                                        for i, name in enumerate(ACTION_NAMES)}
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="NPZ of synchronized observations and N x 3 controls")
    parser.add_argument("--out", required=True, help="output neural policy NPZ")
    parser.add_argument("--inputs", choices=("fusion", "camera"), default="fusion")
    parser.add_argument("--report", help="optional training and development-holdout JSON")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.out, args.report, inputs=args.inputs), indent=2))


if __name__ == "__main__":
    main()
