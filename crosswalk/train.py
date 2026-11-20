"""Fit a speed-command regressor to recorded expert demonstrations using NumPy."""

import argparse
import json
from pathlib import Path

import numpy as np

from .policy import CRUISE_SPEED, FEATURE_NAMES, Model, input_mask, rule_speed


def fit(X, y, *, inputs="fusion", trees=48, depth=2, rate=0.16):
    """Fit shallow least-squares regression trees to successive speed residuals.

    Speed-bin weighting reduces the imbalance between cruising and crossing frames.
    Trees minimize weighted squared error. Successive trees learn residuals as in
    J. H. Friedman, Greedy Function Approximation (2001), doi:10.1214/aos/1013203451.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.shape != (len(y), len(FEATURE_NAMES)) or len(y) < 2:
        raise ValueError(f"expected matching labels and a feature matrix with {len(FEATURE_NAMES)} columns")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("training data contains nonfinite values")
    if (y < 0).any() or (y > CRUISE_SPEED + 1e-5).any():
        raise ValueError("target speed is outside the operating range")
    if np.ptp(y) < 0.3:
        raise ValueError("training labels have insufficient variation to fit a driving policy")
    if trees < 1 or depth < 1 or depth > 6 or not 0 < rate <= 1:
        raise ValueError("invalid boosting parameters")

    bins = np.digitize(y, [0.5, 2.0, 4.0, 5.8])
    weights = 1.0 / np.sqrt(np.bincount(bins)[bins])
    bias = float(np.average(y, weights=weights))
    residual = y - bias
    shape = (trees, 2 ** (depth + 1) - 1)
    splits = np.full(shape, -1, dtype=np.int32)
    thresholds = np.zeros(shape, dtype=np.float64)
    values = np.zeros(shape, dtype=np.float64)
    min_leaf = max(2, min(12, len(y) // 100))
    columns = np.flatnonzero(input_mask(inputs))

    # Sort feature values once. A child keeps the parent order within each column.
    root_order = np.argsort(X[:, columns], axis=0, kind="stable")
    for tree in range(trees):
        update = np.zeros(len(y), dtype=np.float64)

        def grow(order, node, remaining_depth):
            rows = order[:, 0]
            sw = weights[rows].sum()
            sy = np.dot(weights[rows], residual[rows])
            leaf_value = sy / sw * rate
            values[tree, node] = leaf_value
            if remaining_depth == 0 or len(rows) < 2 * min_leaf:
                update[rows] = leaf_value
                return

            sorted_x = X[order, columns]
            left_w = np.cumsum(weights[order], axis=0)[:-1]
            left_y = np.cumsum((weights * residual)[order], axis=0)[:-1]
            # Improvement in weighted SSE for every valid threshold.
            gain = left_y**2 / left_w + (sy - left_y)**2 / (sw - left_w) - sy**2 / sw
            gain[sorted_x[:-1] == sorted_x[1:]] = -np.inf
            gain[:min_leaf - 1] = -np.inf
            gain[len(rows) - min_leaf:] = -np.inf
            cut, local_col = np.unravel_index(gain.argmax(), gain.shape)
            if gain[cut, local_col] < 1e-8:
                update[rows] = leaf_value
                return

            col = columns[local_col]
            edge = sorted_x[cut, local_col]
            threshold = (float(edge) + float(sorted_x[cut + 1, local_col])) / 2
            splits[tree, node] = col
            thresholds[tree, node] = threshold
            goes_left = X[:, col] <= edge
            take_left = goes_left[order]
            # Every column contains the same row IDs in its own sorted order.
            left_order = order.T[take_left.T].reshape(len(columns), -1).T
            right_order = order.T[~take_left.T].reshape(len(columns), -1).T
            grow(left_order, node * 2 + 1, remaining_depth - 1)
            grow(right_order, node * 2 + 2, remaining_depth - 1)

        grow(root_order, 0, depth)
        residual -= update
    return Model(bias, splits, thresholds, values, inputs)


def _metrics(actual, predicted):
    error = actual - np.asarray(predicted)
    return {"mae_mps": float(np.abs(error).mean()), "rmse_mps": float(np.sqrt((error**2).mean()))}


def train(data_path, out_path, report_path=None, *, inputs="fusion"):
    """Report a grouped holdout when IDs exist, then refit the shipping model on all rows."""
    with np.load(data_path, allow_pickle=False) as records:
        X = records["X"]
        y = records["y"].astype(float).reshape(-1)
        group_key = next((k for k in ("episode", "episodes", "episode_ids", "episode_id")
                          if k in records), None)
        groups = records[group_key].reshape(-1) if group_key else None
    report = {"frames": len(y), "inputs": inputs, "features": int(input_mask(inputs).sum()),
              "available_features": [k for k, allowed in zip(FEATURE_NAMES, input_mask(inputs)) if allowed],
              "teacher_speed_min_mps": float(y.min()), "teacher_speed_max_mps": float(y.max())}
    if groups is not None:
        if len(groups) != len(y):
            raise ValueError("episode ids must match frame count")
        unique = np.unique(groups)
        report["episodes"] = len(unique)
        if len(unique) >= 3:
            shuffled = np.random.default_rng(2026).permutation(unique)
            held = shuffled[:max(1, len(unique) // 5)]
            check = np.isin(groups, held)
            if np.ptp(y[~check]) >= 0.3:
                validation_model = fit(X[~check], y[~check], inputs=inputs)
                report["development_holdout_before_final_refit"] = {
                    "episode_ids": held.tolist(),
                    "frames": int(check.sum()),
                    "learned": _metrics(y[check], [validation_model.predict(x) for x in X[check]]),
                    "rules": _metrics(y[check], [rule_speed(x) for x in X[check]]),
                }
    model = fit(X, y, inputs=inputs)
    model.save(out_path)
    used = np.bincount(model.splits[model.splits >= 0], minlength=len(FEATURE_NAMES))
    report["final_model_split_counts"] = {name: int(count) for name, count in zip(FEATURE_NAMES, used)
                                          if count}
    report["final_model_fit_frames"] = len(y)
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="NPZ with X, y and optional episode ids")
    parser.add_argument("--out", required=True, help="output policy NPZ")
    parser.add_argument("--inputs", choices=("fusion", "camera"), default="fusion",
                        help="select features before fitting; camera mode includes RGB and ego speed")
    parser.add_argument("--report", help="optional training and development-holdout JSON")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.out, args.report, inputs=args.inputs), indent=2))


if __name__ == "__main__":
    main()
