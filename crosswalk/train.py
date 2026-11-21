"""Train actuator behavior cloning from native CARLA Traffic Manager demonstrations."""

import argparse
import json
from pathlib import Path

import numpy as np

from .steering_search import refine
from .policy import ACTION_NAMES, FEATURE_NAMES, NAV_NAMES, Model, forward, head_columns, input_mask


def _balanced_weights(bins):
    # Square-root frequency weighting retains frequent cruising and rare braking/turning.
    counts = np.bincount(bins)
    weights = 1.0 / np.sqrt(counts[bins])
    return (weights / weights.mean()).astype(np.float32)


def _drop_scenery(x, columns, rng):
    """Mask background statistics as a group while preserving specific object and lamp cues."""
    # Normalized zero corresponds to the training mean.
    if not len(columns):
        return x
    rows = np.flatnonzero(rng.random(len(x)) < .65)
    if len(rows):
        x[np.ix_(rows, columns)] = 0.
    return x


def _fit_head(X, target, sample_weights, width, epochs, rng, scenery=()):
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
            x = _drop_scenery(X[rows], scenery, rng)
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
    steer_y = y[:, 0]
    # Reflection gives a paired example in the opposite direction. Only lateral
    # route coordinates and their steering labels change sign.
    mirror = np.asarray([-1 if FEATURE_NAMES[i] in NAV_NAMES and FEATURE_NAMES[i].endswith("_y_m")
                         else 1 for i in steer_cols], dtype=np.float32)
    steer_x = np.concatenate((steer_x, steer_x * mirror))
    steer_y = np.concatenate((steer_y, -steer_y))
    drive_y = y[:, 1] - y[:, 2]
    batches = (
        (steer_cols, steer_x, steer_y, np.digitize(np.abs(steer_y), [.02, .08, .25, .55]), 24),
        (drive_cols, X[:, drive_cols], drive_y, np.digitize(drive_y, [-.5, -.05, .05, .5])
         + 5 * (X[:, FEATURE_NAMES.index("ego_speed_mps")] < .5), 48),
    )
    mean = np.zeros((2, len(mask)), dtype=np.float32)
    scale = np.ones_like(mean)
    heads = []
    for head, (columns, observations, actions, bins, width) in enumerate(batches):
        mean[head, columns] = observations.mean(axis=0, dtype=np.float64)
        # A floor avoids magnifying sensor jitter in features nearly constant on a route.
        floor = [0.15 if FEATURE_NAMES[i] in NAV_NAMES else 0.03 for i in columns]
        scale[head, columns] = np.maximum(observations.std(axis=0, dtype=np.float64), floor)
        z = np.clip((observations - mean[head, columns]) / scale[head, columns], -8, 8)
        scenery = [i for i, c in enumerate(columns)
                   if FEATURE_NAMES[c].startswith(("rgb_horizon_", "rgb_near_"))
                   or FEATURE_NAMES[c].endswith(("_luma", "_contrast"))]
        heads.append(_fit_head(z, actions, _balanced_weights(bins), width, epochs,
                               np.random.default_rng(seed), scenery=scenery))
    return Model(inputs, mean, scale, heads)


def _metrics(y, predicted):
    error = predicted - y
    return {name: {"mae": float(np.abs(error[:, i]).mean()),
                   "rmse": float(np.sqrt((error[:, i]**2).mean()))}
            for i, name in enumerate(ACTION_NAMES)}


def _holdout_report(model, X, y):
    predicted = model.predict_batch(X)
    braking = y[:, 2] > .2
    moving_brake = braking & (X[:, FEATURE_NAMES.index("ego_speed_mps")] > .5)
    report = {"frames": len(y), "actuators": _metrics(y, predicted)}
    for label, rows in (("braking", braking), ("braking_while_moving", moving_brake)):
        report[label] = {"frames": int(rows.sum()),
                         "recall_at_brake_0.2": float((predicted[rows, 2] > .2).mean()) if rows.any() else None}
    # This measures functional dependence on input groups on real, withheld
    # observations. Correlated groups can substitute for each other. Only actual
    # matched simulator evaluations measure a change in driving outcomes.
    report["mean_replacement_pedal_sensitivity"] = {}
    groups = {
        "front_object_detections": ("rgb_path_", "rgb_person_", "rgb_vehicle_"),
        "front_signal_candidates": ("rgb_signal_",),
        "front_color_statistics": ("rgb_horizon_", "rgb_near_"),
        "side_camera_statistics_and_candidates": ("rgb_left_", "rgb_right_"),
    }
    if model.inputs == "fusion":
        groups.update({
            "spatial_lidar_density": ("bev_lidar_density_",),
            "spatial_radar_density_and_velocity": ("bev_radar_",),
        })
    for group, prefixes in groups.items():
        indices = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith(prefixes)]
        replaced = X.copy()
        replaced[:, indices] = model.mean[1, indices]
        delta = model.predict_batch(replaced) - predicted
        report["mean_replacement_pedal_sensitivity"][group] = {
            "throttle_mean_absolute_change": float(np.abs(delta[:, 1]).mean()),
            "brake_mean_absolute_change": float(np.abs(delta[:, 2]).mean())}
    return report


def train(data_path, out_path, report_path=None, inputs="fusion", *, route_refinement=True, imitation_path=None):
    with np.load(data_path, allow_pickle=False) as records:
        X, y = records["X"], records["y"]
        if (X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES)
                or ("feature_names" in records and tuple(records["feature_names"]) != FEATURE_NAMES)
                or ("action_names" in records and tuple(records["action_names"]) != ACTION_NAMES)):
            raise ValueError("incompatible demonstration schema; collect synchronized city views")
        group_key = next((k for k in ("seed", "seeds", "episode", "episodes", "episode_ids", "episode_id")
                          if k in records), None)
        groups = records[group_key].reshape(-1) if group_key else None
        routes, route_seeds = [], None
        if route_refinement:
            if not {"route_xy", "route_lengths", "route_seeds"} <= set(records.files):
                raise ValueError("route search requires the recorded training route polylines, lengths and seeds")
            points, lengths, route_seeds = (records[k] for k in ("route_xy", "route_lengths", "route_seeds"))
            if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                    or lengths.ndim != 1 or lengths.dtype.kind not in "iu" or not lengths.size
                    or np.any(lengths < 2) or lengths.sum() != len(points) or route_seeds.shape != lengths.shape
                    or group_key not in ("seed", "seeds") or set(route_seeds) != set(groups)):
                raise ValueError("training route polylines and observation seeds must agree")
            routes = np.split(points, np.cumsum(lengths)[:-1])
    columns = head_columns(inputs)
    report = {"frames": len(y), "inputs": inputs, "outputs": list(ACTION_NAMES),
              "features": int(input_mask(inputs).sum()),
              "head_features": {head: [FEATURE_NAMES[i] for i in indices]
                                for head, indices in zip(("steer", "drive"), columns)}}
    report["steering_supervision"] = "Original demonstrated steering and its left-right reflection"
    report["drive_supervision"] = "Traffic Manager throttle minus brake, weighted by action and stop state"
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
                if route_refinement:
                    withheld = np.isin(route_seeds, held)
                    fit_routes = [r for r, keep in zip(routes, ~withheld) if keep]
                    eval_routes = [r for r, keep in zip(routes, withheld) if keep]
                    report["steering_search_seed_holdout"] = refine(
                        validation_model, fit_routes, evaluation_routes=eval_routes)
                report["development_holdout_before_final_refit"] = {
                    "held_groups": held.tolist(), **_holdout_report(validation_model, X[check], y[check]),
                }
    model = fit(X, y, inputs=inputs)
    if imitation_path:
        model.save(imitation_path)
    if route_refinement:
        report["steering_search_final_fit"] = refine(model, routes)
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
    parser.add_argument("--imitation-out", help="save the imitation-only control heads before steering search")
    parser.add_argument("--imitation-only", action="store_true", help="fit only the demonstrated actions")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.out, args.report, inputs=args.inputs,
                           route_refinement=not args.imitation_only, imitation_path=args.imitation_out), indent=2))


if __name__ == "__main__":
    main()
