"""Train actuator behavior cloning from native CARLA Traffic Manager demonstrations."""

import argparse
import json
from pathlib import Path

import numpy as np

from .steering_search import refine
from .policy import (ACTION_NAMES, FEATURE_NAMES, NAV_NAMES, RANGE_FEATURE_NAMES, Model, expand_range,
                     forward, head_columns, input_mask)

LOCAL_PEDAL_CUES = (
    "ego_speed_mps", "corridor_gap_m", "radar_velocity_mps",
    "lidar_inverse_gap", "radar_inverse_gap", "ego_speed_over_gap",
    "rgb_person_confidence", "rgb_person_height", "rgb_person_center_offset", "rgb_vehicle_height",
    "rgb_path_person_height", "rgb_path_person_offset_in_heights",
    "rgb_path_vehicle_height", "rgb_path_vehicle_offset_in_heights",
    "rgb_signal_red_strength", "rgb_signal_red_diameter", "rgb_signal_green_strength",
    "rgb_left_red_strength", "rgb_right_red_strength", "rgb_left_green_strength", "rgb_right_green_strength",
)


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


def _pedal_context_groups(columns, strength):
    """Keep direct gap, velocity and object observations, decorrelate recurring scene backgrounds."""
    def indices(predicate):
        return [i for i, c in enumerate(columns) if predicate(FEATURE_NAMES[c])]

    groups = [
        (indices(lambda n: n.startswith(("rgb_near_", "rgb_horizon_")) and "center_" not in n), .85),
        (indices(lambda n: n.startswith(("rgb_near_center", "rgb_horizon_center"))), .55),
        (indices(lambda n: n.endswith(("_luma", "_contrast"))), .85),
        (indices(lambda n: n.startswith(("rgb_left_", "rgb_right_")) and n.endswith("_strength")), .45),
        (indices(lambda n: n.startswith("rgb_signal_")), .4),
        (indices(lambda n: n.startswith("route_")), .45),
        (indices(lambda n: n.startswith("bev_") and
                 (n.endswith(("_far_left", "_far_right")) or "_22_36m_" in n or "_36_60m_" in n)), .90),
        (indices(lambda n: n.startswith("bev_") and n.endswith("_center")
                 and "_22_36m_" not in n and "_36_60m_" not in n), .45),
    ]
    return [(cols, chance*strength) for cols, chance in groups if cols]


def _fit_head(X, target, sample_weights, width, epochs, rng, scenery=(), replacements=()):
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
            for columns, probability in replacements:
                selected = np.flatnonzero(rng.random(len(rows)) < probability)
                sources = rng.integers(0, len(X), len(selected))
                x[np.ix_(selected, columns)] = X[np.ix_(sources, columns)]
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


PEDAL_VARIANTS = ("original", "group_resampling", "local_cues", "camera_brake_weighted", "combined")


def _pedal_variant(inputs, requested):
    variant = requested or ("local_cues" if inputs in ("fusion", "camera") else "original")
    if variant not in PEDAL_VARIANTS:
        raise ValueError(f"unknown pedal training method: {variant}")
    if variant == "camera_brake_weighted" and inputs != "camera":
        raise ValueError("camera brake weighting requires camera input mode")
    return variant


def _fit_pedals(z, actions, columns, speed, bins, epochs, seed, variant, measured_gap):
    def original_scenery():
        return [i for i, col in enumerate(columns)
                if FEATURE_NAMES[col].startswith(("rgb_horizon_", "rgb_near_"))
                or FEATURE_NAMES[col].endswith(("_luma", "_contrast"))]

    if variant == "original":
        return _fit_head(z, actions, _balanced_weights(bins), 48, epochs,
                         np.random.default_rng(seed), scenery=original_scenery())
    if variant == "combined":
        # Concatenating hidden units implements the mean of both learned signed
        # demands before clipping. Their input normalization is identical.
        broad = _fit_pedals(z, actions, columns, speed, bins, epochs, seed, "group_resampling", measured_gap)
        local = _fit_pedals(z, actions, columns, speed, bins, epochs, seed, "local_cues", measured_gap)
        return [np.column_stack((broad[0], local[0])), np.r_[broad[1], local[1]],
                .5 * np.vstack((broad[2], local[2])), .5 * (broad[3]+local[3])]

    weights = _balanced_weights(bins)
    launching = (speed < 3.) & (actions > .35)
    moving_brake = (speed > 1.) & (actions < -.25)
    if variant == "local_cues":
        weights[launching] *= 3.
        weights[moving_brake & (measured_gap < 14.)] *= 2.
        keep = [i for i, col in enumerate(columns) if FEATURE_NAMES[col] in LOCAL_PEDAL_CUES]
        scenery = [i for i, col in enumerate(columns[keep])
                   if FEATURE_NAMES[col].startswith(("rgb_signal_", "rgb_left_", "rgb_right_"))]
        compact = _fit_head(z[:, keep], actions, weights/weights.mean(), 28, min(epochs, 55),
                            np.random.default_rng(seed), scenery=scenery)
        # Disconnect unused observations in the common serialized tensor layout.
        w1 = np.zeros((len(columns), 28), dtype=np.float32)
        w1[keep] = compact[0]
        return [w1, *compact[1:]]

    camera_emphasis = variant == "camera_brake_weighted"
    weights[launching] *= 4.
    weights[moving_brake] *= 25. if camera_emphasis else 3.
    if camera_emphasis:
        weights[(speed < .5) & (actions < -.4)] *= 2.
        weights[(speed < 1.5) & (actions > .35)] *= 1.5
    replacements = _pedal_context_groups(columns, .6 if camera_emphasis else 1.)
    return _fit_head(z, actions, weights/weights.mean(), 48, min(65, epochs),
                     np.random.default_rng(seed), replacements=replacements)


def fit(X, y, *, inputs="fusion", epochs=100, seed=2026, pedal_training=None):
    """Learn steering and signed pedal demand from actual Traffic Manager actions."""
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
    variant = _pedal_variant(inputs, pedal_training)
    steer_x = X[:, steer_cols]
    # Reflection pairs each demonstration with an opposite turn.
    mirror = np.asarray([-1 if FEATURE_NAMES[i] in NAV_NAMES and FEATURE_NAMES[i].endswith("_y_m")
                         else 1 for i in steer_cols], dtype=np.float32)
    steer_x = np.concatenate((steer_x, steer_x * mirror))
    steer_y = np.concatenate((y[:, 0], -y[:, 0]))
    drive_y = y[:, 1] - y[:, 2]
    speed = X[:, FEATURE_NAMES.index("ego_speed_mps")]
    mean = np.zeros((2, len(mask)), dtype=np.float32)
    scale = np.ones_like(mean)

    def normalize(head, obs, columns, floor):
        mean[head, columns] = obs.mean(axis=0, dtype=np.float64)
        scale[head, columns] = np.maximum(obs.std(axis=0, dtype=np.float64), floor)
        return np.clip((obs - mean[head, columns])/scale[head, columns], -8, 8)

    def original_floor(columns):
        return [.15 if FEATURE_NAMES[i] in NAV_NAMES else .03 for i in columns]

    steer_z = normalize(0, steer_x, steer_cols, original_floor(steer_cols))
    drive_z = normalize(1, X[:, drive_cols], drive_cols, original_floor(drive_cols) if variant == "original" else .03)
    steer_bins = np.digitize(np.abs(steer_y), [.02, .08, .25, .55])
    drive_bins = np.digitize(drive_y, [-.5, -.05, .05, .5]) + 5*(speed < .5)
    steering = _fit_head(steer_z, steer_y, _balanced_weights(steer_bins), 24, epochs, np.random.default_rng(seed))
    # Both matched policies use the same frame weights from the synchronized
    # demonstration archive. A camera policy cannot read range at inference.
    measured_gap = X[:, FEATURE_NAMES.index("corridor_gap_m")]
    if variant in ("local_cues", "combined") and not np.isfinite(measured_gap).all():
        raise ValueError("matched local training requires paired, measured corridor gaps for sample weighting")
    pedals = _fit_pedals(drive_z, drive_y, drive_cols, speed, drive_bins, epochs, seed, variant, measured_gap)
    return Model(inputs, mean, scale, [steering, pedals])


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
    if model.inputs in ("fusion", "range"):
        groups.update({
            "spatial_lidar_density": ("bev_lidar_density_",),
            "spatial_radar_density_and_velocity": ("bev_radar_",),
        })
    for group, prefixes in groups.items():
        indices = [i for i, name in enumerate(FEATURE_NAMES) if model.mask[i] and name.startswith(prefixes)]
        if not indices:
            continue
        replaced = X.copy()
        replaced[:, indices] = model.mean[1, indices]
        delta = model.predict_batch(replaced) - predicted
        report["mean_replacement_pedal_sensitivity"][group] = {
            "throttle_mean_absolute_change": float(np.abs(delta[:, 1]).mean()),
            "brake_mean_absolute_change": float(np.abs(delta[:, 2]).mean())}
    return report


def train(data_path, out_path, report_path=None, inputs="fusion", *, route_refinement=True, imitation_path=None,
          pedal_training=None):
    with np.load(data_path, allow_pickle=False) as records:
        X, y = records["X"], records["y"]
        names = RANGE_FEATURE_NAMES if inputs == "range" else FEATURE_NAMES
        if (X.ndim != 2 or X.shape[1] != len(names)
                or (inputs == "range" and "feature_names" not in records)
                or ("feature_names" in records and tuple(records["feature_names"]) != names)
                or ("action_names" in records and tuple(records["action_names"]) != ACTION_NAMES)):
            raise ValueError(f"incompatible demonstration schema for {inputs} policy")
        if inputs == "range":
            X = expand_range(X)
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
    if inputs == "range":
        report["packed_feature_names"] = list(RANGE_FEATURE_NAMES)
    report["drive_supervision"] = "Traffic Manager throttle minus brake, weighted by action and stop state"
    report["pedal_training"] = _pedal_variant(inputs, pedal_training)
    if report["pedal_training"] == "local_cues":
        report["head_features"]["drive"] = [n for n in report["head_features"]["drive"] if n in LOCAL_PEDAL_CUES]
        report["available_input_count"] = report["features"]
        report["features"] = len(set(report["head_features"]["steer"] + report["head_features"]["drive"]))
        report["pedal_training_details"] = {
            "hidden_units": 28, "hidden_activation": "tanh", "signed_linear_outputs": 1,
            "epochs": 55, "batch_size": 128, "optimizer": "Adam", "initial_learning_rate": .003,
            "random_seed": 2026, "targets": "unchanged demonstrated throttle minus demonstrated brake",
            "common_frame_weights": {
                "base": "inverse square root of signed-action and stopped/moving bin frequency",
                "action_bin_edges": [-.5, -.05, .05, .5], "stopped_speed_mps": .5,
                "launch_multiplier": 3., "launch_condition": "speed < 3 m/s and signed pedal > 0.35",
                "moving_brake_multiplier": 2.,
                "moving_brake_condition": "speed > 1 m/s, signed pedal < -0.25, measured corridor gap < 14 m",
                "range_use": "same measured training-only corridor gap for both policies' sample weights"},
            "augmentation": "mask lamp cue group to training mean on 65 percent of training rows",
            "input_names": report["head_features"]["drive"],
        }
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
                validation_model = fit(X[~check], y[~check], inputs=inputs, pedal_training=pedal_training)
                if route_refinement:
                    withheld = np.isin(route_seeds, held)
                    fit_routes = [r for r, keep in zip(routes, ~withheld) if keep]
                    eval_routes = [r for r, keep in zip(routes, withheld) if keep]
                    report["steering_search_seed_holdout"] = refine(
                        validation_model, fit_routes, evaluation_routes=eval_routes)
                report["development_holdout_before_final_refit"] = {
                    "held_groups": held.tolist(), **_holdout_report(validation_model, X[check], y[check]),
                }
    model = fit(X, y, inputs=inputs, pedal_training=pedal_training)
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
    parser.add_argument("--inputs", choices=("fusion", "camera", "range"), default="fusion")
    parser.add_argument("--report", help="optional training and development-holdout JSON")
    parser.add_argument("--imitation-out", help="save the imitation-only control heads before steering search")
    parser.add_argument("--imitation-only", action="store_true", help="fit only the demonstrated actions")
    parser.add_argument("--pedal-training", choices=PEDAL_VARIANTS,
                        help="select a longitudinal training variant for a matched development comparison")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.out, args.report, inputs=args.inputs,
                           route_refinement=not args.imitation_only, imitation_path=args.imitation_out,
                           pedal_training=args.pedal_training), indent=2))


if __name__ == "__main__":
    main()
