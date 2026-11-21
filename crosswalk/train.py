"""Train actuator behavior cloning from native CARLA Traffic Manager demonstrations."""

import argparse
import json
from pathlib import Path

import numpy as np

from .policy import ACTION_NAMES, FEATURE_NAMES, NAV_NAMES, Model, forward, head_columns, input_mask


# CARLA's 1.0 ROS 2 bridge publishes steady-state measurements at ~3 m/s
# for the UE5.8 Chaos MKZ. Use samples of this steering response as a prior.
STEER_SOURCE = ("https://github.com/carla-simulator/carla/blob/1.0/LibCarla/source/"
                "carla/ros2/AutowareSteeringCompensation.h")
GEOMETRY_SOURCE = ("https://github.com/autowarefoundation/autoware_universe/blob/main/"
                   "simulator/autoware_carla_interface/README.md")
MKZ_WHEELBASE_M = 2.85
MKZ_WHEEL_ANGLES_RAD = (0., .0495395, .188953, .381975, .559125, .628645)
MKZ_NORMALIZED_STEER = (0., .05, .2, .4, .6, .7)


def _balanced_weights(bins):
    # Square-root frequency weighting retains frequent cruising and rare braking/turning.
    counts = np.bincount(bins)
    weights = 1.0 / np.sqrt(counts[bins])
    return (weights / weights.mean()).astype(np.float32)


def _route_steering(route, speed):
    """Training-only bicycle-model prior for the Chaos Lincoln MKZ.

    The Autoware bridge's nominal wheelbase and midpoint convention locate the
    rear axle. Convert curvature to a tire angle and interpolate CARLA's
    published low-speed sweep; cap auxiliary targets at its measured range.
    """
    fraction = np.clip((3.8 + .28*speed - 3.) / 4., 0., 1.)
    goal = route[:, 0, :] + (route[:, 1, :] - route[:, 0, :]) * fraction[:, None]
    rear_to_goal = goal[:, 0] + MKZ_WHEELBASE_M / 2.
    curvature = 2.*goal[:, 1] / np.maximum(rear_to_goal**2 + goal[:, 1]**2, .25)
    angle = np.arctan(MKZ_WHEELBASE_M * curvature)
    return np.sign(angle) * np.interp(np.abs(angle), MKZ_WHEEL_ANGLES_RAD, MKZ_NORMALIZED_STEER)


def _augmented_steering(observations, actions, rng):
    # Steering columns are ordered speed, x3, y3, x7, y7, x13, y13.
    speed = observations[:, 0]
    route = observations[:, 1:].reshape(-1, 3, 2)
    reference = _route_steering(route, speed)
    # TM can miss the intended intersection branch. Bound disagreement with
    # its route, retaining its demonstrated control around attainable geometry.
    measured = reference + np.clip(actions - reference, -.12, .12)
    routes = [observations]
    labels = [measured]
    for _ in range(2):
        lateral = rng.uniform(-1.7, 1.7, len(route))
        yaw = np.deg2rad(rng.uniform(-16, 16, len(route)))
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        dx, dy = route[:, :, 0], route[:, :, 1] - lateral[:, None]
        moved = np.stack((dx*c + dy*s, -dx*s + dy*c), axis=2)
        routes.append(np.column_stack((speed, moved.reshape(-1, 6))))
        labels.append(_route_steering(moved, speed))
    return np.concatenate(routes).astype(np.float32), np.concatenate(labels).astype(np.float32)


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


def fit(X, y, *, inputs="fusion", epochs=100, seed=2026, route_augmentation=False):
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
    if route_augmentation:
        steer_x, steer_y = _augmented_steering(steer_x, steer_y, np.random.default_rng(seed))
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


def train(data_path, out_path, report_path=None, inputs="fusion", *, route_augmentation=True):
    with np.load(data_path, allow_pickle=False) as records:
        X, y = records["X"], records["y"]
        if (X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES)
                or ("feature_names" in records and tuple(records["feature_names"]) != FEATURE_NAMES)
                or ("action_names" in records and tuple(records["action_names"]) != ACTION_NAMES)):
            raise ValueError("incompatible demonstration schema; collect synchronized city views")
        group_key = next((k for k in ("seed", "seeds", "episode", "episodes", "episode_ids", "episode_id")
                          if k in records), None)
        groups = records[group_key].reshape(-1) if group_key else None
    columns = head_columns(inputs)
    report = {"frames": len(y), "inputs": inputs, "outputs": list(ACTION_NAMES),
              "features": int(input_mask(inputs).sum()),
              "head_features": {head: [FEATURE_NAMES[i] for i in indices]
                                for head, indices in zip(("steer", "drive"), columns)}}
    report["steering_supervision"] = (
        "Traffic Manager with bounded route disagreement, synthetic pose perturbations and bicycle-model "
        "route targets (nominal MKZ geometry and published low-speed Chaos steering response)" if route_augmentation
        else "Traffic Manager demonstrations and mirrored observations")
    if route_augmentation:
        report["steering_prior"] = {
            "nominal_wheelbase_m": MKZ_WHEELBASE_M, "rear_axle_reference": "half wheelbase behind actor origin",
            "geometry_source": GEOMETRY_SOURCE, "steering_response_source": STEER_SOURCE,
            "upstream_calibration_speed_mps": 3., "maximum_auxiliary_steer": MKZ_NORMALIZED_STEER[-1]}
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
                validation_model = fit(X[~check], y[~check], inputs=inputs, route_augmentation=route_augmentation)
                report["development_holdout_before_final_refit"] = {
                    "held_groups": held.tolist(), **_holdout_report(validation_model, X[check], y[check]),
                }
    model = fit(X, y, inputs=inputs, route_augmentation=route_augmentation)
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
