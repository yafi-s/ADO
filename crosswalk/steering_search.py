"""Offline cross-entropy search for a neural steering head on demonstrated route geometry."""

import numpy as np


STEER_SOURCE = ("https://github.com/carla-simulator/carla/blob/1.0/LibCarla/source/"
                "carla/ros2/AutowareSteeringCompensation.h")
GEOMETRY_SOURCE = ("https://github.com/autowarefoundation/autoware_universe/blob/main/"
                   "simulator/autoware_carla_interface/README.md")
CEM_SOURCE = "https://doi.org/10.1023/A:1010091220143"
WHEELBASE = 2.85
TIRE_ANGLE = (0., .0495395, .188953, .381975, .559125, .628645)
NORMALIZED_STEER = (0., .05, .2, .4, .6, .7)
DT = .1
STEPS = 45
PATH_LENGTH = 45
POPULATION = 48
GENERATIONS = 14


def route_bank(routes, seed):
    """Stratify short geometry excerpts by the direction and size of the upcoming bend."""
    rng = np.random.default_rng(seed)
    bins = [[], [], [], [], []]
    for route in routes:
        route = np.asarray(route, dtype=np.float64)
        if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2 or not np.isfinite(route).all():
            raise ValueError("invalid demonstration route polyline")
        lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
        route = route[np.r_[True, lengths > 1.e-4]]
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
        for start in np.arange(0., distance[-1] - PATH_LENGTH + 1.e-6, 4.):
            stations = start + np.arange(PATH_LENGTH + 1)
            path = np.column_stack([np.interp(stations, distance, axis) for axis in route.T])
            initial = path[3] - path[0]
            later = path[23] - path[20]
            cross = initial[0]*later[1] - initial[1]*later[0]
            bend = np.arctan2(cross, np.dot(initial, later))
            bins[np.digitize(bend, [-.65, -.12, .12, .65])].append(path - path[0])
    paths = []
    for subset in bins:
        if subset:
            chosen = rng.choice(len(subset), size=min(5, len(subset)), replace=False)
            paths.extend(subset[i] for i in chosen)
    if not paths:
        raise ValueError(f"route refinement needs a demonstration route at least {PATH_LENGTH} m long")
    paths = np.asarray(paths, dtype=np.float32)
    # A stationary hold carries no steering feedback. Vary the prescribed speed
    # and initial pose to exercise recovery in low-speed urban maneuvers.
    speeds = rng.uniform(3., 6.0, len(paths)).astype(np.float32)
    lateral = rng.uniform(-1.3, 1.3, len(paths)).astype(np.float32)
    yaw_error = np.deg2rad(rng.uniform(-12., 12., len(paths))).astype(np.float32)
    return paths, speeds, lateral, yaw_error


def rollout(bank, hidden_weights, normalization, candidates):
    """Each candidate's own predicted steer advances its own rear-axle bicycle.

    Fixed speed isolates steering. Project each resulting actor center onto
    nearby real path segments and read the same 3/7/13 m navigation inputs
    as deployment. Candidate rows contain the output weights and bias.
    """
    paths, speed, lateral, yaw_error = bank
    segments = np.diff(paths, axis=1)
    direction = segments / np.maximum(np.linalg.norm(segments, axis=2, keepdims=True), 1.e-5)
    cases = np.arange(len(paths))
    shape = (len(candidates), len(paths))
    stations = np.full(shape, 2, dtype=np.int32)
    tangent = direction[:, 2]
    heading = np.broadcast_to(np.arctan2(tangent[:, 1], tangent[:, 0]) + yaw_error, shape).copy()
    center = paths[:, 2] + lateral[:, None] * np.stack((-tangent[:, 1], tangent[:, 0]), axis=-1)
    rear_x = center[:, 0] - WHEELBASE/2 * np.cos(heading)
    rear_y = center[:, 1] - WHEELBASE/2 * np.sin(heading)
    previous = np.zeros(shape, dtype=np.float32)
    accumulated = np.zeros(shape, dtype=np.float32)
    peak_error = np.zeros(shape, dtype=np.float32)
    squared_error = np.zeros(shape, dtype=np.float32)
    goals_ahead = np.array([3., 7., 13.], dtype=np.float32)
    window = np.arange(-2, 3)
    mean, scale = normalization
    w1, b1 = hidden_weights

    def project(cx, cy):
        # Fewer than 0.6 m of forward travel per step, with enough adjacent
        # segments to track both backward motion and corner-cutting.
        nearby = np.clip(stations[:, :, None] + window, 0, paths.shape[1] - 2)
        points = paths[cases[:, None], nearby]
        dirs = direction[cases[:, None], nearby]
        dx = cx[:, :, None] - points[:, :, :, 0]
        dy = cy[:, :, None] - points[:, :, :, 1]
        length = np.linalg.norm(segments[cases[:, None], nearby], axis=-1)
        along = np.clip(dx*dirs[:, :, :, 0] + dy*dirs[:, :, :, 1], 0., length)
        residual = (dx - along*dirs[:, :, :, 0])**2 + (dy - along*dirs[:, :, :, 1])**2
        choice = np.argmin(residual, axis=-1)[:, :, None]
        idx = np.take_along_axis(nearby, choice, axis=-1)[:, :, 0]
        fraction = np.take_along_axis(along / np.maximum(length, 1.e-5), choice, axis=-1)[:, :, 0]
        return idx, idx+fraction, np.sqrt(np.min(residual, axis=-1))

    for _ in range(STEPS):
        c, s = np.cos(heading), np.sin(heading)
        cx, cy = rear_x + WHEELBASE/2*c, rear_y + WHEELBASE/2*s
        stations, progress, error = project(cx, cy)
        peak_error = np.maximum(peak_error, error)
        squared_error += error**2
        target_s = np.minimum(progress[:, :, None] + goals_ahead, paths.shape[1] - 1.0001)
        idx = target_s.astype(np.intp)
        goal = paths[cases[:, None], idx] + segments[cases[:, None], idx] * (target_s - idx)[:, :, :, None]
        dx, dy = goal[:, :, :, 0] - cx[:, :, None], goal[:, :, :, 1] - cy[:, :, None]
        raw = np.empty((*shape, 7), dtype=np.float32)
        raw[:, :, 0] = speed
        raw[:, :, 1::2] = dx*c[:, :, None] + dy*s[:, :, None]
        raw[:, :, 2::2] = -dx*s[:, :, None] + dy*c[:, :, None]
        z = np.clip((raw - mean)/scale, -8., 8.)
        h = np.tanh(np.einsum("pri,ih->prh", z, w1, optimize=False) + b1)
        action = np.clip(np.einsum("prh,ph->pr", h, candidates[:, :-1], optimize=False)
                         + candidates[:, -1, None], -1., 1.)
        accumulated += (action-previous)**2
        previous = action
        wheel_angle = np.sign(action) * np.interp(np.abs(action), NORMALIZED_STEER, TIRE_ANGLE)
        step = DT * speed
        rear_x += step*c
        rear_y += step*s
        heading += step*np.tan(wheel_angle)/WHEELBASE

    cx = rear_x + WHEELBASE/2*np.cos(heading)
    cy = rear_y + WHEELBASE/2*np.sin(heading)
    _, progress, error = project(cx, cy)
    peak_error = np.maximum(peak_error, error)
    rmse = np.sqrt((squared_error + error**2)/(STEPS+1))
    shortfall = np.maximum(0., 2. + STEPS*DT*speed - progress)
    objective = (rmse + .45*peak_error + .3*shortfall + .15*accumulated/STEPS).mean(axis=1)
    return {"objective": objective, "mean_rollout_rmse_m": rmse.mean(axis=1),
            "worst_excursion_m": peak_error.max(axis=1),
            "fraction_within_1_85m": (peak_error <= 1.85).mean(axis=1),
            "mean_progress_shortfall_m": shortfall.mean(axis=1)}


def _single(metrics, index=0):
    return {key: float(value[index]) for key, value in metrics.items()}


def refine(model, training_routes, *, evaluation_routes=(), seed=2026):
    """Search output weights using closed-loop outcomes, without a reference steering command."""
    bank = route_bank(training_routes, seed)
    w1, b1, w2, b2 = model.heads[0]
    columns = model.columns[0]
    norm = (model.mean[0, columns], model.scale[0, columns])
    original = np.concatenate((w2[:, 0], b2))
    incumbent = original.copy()
    initial_metrics = rollout(bank, (w1, b1), norm, original[None])
    best_cost = initial_metrics["objective"][0]
    mean = incumbent.copy()
    sigma = np.full_like(mean, .075)
    rng = np.random.default_rng(seed)
    for _ in range(GENERATIONS):
        # Fixed common initial conditions keep comparisons between populations
        # deterministic. Elitism keeps the unmodified imitation policy eligible.
        population = rng.normal(mean, sigma, (POPULATION, len(mean))).astype(np.float32)
        population = np.clip(population, original - .6, original + .6)
        population[0] = incumbent
        metrics = rollout(bank, (w1, b1), norm, population)
        elite = np.argsort(metrics["objective"])[:POPULATION//6]
        if metrics["objective"][elite[0]] < best_cost:
            best_cost = metrics["objective"][elite[0]]
            incumbent = population[elite[0]].copy()
        mean = .25*mean + .75*population[elite].mean(axis=0)
        sigma = np.maximum(.01, .25*sigma + .75*population[elite].std(axis=0))
    comparison = np.stack((original, incumbent))
    def compare(cases):
        metrics = rollout(cases, (w1, b1), norm, comparison)
        return {"windows": len(cases[0]), "imitation_only": _single(metrics),
                "after_search": _single(metrics, 1)}
    report = {"algorithm": "cross-entropy search over the 24 neural output weights and bias",
              "optimizer_source": CEM_SOURCE, "population": POPULATION, "generations": GENERATIONS,
              "seed": seed, "time_step_s": DT, "ticks_per_window": STEPS,
              "vehicle_model": "fixed-speed rear-axle kinematic bicycle with nominal actor midpoint",
              "nominal_wheelbase_m": WHEELBASE, "geometry_source": GEOMETRY_SOURCE,
              "steering_response_source": STEER_SOURCE, "upstream_calibration_speed_mps": 3.,
              "demonstration_route_count": len(training_routes), "objective": (
                  "mean rollout lateral RMSE + 0.45 peak error + 0.3 forward progress shortfall "
                  "+ 0.15 mean squared steering change"),
              "training_kinematic_rollouts": compare(bank)}
    if len(evaluation_routes):
        report["withheld_kinematic_rollouts"] = compare(route_bank(evaluation_routes, seed+1))
    w2[:, 0] = incumbent[:-1]
    b2[0] = incumbent[-1]
    return report
