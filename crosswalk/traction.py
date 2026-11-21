"""Measure straight-line braking with three CARLA Chaos wheel grip settings."""

import argparse
import json
import math
from pathlib import Path

import carla

from .runtime import MAP_NAME, producer_info, simulator
from .world import Scenario


SEEDS = (101, 102, 103)
FRICTION_SCALES = (1.0, 0.5, 0.25)
DT_SECONDS = 0.05
INITIALIZATION_TICKS = 10
SETTLE_TICKS = 20
ACCELERATION_TICKS = 200
BRAKING_TICKS = 200
THROTTLE = 0.65
TARGET_SPEED_MPS = 8.0
SPEED_TOLERANCE_MPS = 0.25
STOP_SPEED_MPS = 0.15
MAX_LATERAL_M = 1.0


def _point(location):
    return [location.x, location.y, location.z]


def _speed(vehicle):
    velocity = vehicle.get_velocity()
    return math.hypot(velocity.x, velocity.y)


def _coefficients(vehicle):
    return [wheel.friction_force_multiplier for wheel in vehicle.get_physics_control().wheels]


def _matches(actual, requested):
    return len(actual) == len(requested) and all(
        math.isclose(value, target, rel_tol=1e-5, abs_tol=1e-6)
        for value, target in zip(actual, requested)
    )


def braking_trial(world, seed, scale):
    scene = Scenario(world, "clear", seed)
    try:
        return _measure(world, scene, scale)
    finally:
        scene.close()


def _measure(world, scene, scale):
    ego = scene.ego
    brake = carla.VehicleControl(brake=1.0)
    ego.apply_control(brake)
    # Keep the opposing car parked throughout this controlled trial.
    scene.traffic.apply_control(brake)
    # Chaos populates the runtime wheel array after the actor finishes spawning.
    for _ in range(INITIALIZATION_TICKS):
        world.tick()

    physics = ego.get_physics_control()
    wheels = physics.wheels
    original = [wheel.friction_force_multiplier for wheel in wheels]
    if len(original) != 4 or any(not math.isfinite(value) or value <= 0 for value in original):
        raise RuntimeError(f"Unexpected stock wheel grip multipliers for {ego.type_id}: {original}")
    requested = [value * scale for value in original]
    for wheel, value in zip(wheels, requested):
        wheel.friction_force_multiplier = value
    # CARLA returns a copy of the wheel list. Reassign it to retain the other settings.
    physics.wheels = wheels
    ego.apply_physics_control(physics)
    for _ in range(SETTLE_TICKS):
        world.tick()

    trial = dict(
        seed=scene.seed, friction_scale=scale, vehicle=ego.type_id, weather=scene.weather,
        road_id=scene.road_id, lane_id=scene.lane_id,
        route_start_xyz_m=_point(scene.route[0]), route_end_xyz_m=_point(scene.route[-1]),
        original_tire_friction=original, requested_tire_friction=requested,
        applied_tire_friction=_coefficients(ego), final_tire_friction=None,
        initial_speed_mps=None, final_speed_mps=None, braking_start_progress_m=None,
        observed_braking_distance_m=None, observed_braking_duration_s=None,
        stopping_distance_m=None, stopping_duration_s=None, max_lateral_offset_m=None,
    )
    if not _matches(trial["applied_tire_friction"], requested):
        return dict(trial, outcome="friction_readback_mismatch")

    ego.apply_control(carla.VehicleControl(throttle=THROTTLE))
    for _ in range(ACCELERATION_TICKS):
        world.tick()
        speed = _speed(ego)
        progress, lateral = scene._coordinates(ego.get_location())
        if progress >= scene.length or abs(lateral) > MAX_LATERAL_M:
            return dict(trial, outcome="left_route_during_acceleration")
        if speed >= TARGET_SPEED_MPS:
            break
    else:
        return dict(trial, outcome="target_speed_not_reached")

    trial.update(initial_speed_mps=speed, braking_start_progress_m=progress,
                 applied_tire_friction=_coefficients(ego))
    if speed > TARGET_SPEED_MPS + SPEED_TOLERANCE_MPS:
        return dict(trial, outcome="target_speed_overshoot")
    if not _matches(trial["applied_tire_friction"], requested):
        return dict(trial, outcome="friction_readback_mismatch")

    start = progress
    max_lateral = abs(lateral)
    outcome = "braking_time_limit"
    ego.apply_control(brake)
    for ticks in range(1, BRAKING_TICKS + 1):
        world.tick()
        speed = _speed(ego)
        progress, lateral = scene._coordinates(ego.get_location())
        max_lateral = max(max_lateral, abs(lateral))
        if progress >= scene.length or max_lateral > MAX_LATERAL_M:
            outcome = "left_route_during_braking"
            break
        if speed <= STOP_SPEED_MPS:
            outcome = "stopped"
            break

    distance, duration = progress - start, ticks * DT_SECONDS
    trial.update(final_speed_mps=speed, observed_braking_distance_m=distance,
                 observed_braking_duration_s=duration, max_lateral_offset_m=max_lateral,
                 final_tire_friction=_coefficients(ego))
    if not _matches(trial["final_tire_friction"], requested):
        outcome = "friction_readback_mismatch"
    if outcome == "stopped":
        trial.update(stopping_distance_m=distance, stopping_duration_s=duration)
    return dict(trial, outcome=outcome)


def study(world):
    producer = producer_info(world)
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = DT_SECONDS
    trials = []
    pairing_fields = ("road_id", "lane_id", "route_start_xyz_m", "route_end_xyz_m", "weather",
                      "vehicle", "original_tire_friction")
    try:
        world.apply_settings(settings)
        for seed in SEEDS:
            stock = None
            for scale in FRICTION_SCALES:
                trial = braking_trial(world, seed, scale)
                if stock is None:
                    stock = trial
                trial["matches_stock_setup"] = all(trial[key] == stock[key] for key in pairing_fields)
                trials.append(trial)
                print(f"Traction {seed} {scale:g}x {trial['outcome']} "
                      f"{trial['stopping_distance_m']} m", flush=True)
    finally:
        world.apply_settings(original_settings)

    return dict(
        simulator=f"CARLA {producer['server_version']}", map=producer["map"], producer=producer,
        engine="Unreal Engine 5.8", physics_engine="Chaos Vehicles",
        friction_parameter="WheelPhysicsControl.friction_force_multiplier",
        timestep_s=DT_SECONDS, seeds=SEEDS, friction_scales=FRICTION_SCALES,
        acceleration_throttle=THROTTLE, brake_command=1.0, steer_command=0.0,
        target_speed_mps=TARGET_SPEED_MPS, start_speed_tolerance_mps=SPEED_TOLERANCE_MPS,
        stop_speed_mps=STOP_SPEED_MPS, max_lateral_offset_m=MAX_LATERAL_M,
        acceleration_limit_s=ACCELERATION_TICKS * DT_SECONDS,
        braking_limit_s=BRAKING_TICKS * DT_SECONDS, settle_duration_s=SETTLE_TICKS * DT_SECONDS,
        initialization_duration_s=INITIALIZATION_TICKS * DT_SECONDS,
        distance_definition="Horizontal distance projected along the route from brake command to first near-stop sample.",
        comparison="Compare stopped trials with matching stock settings within each seed. All three trials apply wheel physics once at rest, including the stock grip multipliers at 1x.",
        limitations=[
            "The tire friction values are Chaos wheel grip multipliers and have no calibration to pavement grip or weather effects.",
            "Throttle ends at the first sample at or above the target speed, so actual brake-entry speeds and positions can vary slightly.",
            "The experiment samples the near-stop threshold at whole ticks and uses one vehicle model on straight routes.",
        ],
        trials=trials,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=2468)
    parser.add_argument("--attach", action="store_true", help=f"Use an idle local {MAP_NAME} server.")
    parser.add_argument("--output", type=Path, default=Path("results/traction.json"))
    args = parser.parse_args()
    with simulator(args.port, args.attach) as world:
        result = study(world)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + ".partial")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(args.output)


if __name__ == "__main__":
    main()
