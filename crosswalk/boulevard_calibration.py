"""Measure launched-to-rest CARLA vehicle response on the Town03 boulevard.

The route and reference boxes are fixture/evaluation data. Training uses actual
longitudinal speed and travel to a full stop, including the five-tick delay.
"""
import argparse
import math
from pathlib import Path

import carla
import numpy as np

from .control import Navigation
from .experiment import fingerprint, write_json, write_trace
from .runtime import producer_info, simulator
from .traction import _coefficients, _matches, _speed
from .world import _raised


DT = .1
DELAY_STEPS = 5
BRAKE = .45
TRIALS = (
    ("fit-40", 40, 8, 1., BRAKE, "fit"),
    ("fit-50", 50, 8, 1., BRAKE, "fit"),
    ("fit-60", 60, 8, 1., BRAKE, "fit"),
    ("held-55", 55, 3, 1., BRAKE, "held_out"),
    ("stress-55", 55, 3, .5, BRAKE, "grip_stress"),
    ("sign-50", 50, 3, 1., .26, "service_brake"),
    ("sign-60", 60, 3, 1., .26, "service_brake"),
)


def trial(world, name, target, start, scale, brake, split):
    road = world.get_map()
    points = [road.get_waypoint_xodr(67, -2, float(s)) for s in range(start, 250, 2)]
    if any(p is None or p.is_junction or p.lane_type != carla.LaneType.Driving for p in points):
        raise ValueError("Expected continuous boulevard")
    nav = Navigation([p.transform.location for p in points])
    bp = world.get_blueprint_library().find("vehicle.lincoln.mkz")
    ego = world.spawn_actor(bp, _raised(points[0].transform, .35))
    rows = []
    try:
        ego.apply_control(carla.VehicleControl(brake=1.))
        for _ in range(12):
            world.tick()
        physics = ego.get_physics_control()
        wheels = physics.wheels
        grip = [w.friction_force_multiplier * scale for w in wheels]
        for w, value in zip(wheels, grip):
            w.friction_force_multiplier = value
        physics.wheels = wheels
        ego.apply_physics_control(physics)
        for _ in range(12):
            world.tick()
        if len(grip) != 4 or not _matches(_coefficients(ego), grip):
            raise RuntimeError("Wheel grip readback mismatch")
        phase = "launch"
        cue = None
        launch_start = None
        max_side = 0.
        for tick in range(350):
            target_xy = nav.read(ego, (7.,))
            steer = float(np.clip(1.3 * math.atan2(target_xy[1], target_xy[0]), -.35, .35))
            if launch_start is None:
                launch_start = nav.progress
            pedal = .68 if phase == "launch" else 0.
            applied_brake = brake if phase == "brake" else 0.
            ego.apply_control(carla.VehicleControl(throttle=pedal, steer=steer, brake=applied_brake))
            frame = world.tick()
            nav.read(ego)
            speed = _speed(ego)
            loc = ego.get_location()
            wp = road.get_waypoint(loc)
            offset = loc - wp.transform.location
            right = wp.transform.get_right_vector()
            side = abs(offset.x * right.x + offset.y * right.y)
            max_side = max(side, max_side)
            if wp.road_id != 67 or wp.lane_id != -2 or side > .85:
                raise RuntimeError(f"Calibration left the lane at {nav.progress} m")
            rows.append(dict(frame=frame, time_s=round((tick+1)*DT, 2), phase=phase,
                             speed_mps=speed, progress_m=nav.progress, lateral_m=side,
                             throttle=pedal, brake=applied_brake, steer=steer))
            if phase == "launch" and speed * 3.6 >= target:
                cue = rows[-1]
                phase = "response_delay"
                if ego.get_speed_limit()+.1 < speed*3.6:
                    raise RuntimeError(f"Launch outside higher posted zone: {ego.get_speed_limit()}")
                continue
            if phase == "response_delay" and frame-cue["frame"] >= DELAY_STEPS:
                phase = "brake"
            if phase == "brake" and speed <= .05 and applied_brake > 0.:
                final = rows[-1]
                break
        else:
            raise RuntimeError(f"{name} did not stop")
        if not _matches(_coefficients(ego), grip):
            raise RuntimeError("Final grip mismatch")
        return dict(name=name, split=split, road=67, lane=-2, start_station_m=start,
                    grip_scale=scale, grip_readback=grip, brake=brake, target_kmh=target,
                    cue_speed_mps=cue["speed_mps"], launch_time_s=cue["time_s"],
                    launch_distance_m=cue["progress_m"]-launch_start, cue_frame=cue["frame"],
                    final_speed_mps=final["speed_mps"], stop_time_s=final["time_s"]-cue["time_s"],
                    max_lateral_error_m=max_side,
                    stopping_distance_m=final["progress_m"]-cue["progress_m"]), rows
    finally:
        ego.destroy()
        world.tick()


def fit(trials):
    train = [r for r in trials if r["split"] == "fit"]
    v = np.array([r["cue_speed_mps"] for r in train])
    distances = np.array([r["stopping_distance_m"] for r in train])
    # Delay is a measured experimental control. Fit the remaining v-squared
    # term only, without examining the held-out speed/location or reduced grip.
    delay = DT*DELAY_STEPS
    q = float((v**2 @ (distances - delay*v)) / (v**2 @ v**2))
    # Reserve at least another full exposure at the fastest training speed,
    # plus the largest positive fitting residual, before any body standoff.
    guard = float(max(0., np.max(distances-delay*v-q*v**2)) + DT*max(v))
    for r in trials:
        speed = r["cue_speed_mps"]
        predicted = delay*speed + q*speed**2
        if r["brake"] == BRAKE:
            r.update(prediction_m=predicted, actual_minus_prediction_m=r["stopping_distance_m"]-predicted,
                     guarded_margin_m=predicted+guard-r["stopping_distance_m"])
    return dict(delay_s=delay, quadratic_s2_per_m=q, guard_m=guard, max_fit_speed_mps=float(max(v)),
                fit_targets_kmh=[r["target_kmh"] for r in train], trials=trials)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=3718)
    p.add_argument("--attach", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with simulator(args.port, args.attach, "High", physics_only=not args.attach, map_name="Town03_Opt") as world:
        if world.get_actors().filter("vehicle.*") or world.get_actors().filter("walker.*"):
            raise RuntimeError("Calibration requires an empty host")
        original = world.get_weather()
        try:
            world.set_weather(carla.WeatherParameters.ClearNoon)
            results = []
            for case in TRIALS:
                result, rows = trial(world, *case)
                path = args.output / (case[0] + ".jsonl.gz")
                write_trace(path, rows)
                result.update(trace=path.name, trace_sha256=fingerprint(path))
                results.append(result)
                print(result, flush=True)
            report = fit(results)
            write_json(args.output/"stopping-envelope.json", dict(**report, producer=producer_info(world),
                       vehicle="vehicle.lincoln.mkz", weather="ClearNoon", dt_s=DT,
                       source_sha256=fingerprint(Path(__file__)), stop_threshold_mps=.05))
        finally:
            world.set_weather(original)


if __name__ == "__main__":
    main()
