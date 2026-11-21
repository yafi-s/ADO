"""Measure stopping response at independent speeds on a non-junction Town03 development lane."""
import argparse
from pathlib import Path

import carla
import numpy as np

from .experiment import write_json
from .runtime import producer_info, simulator
from .world import _raised
from .traction import _speed


def collect(world, speed_target, start_station, split):
    # The course, requested approach speed, and response delay are experimental controls.
    # The target speed label for fitting comes from vehicle velocity at the cue.
    road = world.get_map()
    lane = road.get_waypoint_xodr(67, -2, float(start_station))
    ego = world.spawn_actor(world.get_blueprint_library().find("vehicle.lincoln.mkz"), _raised(lane.transform, .35))
    trace = []
    try:
        ego.apply_control(carla.VehicleControl(brake=1.))
        for _ in range(8):
            world.tick()
        ego.apply_control(carla.VehicleControl(throttle=.55))
        for _ in range(150):
            world.tick()
            if _speed(ego) >= speed_target:
                break
        else:
            raise RuntimeError("Approach calibration speed was not reached")
        cue = ego.get_location()
        speed = _speed(ego)
        forward = lane.transform.get_forward_vector()
        lateral = lane.transform.get_right_vector()
        for i in range(100):
            # Half a second to establish track motion, then ordinary controlled braking.
            # This does not assume the vehicle can use instantaneous emergency braking.
            brake = 0. if i < 5 else .26
            ego.apply_control(carla.VehicleControl(brake=brake))
            frame = world.tick()
            d = ego.get_location() - cue
            advance = d.x*forward.x+d.y*forward.y
            side = d.x*lateral.x+d.y*lateral.y
            v = _speed(ego)
            trace.append(dict(frame=frame, time_s=round((i+1)*.1, 2), speed_mps=round(v, 5),
                              advance_m=round(advance, 5), side_m=round(side, 5), brake=brake))
            if abs(side) >= .75:
                raise RuntimeError("Calibration left the test lane")
            if v <= .15 and i >= 5:
                break
        else:
            raise RuntimeError("Near-stop velocity was not reached")
        return dict(split=split, road=67, lane=-2, start_station_m=start_station, target_mps=speed_target,
                    cue_speed_mps=speed, stopping_distance_m=advance, trace=trace)
    finally:
        ego.destroy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=3178)
    p.add_argument("--attach", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with simulator(a.port, a.attach, "High", map_name="Town03_Opt") as world:
        trials = [collect(world, speed, start, split)
                  for start, speeds, split in ((55, (2., 4., 6., 8.), "fit"),
                                               (75, (3., 5., 7.), "held_out"))
                  for speed in speeds]
        write_json(a.output, dict(producer=producer_info(world), dt_s=.1, delay_s=.5, brake=.26,
                                  stop_threshold_mps=.15, trials=trials))
    print([(r["target_mps"], round(r["stopping_distance_m"],2)) for r in trials])


if __name__ == "__main__":
    main()
