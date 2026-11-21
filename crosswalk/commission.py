"""Check the CARLA 1.0 vehicle and demonstration scenarios without rendering cameras."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import carla

from .control import vehicle_speed
from .drive import start_demonstration, update_instructor
from .runtime import producer_info, simulator
from .world import Scenario


def xyz(point):
    return [point.x, point.y, point.z]


def inspect_vehicle(ego):
    physics = ego.get_physics_control()
    return dict(blueprint=ego.type_id, mass_kg=physics.mass, bounds_half_extents_m=xyz(ego.bounding_box.extent),
                steering_curve=[[p.x,p.y] for p in physics.steering_curve],
                wheels=[dict(max_steer_deg=w.max_steer_angle, location_raw=xyz(w.location),
                             radius_raw=w.wheel_radius, grip=w.friction_force_multiplier) for w in physics.wheels])


def check_scene(world, teacher, kind, seed):
    with ExitStack() as cleanup:
        teacher.set_random_device_seed(seed)
        scene = Scenario(world, kind, seed, district="train", perturb=True,
                         destination="market" if kind=="trip" else None)
        cleanup.callback(scene.close)
        collisions = []
        collision = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),
                                      carla.Transform(), attach_to=scene.ego)
        cleanup.callback(collision.destroy)
        collision.listen(lambda e: collisions.append((e.frame,e.other_actor.type_id)))
        cleanup.callback(collision.stop)
        for _ in range(8):
            setup_frame = world.tick()
        vehicle = inspect_vehicle(scene.ego)
        cleanup.callback(scene.ego.set_autopilot, False, teacher.get_port())
        start_demonstration(scene, teacher)
        minimum_gap = 60.
        stopped = False
        red_light_associated = False
        crossed = 0
        outcome = "timeout"
        for tick in range(1000 if kind == "trip" else 360):
            scene.before_tick()
            update_instructor(scene, teacher)
            frame = world.tick()
            state = scene.metrics()
            minimum_gap = min(minimum_gap, state["hazard_gap_m"])
            speed = vehicle_speed(scene.ego)
            stopped |= speed < .35 and state["hazard_gap_m"] < 12. and state["progress_m"] > 2.
            if kind == "signal" and state["phase"] == "red signal" and scene.ego.is_at_traffic_light():
                affecting = scene.ego.get_traffic_light()
                red_light_associated |= bool(affecting and affecting.id == scene.signal.id)
            crossed = max(crossed, state["red_light_violations"])
            if any(setup_frame < f <= frame for f, _ in collisions):
                outcome = "collision"
                break
            if crossed or state["route_departure"] or state["progress_m"] >= state["route_length_m"]-.6:
                outcome = "red_light" if crossed else ("lane_departure" if state["route_departure"] else "complete")
                break
        return vehicle, dict(scenario=kind, seed=seed, route=state["route_id"], outcome=outcome,
                             seconds=round((tick+1)*.1,1), progress_m=state["progress_m"],
                             event_triggered=state["event_triggered"], event_cleared=state["event_cleared"],
                             minimum_gap_m=minimum_gap, stopped_for_hazard=stopped, red_light_violations=crossed,
                             yield_violations=state["yield_violations"],
                             ego_reported_controlling_signal_on_red=red_light_associated if kind=="signal" else None,
                             contacts=[name for f,name in collisions if setup_frame < f <= frame])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=2468)
    parser.add_argument("--output", type=Path, default=Path("results/commissioning.json"))
    args = parser.parse_args()
    with simulator(args.port, with_manager=True, physics_only=True) as (world, teacher):
        report = dict(producer=producer_info(world), rendering_disabled=world.get_settings().no_rendering_mode,
                      subject="Built-in demonstration driver and scenario events", trials=[])
        for kind in ("signal","left","occluded","trip"):
            vehicle, trial = check_scene(world, teacher, kind, 11)
            report.update(vehicle=vehicle)
            report["trials"].append(trial)
            print(kind, trial["outcome"], "stopped:", trial["stopped_for_hazard"], flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.output.with_suffix(".partial")
            tmp.write_text(json.dumps(report,indent=2)+"\n", encoding="utf-8")
            tmp.replace(args.output)


if __name__ == "__main__":
    main()
