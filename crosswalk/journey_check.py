"""Commission scenario mechanics using CARLA Traffic Manager without scoring an ADO driver."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import carla

from .drive import start_demonstration, update_instructor
from .control import vehicle_speed
from .journey import JourneyScenario
from .runtime import SUPPORTED_MAPS, producer_info, simulator


def check(world, instructor, seed, destination, output):
    with ExitStack() as cleanup:
        instructor.set_random_device_seed(seed)
        scene = JourneyScenario(world, "journey", seed, destination=destination)
        cleanup.callback(scene.close)
        events = []
        contact = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(), attach_to=scene.ego)
        cleanup.callback(contact.destroy)
        contact.listen(lambda e: events.append((e.frame, e.other_actor.type_id)))
        cleanup.callback(contact.stop)
        for _ in range(8):
            warm = world.tick()
        start_demonstration(scene, instructor)
        cleanup.callback(scene.ego.set_autopilot,False,instructor.get_port())
        phase = ""
        outcome = "timeout"
        lane_change_requested = False
        for i in range(6500):
            scene.before_tick()
            update_instructor(scene, instructor)
            if scene._urban is not None and not lane_change_requested:
                transition = scene._station[scene._urban.lane_change.start]
                if scene._last_progress >= transition:
                    instructor.force_lane_change(scene.ego, False)
                    lane_change_requested = True
            frame = world.tick()
            state = scene.metrics()
            if phase != state["phase"] or i % 300 == 0:
                phase = state["phase"]
                print(i, round(state["progress_m"],1), phase, round(vehicle_speed(scene.ego),2), flush=True)
            if any(f > warm for f, actor in events):
                outcome = "collision"
                break
            if state["route_departure"] or state["red_light_violations"]:
                outcome = "lane_departure" if state["route_departure"] else "red_light"
                break
            if state["progress_m"] >= state["route_length_m"] - .6:
                outcome = "complete"
                break
        report = dict(subject="Native demonstration driver", producer=producer_info(world),seed=seed,
                      outcome=outcome, final_frame=frame, simulated_s=(i+1)*.1, contacts=events, **state)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
        print(outcome, state["progress_m"], "handled", sum(e["handled"] for e in state["events"]), flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=2868)
    parser.add_argument("--map", choices=SUPPORTED_MAPS, default="Town03_Opt")
    parser.add_argument("--attach",action="store_true")
    parser.add_argument("--seed",type=int,default=211)
    parser.add_argument("--destination",choices=("east","south","market"),default=None)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    with simulator(args.port,args.attach,with_manager=True, map_name=args.map) as (world,tm):
        check(world,tm,args.seed,args.destination,args.output)
