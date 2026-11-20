"""Drive an episode, stream synchronized cameras and optionally record a replay."""
import argparse
from contextlib import ExitStack
import json
import math
from time import perf_counter, sleep

import cv2
import numpy as np

from .control import command, vehicle_speed
from .policy import Model, expert_speed, features, rule_speed
from .runtime import simulator
from .sensors import Rig
from .vision import Detector
from .world import Scenario

DT = 0.1


def _sensor_plot(lidar, radar):
    # Drop road plane from display only. Render a bounded sample around the car.
    returns = lidar[(lidar[:, 2] > -2.05) & (lidar[:, 2] < 1.5), :2]
    points = returns[::max(1, math.ceil(len(returns) / 600))]
    v, az, alt, depth = radar[::max(1, math.ceil(len(radar) / 180))].T
    xyv = np.column_stack([2 + depth*np.cos(alt)*np.cos(az), depth*np.cos(alt)*np.sin(az), v])
    return points.round(2).tolist(), xyv.round(2).tolist()


def episode(world, kind, seed, policy, model=None, *, labels=False, live=None,
            recording=None, realtime=False, max_steps=340):
    """Apply t's observation to t+1. Oracle labels and metrics are isolated from inference."""
    cv2.setNumThreads(1)
    data, targets = [], []
    trace = []
    outcome = "timeout"
    with ExitStack() as stack:
        scene = Scenario(world, kind, seed)
        stack.callback(scene.close)
        visuals = live is not None or recording is not None
        rig = Rig(world, scene.ego, visuals=visuals)
        stack.callback(rig.close)
        # Settle the suspension and prime the GPU cameras while all actors wait.
        for _ in range(8):
            rig.read(world.tick())
        detector = Detector() if policy != "rules" or visuals else None
        start = perf_counter()
        for step in range(max_steps):
            scene.before_tick()
            frame = world.tick()
            obs = rig.read(frame)
            speed = vehicle_speed(scene.ego)
            detections = detector.detect(obs["front"]) if detector else []
            x = features(obs["front"], obs["lidar"], obs["radar"], speed, detections)
            if policy == "expert":
                target = expert_speed(scene.expert_gap())
            elif policy == "rules":
                target = rule_speed(x)
            else:
                target = model.predict(x)
            scene.ego.apply_control(command(scene.ego, scene.route, speed, target))
            # Oracle access below this line serves only as ground truth for scoring/collection.
            gap = scene.expert_gap()
            if labels:
                data.append(x)
                targets.append(expert_speed(gap))
            state = scene.metrics()
            contacts = rig.collisions()
            row = {
                "frame": int(frame), "sim_time": round(step * DT, 2), "scenario": kind,
                "seed": seed, "policy": policy, "map": "Town01", "weather": scene.weather,
                "phase": state["phase"], "speed_mps": round(speed, 3), "target_mps": round(target, 3),
                "progress_m": round(state["progress_m"], 3), "route_length_m": round(state["route_length_m"], 3),
                "lat_error_m": round(state["lateral_error_m"], 3), "collision_count": len(contacts),
                "reference_gap_m": round(gap, 3), "expert_target_mps": round(expert_speed(gap), 3),
            }
            trace.append(row)
            if visuals:
                lp, rp = _sensor_plot(obs["lidar"], obs["radar"])
                telemetry = dict(row, lidar=lp, radar=rp, detections=detections)
                if recording:
                    recording.append(telemetry, obs)
                if live:
                    live.publish(telemetry, obs)
            if contacts:
                outcome = "collision"
                break
            if state["lateral_error_m"] > 1.65:
                outcome = "lane_departure"
                break
            if state["progress_m"] >= state["route_length_m"] - 0.6:
                outcome = "complete"
                break
            if realtime:
                sleep(max(0, start + (step + 1)*DT - perf_counter()))
        # Recheck the asynchronous event stream after processing the last camera frame.
        contacts = rig.collisions()
        if contacts:
            outcome = "collision"
        summary = dict(scenario=kind, seed=seed, policy=policy, weather=scene.weather, road_id=scene.road_id,
                       lane_id=scene.lane_id, start_xy=[round(scene.route[0].x, 2), round(scene.route[0].y, 2)],
                       outcome=outcome, frames=len(trace), duration_s=round(len(trace)*DT, 2),
                       progress_m=row["progress_m"], route_length_m=row["route_length_m"],
                       contacts=sorted({actor for _, actor in contacts}), collision_events=len(contacts),
                       max_lateral_error_m=max(r["lat_error_m"] for r in trace),
                       min_hazard_gap_m=min(r["reference_gap_m"] for r in trace),
                       moving_within_1m_frames=sum(r["reference_gap_m"] < 1.0 and r["speed_mps"] > 0.5
                                                   for r in trace),
                       teacher_speed_mae_mps=round(float(np.mean([abs(r["target_mps"] - r["expert_target_mps"])
                                                                   for r in trace])), 4),
                       wall_seconds=round(perf_counter()-start, 3))
        return summary, trace, np.asarray(data, np.float32), np.asarray(targets, np.float32)


def main():
    from .viewio import Live, Recording
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["pedestrian", "lead", "braking", "clear"], default="pedestrian")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--policy", choices=["fusion", "camera", "rules", "expert"], default="fusion")
    parser.add_argument("--model", help="Override the default models/<policy>.npz")
    parser.add_argument("--port", type=int, default=2468)
    parser.add_argument("--attach", action="store_true", help="Use an existing, idle Town01 server")
    parser.add_argument("--serve", action="store_true", help="Watch the run at http://127.0.0.1:8765")
    parser.add_argument("--open-browser", action="store_true", help="Open the local live console")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--record", metavar="DIRECTORY")
    parser.add_argument("--loop", action="store_true", help="Repeat the live scenario until Ctrl+C")
    args = parser.parse_args()
    if args.open_browser and not args.serve:
        parser.error("--open-browser requires --serve")
    if args.loop and (not args.serve or args.record):
        parser.error("--loop requires --serve and a single-run recording requires omitting --loop")
    model = Model.load(args.model or f"models/{args.policy}.npz") if args.policy in ("fusion", "camera") else None
    with ExitStack() as stack:
        live = stack.enter_context(Live(args.http_port)) if args.serve else None
        if live:
            print(f"Live console at http://127.0.0.1:{args.http_port}", flush=True)
            if args.open_browser:
                import webbrowser
                webbrowser.open(live.url)
        world = stack.enter_context(simulator(args.port, args.attach))
        rec = stack.enter_context(Recording(args.record)) if args.record else None
        while True:
            result, *_ = episode(world, args.scenario, args.seed, args.policy, model,
                                 live=live, recording=rec, realtime=bool(live))
            print(json.dumps(result), flush=True)
            if not args.loop:
                break
            sleep(1.)


if __name__ == "__main__":
    main()
