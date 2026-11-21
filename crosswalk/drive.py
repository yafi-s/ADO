"""Drive an episode, stream synchronized cameras and optionally record a replay."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter, sleep

import cv2
import carla
import numpy as np

from .control import Navigation, vehicle_speed
from .policy import Model, features
from .runtime import simulator
from .sensors import Rig
from .vision import Detector
from .world import Scenario

DT = 0.1
SCENARIOS = {
    "trip": "City trip", "pedestrian": "Pedestrian crossing", "lead": "Stopped vehicle",
    "occluded": "Pedestrian behind a van", "signal": "Red light", "left": "Unprotected left turn",
    "braking": "Braking traffic", "turn": "Intersection turn", "clear": "Clear street",
}
DESTINATIONS = {"market": "Market", "station": "Station"}
WEATHER = {"seeded": "From seed", "ClearNoon": "Clear noon", "WetNoon": "Wet noon",
           "CloudySunset": "Cloudy sunset", "ClearNight": "Clear night"}
GRIP = (0.5, 0.75, 1.0)


def _sensor_plot(lidar, radar):
    # Drop road plane from display only. Render a bounded sample around the car.
    returns = lidar[(lidar[:, 2] > -2.05) & (lidar[:, 2] < 1.5), :2]
    points = returns[::max(1, math.ceil(len(returns) / 300))]
    v, az, alt, depth = radar[::max(1, math.ceil(len(radar) / 85))].T
    xyv = np.column_stack([2 + depth*np.cos(alt)*np.cos(az), depth*np.cos(alt)*np.sin(az), v])
    return points.astype(float).round(1).tolist(), xyv.astype(float).round(1).tolist()


def _xy(location):
    return [round(location.x, 2), round(location.y, 2)]


def _observer_map(scene, path):
    pose = scene.ego.get_transform()
    hazard = scene.hazard
    if hasattr(scene, "observer_positions"):
        others = scene.observer_positions()
        hazard_type = next((a["type"] for a in others if hazard and a["actor_id"] == hazard.id), None)
    else:
        others = scene.ambient.observer_positions() if scene.ambient else []
        hazard_type = ("pedestrian" if hazard.type_id.startswith("walker.") else "vehicle") if hazard else None
    return {
        "path": path, "ego": [*_xy(pose.location), round(pose.rotation.yaw, 2)], "ego_z": round(pose.location.z,2),
        "others": others,
        "traffic": _xy(scene.traffic.get_location()) if scene.traffic else None,
        "hazard": _xy(hazard.get_location()) if hazard else None,
        "hazard_type": hazard_type,
    }


def start_demonstration(scene, teacher):
    ego = scene.ego
    ego.set_autopilot(True, teacher.get_port())
    teacher.auto_lane_change(ego, False)
    teacher.set_desired_speed(ego, 21.6)
    teacher.distance_to_leading_vehicle(ego, 3.)
    if scene.kind in ("turn", "trip", "left"):
        # These scenarios stage every approach with a green signal.
        teacher.ignore_lights_percentage(ego, 100)
    teacher.set_path(ego, [*scene.route[5:-1:7], scene.route[-1]])


def update_instructor(scene, teacher):
    speed_limit = 6.
    if scene.kind in ("signal", "left", "journey"):
        # Only the demonstration instructor can consult a stop line. Traffic
        # Manager still selects the actual actuators that become training labels.
        available = max(0., scene.expert_gap() - 2.8)
        speed_limit = min(speed_limit, math.sqrt(4.5 * available))
    if scene.kind in ("turn", "trip", "left", "journey"):
        progress, _ = scene._coordinates(scene.ego.get_location())
        near_junction = any(entry - 13. <= progress <= exit_ + 7.
                            for entry, exit_ in scene._junction_ranges)
        if near_junction:
            speed_limit = min(speed_limit, 3.7)
    teacher.set_desired_speed(scene.ego, 3.6 * speed_limit)
    return speed_limit


def episode(world, kind, seed, policy, model=None, *, teacher=None, live=None,
            recording=None, realtime=False, max_steps=None, district=None, crossing_delay=0.,
            destination=None, run_id=None, change_request=None, weather="seeded", grip=1.0,
            range_only=False, scene_factory=Scenario, observation_factory=None, surround=False, tactical=False,
            evidence_dir=None):
    """Apply t's observation to t+1. Oracle labels and metrics are isolated from inference."""
    cv2.setNumThreads(2)
    data, targets = [], []
    trace = []
    outcome = "timeout"
    if max_steps is None:
        max_steps = 1000 if kind == "trip" else 360
    if range_only and (live is not None or recording is not None):
        raise ValueError("A range-only development run has no cameras to display or export.")
    with ExitStack() as stack:
        if teacher is not None:
            teacher.set_random_device_seed(seed)
        scene = scene_factory(world, kind, seed, perturb=True, district=district,
                         crossing_delay=crossing_delay, destination=destination)
        stack.callback(scene.close)
        if weather != "seeded":
            if weather not in WEATHER:
                raise ValueError("Unknown weather preset")
            scene.weather = weather
            world.set_weather(getattr(carla.WeatherParameters, weather))
        map_name = world.get_map().name.rsplit("/", 1)[-1]
        if not range_only:
            from .robotaxi import BodyKit
            body = BodyKit(world, scene.ego, surround=True) if surround else BodyKit(world, scene.ego)
            stack.callback(body.close)
        visuals = live is not None or recording is not None
        display_path = [_xy(p) for p in scene.route[:-1:4]] + [_xy(scene.route[-1])] if visuals else []
        surround_perception = surround_detector = signals = tactical_pass = None
        if surround:
            if range_only:
                raise ValueError("Surround tracking needs RGB views")
            from .surround import SurroundPerception
            from .camera_geometry import CAMERAS
            from .traffic_cones import detect_views as detect_cones_in_views
            from .surround_vision import SurroundDetector, horizon_region
            from .signal_observer import SignalObserver
            surround_perception = SurroundPerception(evidence_dir=evidence_dir)
            surround_detector = SurroundDetector()
            signals = SignalObserver(world) if visuals else None
            range_tape = None
            if target := os.environ.get("CROSSWALK_RANGE_TAPE"):
                from .radar_geometry import RADARS
                Path(target).mkdir(parents=True, exist_ok=True)
                range_tape = Path(mkdtemp(prefix="sweeps-", dir=target))
        elif visuals:
            from .tracking import Tracker
            spatial_tracker = Tracker()
        pace = getattr(model, "policy", model)
        if tactical:
            from .overtake import TacticalPass
            tactical_pass = (TacticalPass.from_scene_map(scene, measured_hold=True) if hasattr(pace, "tactical_speed_limit")
                             else TacticalPass.from_scene_map(scene))
        include_aux = live is not None or recording is not None and not getattr(recording, "primary_rig_only", False)
        rig_options = {"surround": True} if surround else {}
        rig = Rig(world, scene.ego, visuals=include_aux, range_only=range_only, **rig_options)
        stack.callback(rig.close)
        # Settle the suspension and prime the GPU cameras while all actors wait.
        for _ in range(8):
            warmup_frame = world.tick()
            rig.read(warmup_frame)
        if grip != 1.0:
            if grip not in GRIP:
                raise ValueError("Unknown wheel-grip factor")
            physics = scene.ego.get_physics_control()
            wheels = physics.wheels
            if len(wheels) != 4:
                raise RuntimeError("The sedan has no initialized four-wheel configuration.")
            requested = [w.friction_force_multiplier * grip for w in wheels]
            for wheel, value in zip(wheels, requested):
                wheel.friction_force_multiplier = value
            physics.wheels = wheels
            scene.ego.apply_physics_control(physics)
            warmup_frame = world.tick()
            rig.read(warmup_frame)
            measured = [w.friction_force_multiplier for w in scene.ego.get_physics_control().wheels]
            if not np.allclose(requested, measured):
                raise RuntimeError("CARLA did not apply the requested wheel grip.")
        detector = None if range_only else Detector()
        navigation = Navigation(scene.route)
        signed_limits = None
        if getattr(pace, "uses_signed_speed_preview", False):
            from .speed_signs import SignedRouteLimits
            nose = scene.ego.bounding_box.location.x + scene.ego.bounding_box.extent.x
            signed_limits = SignedRouteLimits.from_map(world.get_map(), scene.route, navigation.arc, nose)
        signed_approach = None
        make_observer = observation_factory or getattr(model, "new_observer", None)
        observer = make_observer() if make_observer is not None else None
        teacher_speed = None
        pending = None
        detections = []
        sensor_lamp = getattr(model, "sensor_lamp_classifier", None)
        if teacher is not None:
            stack.callback(scene.ego.set_autopilot, False, teacher.get_port())
            start_demonstration(scene, teacher)
        else:
            lamp_expert = getattr(model, "lamp_expert", None)
            route_lamps = None
            if (lamp_expert is not None or sensor_lamp is not None) and not range_only:
                from .lamp import RouteLamps
                route_lamps = RouteLamps(world, scene.route)
            if lamp_expert is not None:
                lamp_expert = lamp_expert.for_episode()
        start = perf_counter()
        best_journey_progress = 0.
        last_forward_step = 0
        for step in range(max_steps):
            scene.before_tick()
            if teacher is not None:
                teacher_speed = update_instructor(scene, teacher)
            frame = world.tick()
            obs = rig.read(frame)
            if surround_perception is not None:
                surround_pose = scene.ego.get_transform()
                ego_xy_yaw = (surround_pose.location.x, surround_pose.location.y, surround_pose.rotation.yaw)
                surround_obs = dict(obs)
                # The rider display has longer lidar reach. Preserve the controller's trained 60 m input.
                scan = surround_obs["lidar"]
                obs["lidar"] = scan[np.einsum("ij,ij->i", scan[:, :3], scan[:, :3]) <= 60**2]
                if range_tape is not None:
                    optical = surround_obs.get("front_optical")
                    image = {}
                    if optical is not None and step % 2 == 0:
                        x0,y0,x1,y1 = horizon_region(optical)
                        image = dict(horizon_rgb=optical[y0:y1,x0:x1], horizon_xyxy=[x0,y0,x1,y1],
                                     front_size_wh=[optical.shape[1],optical.shape[0]])
                    np.savez(range_tape / f"{frame}.npz", lidar=scan, ego_xy_yaw=ego_xy_yaw,
                             **{name: surround_obs[name] for name in RADARS}, **image)
            speed = vehicle_speed(scene.ego)
            if observer is None:
                local_route = navigation.read(scene.ego)
            else:
                points = navigation.read(scene.ego, (3., 7., 13., 18., 25., 32., 40.))
                if tactical_pass is not None:
                    points = tactical_pass.preview(scene.ego, obs["lidar"], points, speed)
                    if getattr(tactical_pass, "measured_hold", False):
                        pace.tactical_speed_limit = tactical_pass.speed_limit
                local_route = points[:6]
            if range_only:
                from .policy import range_features
                x = range_features(obs["lidar"], obs["radar"], speed, local_route)
            else:
                if step % 2 == 0:
                    detections = detector.detect(obs["front"])
                if surround_perception is not None:
                    measured_front = surround_detector.detect_front(
                        obs["front"], surround_obs.get("front_optical")) if step % 2 == 0 else []
                    obs["surround_tracks"] = surround_perception.update(
                        surround_detector, surround_obs, measured_front, ego_xy_yaw,
                        step * DT, step)
                if observer is None:
                    x = features(obs["front"], obs["lidar"], obs["radar"], speed, detections,
                                 navigation=local_route, side_views=(obs["left"], obs["right"]))
                elif not hasattr(observer, "observe_and_predict"):
                    junction_gap = next((start - navigation.progress - 2.25
                                         for start, end in scene._junction_ranges if end >= navigation.progress), 60.)
                    x = observer.observe(obs, detections, local_route, speed,
                                         route_preview=points[6:], stopline_gap=max(-2., min(60., junction_gap)))
            if sensor_lamp is not None:
                signal = route_lamps.observe(scene.ego, obs)
                light = (sensor_lamp.red_probability(signal[1]), signal[0]) if signal is not None else (0., 60.)
                x = np.asarray([*x, *light], dtype=np.float32)
            if teacher is not None:
                applied = scene.ego.get_control()
                steer, throttle, brake = applied.steer, applied.throttle, applied.brake
                # The last applied control advanced the preceding observation one tick.
                if pending is not None:
                    data.append(pending)
                    targets.append((steer, throttle, brake))
                pending = x
            else:
                if getattr(model, "uses_posted_limit", False):
                    model.posted_limit_kmh = scene.ego.get_speed_limit()
                    if signed_limits is not None:
                        signed_approach = signed_limits.approach(navigation.progress, speed, model.posted_limit_kmh)
                        pace.signed_speed_limit = signed_approach[0] if signed_approach else None
                if observer is not None and hasattr(observer, "observe_and_predict"):
                    pose = scene.ego.get_transform()
                    predicted = observer.observe_and_predict(obs, detections, local_route, speed,
                        route_preview=points[6:], ego_pose=(pose.location.x, pose.location.y, pose.rotation.yaw),
                        sim_time=step*DT, detection_fresh=step % 2 == 0)
                else:
                    predicted = model.predict_range(x) if range_only else model.predict(x)
                if lamp_expert is not None and route_lamps is not None:
                    predicted = lamp_expert.apply(route_lamps.observe(scene.ego, obs), speed, predicted)
                if tactical_pass is not None:
                    predicted = tactical_pass.apply(predicted, speed)
                steer, throttle, brake = (float(v) for v in predicted)
                scene.ego.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake))
            # Actor geometry below this line is only consulted for scoring.
            state = scene.metrics()
            gap = state["hazard_gap_m"]
            received = rig.collisions()
            if any(f <= warmup_frame for f, _ in received):
                raise RuntimeError(f"Unscored spawn contact in {kind}/{seed}: {received}")
            contacts = [(f, a) for f, a in received if warmup_frame < f <= frame]
            row = {
                "frame": int(frame), "sim_time": round(step * DT, 2), "scenario": kind,
                "seed": seed, "policy": policy, "map": map_name, "weather": scene.weather, "grip":grip,
                "phase": state["phase"], "speed_mps": round(speed, 3), "steer": round(steer, 5),
                "throttle": round(throttle, 5), "brake": round(brake, 5),
                "progress_m": round(state["progress_m"], 3), "route_length_m": round(state["route_length_m"], 3),
                "lat_error_m": round(state["lateral_error_m"], 3), "collision_count": len(contacts),
                "reference_gap_m": round(gap, 3),
                "red_light_violations": state["red_light_violations"],
                "yield_violations": state["yield_violations"],
                "event_triggered": state["event_triggered"], "event_cleared": state["event_cleared"],
            }
            if trace:
                earlier = trace[max(0, len(trace)-5)]
                row["longitudinal_accel_mps2"] = round(
                    (speed-earlier["speed_mps"])/(row["sim_time"]-earlier["sim_time"]), 3)
            if getattr(model, "uses_posted_limit", False):
                row["posted_limit_kmh"] = round(float(model.posted_limit_kmh), 2)
                if signed_approach:
                    cap, upcoming, gap = signed_approach
                    row["upcoming_posted_kmh"] = upcoming
                    row["upcoming_sign_gap_m"] = round(gap, 3)
                    row["signed_approach_cap_mps"] = round(cap, 4)
                speed_request = getattr(model, "last_speed_request", None)
                if speed_request is not None:
                    row["speed_request_mps"] = round(float(speed_request), 5)
            if teacher is not None:
                row["instructor_speed_mps"] = teacher_speed
            if "events" in state:
                row["active_event"] = state["active_event"]
                row["events"] = state["events"]
            if observer is not None and hasattr(observer,"last_info"):
                row["planner"] = observer.last_info
            if tactical_pass is not None:
                row["tactical_pass"] = dict(tactical_pass.info)
            trace.append(row)
            if visuals:
                lp, rp = _sensor_plot(obs["lidar"], obs["radar"])
                pose = scene.ego.get_transform()
                measured = {}
                observer_map = _observer_map(scene, display_path)
                if surround_perception is not None:
                    tracks = obs["surround_tracks"]
                    if observer is not None:
                        # These are the exact local navigation targets handed to the model on this tick.
                        yaw = math.radians(pose.rotation.yaw)
                        rotation = np.asarray([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
                        origin = np.asarray([pose.location.x, pose.location.y])
                        selected = np.asarray(points).reshape(-1, 2) @ rotation + origin
                        measured["plan_xy"] = np.vstack((origin, selected)).round(3).tolist()
                    signal_reads = signals.read(pose, obs, step * DT)
                    surface = surround_obs["lidar"][:, :3].copy()
                    surface[:, 2] += 2.30
                    measured["cone_observations"] = detect_cones_in_views(
                        {view: surround_obs[view] for view in CAMERAS}, surface)
                    surface = surface[(surface[:, 2] > .20) & (surface[:, 2] < 5.0)]
                    sparse = surface[::max(1, math.ceil(len(surface) / 180))]
                    measured.update(perception_version="surround-v1", signals=signal_reads,
                                    barrier_review=surround_perception.barrier_review,
                                    lidar_xyz=sparse.astype(float).round(2).tolist(),
                                    corner_radar=surround_perception.radar_points,
                                    camera_detections=surround_perception.view_detections,
                                    camera_inference_age_s=surround_perception.view_ages.copy())
                    observer_map["signal_reference"] = signals.score_reference(signal_reads)
                else:
                    tracks = spatial_tracker.update(obs["lidar"], obs["radar"], detections if step % 2 == 0 else [],
                                                    (pose.location.x, pose.location.y, pose.rotation.yaw), step*DT)
                display_boxes = (surround_perception.view_detections.get("front", []) if surround_perception is not None
                                 else detections)
                telemetry = dict(row, lidar=lp, radar=rp, detections=display_boxes, detection_age_s=(step % 2)*DT,
                                 run_id=run_id, destination=state["destination"], crossing_delay=crossing_delay,
                                 observer_map=observer_map, tracks=tracks, **measured)
                if recording:
                    recording.append(telemetry, obs)
                if live:
                    live.publish(telemetry, obs)
            if contacts:
                outcome = "collision"
                break
            if state["red_light_violations"]:
                outcome = "red_light"
                break
            if state["route_departure"]:
                outcome = "lane_departure"
                break
            if state["progress_m"] >= state["route_length_m"] - 0.6:
                outcome = "complete"
                break
            if kind == "journey":
                if state["progress_m"] >= best_journey_progress + .5:
                    best_journey_progress = state["progress_m"]
                    last_forward_step = step
                if (step-last_forward_step)*DT >= 45.:
                    outcome = "timeout"
                    break
            if change_request and change_request():
                outcome = "restarted"
                break
            if realtime:
                sleep(max(0, start + (step + 1)*DT - perf_counter()))
        # Read the next synchronized frame as a delivery barrier for asynchronous
        # collision notifications. Count events only through the final scored frame.
        rig.read(world.tick())
        received = rig.collisions()
        if any(f <= warmup_frame for f, _ in received):
            raise RuntimeError(f"Late spawn contact in {kind}/{seed}: {received}")
        contacts = [(f, actor) for f, actor in received if warmup_frame < f <= frame]
        if contacts:
            outcome = "collision"
            by_frame = Counter(f for f, _ in contacts)
            cumulative = 0
            for record in trace:
                cumulative += by_frame[record["frame"]]
                record["collision_count"] = cumulative
        summary = dict(scenario=kind, seed=seed, policy=policy, weather=scene.weather, road_id=scene.road_id,
                       lane_id=scene.lane_id, start_xy=[round(scene.route[0].x, 2), round(scene.route[0].y, 2)],
                       outcome=outcome, frames=len(trace), duration_s=round(len(trace)*DT, 2),
                       progress_m=row["progress_m"], route_length_m=row["route_length_m"],
                       contacts=sorted({actor for _, actor in contacts}), collision_events=len(contacts),
                       max_lateral_error_m=max(r["lat_error_m"] for r in trace),
                       spawn_lateral_m=scene.spawn_lateral_m, spawn_yaw_deg=scene.spawn_yaw_deg,
                       min_hazard_gap_m=min(r["reference_gap_m"] for r in trace),
                       moving_within_1m_frames=sum(r["reference_gap_m"] < 1.0 and r["speed_mps"] > 0.5
                                                   for r in trace),
                       turn_direction=getattr(scene, "turn_direction", None),
                       district=district, destination=state["destination"], crossing_delay_s=crossing_delay,
                       camera_rendering=rig.rendering, ego_blueprint=scene.ego.type_id,
                       background_actors=scene.ambient.counts if scene.ambient else {"walkers":0,"vehicles":0},
                       sensor_configuration=("five-camera-360-lidar-five-radar" if surround else
                                             "lidar-radar" if range_only else "three-camera-lidar-radar"),
                       route_id=state["route_id"], road_ids=state["road_ids"], junction_ids=state["junction_ids"],
                       event_triggered=state["event_triggered"], event_cleared=state["event_cleared"],
                       red_light_violations=state["red_light_violations"],
                       yield_violations=state["yield_violations"],
                       stopped_for_hazard=any(r["speed_mps"] < 0.35 and r["reference_gap_m"] < 12.
                                              and r["progress_m"] > 2. for r in trace),
                       closest_in_path_speed_mps=min(trace, key=lambda r:r["reference_gap_m"])["speed_mps"],
                       wall_seconds=round(perf_counter()-start, 3))
        if teacher is not None:
            summary["route_xy"] = [[p.x, p.y] for p in scene.route]
        if "events" in state:
            summary.update(events=state["events"], turn_counts=state["turn_counts"], district=state["district"],
                           junction_traversals=state["junction_traversals"], turns_completed=state["turns_completed"],
                           clear_road_m=state["clear_road_m"])
            for name in ("sidewalk_activity", "background_traffic_activity"):
                if name in state:
                    summary[name] = state[name]
        return summary, trace, np.asarray(data, np.float32), np.asarray(targets, np.float32)


def main():
    from .viewio import Live, Recording
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="trip")
    parser.add_argument("--destination", choices=list(DESTINATIONS), default="market")
    parser.add_argument("--district", choices=("train","heldout"), default="heldout",
                        help="Use training streets for development or held-out streets for a demonstration.")
    parser.add_argument("--crossing-delay", type=float, default=0., help="Delay the pedestrian's entry by 0–5 simulated seconds")
    parser.add_argument("--weather", choices=list(WEATHER), default="seeded")
    parser.add_argument("--grip", choices=GRIP, type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--policy", choices=["fusion", "camera", "town01-yield-v04", "town01-yield-gated-v04"],
                        default="fusion")
    parser.add_argument("--model", help="Override the selected policy's default checkpoint")
    parser.add_argument("--port", type=int, default=2468)
    parser.add_argument("--attach", action="store_true", help="Use an existing, idle Town01 server")
    parser.add_argument("--serve", action="store_true", help="Watch the run at http://127.0.0.1:8765")
    parser.add_argument("--open-browser", action="store_true", help="Open the local live console")
    parser.add_argument("--interactive", action="store_true", help="Accept new drives after a run completes")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--record", metavar="DIRECTORY")
    parser.add_argument("--loop", action="store_true", help="Repeat the live scenario until Ctrl+C")
    parser.add_argument("--tour", action="store_true", help="Cycle through pedestrian, turn, braking, lead and clear encounters")
    args = parser.parse_args()
    if not 0 <= args.crossing_delay <= 5 or not 0 <= args.seed <= 2147483647:
        parser.error("Use a seed in [0,2147483647] and a crossing delay in [0,5].")
    if args.interactive and (not args.serve or args.record):
        parser.error("Interactive drives require --serve and single-run captures require omitting --interactive.")
    if args.open_browser and not args.serve:
        parser.error("--open-browser requires --serve")
    if args.loop and (not args.serve or args.record):
        parser.error("--loop requires --serve and a single-run recording requires omitting --loop")
    if args.tour and not args.loop:
        parser.error("--tour requires --loop")
    if args.policy == "town01-yield-v04":
        from .town01_interaction import InteractionModel, MODEL
        model = InteractionModel.load(args.model or MODEL)
    elif args.policy == "town01-yield-gated-v04":
        from .town01_interaction_gate import GatedInteractionModel, MODEL
        model = GatedInteractionModel.load(args.model or MODEL)
    else:
        model = Model.load(args.model or f"models/{args.policy}.npz")
    settings = dict(scenario=args.scenario, seed=args.seed, crossing_delay=args.crossing_delay,
                    destination=args.destination, weather=args.weather, grip=args.grip)
    options = dict(scenarios=[dict(id=k, label=v) for k,v in SCENARIOS.items()],
                   destinations=[dict(id=k, label=v) for k,v in DESTINATIONS.items()], defaults=settings.copy(),
                   weather=[dict(id=k,label=v) for k,v in WEATHER.items()],
                   grip=[dict(id=g,label=f"{g:.0%}") for g in GRIP],
                   limits={"seed":{"min":0,"max":2147483647},"crossing_delay":{"min":0.,"max":5.}})
    with ExitStack() as stack:
        live = stack.enter_context(Live(args.http_port, options=options if args.interactive else None)) if args.serve else None
        if live:
            print(f"Live console at http://127.0.0.1:{args.http_port}", flush=True)
            if args.open_browser:
                import webbrowser
                webbrowser.open(live.url)
        world = stack.enter_context(simulator(args.port, args.attach))
        rec = stack.enter_context(Recording(args.record)) if args.record else None
        index = 0
        tour = tuple(SCENARIOS)
        while True:
            if args.tour:
                settings.update(scenario=tour[index % len(tour)], seed=args.seed + index // len(tour))
            pending = None
            def changed():
                nonlocal pending
                if live.stop_requested():
                    return True
                pending = live.next_request()
                return pending is not None
            if live:
                live.clear()
                live.set_status("running", run_id=index, settings=settings, outcome=None, error=None)
            try:
                result, *_ = episode(world, settings["scenario"], settings["seed"], args.policy, model,
                                     live=live, recording=rec, realtime=bool(live), district=args.district,
                                     destination=settings["destination"], crossing_delay=settings["crossing_delay"],
                                     weather=settings.get("weather","seeded"), grip=settings.get("grip",1.0),
                                     run_id=index, change_request=changed if args.interactive else None)
            except Exception as error:
                if not args.interactive:
                    raise
                live.set_status("error", run_id=index, settings=settings, error=str(error), outcome=None)
                result = {"outcome": "error", "error": str(error)}
            else:
                if live and pending is None:
                    live.set_status("complete", run_id=index, settings=settings, outcome=result, error=None)
            print(json.dumps(result), flush=True)
            if args.interactive:
                while pending is None and not live.stop_requested():
                    sleep(0.1)
                    pending = live.next_request()
                if live.stop_requested():
                    break
                settings = pending
            elif not args.loop:
                break
            index += 1
            if args.loop:
                sleep(1.)


if __name__ == "__main__":
    main()
