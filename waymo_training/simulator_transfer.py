"""Private, non-commercial closed-loop CARLA 1.0 test of the Waymo vision checkpoint.

Example from the research checkout (set CARLA_ROOT when launching a server):
  python -m waymo_training.simulator_transfer --checkpoint <private.npz> \
      --output <private-new-directory> --scenario clear pedestrian lead --record
"""
import argparse
from collections import Counter, deque
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from crosswalk.control import Navigation
from crosswalk.steering_search import NORMALIZED_STEER, TIRE_ANGLE

# Axle centers measured in the actor frame on the CARLA 1.0 Lincoln MKZ.
# Rounded planar averages of the left and right wheel centers, in metres.
REAR_X = -1.39706
WHEELBASE = 1.47021 - REAR_X
DT = 0.1
ROLLING_SETUP_TICKS = 3
STEADY_SETUP_TICKS = 48
POLICY = "ADO Waymo-trained trajectory predictor (CARLA research transfer)"
CASES = ("clear", "pedestrian", "lead", "braking", "signal", "occluded", "turn", "left")


def rear_xy(x, y, yaw):
    """Planar rear axle in CARLA world XY; yaw in radians, positive right."""
    return x + REAR_X * math.cos(yaw), y + REAR_X * math.sin(yaw)


def history_input(samples, yaw):
    """Resample measured rear-axle motion into the *current* Waymo ego frame.

    Samples are (sim time, world rear X, world rear Y). Differentiate the rear
    axle itself, avoiding assumptions about the simulator's velocity reference
    point during turns. Central differences use only observations available at
    the decision time; the newest sample has one-sided derivatives.
    """
    samples = np.asarray(samples, dtype=np.float64)
    times, positions = samples[:, 0], samples[:, 1:]
    query = times[-1] + np.arange(-15, 1, dtype=float) / 4
    if (len(times) < 4 or times[0] > query[0] - DT + 1e-6
            or not np.allclose(np.diff(times), DT, rtol=0, atol=1e-5)):
        raise ValueError("Expected a complete, contiguous 10 Hz measured history")
    velocities = np.gradient(positions, times, axis=0, edge_order=2)
    accelerations = np.gradient(velocities, times, axis=0, edge_order=2)
    c, s = math.cos(yaw), math.sin(yaw)
    to_waymo = np.array(((c, s), (s, -c)))
    states = np.column_stack([(xy @ to_waymo) for xy in
                              (positions - positions[-1], velocities, accelerations)])
    return np.column_stack([np.interp(query, times, col) for col in states.T]).astype(np.float32)


def route_intent(nav, position, speed):
    """Quantize change of planned road tangent over the next 10 s of route."""
    offset = np.asarray(position) - nav.points[:-1]
    fraction = np.clip((offset * nav.delta).sum(axis=1) / nav.lengths**2, 0, 1)
    index = int(np.argmin(np.sum((offset - fraction[:, None] * nav.delta)**2, axis=1)))
    progress = nav.arc[index] + fraction[index] * nav.lengths[index]
    ahead = progress + max(10., 10. * max(0., speed))
    future = min(len(nav.delta) - 1, np.searchsorted(nav.arc, ahead, side="right") - 1)
    first, last = nav.delta[index], nav.delta[future]
    bend = math.atan2(first[0] * last[1] - first[1] * last[0], first @ last)
    if abs(bend) < math.radians(30):
        return 1  # GO_STRAIGHT
    return 2 if bend < 0 else 3  # GO_LEFT / GO_RIGHT


def track_path(path, forward_speed):
    """Pure pursuit at 1 s, with speed feedback from the 0.5-to-1 s chord."""
    p = np.asarray(path, dtype=float)
    if p.shape != (20, 2) or not np.isfinite(p).all() or not math.isfinite(forward_speed):
        raise ValueError("Expected 20 finite future XY waypoints and a finite measured speed")
    x, left = p[3]
    length_sq = x*x + left*left
    curvature = 2 * left / length_sq if length_sq >= 0.25**2 and x > 0 else 0.
    tire_angle = -math.atan(WHEELBASE * curvature)  # Positive CARLA steer turns right.
    steer = math.copysign(float(np.interp(abs(tire_angle), TIRE_ANGLE, NORMALIZED_STEER)), tire_angle)
    step = p[3] - p[1]
    target_speed = float(np.linalg.norm(step) / .5) if step[0] > 0 and x > 0 else 0.
    # Speed-error gain 1/s, nominal full-scale +2.5/-4 m/s^2, forward gear only.
    error = target_speed - forward_speed
    return dict(target_xy_m=[float(x), float(left)], curvature_left_per_m=float(curvature),
                target_speed_mps=target_speed, steer=steer,
                throttle=min(1., max(0., error / 2.5)), brake=min(1., max(0., -error / 4.)))


class CalibratedTracker:
    """Follow the predicted speed with bounded acceleration and measured feedback.

    The first pass in the same Lincoln measured about +7.5 m/s² at full
    throttle. Measured acceleration feedback compensates partial-pedal response
    and static resistance. The brake scale is nominal, because run 1 did not
    exercise the brake. The limits apply uniformly to every scenario.
    """
    def __init__(self):
        self.previous_speed = None
        self.acceleration = 0.
        self.bias = 0.

    def request(self, path, speed):
        result = track_path(path, speed)
        error = result["target_speed_mps"] - speed
        # The target is the mean speed over 0.5–1s, centered at 0.75 s.
        desired = float(np.clip(error / .75, -4., 2.))
        if self.previous_speed is not None:
            observed = (speed - self.previous_speed) / DT
            self.acceleration += .4 * (observed - self.acceleration)
        self.previous_speed = speed
        scale = 7.5 if desired >= 0 else 4.
        feedforward = desired / scale
        # Integral acceleration correction in pedal units, bounded and frozen
        # when output saturation would push the integrator farther outward.
        change = DT * (desired - self.acceleration) / (.5 * scale)
        raw = feedforward + self.bias
        if (raw < 1. or change < 0) and (raw > -1. or change > 0):
            self.bias = float(np.clip(self.bias + change, -.45, .45))
        pedal = float(np.clip(feedforward + self.bias, -1., 1.))
        # A non-forward trajectory at rest has zero propulsion even if positive
        # bias remains from an earlier low-speed setpoint.
        if result["target_speed_mps"] == 0 and speed <= .02:
            self.bias = min(0., self.bias)
            pedal = min(0., pedal)
        result.update(throttle=max(0., pedal), brake=max(0., -pedal),
                      requested_accel_mps2=desired, measured_filtered_accel_mps2=self.acceleration,
                      pedal_integral=self.bias)
        return result


def outcome(state, contacts):
    if contacts:
        return "collision"
    if state["red_light_violations"]:
        return "red_light"
    if state["yield_violations"]:
        return "yield_violation"
    if state["route_departure"]:
        return "lane_departure"
    if state["progress_m"] >= state["route_length_m"] - .6:
        return "complete"
    return None


def _sample(world, scene, frame, positions):
    snap = world.get_snapshot()
    if snap.frame != frame:
        raise RuntimeError("CARLA snapshot and camera timestamps differ")
    pose = snap.find(scene.ego.id).get_transform()
    yaw = math.radians(pose.rotation.yaw)
    x, y = rear_xy(pose.location.x, pose.location.y, yaw)
    positions.append((snap.timestamp.elapsed_seconds, x, y))
    return pose, yaw


def _write(path, value):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_episode(world, model, kind, seed, initial_speed, max_steps, dest, record,
                tracker="proportional-v1", rolling_history="impulse"):
    import carla
    from crosswalk.drive import _observer_map, _sensor_plot
    from crosswalk.runtime import producer_info
    from crosswalk.sensors import Rig
    from crosswalk.viewio import Recording
    from crosswalk.world import Scenario

    dest.mkdir(exist_ok=False)
    positions = deque(maxlen=44)
    rows = []
    servo = CalibratedTracker() if tracker == "calibrated-v2" else None
    with ExitStack() as stack:
        scene = Scenario(world, kind, seed)
        stack.callback(scene.close)
        if scene.ego.type_id != "vehicle.lincoln.mkz":
            raise ValueError("The rear-axle calibration requires CARLA 1.0 vehicle.lincoln.mkz")
        if record:
            from crosswalk.tracking import Tracker
            from crosswalk.vision import Detector
            detector, viewer_tracker = Detector(), Tracker()
            recording = stack.enter_context(Recording(dest / "replay"))
        rig = Rig(world, scene.ego, visuals=record)
        stack.callback(rig.close)
        nav = Navigation(scene.route)
        display_path = [[round(p.x, 2), round(p.y, 2)] for p in scene.route]
        # Warm up with the scenario clock paused and collect actual suspension
        # settling and stationary poses. Never fabricate a rolling prehistory.
        scene.ego.apply_control(carla.VehicleControl(brake=1.))
        for _ in range(48):
            frame = world.tick()
            obs = rig.read(frame)
            pose, yaw = _sample(world, scene, frame, positions)
        warmup = frame
        if initial_speed:
            scene.ego.apply_control(carla.VehicleControl())
            steady = rolling_history == "steady"
            if steady:
                scene.ego.enable_constant_velocity(carla.Vector3D(x=initial_speed))
            else:
                scene.ego.set_target_velocity(pose.get_forward_vector() * initial_speed)
            try:
                for _ in range(STEADY_SETUP_TICKS if steady else ROLLING_SETUP_TICKS):
                    frame = world.tick()
                    obs = rig.read(frame)
                    pose, yaw = _sample(world, scene, frame, positions)
            finally:
                if steady:
                    scene.ego.disable_constant_velocity()
            if steady:
                measured = np.diff(np.asarray(positions)[:, 1:], axis=0) / DT
                if np.max(np.abs(np.linalg.norm(measured, axis=1) - initial_speed)) > .3:
                    raise RuntimeError("Rolling initialization did not supply a stable measured speed history")
                warmup = frame  # Contacts during velocity-constrained setup invalidate the scenario.
        first_frame = frame
        labels = []
        previous = None
        for tick in range(max_steps + 1):
            hist = history_input(positions, yaw)
            speed = float(hist[-1, 2])
            intent = route_intent(nav, positions[-1][1:], speed)
            pred = model.predict([obs[k] for k in ("front", "left", "right")], hist, intent)
            requested = servo.request(pred, speed) if servo else track_path(pred, speed)
            # From here, scoring/observer data can terminate the experiment and
            # describe the encounter. They do not select any steering or pedals.
            state = scene.metrics()
            if tick == 0:
                handoff = state["progress_m"]
                if initial_speed and rolling_history == "steady":
                    passed_hazard = (scene.hazard is not None and
                                     scene._coordinates(scene.hazard.get_location())[0] <= handoff)
                    if state["route_departure"] or state["event_triggered"] or state["event_cleared"] or passed_hazard:
                        raise RuntimeError("Rolling initialization compromised the scored scenario")
            contacts = [(f, a) for f, a in rig.collisions() if warmup < f <= frame]
            result = outcome(state, contacts)
            if result is None and tick == max_steps:
                result = "timeout"
            active = result is None
            row = dict(frame=frame, sim_time=round((frame-first_frame)*DT, 2), policy=POLICY,
                       seed=seed, scenario=kind, weather=scene.weather, map=producer_info(world)["map"],
                       phase=state["phase"], intent=intent, speed_mps=speed, progress_m=state["progress_m"],
                       handoff_progress_m=handoff, model_progress_m=state["progress_m"] - handoff,
                       history_input=hist.round(6).tolist(), tracker=tracker,
                       route_length_m=state["route_length_m"], lat_error_m=state["lateral_error_m"],
                       reference_gap_m=state["hazard_gap_m"], event_triggered=state["event_triggered"],
                       event_cleared=state["event_cleared"], red_light_violations=state["red_light_violations"],
                       yield_violations=state["yield_violations"], collision_count=len(contacts),
                       rear_axle_world_xy_m=list(positions[-1][1:]), carla_yaw_deg=pose.rotation.yaw,
                       future_rear_xy_m=np.asarray(pred).round(5).tolist(), applied_for_next_tick=active,
                       previous_control=previous, **requested)
            rows.append(row)
            if record:
                fresh = tick % 2 == 0
                if fresh:
                    labels = detector.detect(obs["front"])
                lp, rp = _sensor_plot(obs["lidar"], obs["radar"])
                tracks = viewer_tracker.update(obs["lidar"], obs["radar"], labels if fresh else [],
                                               (pose.location.x, pose.location.y, pose.rotation.yaw), tick*DT)
                recording.append(dict(row, run_id=dest.name, detections=labels, detection_age_s=(tick%2)*DT,
                                      tracks=tracks, radar=rp, lidar=lp,
                                      observer_map=_observer_map(scene, display_path)), obs)
            if not active:
                break
            previous = {k: requested[k] for k in ("steer", "throttle", "brake")}
            scene.ego.apply_control(carla.VehicleControl(**previous))
            scene.before_tick()
            frame = world.tick()
            obs = rig.read(frame)
            pose, yaw = _sample(world, scene, frame, positions)
        # Sensor callback delivery barrier, excluding contacts after the last scored tick.
        rig.read(world.tick())
        events = rig.collisions()
        if any(f <= warmup for f, _ in events):
            raise RuntimeError(f"Spawn collision during unscored setup in {kind}/{seed}")
        contacts = [(f, a) for f, a in events if warmup < f <= frame]
        counts = Counter(f for f, _ in contacts)
        cumulative = sum(n for f, n in counts.items() if f < first_frame)
        late = False
        for row in rows:
            cumulative += counts[row["frame"]]
            late |= cumulative != row["collision_count"]
            row["collision_count"] = cumulative
        if contacts:
            result = "collision"
        encounter = [row for row in rows if row["reference_gap_m"] < 12.]
        summary = dict(scenario=kind, seed=seed, route_id=state["route_id"], weather=scene.weather,
                       requested_handoff_speed_mps=initial_speed, rolling_history=rolling_history if initial_speed else "stationary",
                       tracker=tracker, outcome=result, frames=len(rows), handoff_progress_m=handoff,
                       model_progress_m=rows[-1]["model_progress_m"],
                       duration_s=rows[-1]["sim_time"], progress_m=rows[-1]["progress_m"],
                       max_lateral_error_m=max(r["lat_error_m"] for r in rows),
                       route_length_m=state["route_length_m"], contacts=contacts,
                       event_triggered=state["event_triggered"], event_cleared=state["event_cleared"],
                       red_light_violations=state["red_light_violations"], yield_violations=state["yield_violations"],
                       min_reference_gap_m=min(r["reference_gap_m"] for r in rows),
                       frames_with_in_path_gap_below_12m=len(encounter),
                       min_speed_inside_12m_mps=min((r["speed_mps"] for r in encounter), default=None),
                       max_applied_brake_inside_12m=max((r["brake"] for r in encounter
                                                       if r["applied_for_next_tick"]), default=None),
                       rendering=rig.rendering, replay="replay/" if record else None)
        _write(dest / "trace.json", rows)
        _write(dest / "summary.json", summary)
    if record and late:
        path = dest / "replay" / "replay.json"
        replay = json.loads(path.read_text(encoding="utf-8"))
        for entry, row in zip(replay, rows, strict=True):
            entry["collision_count"] = row["collision_count"]
        _write(path, replay)
    return summary


def require_private(path):
    """Reject project checkouts, including a linked tree nested in the main checkout."""
    path = path.resolve()
    source = Path(__file__).resolve().parents[1]
    if any(parent == source or ((parent / ".git").exists() and (parent / "crosswalk/runtime.py").is_file())
           for parent in (path, *path.parents)):
        raise ValueError(f"Choose a private path outside all project checkouts for {path.name}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New private directory outside project checkouts")
    parser.add_argument("--scenario", choices=CASES, nargs="+", default=["clear", "pedestrian", "lead"])
    parser.add_argument("--initial-speed", type=float, nargs="+", choices=(0., 3.), default=[0., 3.])
    parser.add_argument("--tracker", choices=("proportional-v1", "calibrated-v2"), default="proportional-v1")
    parser.add_argument("--rolling-history", choices=("impulse", "steady"), default="impulse",
                        help="Steady collects 4.8 s of measured rolling history before releasing the velocity constraint")
    parser.add_argument("--max-steps", type=int, default=300, help="Scored 0.1 s physics steps per episode")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--port", type=int, default=2468)
    parser.add_argument("--attach", action="store_true", help="Use an idle CARLA 1.0 Town01_Opt server")
    parser.add_argument("--record", action="store_true", help="Save synchronized five-view local replay with tracks")
    args = parser.parse_args()
    if (args.max_steps < 1 or len(set(args.scenario)) != len(args.scenario)
            or len(set(args.initial_speed)) != len(args.initial_speed)):
        parser.error("Use a positive episode length and unique scenarios and initial speeds")
    import cv2
    from crosswalk.robotaxi import MOUNTS
    from crosswalk.runtime import producer_info, simulator
    from waymo_training.portable import TrajectoryPolicy
    cv2.setNumThreads(1)
    checkpoint = require_private(args.checkpoint)
    dest = require_private(args.output)
    model = TrajectoryPolicy.load(checkpoint)
    if model.metadata["format"] != "ado-waymo-vision-numpy-v1":
        parser.error("Expected a portable Waymo vision checkpoint")
    with checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    dest.mkdir(parents=True, exist_ok=False)
    report = dict(label="Private non-commercial simulated research transfer", policy=POLICY,
                  checkpoint_sha256=digest, checkpoint_metadata=model.metadata, seed=args.seed,
                  scenarios=args.scenario, initial_speeds_mps=args.initial_speed, max_steps=args.max_steps,
                  dt_s=DT, warmup_s=4.8, rolling_setup_ticks=STEADY_SETUP_TICKS if args.rolling_history=="steady" else ROLLING_SETUP_TICKS,
                  rolling_history=args.rolling_history, tracker_name=args.tracker,
                  actor_rear_x_m=REAR_X, wheelbase_m=WHEELBASE,
                  inputs=["three RGB frames", "rear-axle measured planar pose derivatives", "planned route command"],
                  rig=dict(mounts_carla_xy_right_z_yaw_deg=MOUNTS, horizontal_fov_deg=90,
                           front_pixels=[640, 360], side_pixels=[320, 180],
                           difference="CARLA rendered pinhole cameras on a Lincoln, with fixed poses and FOV. "
                                      "No calibration match to the recorded Waymo cameras."),
                  route_command="Planned tangent change over max(10 m, speed * 10 s), 30 degree turn threshold.",
                  tracker="Pure pursuit at predicted t=1s, speed from 0.5-to-1s chord, forward gear. "
                          "Steer from CARLA 1.0 Lincoln tire-angle table. "
                          + ("Proportional speed error / 2.5 for throttle or / 4 for brake."
                             if args.tracker == "proportional-v1" else
                             "Speed error / 0.75 s; requested acceleration limited to [-4,2] m/s²; "
                             "feedforward normalized by measured full throttle 7.5 or nominal brake 4 m/s² "
                             "and integral correction from observed acceleration. No speed cap."),
                  waypoints="20 XY in metres, forward/left from current rear axle, 0.25 to 5 seconds",
                  observation="LiDAR, radar, detector tracks, modeled thermal and simulator truth are viewer/scoring only.",
                  trials=[])
    with simulator(port=args.port, attach=args.attach) as world:
        report["producer"] = producer_info(world)
        _write(dest / "report.json", report)
        for speed in args.initial_speed:
            for kind in args.scenario:
                name = f"{kind}-seed{args.seed}-start{speed:g}"
                trial = run_episode(world, model, kind, args.seed, speed, args.max_steps, dest / name, args.record,
                                    args.tracker, args.rolling_history)
                report["trials"].append(dict(path=name, **trial))
                _write(dest / "report.json", report)
                print(f"{name}: {trial['outcome']}, {trial['progress_m']:.1f} m", flush=True)


if __name__ == "__main__":
    main()
