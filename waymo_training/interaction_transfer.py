"""Private CARLA research transfer of the WOMD speed forecaster.

The frozen Waymo model predicts logged future ego speeds, not desired speeds
or obstacle-free space. The ADO CARLA model supplies steering. With supported
tracks, a fixed servo maps all three WOMD forecast horizons to pedals. Without
a supported lead/VRU, use the learned CARLA sensor controller, or supply an
explicit, separately fitted WOMD history-only checkpoint. Sources are logged.
The forward sensor rig misses occluded objects and objects outside camera
coverage. Annotation velocities in training are cleaner than sensor estimates.
Use a fresh observer for every drive, under the CARLA Lincoln/10 Hz setup.
"""
from collections import deque
import math

import numpy as np

from crosswalk import behavior_v2, policy
from crosswalk.tracking import Tracker
from .interaction import ActorTrack, FEATURE_NAMES, HORIZONS, MotionSpeedModel, make_features


KINDS = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle",
         "person": "pedestrian", "bicycle": "cyclist", "motorcycle": "vehicle"}


def route_bend(local_route, route_preview):
    """Change of road tangent over about 15 m, counterclockwise positive.

    Use map-to-map chords at stations 3–7 and 13–18 m. Connecting the ego
    origin to the first waypoint would confuse lateral pose error with bend.
    """
    close = np.asarray(local_route, dtype=float).reshape(3, 2)
    far = np.asarray(route_preview, dtype=float).reshape(4, 2)
    first, last = close[1] - close[0], far[0] - close[2]
    if not np.isfinite([first, last]).all() or min(map(np.linalg.norm, (first, last))) < .1:
        return 0., False
    cross = first[0] * last[1] - first[1] * last[0]
    return math.atan2(-cross, first @ last), True


class SensorTracks:
    """Estimate 2-D ground velocity from recent *measured* world positions.

    Track association, position, RGB/lidar corroboration, optional radar match
    and at most 0.5 s coasting come from the same tracker as the ADO viewer.
    A single Doppler return cannot provide two velocity components. Require
    two fresh positions; never invent a stationary target from a first sighting.
    """
    def __init__(self):
        self.tracker = Tracker()
        self.samples = {}

    def read(self, obs, detections, ego_pose, now):
        pose = np.asarray(ego_pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("expected finite ego x, y and yaw in degrees")
        rows = self.tracker.update(obs["lidar"], obs["radar"], detections, pose, now)
        c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
        rotation = np.array([[c, -s], [s, c]])
        active = {r["id"] for r in rows}
        self.samples = {k: v for k, v in self.samples.items() if k in active}
        actors = []
        coasts = 0
        for row in rows:
            kind = KINDS.get(row["label"])
            if kind is None:
                continue
            samples = self.samples.setdefault(row["id"], deque())
            if not row["coasting"]:
                world = pose[:2] + rotation @ [row["x"], row["y"]]
                samples.append((now, world))
                while samples and now - samples[0][0] > .5 + 1e-6:
                    samples.popleft()
            if len(samples) < 2:
                continue
            times, xy = zip(*samples)
            centered = np.array(times) - np.mean(times)
            v_world = centered @ (np.asarray(xy) - np.mean(xy, axis=0)) / (centered @ centered)
            vx, right = rotation.T @ v_world
            actors.append(ActorTrack(kind, row["x"], -row["y"], float(vx), -float(right)))
            coasts += int(row["coasting"])
        return actors, dict(sensor_tracks=len(rows), velocity_tracks=len(actors),
                            coasting_velocity_tracks=coasts,
                            radar_matched_tracks=sum("radar" in row["sources"] for row in rows))


class SpeedServo:
    """CARLA Lincoln calibration from simulator_transfer.CalibratedTracker.

    Match constant acceleration slopes from each forecast origin with weights
    proportional to inverse horizon. Bounds on acceleration and the 7.5 m/s²
    full-throttle scale come from the earlier CARLA calibration. The 4 m/s²
    brake scale is nominal. Acceleration feedback compensates drag and error
    in the inverse dynamics. No obstacle, classifier or route input is used.
    """
    def __init__(self):
        self.previous_speed = None
        self.acceleration = 0.
        self.integral = 0.

    def request(self, forecast, speed, dt):
        forecast = np.asarray(forecast, dtype=float)
        if (forecast.shape != (3,) or not np.isfinite(forecast).all() or (forecast < 0).any()
                or not math.isfinite(speed) or speed < 0 or not math.isfinite(dt) or dt <= 0):
            raise ValueError("expected three nonnegative forecasts, speed and a positive elapsed time")
        horizons = np.asarray(HORIZONS)
        slopes = (forecast - speed) / horizons
        acceleration = float(np.clip(np.average(slopes, weights=1. / horizons), -4., 2.))
        if self.previous_speed is not None:
            self.acceleration += .4 * ((speed - self.previous_speed) / dt - self.acceleration)
        self.previous_speed = speed
        scale = 7.5 if acceleration >= 0 else 4.
        feedforward = acceleration / scale
        change = dt * (acceleration - self.acceleration) / (.5 * scale)
        raw = feedforward + self.integral
        if (raw < 1. or change < 0) and (raw > -1. or change > 0):
            self.integral = float(np.clip(self.integral + change, -.45, .45))
        if np.max(forecast) == 0 and speed <= .02:
            self.integral = min(self.integral, 0.)
        pedal = float(np.clip(feedforward + self.integral, -1., 1.))
        return max(0., pedal), max(0., -pedal), acceleration


class MotionTransfer:
    inputs = "fusion"

    def __init__(self, motion, history_model, steering):
        self.motion = motion
        self.history_model = history_model
        self.steering = steering

    @classmethod
    def load(cls, motion_path, steering_path, *, history_path=None):
        motion = MotionSpeedModel.load(motion_path)
        fallback = MotionSpeedModel.load(history_path) if history_path is not None else None
        if motion.mode != "interaction" or (fallback is not None and fallback.mode != "history"):
            raise ValueError("expected interaction and history-only WOMD checkpoints")
        with np.load(steering_path, allow_pickle=False) as data:
            v2 = str(data["version"].item()) == behavior_v2.VERSION
        steering = (behavior_v2.BehaviorModel if v2 else policy.Model).load(steering_path)
        return cls(motion, fallback, steering)

    def new_observer(self):
        return TransferObserver(self)


class TransferObserver:
    def __init__(self, model):
        self.model = model
        self.tracks = SensorTracks()
        self.speeds = deque(maxlen=12)
        self.servo = SpeedServo()
        factory = getattr(model.steering, "new_observer", None)
        self.steering_observer = factory() if factory else None
        self.last_info = {}

    def observe_and_predict(self, obs, detections, local_route, speed, *, route_preview,
                            ego_pose, sim_time, detection_fresh):
        """One chronological 10 Hz frame, with synchronized captures and ego pose.

        detections may be the last RGB inference for the ADO steering features.
        Only fresh boxes can create a new position/velocity measurement.
        Diagnostics are available in last_info immediately after this call.
        """
        now, speed = float(sim_time), float(speed)
        if not math.isfinite(now) or not math.isfinite(speed) or speed < 0.:
            raise ValueError("expected finite simulation time and nonnegative measured speed")
        dt = now - self.speeds[-1][0] if self.speeds else .1
        if not math.isclose(dt, .1, rel_tol=0., abs_tol=1e-5):
            raise ValueError("expected contiguous 10 Hz measurements; start a fresh observer after a gap")
        self.speeds.append((now, speed))
        if self.steering_observer is None:
            steering_x = policy.features(obs["front"], obs["lidar"], obs["radar"], speed, detections,
                                         navigation=local_route, side_views=(obs["left"], obs["right"]))
        else:
            steering_x = self.steering_observer.observe(obs, detections, local_route, speed,
                                                        route_preview=route_preview)
        steer, ado_throttle, ado_brake = map(float, self.model.steering.predict(steering_x))
        actors, counts = self.tracks.read(obs, detections if detection_fresh else (), ego_pose, now)
        self.last_info = dict(**counts, detection_fresh=bool(detection_fresh))
        if now - self.speeds[0][0] < 1. - 1e-5:
            self.last_info.update(forecast_source="measured_history_warmup", forecast_mps=None)
            return steer, 0., 1.
        times, values = zip(*self.speeds)
        history = np.interp(now + np.array([-1., -.5, 0.]), times, values)
        bend, valid = route_bend(local_route, route_preview)
        x = make_features(history, actors, bend, valid)
        supported = bool(x[FEATURE_NAMES.index("lead_present")] or x[FEATURE_NAMES.index("vru_present")])
        # Absence of a supported detection is censored observation, not evidence
        # the annotated lane is empty. With only the interaction artifact,
        # use ADO's learned scene controller until a qualified track is found.
        if not supported and self.model.history_model is None:
            self.servo = SpeedServo()  # Prior forecast tracking has ended.
            self.last_info.update(forecast_source="ado_sensor_fallback", forecast_mps=None,
                                  features=x.tolist(), requested_accel_mps2=None)
            return steer, ado_throttle, ado_brake
        model = self.model.motion if supported else self.model.history_model
        forecast = model.predict(x)
        throttle, brake, acceleration = self.servo.request(forecast, speed, dt)
        self.last_info.update(forecast_source="interaction" if supported else "history_only",
                              forecast_mps=np.asarray(forecast).tolist(), features=x.tolist(),
                              requested_accel_mps2=acceleration)
        return steer, throttle, brake


def main():
    import argparse
    from collections import Counter
    import gzip
    import hashlib
    import json
    from pathlib import Path
    import shutil

    from crosswalk.drive import episode
    from crosswalk.runtime import producer_info, simulator

    parser = argparse.ArgumentParser(description="Private Town01 development trial of the WOMD/ADO hybrid.")
    parser.add_argument("--motion", required=True, type=Path, help="private motion-speed.npz from Colab")
    parser.add_argument("--steering", required=True, type=Path, help="frozen CARLA ADO policy checkpoint")
    parser.add_argument("--history", type=Path, help="optional matching independent WOMD history-only checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="new private output directory")
    parser.add_argument("--port", type=int, default=2868)
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--scenarios", nargs="+", choices=("clear", "lead", "pedestrian", "occluded"),
                        default=("clear", "lead", "pedestrian", "occluded"))
    args = parser.parse_args()
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("each scenario must appear once")
    model = MotionTransfer.load(args.motion, args.steering, history_path=args.history)
    args.output.mkdir(parents=True, exist_ok=False)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    source = Path(__file__).resolve()
    paths = [source, source.with_name("interaction.py"), *[source.parents[1]/"crosswalk"/(name + ".py")
             for name in ("drive", "tracking", "vision", "control", "world", "policy", "behavior_v2")]]
    manifest = dict(motion_sha256=digest(args.motion), steering_sha256=digest(args.steering),
                    history_sha256=digest(args.history) if args.history else None, seed=args.seed, district="train",
                    stage="development", scenarios=args.scenarios, source_sha256={p.name:digest(p) for p in paths},
                    attribution="Waymo Open Motion Dataset v1.3.1 speed forecasts; ADO CARLA learned steering "
                                "and no-track fallback pedals; calibrated speed servo; NanoDet/RGB + lidar tracks "
                                "with optional radar match. Non-commercial research.")
    for name in ("WAYMO_DATASET_AGREEMENT.txt", "NOTICE.md"):
        notice = args.motion.parent / name
        if notice.is_file():
            shutil.copyfile(notice, args.output/name)

    def save(name, data):
        dest = args.output / name
        partial = dest.with_suffix(dest.suffix + ".partial")
        partial.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        partial.replace(dest)

    summaries = []
    with simulator(args.port, args.attach) as world:
        save("protocol.json", dict(manifest, producer=producer_info(world)))
        for scenario in args.scenarios:
            result, trace, _, _ = episode(world, scenario, args.seed, "WOMD interaction + ADO", model,
                                          district="train")
            result["forecast_sources"] = dict(Counter(r["planner"]["forecast_source"] for r in trace))
            trace_path = args.output / (scenario + ".jsonl.gz")
            with gzip.open(trace_path, "wt", encoding="utf-8") as out:
                for row in trace:
                    out.write(json.dumps(row, allow_nan=False) + "\n")
            result["trace_sha256"] = digest(trace_path)
            summaries.append(result)
            save("results.json", summaries)
            print(scenario, result["outcome"], result["progress_m"], result["forecast_sources"], flush=True)


if __name__ == "__main__":
    main()
