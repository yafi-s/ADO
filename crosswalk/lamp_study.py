"""Matched stationary red/green RGB measurements on CARLA training-district roads."""

import argparse
from collections import deque
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .lamp import CAMERAS, LampClassifier, LampExpert, RouteLamps, calibrate, color_features, head_vertices, select_crop


FIT_SEEDS = (11, 12, 13, 14)
VALIDATION_SEEDS = (31, 32)
EXTRA_SEEDS = (41, 42, 43, 44)
DISTANCES_M = (17., 10., 5., 2.)


def _park(scene, distance, z_offset):
    """Place the rigid camera platform on the route, measuring from its front bumper."""
    import carla
    xyz = np.asarray([(p.x, p.y, p.z) for p in scene.route])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))]
    nose = scene.ego.bounding_box.extent.x + scene.ego.bounding_box.location.x
    progress = np.clip(scene._stop_s - distance - nose, 0., arc[-1])
    idx = min(int(np.searchsorted(arc, progress, side="right") - 1), len(xyz)-2)
    x, y, z = [np.interp(progress, arc, axis) for axis in xyz.T]
    dx, dy = xyz[idx+1, :2] - xyz[idx, :2]
    rotation = carla.Rotation(yaw=float(np.degrees(np.arctan2(dy, dx))))
    scene.ego.set_transform(carla.Transform(carla.Location(x=x, y=y, z=z+z_offset), rotation))


def _capture(world, output, seed, probe):
    import carla
    from .sensors import Rig
    from .world import Scenario
    start = perf_counter()
    with ExitStack() as stack:
        scene = Scenario(world, "signal", seed, district="train")
        stack.callback(scene.close)
        rig = Rig(world, scene.ego)
        stack.callback(rig.close)
        cameras = dict(zip(CAMERAS, [s for s in rig.sensors if s.type_id == "sensor.camera.rgb"], strict=True))
        heads = head_vertices(scene.signal)
        pole = scene.signal.get_location()
        report = dict(seed=seed, district=scene.district, road=scene.road_id, lane=scene.lane_id,
                      weather=scene.weather, signal_xyz=[round(x, 2) for x in (pole.x, pole.y, pole.z)],
                      signal_opendrive=scene.signal.get_opendrive_id(),
                      static_world_boxes=[verts.round(3).tolist() for verts in heads],
                      route=scene.route_id, rendering=rig.rendering, pairs=[])
        scene.ego.apply_control(carla.VehicleControl(hand_brake=True))
        for _ in range(7):
            rig.read(world.tick())
        z_offset = scene.ego.get_location().z - scene.route[0].z
        scene.ego.set_simulate_physics(False)
        distances = (17., 7., 2.) if probe else DISTANCES_M
        if seed in EXTRA_SEEDS:
            available = -scene._bumper_line_distance()
            distances = tuple(fraction*available for fraction in (.95, .66, .38, .14))
        for index, distance in enumerate(distances):
            _park(scene, distance, z_offset)
            pair = dict(requested_gap_m=distance, observations={})
            # Alternate state order, counterbalanced across seeds and distances.
            colors = (("red", carla.TrafficLightState.Red), ("green", carla.TrafficLightState.Green))
            for color, state in (colors if (seed+index) % 2 else reversed(colors)):
                scene.signal.set_state(state)
                for _ in range(4):
                    frame = world.tick()
                    images = rig.read(frame)
                if scene.signal.get_state() != state:
                    raise RuntimeError("Simulator did not apply the requested training label")
                transforms = {name: camera.get_transform().get_inverse_matrix() for name, camera in cameras.items()}
                selection = select_crop(heads, images, transforms)
                prefix = f"{seed}-{index}-{color}"
                obs = dict(frame=frame, crop=None)
                if selection is not None:
                    name, head_index, rect, crop = selection
                    filename = f"{prefix}-{name}.png"
                    if not cv2.imwrite(str(output / filename), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
                        raise OSError(f"Could not write {filename}")
                    obs.update(camera=name, head=head_index, rect=rect, crop=filename)
                if probe:
                    # Keep full frames when there is no projection or the light appears unlit.
                    for name in CAMERAS:
                        if not cv2.imwrite(str(output / f"{prefix}-{name}-full.png"),
                                           cv2.cvtColor(images[name], cv2.COLOR_RGB2BGR)):
                            raise OSError("Could not write probe frame")
                pair["observations"][color] = obs
            pair["actual_gap_m"] = round(-scene._bumper_line_distance(), 3)
            report["pairs"].append(pair)
    report["wall_seconds"] = round(perf_counter() - start, 2)
    return report


def _drive_lesson(world, teacher, seed, start_green):
    """Return image-derived features at t and actual teacher controls that advance t+1."""
    import carla
    from .control import vehicle_speed
    from .drive import start_demonstration, update_instructor
    from .sensors import Rig
    from .world import Scenario
    rows = []
    with ExitStack() as stack:
        teacher.set_random_device_seed(seed)
        scene = Scenario(world, "signal", seed, district="train", perturb=True)
        stack.callback(scene.close)
        if start_green:
            scene.signal.set_state(carla.TrafficLightState.Green)
            scene.phase = "green signal"
            scene._cleared = True
        rig = Rig(world, scene.ego)
        stack.callback(rig.close)
        mapped = RouteLamps(world, scene.route)
        for _ in range(8):
            rig.read(world.tick())
        stack.callback(scene.ego.set_autopilot, False, teacher.get_port())
        start_demonstration(scene, teacher)
        pending = None
        for step in range(175):
            scene.before_tick()
            update_instructor(scene, teacher)
            frame = world.tick()
            images = rig.read(frame)
            controls = scene.ego.get_control()
            if pending is not None:
                rows.append(dict(**pending, throttle=controls.throttle, brake=controls.brake))
            if scene._bumper_line_distance() > 2. or rig.collisions():
                break
            obs = mapped.observe(scene.ego, images)
            pending = None if obs is None else dict(
                frame=frame, gap=obs[0], color=obs[1].tolist(), speed=vehicle_speed(scene.ego),
                # Label retained for analysis only. Fit actuator models on color, gap, speed.
                label=str(scene.signal.get_state()))
    return dict(seed=seed, district="train", green_from_start=start_green, rows=rows)


def _features(output, pair):
    rows = [pair["observations"][color] for color in ("red", "green")]
    if any(row["crop"] is None for row in rows):
        return None
    if any(rows[0][key] != rows[1][key] for key in ("camera", "head", "rect")):
        return None
    features = []
    for row in rows:
        bgr = cv2.imread(str(output / row["crop"]))
        if bgr is None:
            raise OSError(f"Missing crop {row['crop']}")
        features.append(color_features(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return np.asarray(features)


def summarize(output):
    records = [json.loads(file.read_text()) for file in sorted(output.glob("seed-*.json"))]
    training, evaluation, paired_observations = [], [], []
    fitted_poles = {tuple(r["signal_xyz"]) for r in records if r["seed"] in FIT_SEEDS}
    overlaps = []
    for record in records:
        seed = record["seed"]
        if record["district"] != "train" or seed not in FIT_SEEDS + VALIDATION_SEEDS + EXTRA_SEEDS:
            raise ValueError("Unexpected route in lamp diagnostic")
        if seed not in FIT_SEEDS and tuple(record["signal_xyz"]) in fitted_poles:
            overlaps.append(seed)
            continue
        for pair in record["pairs"]:
            item = _features(output, pair)
            observation = pair["observations"]["red"]
            paired_observations.append(dict(seed=seed, gap_m=pair["actual_gap_m"],
                                            camera=observation.get("camera"), box=observation.get("rect"),
                                            max_color_feature_change=float(np.max(np.abs(item[0]-item[1])))
                                            if item is not None else None))
            if seed in FIT_SEEDS:
                if item is not None:
                    training.append(item)
            else:
                evaluation.append((seed, pair["actual_gap_m"], item))
    model = LampClassifier.fit(training) if training else None
    scores = []
    for seed, gap, item in evaluation:
        margins = [model.score(row) for row in item] if model is not None and item is not None else None
        scores.append(dict(seed=seed, gap_m=gap, red_margin=margins[0] if margins else None,
                           green_margin=margins[1] if margins else None))
    covered = [s for s in scores if s["red_margin"] is not None]
    red_hits = sum(s["red_margin"] > 0 for s in covered)
    green_hits = sum(s["green_margin"] < 0 for s in covered)
    denominator = len(scores)
    result = dict(district="train", fit_seeds=[r["seed"] for r in records if r["seed"] in FIT_SEEDS],
                  validation_seeds=sorted({s["seed"] for s in scores}),
                  excluded_shared_pole_seeds=overlaps, fit_pairs=len(training),
                  validation_pairs=denominator, scored_pairs=len(covered), scores=scores,
                  paired_observations=paired_observations,
                  red_recall=red_hits/denominator if denominator else None,
                  green_recall=green_hits/denominator if denominator else None,
                  balanced_accuracy_all_pairs=(red_hits+green_hits)/(2*denominator) if denominator else None,
                  correctly_ordered_pairs=sum(s["red_margin"] > s["green_margin"] for s in covered),
                  capture_seconds=sum(r["wall_seconds"] for r in records))
    result["meets_probe_criterion"] = bool(denominator and len(result["validation_seeds"]) >= 2 and
                                          min(result["red_recall"], result["green_recall"]) >= .90)
    # The signed score is a diagnostic output. Driving behavior needs separate validation.
    if model is not None:
        result["classifier"] = dict(center=model.center.tolist(), weights=model.weights.tolist())
    return result


def fit_expert(study, lessons, output):
    samples = {}
    training_poles = set()
    for seed in FIT_SEEDS:
        record = json.loads((study / f"seed-{seed}.json").read_text())
        if record["district"] != "train" or record["seed"] != seed:
            raise ValueError("Signal training requires an identified training-district record.")
        training_poles.add(tuple(record["signal_xyz"]))
        samples[seed] = [x for pair in record["pairs"] if (x := _features(study, pair)) is not None]
    if len(training_poles) != len(FIT_SEEDS):
        raise ValueError("Calibration folds require distinct physical signal poles")
    all_pairs = [pair for group in samples.values() for pair in group]
    classifier = LampClassifier.fit(all_pairs)
    margins = []
    for held in FIT_SEEDS:
        other_pairs = [pair for seed, group in samples.items() if seed != held for pair in group]
        fold = LampClassifier.fit(other_pairs)
        margins.extend(fold.score(z) for pair in samples[held] for z in pair)
    calibration = calibrate(margins, np.tile([1., 0.], len(margins)//2))
    expert = LampExpert(classifier, calibration, [], [])
    points, pedals = [], []
    gate_scores, approach_labels = [], []
    drives = []
    for seed in FIT_SEEDS:
        for condition in ("cycle", "green"):
            record = json.loads((lessons / f"drive-{seed}-{condition}.json").read_text())
            if (record["district"] != "train" or record["seed"] != seed or
                    record["green_from_start"] != (condition == "green") or not record["rows"]):
                raise ValueError("Signal pedal fit requires nonempty designated training drives.")
            window, peak = deque(maxlen=5), 0.
            for row in record["rows"]:
                probability = expert.red_probability(row["color"])
                points.append([5.*probability, row["gap"], 2.*row["speed"]])
                pedals.append(row["throttle"]-row["brake"])
                window.append(probability)
                peak = max(peak, float(np.median(window)))
                gate_scores.append(peak)
                approach_labels.append(not record["green_from_start"])
            drives.append(dict(seed=seed, condition=condition, frames=len(record["rows"])))
    expert = LampExpert(classifier, calibration, points, pedals)
    expert.save(output / "lamp.npz")
    gate_parameters = dict(version=1, median_frames=5, weights=calibrate(gate_scores, approach_labels).tolist())
    (output / "approach-gate.json").write_bytes(
        (json.dumps(gate_parameters, sort_keys=True, separators=(",", ":"))+"\n").encode("utf-8"))
    truth, probabilities = [], []
    validation = []
    for root, seeds in ((study, VALIDATION_SEEDS), (lessons, EXTRA_SEEDS)):
        for seed in seeds:
            record = json.loads((root / f"seed-{seed}.json").read_text())
            shared = tuple(record["signal_xyz"]) in training_poles
            observations = []
            for pair in record["pairs"]:
                item = _features(root, pair)
                probs = [expert.red_probability(row) for row in item] if item is not None else None
                observations.append(dict(gap_m=pair["actual_gap_m"], probabilities_red_green=probs))
                if not shared and probs is not None:
                    truth.extend([1., 0.])
                    probabilities.extend(probs)
            validation.append(dict(seed=seed, pole=record["signal_xyz"], shared_fit_pole=shared,
                                   observations=observations))
    y, p = np.asarray(truth), np.asarray(probabilities)
    expected_pairs = sum(len(d["observations"]) for d in validation if not d["shared_fit_pole"])
    report = dict(fit_seeds=FIT_SEEDS, fit_static_pairs=len(all_pairs), lessons=drives,
                  fit_frames=len(pedals), validation=validation,
                  validation_distinct_unseen_poles=len({tuple(d["pole"]) for d in validation if not d["shared_fit_pole"]}),
                  red_correct=int(np.sum((y==1)&(p>=.5))), green_correct=int(np.sum((y==0)&(p<.5))),
                  trials_per_color=expected_pairs, coverage=len(p)/(2*expected_pairs),
                  held_pole_brier_score=float(np.mean((p-y)**2)), calibration=calibration.tolist())
    (output / "fit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k not in ("validation", "lessons")}, indent=2))


def _validate_drive(world, kind, seed, start_green, model, expert):
    """Use the ordinary episode loop; wrap its public sensor/actuator boundary only."""
    import carla
    from unittest.mock import patch
    from . import drive
    from .control import vehicle_speed
    observations = []
    class TestScene(drive.Scenario):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if start_green:
                self.signal.set_state(carla.TrafficLightState.Green)
                self.phase = "green signal"
                self._cleared = True
            gate.routes = RouteLamps(world, self.route)
    class TestRig(drive.Rig):
        def __init__(self, world, ego, *args, **kwargs):
            super().__init__(world, ego, *args, **kwargs)
            gate.ego = ego
        def read(self, *args, **kwargs):
            images = super().read(*args, **kwargs)
            gate.observation = gate.routes.observe(gate.ego, images)
            return images
    class ActuatorTrace:
        def predict(self, x):
            base = model.predict(x)
            selected = expert.apply(self.observation, vehicle_speed(self.ego), base)
            obs = self.observation
            observations.append(dict(gap_m=obs[0] if obs else None,
                                     p_red=expert.red_probability(obs[1]) if obs else None,
                                     base=list(base), selected=list(selected)))
            return selected
    gate = ActuatorTrace()
    with patch.object(drive, "Scenario", TestScene), patch.object(drive, "Rig", TestRig):
        summary, trace, *_ = drive.episode(world, kind, seed, "fusion", gate, district="train")
    if len(trace) != len(observations):
        raise RuntimeError("Sensor and actuator verification streams have different lengths.")
    for row, obs in zip(trace, observations):
        row["expert"] = obs
    red = [r for r in trace if r["phase"] == "red signal"]
    green = [r for r in trace if r["phase"] == "green signal"]
    stopped = [r for r in red if r["speed_mps"] <= .35 and r["progress_m"] > 1.]
    summary.update(green_from_start=start_green,
                   engaged_frames=sum(r["gap_m"] is not None for r in observations),
                   modified_frames=sum(r["base"] != r["selected"] for r in observations),
                   red_stopped_frames=len(stopped),
                   smallest_red_gap_m=min((r["expert"]["gap_m"] for r in red
                                            if r["expert"]["gap_m"] is not None), default=None),
                   final_red_speed_mps=red[-1]["speed_mps"] if red else None,
                   first_green_speed_mps=green[0]["speed_mps"] if green else None,
                   maximum_green_speed_mps=max((r["speed_mps"] for r in green), default=None))
    return summary, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "study", "lessons", "summarize", "fit", "validate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attach", action="store_true", help="Use an idle, rendering-capable server")
    parser.add_argument("--port", type=int, default=2468)
    parser.add_argument("--study", type=Path, help="Stationary study input for offline fitting")
    parser.add_argument("--lessons", type=Path, help="Additional lesson input for offline fitting")
    parser.add_argument("--expert", type=Path, help="Learned signal model for closed-loop validation")
    parser.add_argument("--base", type=Path, default=Path("models/fusion.npz"))
    args = parser.parse_args()
    if args.mode == "validate":
        import gzip
        import hashlib
        from .policy import Model
        from .runtime import simulator
        if args.expert is None:
            parser.error("Validation requires --expert")
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            parser.error("Use a fresh closed-loop validation output directory")
        expert = LampExpert.load(args.expert)
        base = Model.load(args.base)
        hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in (("expert",args.expert),("base",args.base))}
        summaries = []
        cases = [("signal", seed, green) for seed in VALIDATION_SEEDS for green in (False, True)]
        with simulator(args.port, args.attach) as world:
            for kind, seed, green in cases + [("clear", 31, False)]:
                result, trace = _validate_drive(world, kind, seed, green, base, expert)
                name = f"{kind}-{seed}-{'green' if green else 'cycle'}"
                with gzip.open(args.output / f"{name}.jsonl.gz", "wt", encoding="utf-8") as f:
                    for row in trace:
                        f.write(json.dumps(row) + "\n")
                summaries.append(result)
                (args.output / "results.json").write_text(json.dumps(dict(hashes=hashes, trials=summaries), indent=2))
                print(name, result["outcome"], "stopped", result["red_stopped_frames"],
                      "modified", result["modified_frames"], flush=True)
        return
    if args.mode == "fit":
        if args.study is None or args.lessons is None:
            parser.error("Fitting requires --study and --lessons")
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            parser.error("Use a fresh or empty model output directory")
        fit_expert(args.study, args.lessons, args.output)
        return
    if args.mode == "summarize":
        if not list(args.output.glob("seed-*.json")):
            parser.error("No completed seed records at --output")
    else:
        from .runtime import producer_info, simulator
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            parser.error("Use a fresh or empty --output directory for capture")
        lessons = args.mode == "lessons"
        seeds = EXTRA_SEEDS if lessons else FIT_SEEDS[:1] if args.mode == "probe" else FIT_SEEDS + VALIDATION_SEEDS
        with simulator(args.port, args.attach, with_manager=lessons) as session:
            world = session[0] if lessons else session
            for seed in seeds:
                report = _capture(world, args.output, seed, args.mode == "probe")
                report["producer"] = producer_info(world)
                (args.output / f"seed-{seed}.json").write_text(json.dumps(report, indent=2))
                print(f"Seed {seed}: {len(report['static_world_boxes'])} mapped boxes, "
                      f"{sum(p['observations']['red']['crop'] is not None for p in report['pairs'])}"
                      f"/{len(report['pairs'])} visible poses in {report['wall_seconds']} s", flush=True)
            if lessons:
                for seed in FIT_SEEDS:
                    for start_green in (False, True):
                        data = _drive_lesson(world, session[1], seed, start_green)
                        name = f"drive-{seed}-{'green' if start_green else 'cycle'}.json"
                        (args.output / name).write_text(json.dumps(data))
                        print(f"{name}: {len(data['rows'])} visible labelled controls", flush=True)
    if args.mode == "lessons":
        return
    summary = summarize(args.output)
    if args.mode != "summarize":
        (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "classifier"}, indent=2))


if __name__ == "__main__":
    main()
