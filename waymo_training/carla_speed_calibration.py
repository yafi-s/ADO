"""Private simulator research. Compare matched CARLA speed planners with and without WOMD.

The archived teacher lessons store synchronized detector boxes as heights/bearings
and nearest route-relative ranges, but not raw point clouds or object identities.
Both arms use the same causal, reduced observation adapter in fitting and driving.
"""
from collections import deque
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from crosswalk import behavior_v2 as behavior
from crosswalk.camera_geometry import CAMERAS
from crosswalk.policy import BUMPER_X
from .interaction import ActorTrack, MotionSpeedModel, make_features

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "data/behavior-v2/lessons.zip"
STEERING = ROOT / "models/behavior-v2/candidate.npz"
COL = {name: i for i, name in enumerate(behavior.FEATURE_NAMES)}
CASES = ("clear", "lead", "pedestrian", "occluded")
SEEDS = (43, 44, 47, 48)
ARMS = ("carla_only", "with_waymo")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CompactTracks:
    """Recover coarse straight-road tracks only when lidar supports a camera box."""

    def __init__(self):
        self.speeds = deque(maxlen=11)
        self.positions = {}
        self.distance = 0.
        self.tick = 0

    def observe(self, x):
        speed = float(x[COL["ego_speed_mps"]])
        if self.speeds:
            self.distance += .05 * (speed + self.speeds[-1])
        self.speeds.append(speed)
        self.tick += 1
        camera = CAMERAS["front"]
        actors = []
        for group, kind, nominal_height in (("person", "pedestrian", 1.7), ("vehicle", "vehicle", 1.52)):
            height = x[COL[f"rgb_{group}_height"]]
            if height <= 0:
                continue
            tangent = x[COL[f"rgb_{group}_center_offset"]] / (2 * camera.fx)
            optical_depth = camera.fy * nominal_height / height
            surfaces = []
            for name, low, high in (("path_body_lidar", -1.08, 1.08),
                                    ("left_edge_lidar", -2.5, -1.08), ("right_edge_lidar", 1.08, 2.5)):
                gap = x[COL[name]]
                forward = float(gap + BUMPER_X)
                depth = forward - camera.x
                right = depth * tangent
                if (gap < behavior.HORIZON and low - .4 <= right <= high + .4
                        and .55 * optical_depth <= depth <= 1.6 * optical_depth):
                    surfaces.append((abs(depth - optical_depth), forward, -right))
            if not surfaces:
                continue
            _, forward, left = min(surfaces)
            # Repeated boxes can remain unchanged on the 10 Hz detector's intermediate tick.
            # Use a half-second position baseline for velocity, reject discontinuous matches.
            samples = self.positions.setdefault(kind, deque())
            while samples and self.tick - samples[0][0] > 5:
                samples.popleft()
            world = np.asarray([self.distance + forward, left])
            if samples and np.linalg.norm(world - samples[-1][1]) > 2.5:
                samples.clear()
            samples.append((self.tick, world))
            dt = .1 * (self.tick - samples[0][0])
            if dt >= .2:
                vx, vy = (world - samples[0][1]) / dt
                actors.append(ActorTrack(kind, forward, left, vx, vy))
        if len(self.speeds) < 11:
            return None
        # Straight-street study. The first signed map tangent change is recorded
        # with opposite lateral sign to the WOMD frame.
        return make_features([self.speeds[0], self.speeds[5], speed], actors,
                             -float(x[COL["bend_7_13"]]), True)


def read_lessons(path=LESSONS):
    records = []
    with zipfile.ZipFile(path) as archive:
        for filename in sorted(archive.namelist()):
            case, seed = Path(filename).stem.rsplit("-", 1)
            if case.removesuffix("-delayed") not in CASES or int(seed) not in SEEDS:
                continue
            with np.load(io.BytesIO(archive.read(filename)), allow_pickle=False) as item:
                if tuple(item["feature_names"]) != behavior.FEATURE_NAMES:
                    raise ValueError("Unexpected demonstration schema")
                lesson = dict(item)
            x = lesson["X"]
            if len(set(lesson["episode_id"])) != 1 or set(lesson["seed"]) != {int(seed)}:
                raise ValueError("Unexpected episode boundary or seed")
            tracks = CompactTracks()
            motion = [tracks.observe(row) for row in x]
            # Supervision only. Reference gaps, future occupancy, scenario names
            # and instructor controls never enter the feature matrix.
            target = behavior._targets(lesson)
            records.append((case, int(seed), x[10:], np.stack(motion[10:]), target[10:]))
    if len(records) != 20:
        raise ValueError("Expected five complete lessons per seed")
    return records


def inputs(x, motion, prior=None):
    # Both conditions receive all the measurements used to compute the prior.
    parts = [np.atleast_2d(x)[:, behavior.PLAN_IDX], np.atleast_2d(motion)]
    if prior is not None:
        parts.append(np.atleast_2d(prior.predict(motion)))
    return np.column_stack(parts)


def fit(records, prior):
    from sklearn.ensemble import ExtraTreesRegressor
    x = np.vstack([r[2] for r in records])
    motion = np.vstack([r[3] for r in records])
    target = np.concatenate([r[4] for r in records])
    weight = np.where(target < 1., 3., np.where(target < 4.5, 1.7, 1.))
    model = ExtraTreesRegressor(n_estimators=40, min_samples_leaf=4, max_depth=12,
                                max_features=1., random_state=4141, n_jobs=1)
    return model.fit(inputs(x, motion, prior), target, sample_weight=weight)


def cross_validate(records, prior):
    rows = []
    for seed in SEEDS:
        train = [r for r in records if r[1] != seed]
        models = [fit(train, m) for m in (None, prior)]
        for case, heldout, x, motion, target in records:
            if heldout != seed:
                continue
            forecasts = prior.predict(motion)
            row = dict(seed=seed, case=case, frames=len(target), stop_frames=int((target < 1.).sum()),
                       lead_frames=int(motion[:, 3].sum()), pedestrian_frames=int(motion[:, 7].sum()))
            for name, model, source in zip(ARMS, models, (None, prior)):
                design = inputs(x, motion, source)
                pred = model.predict(design)
                err = np.abs(pred - target)
                row[name] = dict(mae=float(err.mean()), stop_mae=float(err[target < 1.].mean())
                                 if (target < 1.).any() else None)
                if source is not None:
                    # Within-episode cyclic triplet swap, keeping all other sensors fixed.
                    design[:, -3:] = np.roll(forecasts, len(forecasts) // 2, axis=0)
                    swap = model.predict(design)
                    row[name]["forecast_swap_mean_abs_change_mps"] = float(np.abs(pred - swap).mean())
            rows.append(row)
    return rows


def collision_probes(prior, archived):
    reference = json.loads(Path(__file__).with_name("motion_transfer_development.json").read_text())
    out = {}
    for case in reference["cases"]:
        name = case["scenario"]
        path = archived / (name + ".jsonl.gz")
        if digest(path) != case["trace_sha256"]:
            raise ValueError(f"Archived {name} trace hash differs")
        with gzip.open(path, "rt") as f:
            rows = [json.loads(line) for line in f]
        selected = [r for r in rows if not r["collision_count"] and r["planner"]["forecast_source"] == "interaction"]
        if not selected:
            continue
        x = np.asarray([r["planner"]["features"] for r in selected])
        original = prior.predict(x)
        if not np.allclose(original, [r["planner"]["forecast_mps"] for r in selected], rtol=0, atol=1e-8):
            raise ValueError("Could not reproduce archived forecast")
        masked = x.copy()
        masked[:, 3:13] = 0.
        removed = prior.predict(masked)
        out[name] = dict(frames=len(x), last_frame=selected[-1]["frame"], speed_mps=x[-1, 0],
                         forecast_mps=original[-1].tolist(), no_actor_mps=removed[-1].tolist(),
                         mean_abs_actor_effect_mps=np.abs(original - removed).mean(axis=0).tolist())
    return out


class DrivingModel:
    inputs = "fusion"

    def __init__(self, records, prior, base):
        self.base = base
        self.prior = prior
        self.model = fit(records, prior)

    def new_observer(self):
        self.tracks = CompactTracks()
        self.observer = behavior.Observer()
        self.observer.last_info = {}
        return self.observer

    def predict(self, x):
        steer = self.base.predict(x)[0]
        motion = self.tracks.observe(x)
        if motion is None:
            self.observer.last_info = {"history_warmup": True}
            return steer, 0., 1.
        target = float(self.model.predict(inputs(x, motion, self.prior))[0])
        speed = float(x[COL["ego_speed_mps"]])
        pedal = float(np.clip(behavior._feedback_features(target, speed) @ self.base.feedback, -1., 1.))
        self.observer.last_info = dict(target_speed_mps=target, history_warmup=False,
                                       prior_mps=self.prior.predict(motion).tolist() if self.prior else None)
        return steer, max(0., pedal), max(0., -pedal)


def main():
    import argparse
    from sklearn import __version__ as sklearn_version
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archived", type=Path)
    parser.add_argument("--native-port", type=int, help="Attach only to a reserved, idle Town01 server")
    args = parser.parse_args()
    reference = json.loads(Path(__file__).with_name("motion_transfer_development.json").read_text())
    if digest(args.motion) != reference["motion_sha256"] or digest(STEERING) != reference["steering_sha256"]:
        raise ValueError("This study requires the archived research checkpoints")
    prior = MotionSpeedModel.load(args.motion)
    records = read_lessons()
    report = dict(checkpoint_sha256=digest(args.motion), lessons_sha256=digest(LESSONS),
                  base_sha256=digest(STEERING), source_sha256=digest(__file__),
                  sklearn=sklearn_version, numpy=np.__version__, seeds=SEEDS,
                  attribution=reference["attribution"], license=reference["license"])
    if args.output.exists():
        parser.error("Choose a new output path")
    if args.native_port:
        from crosswalk.drive import episode
        from crosswalk.runtime import simulator, producer_info
        base = behavior.BehaviorModel.load(STEERING)
        models = {arm: DrivingModel(records, source, base) for arm, source in zip(ARMS, (None, prior))}
        report["runs"] = []
        with simulator(args.native_port, attach=True) as world:
            report["producer"] = producer_info(world)
            for n, case in enumerate(CASES):
                for arm in (ARMS if n % 2 == 0 else ARMS[::-1]):
                    summary, trace, _, _ = episode(world, case, 31, arm, models[arm], district="train")
                    folder = args.output.parent
                    path = folder / f"{args.output.stem}-{case}-{arm}.jsonl.gz"
                    with gzip.open(path, "xt", encoding="utf-8") as f:
                        for row in trace:
                            f.write(json.dumps(row, allow_nan=False) + "\n")
                    keys = ("outcome", "progress_m", "collision_events", "yield_violations", "red_light_violations",
                            "route_length_m", "duration_s", "event_cleared")
                    result = dict(case=case, arm=arm, seed=31, trace_sha256=digest(path),
                                  **{k: summary[k] for k in keys})
                    report["runs"].append(result)
                    print(case, arm, result["outcome"], flush=True)
                    partial = args.output.with_suffix(".partial")
                    partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                    partial.replace(args.output)
    else:
        report["folds"] = cross_validate(records, prior)
        if args.archived:
            report["original_actor_ablation"] = collision_probes(prior, args.archived)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")


if __name__ == "__main__":
    main()
