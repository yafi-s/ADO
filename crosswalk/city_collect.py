"""Collect Town03 imitation lessons with a training-only privileged speed instructor."""
import argparse
from pathlib import Path

import numpy as np

from .behavior_v2 import BehaviorModel, FEATURE_NAMES as BASE_NAMES, Observer, _feedback_features
from .drive import episode
from .experiment import PROJECT, save_arrays, write_json, write_trace
from .journey import JourneyScenario
from .runtime import simulator
from .world import _raised

FEATURE_NAMES = (*BASE_NAMES, "rgb_lamp_red_probability", "mapped_lamp_gap_m")


class CityInstructor:
    new_observer = Observer
    lamp_expert = None

    def __init__(self, base, start_m):
        self.start_m = start_m
        self.base = base
        self.sensor_lamp_classifier = base.lamp_expert
        self.scene = None
        self.inputs_seen, self.requests, self.actions = [], [], []

    def create_scene(self, *args, **kwargs):
        self.scene = JourneyScenario(*args, **kwargs)
        if self.start_m:
            if not 0. < self.start_m < self.scene.length - 20.:
                self.scene.close()
                raise ValueError("Training departure must be on the course with at least 20 m remaining")
            index = min(range(len(self.scene._station)), key=lambda i: abs(self.scene._station[i]-self.start_m))
            self.scene.ego.set_transform(_raised(self.scene._points[index].transform, .35))
        return self.scene

    def predict(self, x):
        # Privileged distance and junction labels are confined to this instructor.
        state = self.scene.metrics()
        gap = state["hazard_gap_m"]
        progress = state["progress_m"]
        on_bend = any(a - 8. < progress < b + 3. for a, b in self.scene._junction_ranges)
        cruise = 3.6 if on_bend else 5.2
        # Constant-deceleration stopping envelope with four metres of standstill space.
        requested = min(cruise, np.sqrt(2. * 1.8 * max(0., gap - 4.0)))
        steer = self.base.predict(x[:len(BASE_NAMES)])[0]
        pedal = np.clip(_feedback_features(requested, x[6]) @ self.base.feedback, -1., 1.)
        action = (steer, max(0., pedal), max(0., -pedal))
        self.inputs_seen.append(x.copy())
        self.requests.append(requested)
        self.actions.append(action)
        if len(self.actions) % 350 == 0:
            print(f'{progress:.1f} m   {state["phase"]}', flush=True)
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=2868)
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--start-m", type=float, default=0., help="Training-only departure station in metres")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = BehaviorModel.load(PROJECT/"models/behavior-v2/candidate.npz")
    model = CityInstructor(base, args.start_m)
    with simulator(args.port, args.attach, map_name="Town03_Opt") as world:
        summary, trace, _, _ = episode(world, "journey", args.seed, "privileged-city-instructor", model,
            scene_factory=model.create_scene, destination="market", max_steps=8500)
    summary.update(training_departure_m=args.start_m, purpose="privileged demonstration only")
    frames = np.array([r["frame"] for r in trace])
    assert len(frames) == len(model.inputs_seen)
    save_arrays(args.output/"lesson.npz", X=model.inputs_seen, y=model.actions, frame=frames,
                teacher_speed=model.requests, feature_names=FEATURE_NAMES, seed=np.full(len(frames),args.seed),
                episode_id=np.full(len(frames),f"city-instructor-{args.seed}-{args.start_m:g}"))
    write_trace(args.output/"trace.jsonl.gz",trace)
    write_json(args.output/"summary.json",summary)
    print(summary["outcome"],summary["progress_m"],"handled",sum(e["handled"] for e in summary["events"]),flush=True)


if __name__ == "__main__":
    main()
