"""Run one Town03 journey with the learned driver and retain its measured outcome."""
import argparse
from contextlib import nullcontext
from pathlib import Path
from tempfile import mkdtemp
import webbrowser

import numpy as np

from .behavior_v2 import BehaviorModel, FEATURE_NAMES
from .city_map import geometry
from .drive import episode
from .experiment import PROJECT, fingerprint, save_arrays, write_json, write_trace
from .journey import JourneyScenario
from .runtime import producer_info, simulator
from .viewio import Live


def load_driver(path):
    with np.load(path, allow_pickle=False) as data:
        version = str(data["version"].item())
    if version == "ado-behavior-v2":
        return BehaviorModel.load(path), FEATURE_NAMES, version
    if version == "ado-city-policy-v1":
        from .city_policy import CityModel, FEATURE_NAMES as city_names
        return CityModel.load(path), city_names, version
    raise ValueError(f"Unknown driving model: {version}")


class LoggedDriver:
    """Keep measured features and the corresponding speed request for diagnosis."""
    def __init__(self, policy):
        self.policy = policy
        self.new_observer = policy.new_observer
        self.lamp_expert = policy.lamp_expert
        self.sensor_lamp_classifier = getattr(policy, "sensor_lamp_classifier", None)
        self.observations = []
        self.requested_speed = []

    def predict(self, x):
        action = self.policy.predict(x)
        self.observations.append(x.copy())
        if hasattr(self.policy, "last_speed_request"):
            self.requested_speed.append(self.policy.last_speed_request)
        else:
            # The legacy tree planner is stateless and can be queried for diagnostics.
            if not isinstance(self.policy, BehaviorModel):
                raise RuntimeError("City policy did not expose its applied speed request")
            self.requested_speed.append(self.policy._plan(x))
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT / "models/city-v1/candidate-73-74.npz")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--output", type=Path, help="Parent directory for a new attempt")
    parser.add_argument("--quality", choices=("Low", "Medium", "High"), default="Low")
    parser.add_argument("--port", type=int, default=2468)
    parser.add_argument("--attach", action="store_true", help="Use an idle Town03_Opt server")
    parser.add_argument("--serve", action="store_true", help="Stream the live cameras and measured controls")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if args.open_browser and not args.serve:
        parser.error("--open-browser requires --serve")
    if args.max_steps < 1 or not 0 <= args.seed <= 2147483647:
        parser.error("Choose a positive step limit and a seed in [0, 2147483647]")
    model_sha = fingerprint(args.model)
    driver, names, schema = load_driver(args.model)
    model = LoggedDriver(driver)
    root = args.output or PROJECT / "out/city-drives"
    root.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(prefix=f"city-{args.seed}-", dir=root))
    print(f"City drive evidence: {folder}", flush=True)
    with simulator(args.port, args.attach, args.quality, map_name="Town03_Opt") as world:
        producer = producer_info(world)
        with (Live(args.http_port, map_geometry=geometry(world)) if args.serve else nullcontext()) as live:
            if live:
                url = live.url
                print(f"Live cameras and telemetry: {url}", flush=True)
                if args.open_browser:
                    webbrowser.open(url)
            summary, trace, _, _ = episode(world, "journey", args.seed, schema, model,
                live=live, realtime=bool(live), max_steps=args.max_steps, destination="market", run_id=folder.name,
                scene_factory=JourneyScenario)
    write_trace(folder / "trace.jsonl.gz", trace)
    if len(model.observations) != len(trace):
        raise RuntimeError("Scored frames and model observations differ")
    save_arrays(folder / "observations.npz", frame=[row["frame"] for row in trace],
                features=np.asarray(model.observations), feature_names=names,
                speed_request_mps=model.requested_speed)
    summary.update(producer=producer, model_sha256=model_sha, policy_schema=schema)
    write_json(folder / "summary.json", summary)
    passed = sum(event["handled"] for event in summary["events"])
    print(f'{summary["outcome"]}: {summary["progress_m"]:.1f}/{summary["route_length_m"]:.1f} m, '
          f'{passed}/{len(summary["events"])} encounters handled', flush=True)


if __name__ == "__main__":
    main()
