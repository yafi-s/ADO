"""Run one Town03 journey with the learned driver and retain its measured outcome."""
import argparse
from contextlib import nullcontext
from functools import partial
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
        revision = int(data["revision"].item()) if "revision" in data else None
    if version == "ado-behavior-v2":
        return BehaviorModel.load(path), FEATURE_NAMES, version
    if version == "ado-city-policy-v1":
        from .city_policy import CityModel, FEATURE_NAMES as city_names
        return CityModel.load(path), city_names, version
    if version == "ado-city-cruise-v1":
        from .city_cruise import UrbanCruiseModel, FEATURE_NAMES as city_names
        return UrbanCruiseModel.load(path), city_names, version
    if version == "ado-city-pace-v1":
        from .city_pace import CityPaceModel, FEATURE_NAMES as city_names
        return CityPaceModel.load(path), city_names, version
    if version == "ado-city-clearance-v1":
        from .city_clearance import ClearanceModel, FEATURE_NAMES as clearance_names
        return ClearanceModel.load(path), clearance_names, version
    if version == "ado-city-maneuvers-v1":
        from .city_maneuvers import ManeuverModel, FEATURE_NAMES as maneuver_names
        return ManeuverModel.load(path), maneuver_names, version
    if version == "ado-city-progress-v1":
        if revision == 4:
            from .city_grade import GradeModel, FEATURE_NAMES as grade_names
            return GradeModel.load(path), grade_names, version
        from .city_progress import ProgressModel, FEATURE_NAMES as progress_names
        return ProgressModel.load(path), progress_names, version
    raise ValueError(f"Unknown driving model: {version}")


class LoggedDriver:
    """Keep measured features and the corresponding speed request for diagnosis."""
    def __init__(self, policy):
        self.policy = policy
        self.new_observer = policy.new_observer
        self.lamp_expert = policy.lamp_expert
        self.sensor_lamp_classifier = getattr(policy, "sensor_lamp_classifier", None)
        self.uses_posted_limit = getattr(policy, "uses_posted_limit", False)
        self.observations = []
        self.requested_speed = []
        self.posted_limits = []

    @property
    def posted_limit_kmh(self):
        return self.policy.posted_limit_kmh

    @posted_limit_kmh.setter
    def posted_limit_kmh(self, value):
        self.policy.posted_limit_kmh = value

    @property
    def last_speed_request(self):
        return self.requested_speed[-1] if self.requested_speed else None

    def predict(self, x):
        action = self.policy.predict(x)
        self.observations.append(x.copy())
        if self.uses_posted_limit:
            self.posted_limits.append(self.posted_limit_kmh)
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
    parser.add_argument("--model", type=Path, default=PROJECT / "models/city-maneuvers/candidate.npz")
    parser.add_argument("--destination", choices=("market", "market-loop"),
                        help="Choose the original Market route or continue through the return boulevard")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--crossing-delay", type=float, default=0., help="Additional pedestrian departure delay in simulated seconds")
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--output", type=Path, help="Parent directory for a new attempt")
    parser.add_argument("--quality", choices=("Low", "Medium", "High"), default="Low")
    parser.add_argument("--port", type=int, default=2468)
    parser.add_argument("--attach", action="store_true", help="Use an idle Town03_Opt server")
    parser.add_argument("--serve", action="store_true", help="Stream the live cameras and measured controls")
    parser.add_argument("--population-profile", choices=("standard", "city-life", "city-maneuvers"),
                        help="Street population and staged encounters for this drive")
    parser.add_argument("--urban-cap", action="store_true",
                        help="Keep the City Maneuvers driver at its urban cruise ceiling on high-posted roads")
    parser.add_argument("--surround", action=argparse.BooleanOptionalAction, default=None,
                        help="Run the five-camera perception rig (on by default for the live urban cruise driver)")
    parser.add_argument("--tactical", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable the tactical pass planner (on by default for the live urban cruise driver)")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if args.open_browser and not args.serve:
        parser.error("--open-browser requires --serve")
    if args.max_steps < 1 or not 0 <= args.seed <= 2147483647:
        parser.error("Choose a positive step limit and a seed in [0, 2147483647]")
    if not np.isfinite(args.crossing_delay) or args.crossing_delay < 0.:
        parser.error("Crossing delay must be finite and nonnegative")
    model_sha = fingerprint(args.model)
    driver, names, schema = load_driver(args.model)
    if args.urban_cap and hasattr(driver, "allow_high_speed"):
        driver.allow_high_speed = False
    pace_schema = schema in ("ado-city-pace-v1", "ado-city-clearance-v1", "ado-city-progress-v1", "ado-city-maneuvers-v1")
    maneuver_driver = schema == "ado-city-maneuvers-v1"
    default_rig_and_planner = maneuver_driver or (args.serve and (pace_schema or schema == "ado-city-cruise-v1"))
    options = {key: True for key, value in (("surround", args.surround), ("tactical", args.tactical))
               if (default_rig_and_planner if value is None else value)}
    tactical_sha = None
    if options.get("tactical"):
        if schema == "ado-city-maneuvers-v1":
            from .maneuver_choice import MODEL as tactical_checkpoint
        else:
            from .overtake import MODEL as tactical_checkpoint
        tactical_sha = fingerprint(tactical_checkpoint)
    model = LoggedDriver(driver)
    live_populations = {"ado-city-clearance-v1": "city-life", "ado-city-progress-v1": "city-life",
                        "ado-city-maneuvers-v1": "city-maneuvers"}
    population_profile = args.population_profile or (
        live_populations.get(schema, "standard") if args.serve or maneuver_driver else "standard")
    destination = args.destination or ("market-loop" if maneuver_driver and population_profile == "city-maneuvers"
                                       else "market")
    scene_factory = partial(JourneyScenario, population_profile=population_profile,
                            barriers=pace_schema)
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
                live=live, realtime=bool(live), max_steps=args.max_steps, destination=destination, run_id=folder.name,
                scene_factory=scene_factory, crossing_delay=args.crossing_delay, **options)
    write_trace(folder / "trace.jsonl.gz", trace)
    if len(model.observations) != len(trace):
        raise RuntimeError("Scored frames and model observations differ")
    save_arrays(folder / "observations.npz", frame=[row["frame"] for row in trace],
                features=np.asarray(model.observations), feature_names=names,
                speed_request_mps=model.requested_speed,
                **({"posted_limit_kmh": model.posted_limits} if model.uses_posted_limit else {}))
    summary.update(producer=producer, model_sha256=model_sha, policy_schema=schema, population_profile=population_profile,
                   surround=bool(options.get("surround")), tactical=bool(options.get("tactical")),
                   tactical_model_sha256=tactical_sha)
    if schema == "ado-city-maneuvers-v1":
        summary["open_road_cruise_enabled"] = bool(driver.allow_high_speed)
    write_json(folder / "summary.json", summary)
    passed = sum(event["handled"] for event in summary["events"])
    print(f'{summary["outcome"]}: {summary["progress_m"]:.1f}/{summary["route_length_m"]:.1f} m, '
          f'{passed}/{len(summary["events"])} encounters handled', flush=True)


if __name__ == "__main__":
    main()
