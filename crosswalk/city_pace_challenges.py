"""Run two short, prespecified urban conflict challenges against a frozen city pace checkpoint."""
import argparse
from pathlib import Path

import numpy as np

from .city_drive import LoggedDriver, load_driver
from .drive import episode
from .experiment import PROJECT, fingerprint, save_arrays, write_json, write_trace
from .journey import JourneyScenario
from .runtime import producer_info, simulator
from .world import _raised


# All stations and timing changes belong to test setup; the policy receives its normal sensors.
CHALLENGES = (
    ("late-entry", 314, 0., .75, ("pedestrian", "occluded")),
    ("late-route", 315, 925., .75, ("braking", "left", "cyclist", "pedestrian-delayed")),
)


def run(world, checkpoint, output, case):
    label, seed, start_m, delay, targets = case
    folder = output / label
    folder.mkdir(parents=True, exist_ok=False)
    driver, names, schema = load_driver(checkpoint)
    model = LoggedDriver(driver)
    scenes = []

    def make_scene(*args, **kwargs):
        scene = JourneyScenario(*args, **kwargs, barriers=True)
        if start_m:
            index = int(np.argmin(abs(np.asarray(scene._station) - start_m)))
            scene.ego.set_transform(_raised(scene._points[index].transform, .35))
        scenes.append(scene)
        return scene

    # End when every predeclared exposure has been passed, including any failed exposure.
    # Keep the original episode grade as well as the independently verified target grades.
    def finished():
        return all("passed_s" in event.observed for event in scenes[0].events if event.kind in targets)

    summary, trace, _, _ = episode(world, "journey", seed, schema, model, max_steps=2200,
        destination="market", scene_factory=make_scene, crossing_delay=delay, tactical=True, change_request=finished)
    observed = {e["kind"]: e for e in summary["events"] if e["kind"] in targets}
    clean = (summary["collision_events"] == 0 and summary["red_light_violations"] == 0 and
             summary["yield_violations"] == 0 and all(observed[k]["handled"] for k in targets))
    summary.update(test_case=label, departure_m=start_m, target_events=list(targets), target_pass=clean,
                   policy_schema=schema, model_sha256=fingerprint(checkpoint), producer=producer_info(world))
    write_trace(folder / "trace.jsonl.gz", trace)
    write_json(folder / "summary.json", summary)
    save_arrays(folder / "observations.npz", features=model.observations, feature_names=names,
                frame=[r["frame"] for r in trace], speed_request_mps=model.requested_speed,
                posted_limit_kmh=model.posted_limits)
    print(label, clean, summary["progress_m"], {k:r["handled"] for k,r in observed.items()}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=PROJECT / "models/city-pace/candidate.npz")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=2968)
    p.add_argument("--attach", action="store_true")
    args = p.parse_args()
    with simulator(args.port, args.attach, "Low", map_name="Town03_Opt") as world:
        for case in CHALLENGES:
            run(world, args.model, args.output, case)


if __name__ == "__main__":
    main()
