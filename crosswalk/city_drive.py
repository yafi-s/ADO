"""Run one Town03 journey with the learned driver and retain its measured outcome."""
import argparse
from contextlib import nullcontext
from pathlib import Path
from tempfile import mkdtemp
import webbrowser

from .behavior_v2 import BehaviorModel
from .city_map import geometry
from .drive import episode
from .experiment import PROJECT, fingerprint, write_json, write_trace
from .journey import JourneyScenario
from .runtime import producer_info, simulator
from .viewio import Live


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT / "models/behavior-v2/candidate.npz")
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
    model = BehaviorModel.load(args.model)
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
            summary, trace, _, _ = episode(world, "journey", args.seed, "behavior-v2", model,
                live=live, realtime=bool(live), max_steps=args.max_steps, destination="market", run_id=folder.name,
                scene_factory=JourneyScenario)
    write_trace(folder / "trace.jsonl.gz", trace)
    summary.update(producer=producer, model_sha256=fingerprint(args.model))
    write_json(folder / "summary.json", summary)
    passed = sum(event["handled"] for event in summary["events"])
    print(f'{summary["outcome"]}: {summary["progress_m"]:.1f}/{summary["route_length_m"]:.1f} m, '
          f'{passed}/{len(summary["events"])} encounters handled', flush=True)


if __name__ == "__main__":
    main()
