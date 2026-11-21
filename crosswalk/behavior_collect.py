"""Record synchronized sensor observations before each demonstrated actuator command."""
import argparse
import json
from pathlib import Path

import numpy as np

from .behavior_v2 import FEATURE_NAMES, Observer
from .drive import episode
from .runtime import producer_info, simulator


def collect(world, instructor, output, seeds):
    output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for case in ("occluded", "left", "pedestrian-delayed", "turn", "clear", "lead", "pedestrian", "trip"):
            trial = f"{case}-{seed}"
            path = output / f"{trial}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as sample:
                    if list(sample["feature_names"]) != list(FEATURE_NAMES):
                        raise ValueError(f"Incompatible observation columns in {path}")
                continue
            kind = case.removesuffix("-delayed")
            print("Starting", trial, flush=True)
            summary, trace, x, y = episode(world, kind, seed, "demonstrator-v2", teacher=instructor,
                district="train", destination="market" if seed % 2 else "station",
                crossing_delay=1.0 if case.endswith("-delayed") else 0., observation_factory=Observer)
            if len(x) != len(trace)-1 or len(x) != len(y):
                raise RuntimeError(f"Demonstration timestamps disagree for {trial}")
            summary.update(case=case, producer=producer_info(world))
            (output / f"{trial}.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
            rows = trace[:-1]
            temp = path.with_suffix(".partial")
            with temp.open("wb") as stream:
                np.savez_compressed(stream, X=x, y=y, feature_names=FEATURE_NAMES,
                    seed=np.full(len(y),seed), scenario=np.full(len(y),kind), episode_id=np.full(len(y),trial),
                    expert_gap_m=[r["reference_gap_m"] for r in rows], progress_m=[r["progress_m"] for r in rows],
                    teacher_speed=[r["instructor_speed_mps"] for r in trace[1:]],
                    crossing_delay=np.full(len(y),summary["crossing_delay_s"]),
                    route_xy=np.asarray(summary["route_xy"],dtype=np.float32))
            temp.replace(path)
            print(trial, summary["outcome"], len(y), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--port",type=int,default=2868)
    cli.add_argument("--attach",action="store_true")
    cli.add_argument("--output",type=Path,default=Path("out/behavior-v2-lessons"))
    cli.add_argument("--seeds",type=int,nargs="+",default=[43,44,47,48])
    args=cli.parse_args()
    with simulator(port=args.port,attach=args.attach,with_manager=True) as (world,tm):
        collect(world,tm,args.output,args.seeds)


if __name__ == "__main__":
    main()
