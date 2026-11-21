"""Run the normal Town01 interactive entry point and retain its two startup-trip traces."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch
import zipfile

from crosswalk import drive, experiment as ex, report, runtime
from crosswalk.sensors import render_profile
from crosswalk.town01_interaction_gate import MODEL, VERSION, GatedInteractionModel
from crosswalk.town01_interaction import ROOT, digest


MATCH_KEYS = ("district","crossing_delay_s","start_xy","background_actors","sensor_configuration","producer","render_profile")
METRICS = ("outcome","frames","duration_s","progress_m","route_length_m","collision_events","yield_violations","red_light_violations")


def matches(previous, score):
    return report.matching_setup(previous,score) and all(previous[k] == score[k] for k in MATCH_KEYS)


def historical_fusion(score):
    source = "results/development-baseline/episodes.json" if score["district"] == "train" else "results/primary.json"
    earlier = next(r for r in ex.read_json(ROOT/source) if r["seed"] == score["seed"]
                   and r["scenario"] == "trip" and r["policy"] == "fusion")
    return dict(source=source, matched=matches(earlier,score), clean=report.clean(earlier),
                **{k:earlier[k] for k in METRICS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    summaries = []
    with zipfile.ZipFile(ROOT/"results/behavior-v2/closed-loop-evidence.zip") as z:
        market = json.loads(z.read("regression/103-trip-candidate/summary.json"))
    station = ex.read_json(ROOT/"results/city-review/station-32/summary.json")
    for dest, seed, district, previous in [("station",32,"train",station),("market",103,"heldout",market)]:
        folder = args.output/f"{seed}-{dest}"
        folder.mkdir(parents=True,exist_ok=True)
        actual = drive.episode
        def record(world, kind, trial_seed, policy, model, **kwargs):
            live = kwargs["live"]
            try:
                if policy != VERSION or not isinstance(model,GatedInteractionModel):
                    raise RuntimeError("The launcher selected an unexpected policy")
                row, trace, x, y = actual(world,kind,trial_seed,policy,model,**kwargs)
                if live.frame is None or not kwargs["realtime"] or kwargs["change_request"] is None:
                    raise RuntimeError("Interactive frames were not delivered")
                row.update(producer=runtime.producer_info(world),render_profile=render_profile())
                ex.write_json(folder/"summary.json",row)
                ex.write_trace(folder/"trace.jsonl.gz",trace)
                score = ex.read_json(folder/"summary.json")
                summaries.append(dict(seed=seed,destination=dest,district=district,matched=matches(previous,score),
                    clean=report.clean(score),baseline_clean=report.clean(previous),
                    gated={k:score[k] for k in METRICS},v02={k:previous[k] for k in METRICS},
                    older_fusion=historical_fusion(score),
                    published_final_frame=True,model_sha256=digest(MODEL),
                    trace_sha256=ex.fingerprint(folder/"trace.jsonl.gz"),summary_sha256=ex.fingerprint(folder/"summary.json")))
                ex.write_json(args.output/"comparison.json",summaries)
                return row,trace,x,y
            finally:
                # Normal interactive shutdown after the scored episode.
                live.request_stop()
        argv = ["ado drive","--attach","--port",str(args.port),"--serve","--interactive","--http-port","0",
                "--policy",VERSION,"--scenario","trip","--destination",dest,"--seed",str(seed),"--district",district]
        with patch.object(sys,"argv",argv), patch.object(drive,"episode",record):
            drive.main()
        if len(summaries) != (1 if dest=="station" else 2):
            raise RuntimeError(f"Missing launcher trace for {dest}")


if __name__=="__main__":
    main()
