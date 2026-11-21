"""Audit a frozen interaction study against the published v0.2 evidence."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

from . import experiment as ex, report
from .revision_benchmark import read_case
from .town01_interaction import ROOT


def rows(raw):
    return [json.loads(line) for line in gzip.decompress(raw).splitlines()]


def audit(output, archive):
    manifest = ex.read_json(output/"manifest.json")
    version = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    grouped = defaultdict(list)
    entries = []
    with zipfile.ZipFile(archive) as z:
        for group, pairs in manifest["schedule"].items():
            for seed, case in pairs:
                trial = ex.Trial(seed, case, manifest.get("policy","interaction-v04"), "heldout", group)
                folder = output/trial.trial_id
                old = {}
                if group != "fresh_challenge":
                    stage = "fresh" if seed >= 2000 else "regression"
                    prefix = f"{stage}/{seed}-{case}-candidate/"
                    raw_summary, raw_trace = z.read(prefix+"summary.json"), z.read(prefix+"trace.jsonl.gz")
                    check = json.loads(z.read(prefix+"checkpoint.json"))["sha256"]
                    for name, content in (("summary.json",raw_summary), ("trace.jsonl.gz",raw_trace)):
                        if hashlib.sha256(content).hexdigest() != check[name]:
                            raise ValueError(f"Invalid archived digest for {prefix}{name}")
                    old = json.loads(raw_summary)
                    previous = rows(raw_trace)
                summary = read_case(folder, version, trial)
                if summary is None:
                    raise ValueError(f"Missing {trial.trial_id}")
                trace = rows((folder/"trace.jsonl.gz").read_bytes())
                if (len(trace) != summary["frames"] or any(b["frame"] != a["frame"]+1 for a,b in zip(trace,trace[1:]))
                        or any(abs(r["sim_time"]-.1*i)>.00001 for i,r in enumerate(trace))
                        or trace[-1]["yield_violations"] != summary["yield_violations"]
                        or trace[-1]["red_light_violations"] != summary["red_light_violations"]
                        or max(r["collision_count"] for r in trace) != summary["collision_events"]):
                    raise ValueError(f"Trace and summary disagree for {trial.trial_id}")
                item = dict(seed=seed, case=case, clean=report.clean(summary), outcome=summary["outcome"],
                            duration_s=summary["duration_s"], yields=summary["yield_violations"],
                            reds=summary["red_light_violations"], contacts=summary["collision_events"],
                            min_reference_gap_m=summary["min_hazard_gap_m"], weather=summary["weather"],
                            event_triggered=summary["event_triggered"], event_cleared=summary["event_cleared"],
                            route_id=summary["route_id"], interventions=dict(Counter(
                                r["planner"]["cue"] for r in trace
                                if r["planner"]["target_mps"] < r["planner"]["base_target_mps"]-.01)))
                if old:
                    settings = ("district","crossing_delay_s","start_xy","background_actors","sensor_configuration","producer","render_profile")
                    item["matched"] = report.matching_setup(old, summary) and all(old[k]==summary[k] for k in settings)
                    item["v02"] = dict(clean=report.clean(old), outcome=old["outcome"], duration_s=old["duration_s"],
                                       contacts=old["collision_events"], yields=old["yield_violations"],
                                       min_reference_gap_m=old["min_hazard_gap_m"])
                    failures = [r for r in previous if r["yield_violations"] or r["collision_count"]]
                    if failures:
                        first = failures[0]
                        item["v02_first_event"] = {k:first[k] for k in ("sim_time","phase","speed_mps","brake","throttle")}
                        clear = next((r["sim_time"] for r in previous if r["event_cleared"]), None)
                        if clear is not None:
                            item["v02_first_event"]["seconds_before_clear"] = round(clear-first["sim_time"],2)
                entries.append(dict(group=group, **item))
                grouped[group].append(item)
    return dict(study=version, v02_archive_sha256=ex.fingerprint(archive), cohort={g:dict(n=len(v), clean=sum(x["clean"] for x in v),
                contacts=sum(x["contacts"] for x in v), yields=sum(x["yields"] for x in v),
                reds=sum(x["reds"] for x in v), routes=len({x["route_id"] for x in v})) for g,v in grouped.items()},
                archived_setups_matched=sum(x.get("matched",False) for x in entries),
                archived_setups_compared=sum("matched" in x for x in entries), drives=entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/town01-yield-v04")
    parser.add_argument("--archive",type=Path,default=ROOT/"results/behavior-v2/closed-loop-evidence.zip")
    args = parser.parse_args()
    result = audit(args.output,args.archive)
    ex.write_json(args.output/"audit.json",result)
    print(json.dumps(result["cohort"],indent=2))
