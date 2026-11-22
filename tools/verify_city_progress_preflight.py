"""Grade a complete alternate-seed populated drive and locate every sustained standstill."""
import argparse
from collections import Counter
import gzip
from itertools import groupby
import json
from pathlib import Path

from crosswalk.experiment import fingerprint


def grade(folder, checkpoint):
    summary = json.loads((folder / "summary.json").read_bytes())
    with gzip.open(folder / "trace.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    stops = []
    for stopped, group in groupby(rows, lambda row: row["speed_mps"] < .15):
        batch = list(group)
        if stopped and len(batch) >= 10:
            stops.append(dict(seconds=round(.1*len(batch), 1), first_frame=batch[0]["frame"],
                station_m=batch[0]["progress_m"], phases=dict(Counter(r["phase"] for r in batch)),
                constraints=dict(Counter(r.get("progress_evidence", {}).get("binding", "legacy policy") for r in batch))))
    activity = summary["background_traffic_activity"]
    checksum = fingerprint(checkpoint)
    conditions = (summary["seed"] == 213 and summary["crossing_delay_s"] == .75 and
                  summary["population_profile"] == "city-life" and not summary["surround"] and summary["tactical"] and
                  summary["sensor_configuration"] == "three-camera-lidar-radar" and checksum == summary["model_sha256"])
    hazards = {e["kind"]: {k: e[k] for k in ("handled", "passed", "exposed", "violation", "minimum_gap_m")}
               for e in summary["events"]}
    junctions = [dict(id=j["junction_id"], direction=j["direction"], completed="completed_s" in j)
                 for j in summary["junction_traversals"]]
    clean = bool(conditions and len(rows) == summary["frames"] and summary["outcome"] == "complete"
        and summary["progress_m"] >= summary["route_length_m"]-.6 and len(hazards) == 11
        and all(e["handled"] and e["exposed"] and e["passed"] and not e["violation"] for e in hazards.values())
        and len(junctions) == 11 and all(j["completed"] for j in junctions)
        and not activity["participant_contacts"] and not any(v["left_corridor"] for v in activity["vehicles"])
        and not any(summary[k] for k in ("collision_events", "red_light_violations", "yield_violations")))
    result = dict(passed=clean, model_sha256=checksum, seed=summary["seed"], crossing_delay_s=summary["crossing_delay_s"],
        producer=summary["producer"], population=summary["background_actors"], duration_s=summary["duration_s"],
        progress_m=summary["progress_m"], route_length_m=summary["route_length_m"], frames=len(rows),
        outcome=summary["outcome"], encounters=hazards, junctions=junctions, standstills_below_015_mps=stops,
        participant_contacts=activity["participant_contacts"], collisions=summary["collision_events"],
        missed_yields=summary["yield_violations"], red_violations=summary["red_light_violations"],
        trace_sha256=fingerprint(folder / "trace.jsonl.gz"), features_sha256=fingerprint(folder / "observations.npz"))
    out = folder / "grade.json"
    if out.exists() and json.loads(out.read_bytes()) != result:
        raise ValueError("Preflight evidence changed")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("checkpoint", type=Path)
    args = p.parse_args()
    result = grade(args.folder, args.checkpoint)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
