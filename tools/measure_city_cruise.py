"""Measure achieved speeds from a scored city drive and aligned native posted limits."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def measure(folder):
    if (folder/"summary.json").exists():
        summary = json.loads((folder/"summary.json").read_bytes())
        with gzip.open(folder/"trace.jsonl.gz", "rt", encoding="utf-8") as f:
            trace = [json.loads(line) for line in f]
    else:
        manifest = json.loads((folder/"manifest.json").read_bytes())
        summary = manifest["summary"]
        trace = json.loads((folder/"telemetry.json").read_bytes())
    speed = np.array([r["speed_mps"]*3.6 for r in trace])
    station = np.array([r["progress_m"] for r in trace])
    open_road = np.array([not r.get("active_event") and r["reference_gap_m"] >= 35. for r in trace])
    turns = np.zeros(len(trace), bool)
    for item in summary["junction_traversals"]:
        a, b = item["entry_m"], item["exit_m"]
        open_road &= (station < a-25.) | (station > b+8.)
        if item["direction"] in ("left", "right"):
            turns |= (station >= a) & (station <= b)
    duration = summary["duration_s"]/len(trace)

    def stats(mask):
        speeds = speed[mask]
        if not len(speeds):
            return {"frames": 0}
        fast = mask & (speed >= 35.)
        endpoints = np.flatnonzero(np.diff(np.r_[False, fast, False])).reshape(-1, 2)
        longest = max((b-a for a, b in endpoints), default=0)
        interval = None
        if longest:
            start, end = next((a,b) for a,b in endpoints if b-a == longest)
            interval = {"from_m": round(station[start], 1), "to_m": round(station[end-1], 1),
                        "from_s": trace[start]["sim_time"], "to_s": trace[end-1]["sim_time"]}
        return dict(frames=len(speeds), seconds=round(len(speeds)*duration, 1),
                    median_kmh=round(float(np.median(speeds)), 2),
                    p90_kmh=round(float(np.quantile(speeds, .90)), 2),
                    peak_kmh=round(float(np.max(speeds)), 2),
                    seconds_at_or_above_35=round(int(fast.sum())*duration, 1),
                    longest_continuous_seconds_at_or_above_35=round(longest*duration, 1),
                    longest_interval=interval)

    result = dict(seed=summary["seed"], frames=len(trace), duration_s=summary["duration_s"],
                  outcome=summary["outcome"], collision_events=summary["collision_events"],
                  progress_m=summary["progress_m"], route_length_m=summary["route_length_m"],
                  red_light_violations=summary["red_light_violations"],
                  yield_violations=summary["yield_violations"],
                  moving_within_1m_frames=summary["moving_within_1m_frames"],
                  turns_completed=summary["turns_completed"], clear_straights=stats(open_road),
                  inside_left_and_right_turns=stats(turns), whole_drive=stats(np.ones(len(trace), bool)),
                  events={e["kind"]: dict(handled=e["handled"], stopped=e["stopped_for_event"])
                          for e in summary["events"]})
    archive = folder/"observations.npz"
    if archive.is_file():
        with np.load(archive, allow_pickle=False) as data:
            if "posted_limit_kmh" in data.files:
                np.testing.assert_array_equal(data["frame"], [r["frame"] for r in trace])
                limits = data["posted_limit_kmh"]
                result["native_limit_frames"] = {str(k): int((limits==k).sum()) for k in np.unique(limits)}
                result["permissive_clear_straights"] = stats(open_road & (limits >= 35.))
                result["max_request_above_posted_kmh"] = round(float(np.max(
                    data["speed_request_mps"]*3.6-limits)), 3)
                result["seconds_measured_above_posted_plus_1_kmh"] = round(
                    int(np.count_nonzero(speed > limits+1.))*duration, 1)
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("folder", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    content = json.dumps(measure(args.folder), indent=2)+"\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content)
