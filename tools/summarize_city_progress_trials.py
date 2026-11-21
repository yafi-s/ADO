"""Summarize matched native driving trials without dropping an unsuccessful safety check."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


CASES = ("through", "bus", "bus-crossing", "adjacent", "adjacent-camera", "no-lead", "departing-lead",
         "left", "occluded", "delayed", "signal", "reopened-board")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(root, roots, case):
    candidates = [root / base / case for base in roots if (root / base / case / "summary.json").is_file()]
    if not candidates:
        raise ValueError(f"No source for {case} among {roots}")
    # The caller supplies a frozen priority order (e.g. the corrected physical fixture before the pilot).
    folder = candidates[0]
    path = folder / "summary.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    data = dict(directory=folder.relative_to(root).as_posix(), summary_sha256=digest(path),
                model_sha256=raw["model_sha256"], producer=raw["producer"])
    if case.startswith("adjacent") or case in ("no-lead", "departing-lead"):
        data.update(raw["score"])
        data["geometry"] = raw["geometry"]
    else:
        data.update({k: raw[k] for k in ("frames", "outcome", "collision_events", "red_light_violations",
                                         "yield_violations", "duration_s", "seed", "progress_m")})
        if case == "reopened-board":
            data.update({k: raw[k] for k in ("longest_stop_s", "minimum_stopped_gap_m", "intersects_board")})
            data["passed"] = bool(raw["outcome"] == "complete" and not raw["collision_events"] and
                                  not raw["intersects_board"] and raw["longest_stop_s"] >= 1. and
                                  raw["minimum_stopped_gap_m"] > 0)
            with gzip.open(folder / "trace.jsonl.gz", "rt", encoding="utf-8") as stream:
                trace = [json.loads(line) for line in stream]
            stopped = next(i for i, r in enumerate(trace) if r["progress_m"] > 5 and r["speed_mps"] < .3
                           and r["board_in_ego_path"] and r["bumper_gap_m"] < 20)
            clear_index = next(i for i in range(stopped, len(trace)) if not trace[i]["board_in_ego_path"])
            release = trace[clear_index]
            data["path_reopening"] = dict(time_s=release["sim_time"], frame=release["frame"],
                seconds_to_kmh={speed: next((round(r["sim_time"]-release["sim_time"], 2) for r in trace[clear_index:]
                                           if r["speed_mps"]*3.6 >= speed), None) for speed in (5, 10, 15, 20)})
        else:
            data.update(passed=raw["target_pass"], windows=raw["windows"], target_event=raw["target_event"],
                        events=[e for e in raw["events"] if e["kind"] == raw["target_event"]],
                        nearest_target_footprint=raw.get("nearest_target_footprint"),
                        source_front_sha256=digest(folder / "front.mp4"))
            if "crossing" in raw:
                data["crossing"] = raw["crossing"]
    return data


def summarize(root, before, after, output):
    results = []
    for case in CASES:
        old, new = (load(root, arms, case) for arms in (before, after))
        if old["producer"] != new["producer"] or old.get("seed") != new.get("seed"):
            raise ValueError(f"Mismatched simulation settings in {case}")
        pair = dict(case=case, before=old, after=new)
        if "windows" in new:
            differences = []
            for a, b in zip(old["windows"], new["windows"], strict=True):
                if [a["start_m"], a["end_m"]] != [b["start_m"], b["end_m"]]:
                    raise ValueError(f"Different geography in {case}")
                differences.append(dict(start_m=a["start_m"], end_m=a["end_m"],
                    seconds_saved=round(a["seconds"]-b["seconds"], 2),
                    median_change_kmh=round(b["median_kmh"]-a["median_kmh"], 2)
                                      if a["frames"] and b["frames"] else None))
            pair["differences"] = differences
        results.append(pair)
    result = dict(physics="CARLA 1.0 Town03_Opt, High, fixed 0.1 s", cases=results,
                  passed=all(pair["after"]["passed"] for pair in results))
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--before", nargs="+", required=True)
    p.add_argument("--after", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    summarize(a.root, a.before, a.after, a.output)
