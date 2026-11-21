"""Score native closed-loop roadside pace at identical course stations."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def measure(folder):
    summary = json.loads((folder / "summary.json").read_bytes())
    with gzip.open(folder / "trace.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    result = {k: summary[k] for k in ("outcome", "frames", "duration_s", "progress_m", "collision_events",
              "red_light_violations", "yield_violations", "moving_within_1m_frames")}
    events = {e["kind"]: e for e in summary.get("events", [])}
    for label, fields in {"roadworks": ("handled", "cone_count", "minimum_cone_clearance_m", "max_cone_displacement_m"),
                          "occluded": ("handled", "stopped_for_event", "minimum_gap_m"),
                          "barrier": ("handled", "minimum_barrier_clearance_m"),
                          "lead": ("handled", "stopped_for_event", "minimum_gap_m")}.items():
        if label in events and events[label]["exposed"]:
            result[label] = {k: events[label].get(k) for k in fields}
    if events:
        result["encounters_handled"] = sum(e["handled"] for e in events.values())
        result["junctions_completed"] = sum(summary["turns_completed"].values())
    for key in ("longest_stop_s", "minimum_stopped_gap_m", "intersects_board"):
        if key in summary:
            result[key] = summary[key]
    if not events or "roadworks" not in result:
        return result
    position = np.asarray([r["progress_m"] for r in rows])
    speed = np.asarray([r["speed_mps"] * 3.6 for r in rows])
    request = np.asarray([r["speed_request_mps"] * 3.6 for r in rows])
    signed = np.asarray([r["posted_limit_kmh"] for r in rows])
    observations = folder / "observations.npz"
    if observations.exists():
        with np.load(observations) as data:
            assert np.array_equal(data["frame"], [r["frame"] for r in rows])
            X = data["features"]
            ix = {s: i for i, s in enumerate(data["feature_names"])}
    first = events["roadworks"]["station_m"]
    # Six cone locations lie at 0, 4, 8, 12, 16 and 20 m. The buffer after
    # the last cone is scored separately to avoid mixing recovery into passage.
    for label, lo, hi in (("alongside", first, first+20.), ("recovery", first+20., first+45.),
                          ("open_boulevard", first+45., first+125.)):
        use = (position >= lo) & (position < hi)
        if not use.any() or position[0] > lo or position[-1] < hi:
            continue
        selected = [r for r, keep in zip(rows, use) if keep]
        controls = [r.get("commanded_control", r) for r in selected]
        metrics = dict(stations_m=[round(lo, 2), round(hi, 2)], seconds=round(.1 * sum(map(int, use)), 1),
                       speed_min_kmh=round(float(min(speed[use])), 2),
                       median_speed_kmh=round(float(np.median(speed[use])), 2),
                       median_request_kmh=round(float(np.median(request[use])), 2),
                       peak_speed_kmh=round(float(max(speed[use])), 2),
                       posted_limits_kmh=sorted(set(float(n) for n in signed[use])),
                       seconds_below_8_kmh=round(.1*int(np.count_nonzero(speed[use] < 8.)), 1),
                       brake_above_005_s=round(.1*sum(r["brake"]>.05 for r in controls), 1),
                       median_commanded_throttle=round(float(np.median([r["throttle"] for r in controls])), 3))
        if observations.exists():
            side = np.min(X[use][:, [ix[n] for n in ("left_edge_lidar", "right_edge_lidar")]], axis=1)
            inner = np.min(X[use][:, [ix[n] for n in ("path_body_lidar", "path_body_radar")]], axis=1)
            metrics.update(median_body_range_m=round(float(np.median(inner)), 2),
                           median_edge_range_m=round(float(np.median(side)), 2))
            if "open_edge_margin_m" in ix:
                metrics["open_edge_evidence_s"] = round(.1*int(np.count_nonzero(X[use, ix["open_edge_margin_m"]] >= 0.)), 1)
        result[label] = metrics
    result["seconds_measured_above_posted_plus_1"] = round(.1*int(np.count_nonzero(speed > signed+1.)), 1)
    result["max_requested_excess_above_posted_kmh"] = round(float(np.maximum(request-signed, 0.).max()), 3)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folders", type=Path, nargs="+")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    data = {str(f).replace("\\", "/"): measure(f) for f in a.folders}
    output = json.dumps(data, indent=2) + "\n"
    if a.output:
        a.output.write_text(output, encoding="utf-8")
    else:
        print(output)
