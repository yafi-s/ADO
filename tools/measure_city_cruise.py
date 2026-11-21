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
        staged = (folder/"capture.json").is_file()
        manifest = json.loads((folder/("capture.json" if staged else "manifest.json")).read_bytes())
        summary = manifest["summary"]
        telemetry = folder/"raw/telemetry.json" if staged else folder/"telemetry.json"
        trace = json.loads(telemetry.read_bytes())
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
    time = np.array([r["sim_time"] for r in trace])
    if not np.allclose(np.diff(time), duration, atol=.001):
        raise ValueError("Expected uninterrupted, evenly spaced native velocity measurements")

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
        in_band = mask & (speed >= 35.) & (speed <= 40.)
        spans = np.flatnonzero(np.diff(np.r_[False, in_band, False])).reshape(-1, 2)
        same_section = mask[1:] & mask[:-1]
        changes = np.diff(speed)[same_section]
        return dict(frames=len(speeds), seconds=round(len(speeds)*duration, 1),
                    p10_kmh=round(float(np.quantile(speeds, .1)), 2),
                    median_kmh=round(float(np.median(speeds)), 2),
                    p90_kmh=round(float(np.quantile(speeds, .90)), 2),
                    peak_kmh=round(float(np.max(speeds)), 2),
                    sd_kmh=round(float(np.std(speeds)), 3),
                    variance_kmh2=round(float(np.var(speeds)), 3),
                    mean_absolute_change_kmh_per_s=round(float(np.mean(abs(changes)) / duration), 3) if len(changes) else None,
                    seconds_35_to_40_kmh=round(float(in_band.sum()*duration), 1),
                    longest_continuous_seconds_35_to_40_kmh=round(float(max((b-a for a,b in spans), default=0)*duration), 1),
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
    # Common physical interval registered on the v0.3 route before testing v0.4.
    # It omits roadworks departure and the approach to the new roadside boards.
    sites = {e["kind"]: e["station_m"] for e in summary["events"]}
    if "roadworks" in sites and "lead" in sites:
        start, end = sites["roadworks"] + 45., sites["lead"] - 75.
        section = (station >= start) & (station <= end)
        result["matched_boulevard_interior"] = dict(
            from_m=round(start, 2), to_m=round(end, 2),
            active_encounter_frames=int(np.count_nonzero(section & np.array([bool(r.get("active_event")) for r in trace]))),
            **stats(section))
    for event in summary["events"]:
        row = result["events"][event["kind"]]
        row.update(exposed=event.get("exposed", False), passed=event.get("passed", False),
                   violation=event.get("violation", False), minimum_gap_m=event.get("minimum_gap_m"))
        for name in ("minimum_barrier_clearance_m", "minimum_cone_clearance_m", "minimum_bus_clearance_m",
                     "peak_lead_deceleration_mps2"):
            if name in event:
                row[name] = event[name]
        if row["stopped"] and "cleared_s" in event and "passed_s" in event:
            # A zero delay means the vehicle was already rolling at the first clear frame.
            resumed = np.flatnonzero((time >= event["cleared_s"]) & (time <= event["passed_s"]) & (speed >= 5.))
            row["seconds_from_clear_to_5_kmh"] = round(float(time[resumed[0]] - event["cleared_s"]), 2) if len(resumed) else None
    limits = None
    requests = None
    if all("posted_limit_kmh" in r for r in trace):
        limits = np.asarray([r["posted_limit_kmh"] for r in trace], dtype=float)
    if all("speed_request_mps" in r for r in trace):
        requests = np.asarray([r["speed_request_mps"] for r in trace], dtype=float)
    archive = folder/"observations.npz"
    if archive.is_file():
        with np.load(archive, allow_pickle=False) as data:
            if "posted_limit_kmh" in data.files:
                np.testing.assert_array_equal(data["frame"], [r["frame"] for r in trace])
                if limits is not None:
                    np.testing.assert_allclose(data["posted_limit_kmh"], limits, atol=.01)
                limits = data["posted_limit_kmh"]
                if requests is not None:
                    np.testing.assert_allclose(data["speed_request_mps"], requests, atol=.001)
                requests = data["speed_request_mps"]
    if limits is not None:
        if requests is not None:
            result["max_request_above_posted_kmh"] = round(float(np.max(requests*3.6-limits)), 3)
        result["native_limit_frames"] = {str(v): int(np.sum(limits == v)) for v in np.unique(limits)}
        result["permissive_clear_straights"] = stats(open_road & (limits >= 35.))
        result["seconds_measured_above_posted_plus_1_kmh"] = round(
            int(np.count_nonzero(speed > limits+1.))*duration, 1)
        result["seconds_measured_above_posted"] = round(int(np.count_nonzero(speed > limits))*duration, 1)
        result["max_measured_excess_above_posted_kmh"] = round(float(max(0., np.max(speed-limits))), 3)
    # Report native speed differences and a half-second mean, retaining urgent braking.
    window = max(1, round(.5/duration))
    accel = np.diff(speed/3.6) / duration
    averaged = (speed[window:] - speed[:-window]) / (window * duration * 3.6)
    if len(averaged):
        strongest = int(np.argmin(averaged)) + window
        result["longitudinal_comfort"] = dict(
            raw_peak_deceleration_mps2=round(float(max(0., -accel.min())), 3),
            half_second_peak_deceleration_mps2=round(float(max(0., -averaged.min())), 3),
            half_second_peak_acceleration_mps2=round(float(max(0., averaged.max())), 3),
            raw_seconds_braking_harder_than_3_mps2=round(float((accel < -3.).sum()*duration),1),
            clear_straight_seconds_braking_harder_than_3_mps2=round(float(
                ((accel < -3.) & open_road[:-1] & open_road[1:]).sum() * duration), 1),
            half_second_seconds_braking_harder_than_3_mps2=round(float((averaged < -3.).sum()*duration),1),
            strongest_half_second_end=dict(sim_time_s=float(time[strongest]), progress_m=float(station[strongest]),
                                           active_event=trace[strongest].get("active_event")),
            moving_p95_abs_half_second_acceleration_mps2=round(float(np.quantile(abs(averaged[speed[window:]>1.]),.95)),3)
            if (speed[window:]>1.).any() else None)
    starts = np.flatnonzero(speed >= 10.)
    result["first_10_kmh_at_s"] = round(float(time[starts[0]]), 2) if len(starts) else None
    result["seconds_measured_above_40_kmh"] = round(float((speed>40.).sum()*duration),1)
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
