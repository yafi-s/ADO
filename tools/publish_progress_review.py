"""Publish the final station-matched pace audit alongside the recorded drives."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/city-progress"
TAG = "https://github.com/yafi-s/ADO/blob/v0.6.0/results/city-progress/"


def read(path):
    return json.loads(path.read_bytes())


def verified_run(name, audit_file):
    manifest = read(ROOT / "web/media" / name / "manifest.json")
    report = read(STUDY / "audit" / audit_file)
    summary = manifest["summary"]
    if (not manifest.get("passed") or manifest["trial_id"] != name
            or report["source"]["policy_sha256"] != manifest["policy"]["sha256"]
            or report["source"]["telemetry_sha256"] != manifest["sha256"]["telemetry.json"]
            or report["grade"]["duration_s"] != summary["duration_s"]
            or report["whole"]["frames"] != summary["frames"]):
        raise ValueError(f"Recorded drive and pace audit disagree for {name}")
    return manifest, report


def section(report, start, end):
    span, = (row for row in report["sections"] if row["from_m"] == start and row["to_m"] == end)
    return span


def camera(run, frame, view="front"):
    return f"./?run={run}&frame={frame}&camera={view}"


def create():
    _, old = verified_run("city-life", "v05.json")
    manifest, new = verified_run("city-progress", "v06.json")
    for field in ("route", "map", "seed", "population_profile", "crossing_delay_s", "fixed_delta_seconds"):
        if old["source"][field] != new["source"][field]:
            raise ValueError(f"Route comparison differs in {field}")
    checks = []

    def add(title, result, *links):
        checks.append({"test": title, "result": result,
                       "evidence": [{"label": label, "url": url} for label, url in links]})

    grade = new["grade"]
    events = manifest["summary"]["events"]
    add("Populated route",
        f'{grade["progress_m"]/1000:.2f} km in {grade["duration_s"]:.1f} s, previously {old["grade"]["duration_s"]:.1f} s. '
        f'{sum(bool(e["handled"]) for e in events)}/{len(events)} encounters handled. '
        f'{grade["collision_events"]} ego contacts, {grade["red_light_violations"]} red-light violations '
        f'and {grade["yield_violations"]} missed yields.',
        ("Watch the drive", "./?run=city-progress"), ("All measurements", TAG+"audit/README.md"))
    a, b = (run["near_straight_junctions"] for run in (old, new))
    if sorted(a["junction_ids"]) != sorted(b["junction_ids"]):
        raise ValueError("Straight junction groups differ across the two drives")
    add("Straight junctions",
        f'Across the same {len(a["junction_ids"])} nearly straight junctions, measured median speed was '
        f'{a["median_kmh"]:.1f} km/h before and {b["median_kmh"]:.1f} km/h after. '
        f'Total time inside them was {a["seconds"]:.1f} s before and {b["seconds"]:.1f} s after.',
        ("Every junction", TAG+"audit/README.md"), ("Test protocol", TAG+"audit/PROTOCOL.md"))
    heldout = read(STUDY / "junction-transfer/transfer-r3-first/audit.json")
    arms = ("candidate-clear", "candidate-crossing", "candidate-stops-short")
    if any(heldout[arm]["policy_sha256"] != manifest["policy"]["sha256"] for arm in arms):
        raise ValueError("Off-route comparison used a different checkpoint")
    old_approach = heldout["baseline-clear"]["exact_station_speeds"]["-12.0"]["actual_kmh"]
    new_approach = heldout["candidate-clear"]["exact_station_speeds"]["-12.0"]["actual_kmh"]
    crossing = heldout["candidate-crossing"]["grade"]
    short = heldout["candidate-stops-short"]["grade"]
    add("Cross traffic on a different road",
        f'On the empty approach, speed 12 m before the conflict rose from {old_approach:.1f} to '
        f'{new_approach:.1f} km/h. When a Mini crossed, ADO yielded, leaving '
        f'{crossing["physical_min_body_gap_m"]:.2f} m minimum clearance. When the Mini stopped short, '
        f'ADO continued at a minimum of {short["near_center_min_speed_kmh"]:.1f} km/h near the crossing.',
        ("Matched camera evidence", TAG+"junction-transfer/README.md"),
        ("Five trials", TAG+"junction-transfer/transfer-r3-first/audit.json"))
    for title, lo, hi, view in (("Beside the bus", 858, 872, "chase"), ("Past the bus", 872, 882, "front")):
        a, b = section(old, lo, hi), section(new, lo, hi)
        result = (f'At {lo}–{hi} m along the route, the car drove at a median '
                  f'{a["median_kmh"]:.1f} km/h before and {b["median_kmh"]:.1f} km/h after. '
                  f'Elapsed time was {a["seconds"]:.1f} s before and {b["seconds"]:.1f} s after.')
        add(title, result, ("Earlier recording", camera("city-life", a["middle_frame"], view)),
            ("Revised recording", camera("city-progress", b["middle_frame"], view)))
    a, b = old["whole"]["below_kmh_s"], new["whole"]["below_kmh_s"]
    add("Slow sections across town",
        f'Time below 15 km/h across the full trip changed from {a["15"]:.1f} to {b["15"]:.1f} s. '
        f'Time below 10 km/h changed from {a["10"]:.1f} to {b["10"]:.1f} s. Both include required stops.',
        ("All 25 m sections", TAG+"audit/v06.json"), ("Driving analysis", TAG+"README.md"))
    a, b = old["comfort"], new["comfort"]
    add("Braking and speed limits",
        f'Strongest half-second deceleration was {a["max_decel_half_s"]:.2f} m/s² before and '
        f'{b["max_decel_half_s"]:.2f} m/s² after. The revised car spent '
        f'{b["overspeed_posted_plus_1_seconds"]:.1f} s more than 1 km/h above the posted limit.',
        ("Controls and limitations", TAG+"audit/README.md"))
    return {"checks": checks, "scope": "The two drivers ran the same populated route. Arriving at a different time "
            "can change which people and vehicles occupy a junction. Matched open and occupied trials supplement the route comparison."}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=ROOT / "web/media/progress-review.json")
    path = cli.parse_args().output
    path.write_text(json.dumps(create(), ensure_ascii=False, indent=2) + "\n", encoding="utf8", newline="\n")
