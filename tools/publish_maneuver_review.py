"""Build the public maneuver review from native event witnesses and the independent route audit."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "web/media"
STUDY = ROOT / "results/city-maneuvers"
TAG = "https://github.com/yafi-s/ADO/blob/v0.7.0/results/city-maneuvers/"


def read(path):
    return json.loads(path.read_bytes())


def create():
    manifest = read(MEDIA / "city-maneuvers/manifest.json")
    old = read(STUDY / "audit/v06.json")
    new = read(STUDY / "audit/v07.json")
    if (not manifest["passed"] or manifest["trial_id"] != "city-maneuvers"
            or manifest["policy"]["sha256"] != new["source"]["policy_sha256"]
            or manifest["sha256"]["telemetry.json"] != new["source"]["telemetry_sha256"]
            or manifest["summary"]["frames"] != new["grade"]["frames"]):
        raise ValueError("Maneuver review needs the audited full recording")
    summary = manifest["summary"]
    events = {e["kind"]: e for e in summary["events"]}
    obstacle, lead = events["in-lane-obstruction"], events["slow-traffic"]
    sequence = manifest["maneuver_sequence"]
    if not (sequence["passed"] and sequence["actual_wait_policy_frames"] > 0 and obstacle["handled"] and lead["handled"] and
            obstacle["returned_to_route_s"] < lead["occupied_gap_wait_s"] < lead["moving_alongside_s"] < lead["returned_to_route_s"]):
        raise ValueError("The recording lacks the two distinct completed maneuvers")

    def exposure(frame, view):
        if not manifest["first_frame"] <= frame <= manifest["last_frame"]:
            raise ValueError("Event outside the camera recording")
        return f"./?run=city-maneuvers&frame={frame}&camera={view}"

    def camera(stamp, view):
        return exposure(manifest["first_frame"] + round(stamp / manifest["frame_period_s"]), view)

    checks = []

    def add(title, result, *links):
        checks.append({"test": title, "result": result,
                       "evidence": [{"label": label, "url": url} for label, url in links]})

    add("Full populated drive",
        f'{summary["progress_m"]/1000:.2f} km, {summary["frames"]:,} native exposures and '
        f'{sum(bool(e["handled"]) for e in events.values())}/{len(events)} encounters handled. '
        f'{summary["collision_events"]} ego contacts, {summary["red_light_violations"]} red-light violations '
        f'and {summary["yield_violations"]} missed yields.',
        ("Watch from the beginning", "./?run=city-maneuvers"),
        ("Original grade", "./media/city-maneuvers/manifest.json"))
    add("Obstruction in the lane",
        f'ADO passed {obstacle["target_count"]} physical props in the neighboring lane, with '
        f'{obstacle["alongside_minimum_body_gap_m"]:.2f} m minimum body clearance while alongside. '
        f'It then returned to the original lane, {obstacle["rejoin_rear_clear_m"]:.1f} m clear of the obstruction.',
        ("See the obstruction", camera(obstacle["exposed_s"], "front")),
        ("Alongside", camera(obstacle["alternate_alongside_s"], "chase")),
        ("Back in lane", camera(obstacle["returned_to_route_s"], "rear_right")))
    add("Yield and overtake",
        f'The vehicle ahead was moving at {lead["alongside_target_mps"]*3.6:.1f} km/h at first measured moving overlap. '
        f'For {sequence["actual_wait_policy_frames"] * manifest["frame_period_s"]:.1f} s of native exposures '
        f'ADO selected yield while adjacent traffic occupied the gap. It then passed the lead and returned '
        f'{lead["rejoin_rear_clear_m"]:.1f} m ahead, measured between the two bodies.',
        ("Yield to adjacent traffic", exposure(sequence["first_actual_wait_frame"], "rear_left")),
        ("Pass the moving car", camera(lead["moving_alongside_s"], "chase")),
        ("Return", camera(lead["returned_to_route_s"], "rear_right")))
    before, after = old["posted_90"], new["posted_90"]
    add("90 km/h posted section",
        f'On the featured drive, actual speed on this section peaked at {after["kmh_p10_p50_p95_max"][3]:.1f} km/h '
        f'and spent {after["over_40_kmh_seconds"]:.1f} s above 40 km/h. The previous drive peaked at '
        f'{before["kmh_p10_p50_p95_max"][3]:.1f} km/h. The two recordings have different traffic and obstructions.',
        ("Matched road sections", TAG + "audit/README.md"),
        ("Stopping and higher-speed tests", TAG + "speed/README.md"))
    speed_study = read(STUDY / "speed/comparison.json")["town03"]
    capped, higher = speed_study["capped"], speed_study["higher"]
    add("Higher-speed control",
        f'On a separate open boulevard, the matched straight-section median changed from '
        f'{capped["actual_section_median_kmh"]:.1f} to {higher["actual_section_median_kmh"]:.1f} km/h. '
        f'The faster configuration peaked at {higher["actual_peak_kmh"]:.1f} km/h and held above 40 for '
        f'{higher["longest_above_40_in_90_s"]:.1f} consecutive seconds in the 90 zone. '
        f'In the occupied trial it stopped {speed_study["occupied"]["stationary_queue"]["final_gap_m"]:.2f} m behind the queue.',
        ("Cruise, lower limits and braking", TAG + "speed/README.md"),
        ("Full matched measurements", TAG + "speed/comparison.json"))
    a, b = old["moving_population_exposure"]["50"], new["moving_population_exposure"]["50"]
    add("Moving city traffic",
        f'The simulator spawned {new["grade"]["ambient_vehicles_spawned"]} background vehicles, '
        f'compared with {old["grade"]["ambient_vehicles_spawned"]} previously. '
        f'At least one moving vehicle and one moving pedestrian were within 50 m for '
        f'{100*b["both"]/b["scored"]:.1f}% of eligible frames, compared with '
        f'{100*a["both"]/a["scored"]:.1f}% on the earlier drive. These measurements include occluded actors.',
        ("Native proof and controls", TAG + "README.md"),
        ("Explore the traffic timeline", "./benchmarks.html#traffic-study"))
    return {"checks": checks, "scope": "The native cameras and vehicle bodies establish the order and clearance of each tested pass. "
            "The posted limit, fitted operating ceiling, requested speed and measured speed are separate quantities."}


if __name__ == "__main__":
    path = MEDIA / "maneuvers-review.json"
    path.write_text(json.dumps(create(), ensure_ascii=False, indent=2) + "\n", encoding="utf8", newline="\n")
