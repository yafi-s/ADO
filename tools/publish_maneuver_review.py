"""Build the public maneuver review from native event witnesses and the independent route audit."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "web/media"
STUDY = ROOT / "results/city-maneuvers"
TAG = "https://github.com/yafi-s/ADO/blob/v0.7.0/results/city-maneuvers/"


def read(path):
    return json.loads(path.read_bytes())


def event_frame(rows, event, mark):
    """Find the first camera exposure whose scorer has recorded this milestone."""
    stamp = event[mark]
    if stamp is not None:
        for row in rows:
            for measured in row["events"]:
                if measured["kind"] == event["kind"] and measured.get(mark) == stamp:
                    return row["frame"]
    raise ValueError(f'Missing native exposure for {event["kind"]}/{mark}')


def create():
    manifest = read(MEDIA / "city-maneuvers/manifest.json")
    rows = read(MEDIA / "city-maneuvers/telemetry.json")
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

    def camera(event, mark, view):
        return exposure(event_frame(rows, event, mark), view)

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
        ("See the obstruction", camera(obstacle, "exposed_s", "front")),
        ("Alongside", camera(obstacle, "alternate_alongside_s", "chase")),
        ("Passenger-side camera", camera(obstacle, "alternate_alongside_s", "right")),
        ("Back in lane", camera(obstacle, "returned_to_route_s", "rear_right")))
    add("Yield and overtake",
        f'The vehicle ahead was moving at {lead["alongside_target_mps"]*3.6:.1f} km/h at first measured moving overlap. '
        f'For {sequence["actual_wait_policy_frames"] * manifest["frame_period_s"]:.1f} s of native exposures '
        f'ADO selected yield while adjacent traffic occupied the gap. It then passed the lead and returned '
        f'{lead["rejoin_rear_clear_m"]:.1f} m ahead, measured between the two bodies.',
        ("Yield to adjacent traffic", exposure(sequence["first_actual_wait_frame"], "rear_left")),
        ("Pass the moving car", camera(lead, "moving_alongside_s", "chase")),
        ("Passenger-side camera", camera(lead, "moving_alongside_s", "right")),
        ("Return", camera(lead, "returned_to_route_s", "rear_right")))
    variant = read(STUDY / "qualification/timing-variation/grade.json")
    variant_run = read(STUDY / "qualification/timing-variation/summary.json")
    variant_lead = next(event for event in variant["events"] if event["kind"] == "slow-traffic")
    if variant["passed"] or variant_lead.get("returned_to_route_s") is not None:
        raise ValueError("The recorded timing-variation outcome changed")
    add("Later gap and faster lead",
        f'When the adjacent gap opened later and the lead traveled at {variant_lead["alongside_target_mps"]*3.6:.1f} km/h, '
        f'ADO yielded and overtook, then stopped in the passing lane. The second return failed. '
        f'The trial timed out after {variant_run["duration_s"]:.1f} s with {variant_run["collision_events"]} recorded ego contacts.',
        ("Native grade and events", TAG + "qualification/timing-variation/grade.json"),
        ("Stopped position and controls", TAG + "qualification/timing-variation/summary.json"))
    early = read(STUDY / "audit/high01-failed.json")
    registration = read(STUDY / "qualification/spaced-registration.json")
    if events["lead"].get("stopped_lead_layout") == "junction-approach":
        add("Why the second layout differs",
            f'In the first High recording ADO passed the slow car, then stopped in the passing lane at '
            f'{early["grade"]["progress_m"]:.1f} m with no viable return gap on the measured trajectory. '
            f'The featured course places the next stopped sedan {registration["revised_course"]["held_sedan_shift_m"]:.0f} m farther '
            f'down the same lane. It makes the two encounters distinct. The original course remains a failed test.',
            ("Watch the failed return", TAG.replace("/blob/", "/raw/") + "attempts/high01/incomplete-return.mp4"),
            ("First High recording audit", TAG + "audit/high01-failed.json"),
            ("Registered scene change", TAG + "qualification/spaced-registration.json"))
    comparison = new.get("route_comparison")
    if new["source"]["route_id"] != old["source"]["route_id"] and (
            not comparison or comparison["original_route_id"] != old["source"]["route_id"]):
        raise ValueError("A changed course needs an audited shared route before comparison")
    shared = comparison["shared_prefix"] if comparison else new
    before, after = old["posted_90"], shared["posted_90"]
    add("Original 90 km/h posted section",
        f'On the shared stretch, actual speed in the featured drive peaked at {after["kmh_p10_p50_p95_max"][3]:.1f} km/h '
        f'and spent {after["over_40_kmh_seconds"]:.1f} s above 40 km/h. The previous drive peaked at '
        f'{before["kmh_p10_p50_p95_max"][3]:.1f} km/h. The two recordings have different traffic and obstructions.',
        ("Matched road sections", TAG + "audit/README.md"),
        ("Stopping and higher-speed tests", TAG + "speed/README.md"))
    if comparison:
        extension = comparison["extension"]
        add("Why the passes are on separate blocks",
            f'The first High drive on the shorter layout timed out at {early["grade"]["progress_m"]:.1f} m '
            f'after overtaking the slow sedan. The featured route retains the original streets and continues '
            f'{summary["route_length_m"]-comparison["shared_route_to_m"]:.0f} m farther. The moving lead and its '
            f'adjacent traffic are on the return carriageway. The in-lane drums remain on the outbound road.',
            ("Watch the failed return", TAG.replace("/blob/", "/raw/") + "attempts/high01/incomplete-return.mp4"),
            ("Original and extended geometry", TAG + "audit/market-loop-survey.json"))
        speed = extension["posted_90"]
        if speed and speed.get("frames"):
            add("Return boulevard",
                f'The extension starts beyond the earlier film at {comparison["shared_route_to_m"]:.0f} m. '
                f'In its own 90 km/h zone, the car spent {speed["seconds"]:.1f} s, '
                f'reached {speed["kmh_p10_p50_p95_max"][3]:.1f} km/h and drove above 40 km/h for '
                f'{speed["over_40_kmh_seconds"]:.1f} s.',
                ("Every posted limit and speed interval", TAG + "audit/v07.json"))
    matched = {case["name"]: case for case in read(STUDY / "pace/comparison.json")["cases"]}
    bend, boards, drums = (matched[name] for name in ("J1932 bend", "Original roadside boards", "Original roadside drums"))
    add("Original bend and roadside objects",
        f'In paired runs, median speed on the bend changed from {bend["v06"]["median_kmh"]:.1f} to '
        f'{bend["v07"]["median_kmh"]:.1f} km/h. At the roadside boards it changed from '
        f'{boards["v06"]["median_kmh"]:.1f} to {boards["v07"]["median_kmh"]:.1f} km/h. The six roadside '
        f'drums measured {drums["v06"]["median_kmh"]:.1f} and {drums["v07"]["median_kmh"]:.1f} km/h respectively.',
        ("Paired tests and explanations", TAG + "pace/README.md"),
        ("Native speed and clearance", TAG + "pace/comparison.json"))
    speed_study = read(STUDY / "speed/comparison.json")["town03"]
    capped, higher, occupied = (speed_study[name] for name in ("capped", "higher", "occupied"))
    add("Higher-speed control",
        f'On a separate open boulevard, the matched straight-section median changed from '
        f'{capped["actual_section_median_kmh"]:.1f} to {higher["actual_section_median_kmh"]:.1f} km/h. '
        f'The faster configuration peaked at {higher["actual_peak_kmh"]:.1f} km/h and held above 40 for '
        f'{higher["longest_above_40_in_90_s"]:.1f} consecutive seconds in the 90 zone. '
        f'With a queue ahead it peaked at {occupied["actual_peak_kmh"]:.1f} km/h, spent '
        f'{occupied["seconds_above_50_in_90"]:.1f} s above 50 and stopped '
        f'{occupied["stationary_queue"]["final_gap_m"]:.2f} m behind the queue.',
        ("Native higher-speed footage", "./media/city-maneuvers-tests/boulevard.mp4"),
        ("Cruise, lower limits and braking", TAG + "speed/README.md"),
        ("Full matched measurements", TAG + "speed/comparison.json"))
    a, b = (run["whole"]["moving_population_exposure"]["50"] for run in (old, shared))
    add("Moving city traffic",
        f'The simulator spawned {new["grade"]["ambient_vehicles_spawned"]} background vehicles, '
        f'compared with {old["grade"]["ambient_vehicles_spawned"]} previously. '
        f'On the shared streets, at least two moving vehicles and two moving pedestrians were within 50 m for '
        f'{100*b["two_of_each_frames"]/b["scored"]:.1f}% of eligible frames, compared with '
        f'{100*a["two_of_each_frames"]/a["scored"]:.1f}% on the earlier drive. Counts include occluded actors.',
        ("Native proof and controls", TAG + "README.md"),
        ("Explore the traffic timeline", "./benchmarks.html#traffic-study"))
    screen = read(STUDY / "display.json")
    if not screen["capture_passed"] or screen["telemetry_sha256"] != new["source"]["telemetry_sha256"]:
        raise ValueError("Rider display audit must use the featured camera recording")
    links = []
    for key, label in (("obstruction_right", "Cone in the side camera"),
                       ("yielding_to_adjacent_rear", "Approaching car while ADO yields")):
        witness = screen["publication_witnesses"].get(key)
        if witness:
            links.append((label, exposure(witness["frame"], witness["view"])))
    lead_display = screen["roles"]["slow-traffic"]
    cone_display = screen["roles"]["in-lane-obstruction"]
    add("What the rider display actually saw",
        f'Within 35 m the display placed the slow vehicle within 3 m of its native body on '
        f'{lead_display["matched_drawings"]}/{lead_display["nearby_actor_frames"]} exposures. '
        f'For the three staged cones, {cone_display["matched_drawings"]}/{cone_display["nearby_actor_frames"]} '
        f'individual cone observations had a symbol within 0.65 m. Counts include occluded bodies and remembered symbols.',
        *links, ("Camera and range checks", TAG + "display.md"))
    return {"checks": checks, "scope": "The native cameras and vehicle bodies establish the order and clearance of each tested pass. "
            "The posted limit, fitted operating ceiling, requested speed and measured speed are separate quantities."}


if __name__ == "__main__":
    path = MEDIA / "maneuvers-review.json"
    path.write_text(json.dumps(create(), ensure_ascii=False, indent=2) + "\n", encoding="utf8", newline="\n")
