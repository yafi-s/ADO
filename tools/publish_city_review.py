"""Build the Results page's v0.4 scorecard from the recorded studies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/yafi-s/ADO/blob/v0.4.0/"
MEDIA = "web/media/urban-pace/"
PERCEPTION = "results/urban-pace/perception.json"


def read(path):
    return json.loads((ROOT / path).read_bytes())


def validate_drive(hero, speeds, turn, perception, barriers, signals):
    """Require the featured recording and the audits of its final rider display."""
    score = hero["summary"]
    start, end = hero["first_frame"], hero["last_frame"]
    frame_count = end - start + 1
    source = perception["source"]
    if (hero["trial_id"] != "urban-pace" or not hero["passed"] or hero["outcome"] != "complete"
            or turn["trial_id"] != hero["trial_id"] or turn["capture"] != hero["capture_id"]
            or signals["capture_id"] != hero["capture_id"]):
        raise ValueError("The scorecard requires the completed urban-pace recording and its own audits")
    if (turn["native_frames"] != [start, end] or [source["first_frame"], source["last_frame"]] != [start, end]
            or [signals["first_frame"], signals["last_frame"]] != [start, end]
            or any(count != frame_count for count in (score["frames"], speeds["frames"], turn["exposures"],
                    turn["joined_camera_timestamps"], source["frames"], barriers["capture_frames"],
                    signals["frame_count"], signals["verified_recorded_signal_rows"]))):
        raise ValueError("Featured drive metrics disagree on the native frame interval")
    raw = hero["raw_sha256"]
    if (turn["raw_replay_sha256"] != raw["replay.json"] or source["sha256"] != raw["perception.jsonl"]
            or barriers["native_journal_sha256"] != raw["perception.jsonl"]
            or signals["producer"]["revised_reader_sha256"] != hero["policy"]["code_sha256"]["signal_observer.py"]):
        raise ValueError("Featured drive metrics refer to different capture inputs")
    review = (ROOT / MEDIA / "barrier-review.json").read_bytes()
    if hashlib.sha256(review).hexdigest() != source["barrier_review_sha256"] or (
            hero["barrier_review_sha256"] != source["barrier_review_sha256"]):
        raise ValueError("Re-audit the featured drive after changing barrier review decisions")
    for path, expected in source["code_sha256_lf"].items():
        if not path.startswith("web/"):
            continue
        actual = hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError(f"Re-audit the featured drive after changing {path}")


def _clean(trial):
    return trial["outcome"] == "complete" and not any(
        trial[key] for key in ("collision_events", "red_light_violations", "yield_violations"))


def generate():
    hero = read(MEDIA + "manifest.json")
    score = hero["summary"]
    speeds = read("results/urban-pace/speeds.json")
    turn = read("results/urban-pace/follow-turns.json")
    perception = read(PERCEPTION)
    barriers = read("results/barrier-recognition/hero-journey.json")
    signal_drive = read("results/surround-city/signal-native-drive-v04.json")
    validate_drive(hero, speeds, turn, perception, barriers, signal_drive)
    before = read("results/city-pace/boulevard/city-cruise/speeds.json")["interior"]
    after = read("results/city-pace/boulevard/city-pace/speeds.json")["interior"]
    entry = read("results/city-pace/signed-transition/with-preview/speeds.json")["lower_zone_entries"][-1]
    closed = read("results/workzone-stop/closed-then-open.json")
    curb = read("results/workzone-stop/curbside.json")
    wet = read("results/workzone-stop/wet-grip-075/closed-then-open.json")
    wet_curb = read("results/workzone-stop/wet-grip-075/curbside.json")
    boards = read("results/barrier-recognition/validation.json")["totals"]["holdout"]
    chains = read("results/barrier-recognition/post-appearance-fit.json")["fixtures"]["high"]
    chain_conditions = read("results/barrier-recognition/chain-high.json")["by_condition"]
    chain_controls = [value for key, value in chain_conditions.items() if key not in ("chain", "barricade")]
    board_counts = barriers["raw_counts"]
    accepted = board_counts["breadth_confirmed_striped_boards"]
    board_matches = board_counts["confirmed_striped_scene_matches"] + board_counts["confirmed_striped_map_asset_matches"]
    display = perception["barriers"]
    heading = turn["visible_movie"]
    old, new = (heading[k]["angular_error_deg"]["p95"] for k in ("previous", "updated"))
    previous, revised = (signal_drive["comparisons"][arm] for arm in ("original", "revised"))
    previous_count, revised_count = (arm["counts"] for arm in (previous, revised))
    first_green = [next(item["first_agreement_frames"] for item in arm["transitions"]
                        if item["from_state"] == "red" and item["to_state"] == "green") * hero["frame_period_s"]
                   for arm in (previous, revised)]
    signal_ranges = read("results/surround-city/signal-native-ranges.json")["native"]
    at_25m = next(arms["revised_reader"] for clip, arms in signal_ranges.items() if clip.endswith("_25m"))
    objects = read("results/perception/object-continuity/same-tape.json")["replays"]["surround-city"]
    first, last = (objects[k]["overlay"]["with_parked"] for k in ("before", "after"))
    displayed = perception["including_registered_parked"]["rider"]["overall"]
    cyclist_audits = [read(f"results/perception/cycle-range-v04/{file}.json")
                      for file in ("high212-initial-display", "high212")]
    if any(audit["capture_id"] != hero["capture_id"] or audit["source_sha256"] != perception["source"]["sha256"]
           for audit in cyclist_audits):
        raise ValueError("Cyclist comparisons must use the same featured recording")
    cyclist_before, cyclist = [audit["principal_cyclist"]["display"]["nearby"] for audit in cyclist_audits]
    if cyclist["bicycle_frames"] != perception["roles_including_parked"]["cyclist"]["rider"]["exact_label_matches"]:
        raise ValueError("The cyclist and whole-drive audits disagree on the final rider display")
    gated = read("results/town01-yield-gated-v04/audit.json")["cohort"]
    launcher = read("results/town01-yield-gated-v04/launcher/comparison.json")
    transfer = read("results/tactical-pass/v04-road67/paired-jo9p4yw0/summary.json")
    waymo = read("waymo_training/carla_speed_native_development.json")["runs"]
    checks = []

    def add(test, result, *links):
        checks.append(dict(test=test, result=result, evidence=[dict(label=label, url=url) for label, url in links]))

    add("City drive", f'{score["progress_m"]/1000:.2f} km in {score["duration_s"]:.1f} simulated seconds. '
        f'{sum(e["handled"] for e in score["events"])}/{len(score["events"])} encounters handled, '
        f'{score["collision_events"]} contacts and {score["red_light_violations"]+score["yield_violations"]} traffic violations.',
        ("Watch the drive", "./?run=urban-pace"), ("Inspect frames", "./inspect.html?run=urban-pace"))
    add("Cruising pace", f'Median speed on the empty boulevard rose from {before["median_kmh"]:.1f} to '
        f'{after["median_kmh"]:.1f} km/h. Standard deviation fell from {before["sd_kmh"]:.2f} to '
        f'{after["sd_kmh"]:.2f} km/h. Clear-straight median in the featured drive was '
        f'{speeds["clear_straights"]["median_kmh"]:.1f} km/h across all speed zones.',
        ("Speed traces", BASE + "results/urban-pace/speed-traces.png"), ("Study", BASE + "results/city-pace/README.md"))
    add("Lower speed limit", f'Began slowing {entry["preview_distance_before_entry_m"]:.1f} m upstream and entered '
        f'the 30 km/h zone at {entry["entry_speed_kmh"]:.2f} km/h in a matched native test.',
        ("Sign test", BASE + "results/city-pace/signed-transition/with-preview/speeds.json"))
    add("Stopping and restarting", f'Stopped {closed["longest_stop_before_blocked_board_s"]:.1f} s behind the board, '
        f'then reached {closed["max_speed_after_reopening_mps"]*3.6:.1f} km/h. The wet test also held for '
        f'{wet["longest_stop_before_blocked_board_s"]:.1f} s. Together the curbside controls recorded '
        f'{curb["longest_stop_before_blocked_board_s"] + wet_curb["longest_stop_before_blocked_board_s"]:.0f} s stopped on approach.',
        ("Blocked lane", "./media/workzone-tests/closed-then-open.mp4"),
        ("Wet and curbside controls", BASE + "results/workzone-stop/README.md"))
    add("Striped boards", f'{boards["true_positive"]}/{boards["positive"]} held-out board exposures identified, with '
        f'{boards["false_positive"]} false detections across {boards["negative_exposures"]} controls. In the featured '
        f'drive, {board_matches} of {accepted} qualified fresh board measurements matched staged or mapped boards '
        f'within 0.65 m. The other {accepted - board_matches} estimates landed farther from their references.',
        ("Inspect the two boards", f'./inspect.html?run=urban-pace&frame={perception["evidence"]["board"]["frame"]}&camera=front'),
        ("Native board evidence", BASE + "results/barrier-recognition/hero-journey.json"))
    add("Post-and-chain display", f'{chains["raw_chain_tracks"]}/{chains["actual_chain_exposures"]} linked-chain exposures '
        f'had raw support in the independent High test, and the final display retained '
        f'{chains["causal_confirmed_chain_tracks"]}/{chains["actual_chain_exposures"]}. The detector made '
        f'{sum(item["unmatched_predictions"] for item in chain_controls)} false calls across '
        f'{sum(item["exposures"] for item in chain_controls)} controls. On the featured drive, '
        f'{display["raw_chain_proposals"]} proposals produced {display["visible_chain_symbol_ticks"]} visible chain symbols.',
        ("Chain tests and limits", BASE + "results/barrier-recognition/README.md"))
    add("Signal colors", f'On the same live RGB, wrong colors fell from {len(previous["mismatches"])} to '
        f'{len(revised["mismatches"])} and correct colors from {previous_count["correct"]:,} to {revised_count["correct"]:,}. '
        f'The revised display left {revised_count["unknown"]:,} head observations neutral and confirmed green '
        f'at {first_green[1]:.1f} s versus {first_green[0]:.1f} s. The separate 25 m test still gave '
        f'{len(at_25m["mismatches"])} wrong colors in {at_25m["counts"]["mapped_head_frames"]} head observations.',
        ("Native signal audit", BASE + "results/surround-city/signal-reader-transition.md"))
    sharpest_turn = max(turn["turns"], key=lambda window: window["peak_native_yaw_rate_deg_s"])
    add("Follow camera on turns", f'Across {len(turn["turns"])} turns in the featured drive, the 95th-percentile heading '
        f'error fell from {old:.2f}° to {new:.2f}°. Fitted lag fell from {heading["previous"]["best_fit_lag_ms"]:.0f} to '
        f'{heading["updated"]["best_fit_lag_ms"]:.0f} ms at {1000 / turn["modelled_display_hz"]:.1f} ms resolution. '
        f'All {turn["exposures"]:,} native exposures aligned.',
        ("Watch a featured turn", f'./?run=urban-pace&at={sharpest_turn["seconds"][0]:g}'),
        ("Turn measurements", BASE + "results/urban-pace/follow-turns.md"))
    add("Object continuity", 'On the identical v0.3 tape including registered parked vehicles, matched drawings '
        f'increased from {first["hits"]:,} to {last["hits"]:,}. Unmatched drawings fell from '
        f'{first["unmatched"]:,} to {last["unmatched"]:,}. The featured drive matched '
        f'{displayed["matched"]:,}/{displayed["actor_frames"]:,} nearby actor-frames, including parked vehicles. '
        f'{displayed["unmatched_predictions"]:,}/{displayed["prediction_frames"]:,} drawings had no reference match. '
        'The reference omits some roadside objects.',
        ("Same-tape comparison", BASE + "results/perception/object-continuity/README.md"),
        ("Featured drive audit", BASE + PERCEPTION))
    add("Cyclist display", f'The featured drive drew the bicycle correctly in {cyclist["bicycle_frames"]}/'
        f'{cyclist["frames"]} frames within {cyclist_audits[1]["nearby_range_m"]:g} m. Duplicate person/bicycle drawings '
        f'fell from {cyclist_before["duplicate_same_source_person_bicycle_frames"]} frames to '
        f'{cyclist["duplicate_same_source_person_bicycle_frames"]} on the same recording.',
        ("Cyclist and range evidence", BASE + "results/perception/cycle-range-v04/README.md"))
    add("Passing on a second road", f'{sum(case["passed"] for case in transfer["cases"])}/{len(transfer["cases"])} '
        f'cases passed on a second configured Town03 corridor. The cases tested a clear lane, an occupied lane '
        f'and a no-lead control across {sum(case["frames"] for case in transfer["cases"])} frames. '
        'Both passing corridors use configured straight lanes with legal lane changes.',
        ("Second-road scores and videos", BASE + "results/tactical-pass/v04-road67/README.md"))
    add("Yield and crossing failures", f'{gated["known_failure"]["clean"]}/{gated["known_failure"]["n"]} previous failures, '
        f'{gated["known_control"]["clean"]}/{gated["known_control"]["n"]} preserved controls and '
        f'{gated["fresh_challenge"]["clean"]}/{gated["fresh_challenge"]["n"]} new Town01 challenges finished cleanly. '
        f'The interactive launcher finished {sum(trip["clean"] for trip in launcher)}/{len(launcher)} named trips cleanly.',
        ("Cases and limitations", BASE + "results/town01-yield-gated-v04/README.md"))
    pair_count = len({trial["case"] for trial in waymo})
    waymo_clean = sum(_clean(trial) for trial in waymo if trial["arm"] == "with_waymo")
    carla_clean = sum(_clean(trial) for trial in waymo if trial["arm"] == "carla_only")
    add("Waymo forecast transfer", f'In {len(waymo)} separate native research runs on one Town01 seed, CARLA calibration '
        f'finished {carla_clean}/{pair_count} cases cleanly and calibration with the Waymo forecast finished '
        f'{waymo_clean}/{pair_count}.',
        ("Paired research trials", BASE + "waymo_training/CARLA_SPEED_CALIBRATION.md"))
    return dict(checks=checks, scope="The city course, configured passing corridors and named Town01 cases are controlled "
        "simulator tests. The evidence covers those routes and fixtures. Thin chains, uncertain signal colors and "
        "partly hidden road users remain harder to identify.")


if __name__=="__main__":
    output=ROOT/"web/media/urban-review.json"
    output.write_text(json.dumps(generate(),indent=2,ensure_ascii=False)+"\n",encoding="utf8",newline="\n")
