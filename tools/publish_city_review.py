"""Build the Results page's v0.4 scorecard from the recorded studies."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/yafi-s/ADO/blob/v0.4.0/"


def read(path):
    return json.loads((ROOT/path).read_bytes())


def generate():
    hero = read("web/media/urban-pace/manifest.json")
    score = hero["summary"]
    speeds = read("results/urban-pace/speeds.json")
    before = read("results/city-pace/boulevard/city-cruise/speeds.json")["interior"]
    after = read("results/city-pace/boulevard/city-pace/speeds.json")["interior"]
    entry = read("results/city-pace/signed-transition/with-preview/speeds.json")["lower_zone_entries"][-1]
    closed = read("results/workzone-stop/closed-then-open.json")
    curb = read("results/workzone-stop/curbside.json")
    boards = read("results/barrier-recognition/validation.json")["totals"]["holdout"]
    chains = read("results/barrier-recognition/chain-validation.json")["by_condition"]["chain"]
    heading = read("results/perception/rider-turns/turns.json")["visible_movie"]
    old,new = (heading[k]["angular_error_deg"]["p95"] for k in ("previous","updated"))
    objects = read("results/perception/object-continuity/same-tape.json")["replays"]["surround-city"]
    first,last = (objects[k]["overlay"]["with_parked"] for k in ("before","after"))
    gated = read("results/town01-yield-gated-v04/audit.json")["cohort"]
    checks = []
    def add(test, result, *links):
        checks.append(dict(test=test, result=result, evidence=[dict(label=label,url=url) for label,url in links]))
    add("City drive", f'{score["progress_m"]/1000:.2f} km in {score["duration_s"]:.1f} simulated seconds. '
        f'{sum(e["handled"] for e in score["events"])}/{len(score["events"])} encounters handled, '
        f'{score["collision_events"]} contacts and {score["red_light_violations"]+score["yield_violations"]} traffic violations.',
        ("Watch the drive","./?run=urban-pace"),("Inspect frames","./inspect.html?run=urban-pace"))
    add("Cruising pace", f'Median speed on the empty boulevard rose from {before["median_kmh"]:.1f} to '
        f'{after["median_kmh"]:.1f} km/h. Standard deviation fell from {before["sd_kmh"]:.2f} to '
        f'{after["sd_kmh"]:.2f} km/h. Clear-straight median in the featured drive was '
        f'{speeds["clear_straights"]["median_kmh"]:.1f} km/h across all speed zones.',
        ("Speed traces",BASE+"results/urban-pace/speed-traces.png"),("Study",BASE+"results/city-pace/README.md"))
    add("Lower speed limit",f'Began slowing {entry["preview_distance_before_entry_m"]:.1f} m upstream and entered '
        f'the 30 km/h zone at {entry["entry_speed_kmh"]:.2f} km/h in a matched native test.',
        ("Sign test",BASE+"results/city-pace/signed-transition/with-preview/speeds.json"))
    add("Stopping and restarting", f'Stopped {closed["longest_stop_before_blocked_board_s"]:.1f} s behind the board, '
        f'resumed to {closed["max_speed_after_reopening_mps"]*3.6:.1f} km/h, and recorded '
        f'{curb["longest_stop_before_blocked_board_s"]:.0f} s stopped on approach to the curbside control.',
        ("Blocked lane","./media/workzone-tests/closed-then-open.mp4"),("Curbside","./media/workzone-tests/curbside.mp4"))
    add("Barrier perception", f'{boards["true_positive"]}/{boards["positive"]} striped boards identified on held-out exposures '
        f'with {boards["false_positive"]} false detections across {boards["negative_exposures"]} matched controls. '
        f'{chains["matched"]}/{chains["exposures"]} linked-chain exposures had raw geometric and camera support.',
        ("Native examples",BASE+"results/barrier-recognition/README.md"))
    add("Turns",f'On eight recorded turn windows, follow-camera heading error at the 95th percentile '
        f'fell from {old:.2f}° to {new:.2f}°. The native camera exposures and car poses stayed aligned.',
        ("Watch the turn","./media/turn-camera.mp4"),("Method",BASE+"results/perception/rider-turns/README.md"))
    add("Object continuity",f'On the identical v0.3 tape including registered parked vehicles, matched drawings '
        f'increased from {first["hits"]:,} to {last["hits"]:,}. Unmatched drawings fell from '
        f'{first["unmatched"]:,} to {last["unmatched"]:,}.',("Full comparison",BASE+"results/perception/object-continuity/README.md"))
    add("Yield and crossing failures",f'{gated["known_failure"]["clean"]}/{gated["known_failure"]["n"]} previous failures, '
        f'{gated["known_control"]["clean"]}/{gated["known_control"]["n"]} preserved controls and '
        f'{gated["fresh_challenge"]["clean"]}/{gated["fresh_challenge"]["n"]} new Town01 challenges finished cleanly.',
        ("Cases and limitations",BASE+"results/town01-yield-gated-v04/README.md"))
    return dict(checks=checks, scope="The city course and small matched challenges are development tests. "
        "Thin chains, uncertain signal colors and partly hidden road users remain harder to identify. "
        "Each study reports its missed observations, controls and test conditions.")


if __name__=="__main__":
    output=ROOT/"web/media/urban-review.json"
    output.write_text(json.dumps(generate(),indent=2,ensure_ascii=False)+"\n",encoding="utf8",newline="\n")
