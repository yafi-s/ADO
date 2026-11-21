"""Build the City Life scorecard from the two recordings and matched native studies."""
import json
from pathlib import Path

from tools.measure_roadworks_pace import measure

ROOT = Path(__file__).resolve().parents[1]
TAG = "https://github.com/yafi-s/ADO/blob/v0.5.0/"


def read(name):
    return json.loads((ROOT / name).read_bytes())


def scorecard():
    new = read("web/media/city-life/manifest.json")
    summary = new["summary"]
    timelines = {d["trial"]: d for d in read("web/media/city-traffic.json")["drives"]}
    old_timeline, timeline = timelines["urban-pace"], timelines["city-life"]
    if timeline["frames"] != summary["frames"] or timeline["source_sha256"] != new["sha256"][new["replay"]]:
        raise ValueError("The drive timeline differs from its source recording")
    new_density = read("results/city-life/population-score.json")["radii_m"]["50"]
    old_density = read("results/city-life/baseline-population-score.json")["radii_m"]["50"]
    for timeline_score, population in [(timeline, new_density), (old_timeline, old_density)]:
        if (timeline_score["nearby"]["eligible_frames"] != population["scored_frames"]
                or timeline_score["nearby"]["moving_car_and_ped_frames"] != population["both_frames"]):
            raise ValueError("Independent moving-traffic measurements disagree")
    pace_dir = ROOT / "results/roadworks-pace"
    slower = measure(pace_dir / "right-taper-v04")["alongside"]
    faster = measure(pace_dir / "right-taper")
    blocked = measure(pace_dir / "blocked")
    cones = read("results/cone-perception/metrics.json")["journey"]
    old_frame = next(row[1] for row in old_timeline["samples"] if row[0] == old_timeline["cones_at_s"])
    checks = []

    def add(title, result, *links):
        checks.append({"test": title, "result": result, "evidence": [
            {"label": label, "url": url} for label, url in links]})

    add("Closed-loop drive",
        f'{summary["progress_m"]/1000:.2f} km and {sum(e["handled"] for e in summary["events"])}/'
        f'{len(summary["events"])} staged encounters in {summary["duration_s"]:.1f} simulated seconds. '
        f'CARLA recorded {summary["collision_events"]} ego contacts, {summary["red_light_violations"]} '
        f'red-light violations and {summary["yield_violations"]} missed yields.',
        ("Watch City Life", "./?run=city-life"), ("Recorded grade", "./media/city-life/manifest.json"))
    pct = lambda d: 100 * d["both_frames"] / d["scored_frames"]
    peak = new_density["peak"]
    add("Moving traffic",
        f'A moving vehicle and a moving pedestrian were both within 50 m for {pct(new_density):.1f}% of '
        f'the drive, versus {pct(old_density):.1f}% previously. One exposure had '
        f'{peak["simultaneous_cars"]} moving vehicles and {peak["simultaneous_pedestrians"]} moving pedestrians nearby.',
        ("See mixed traffic", f'./?run=city-life&at={timeline["mixed_at_s"]}'),
        ("Native movement", TAG + "results/city-life/POPULATION.md"))
    add("Roadside pace",
        f'In a matched test, median speed alongside the six drums rose from {slower["median_speed_kmh"]:.1f} to '
        f'{faster["alongside"]["median_speed_kmh"]:.1f} km/h. Minimum body clearance was '
        f'{faster["roadworks"]["minimum_cone_clearance_m"]:.2f} m. The driver held a stop for '
        f'{blocked["longest_stop_s"]:.1f} seconds at the blocked lane.',
        ("Synchronized before and after", "./media/roadworks-pace/roadside-comparison.mp4"),
        ("Conditions and limits", TAG + "results/roadworks-pace/README.md"))
    sections = read("results/roadworks-pace/matched-route-intervals.json")
    old_speed = sections["v04_high"]["shared_boulevard"]["speed_kmh"]
    new_speed = sections["clearance_high"]["shared_boulevard"]["speed_kmh"]
    add("Remaining slowdown",
        f'On the original route, median speed on a different boulevard segment fell from {old_speed:.1f} '
        f'to {new_speed:.1f} km/h. At 434 m the front detector reported a person. The three-camera test '
        'retained its input features and controls but has no image to verify that classification.',
        ("Frame-level measurements", TAG + "results/roadworks-pace/optical-slowdown-434m.json"),
        ("Other route sections", TAG + "results/roadworks-pace/matched-route-intervals.json"))
    add("Cone identification",
        f'On the original film, {cones["matched"]} fresh camera-and-lidar readings matched the six drums. '
        f'{len(cones["unmatched"])} were unmatched. Fresh readings covered '
        f'{cones["matched_within_20m"]}/{cones["within_20m_reference_actor_frames"]} nearby drum-frames.',
        ("Inspect the original exposure", f"./inspect.html?run=urban-pace&frame={old_frame}&camera=front"),
        ("Held-out checks", TAG + "results/cone-perception/README.md"))
    prior = read("results/distant-awareness/high-before.json")
    updated = read("results/distant-awareness/high-after.json")
    lead_old, = (e for e in prior["encounters"] if e["role"] == "lead")
    lead_new, = (e for e in updated["encounters"] if e["id"] == lead_old["id"])
    before = lead_old["first"]["display"]
    after = lead_new["first"]["display"]
    add("First ranged vehicle",
        f'On the same Urban Pace sensor tape, the lead car first appeared as a correctly placed 3D body at '
        f'{after["range_m"]:.1f} m, versus {before["range_m"]:.1f} m with the older display. The front '
        'camera can also show a bearing before the sensors have enough range evidence for a 3D body.',
        ("See the selected lead car", f'./?run=urban-pace&frame={after["frame"]}&track={after["source"]}&camera=front'),
        ("By class and distance", TAG + "results/distant-awareness/README.md"))
    return {"checks": checks, "scope": "The site replays closed-loop simulated drives. "
            "Population counts include occluded traffic. A 3D symbol needs measured range. "
            "First-sighting distances depend on the individual road user, view and occlusion."}


if __name__ == "__main__":
    path = ROOT / "web/media/city-life-review.json"
    path.write_text(json.dumps(scorecard(), ensure_ascii=False, indent=2) + "\n", encoding="utf8", newline="\n")
