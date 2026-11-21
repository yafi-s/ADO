"""Index an installed CARLA city recording in the two public drive catalogs."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf8", newline="\n")
    temp.replace(path)


def publish(web, directory, *, feature=False):
    media = (web / "media").resolve()
    recording = (media / directory).resolve()
    folder = recording.relative_to(media).as_posix()
    manifest = json.loads((recording / "manifest.json").read_bytes())
    entry = manifest["catalog_entry"]
    if not entry or entry["path"] != folder or entry["trial_id"] != manifest["trial_id"]:
        raise ValueError("Install the capture with a matching web/media destination first")
    for name, expected in manifest["sha256"].items():
        if Path(name).name != name or digest(recording / name) != expected:
            raise ValueError(f"Recording artifact differs: {name}")

    if manifest["policy"]["schema"] == "ado-city-policy-v1":
        entry["selection_note"] = ("One continuous Town03 development drive on the city training course. "
                                   "Navigation supplied the lane-change path around the bus.")
    base = f"./media/{folder}"
    summary = manifest["summary"]
    show = {key: summary[key] for key in ("scenario", "policy", "seed", "destination", "district", "outcome",
                                          "frames", "duration_s", "route_length_m", "progress_m",
                                          "collision_events", "red_light_violations", "yield_violations")}
    show.update(label=entry["label"], trial_id=entry["trial_id"], case_id=summary["scenario"],
                recording_path=folder, model_sha256=manifest["policy"]["sha256"], benchmark_member=False,
                selection_note=entry["selection_note"], cohort=entry["cohort"],
                quality_requested=manifest["quality_requested"], map=entry["map"], map_url=entry["map_url"],
                width=manifest["video"]["width"], height=manifest["video"]["height"],
                manifest_url=entry["manifest"], log_url=f"{base}/telemetry.json", video_url=f"{base}/chase.mp4",
                encounters_handled=sum(e["handled"] for e in summary["events"]),
                encounters_planned=len(manifest["required_event_kinds"]))
    pack = recording / "replay-pack.json.gz"
    if pack.is_file():
        entry["replay_pack_url"] = show["replay_pack_url"] = f"{base}/{pack.name}"
        entry["replay_pack_sha256"] = digest(pack)
    build_path = media / "build.json"
    catalog_path = media / "viewer-catalog.json"
    build = json.loads(build_path.read_bytes())
    build["showcases"] = [show, *(r for r in build["showcases"] if r["trial_id"] != show["trial_id"])]
    catalog = [e for e in json.loads(catalog_path.read_bytes()) if e["trial_id"] != entry["trial_id"]]
    catalog.insert(0 if feature else len(catalog), entry)
    write_json(build_path, build)
    write_json(catalog_path, catalog)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", help="Installed path relative to web/media")
    cli.add_argument("--web", type=Path, default=Path(__file__).resolve().parents[1] / "web")
    cli.add_argument("--feature", action="store_true", help="Make this the default recording")
    args = cli.parse_args()
    publish(args.web, args.directory, feature=args.feature)
