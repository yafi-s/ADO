"""Attach a source-verified cone review to an already indexed native recording."""
import argparse
import json
from pathlib import Path

from tools.index_city_film import digest, write_json


def index(web, name):
    media = (web / "media").resolve()
    folder = (media / name).resolve()
    relative = folder.relative_to(media).as_posix()
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    file = folder / "cone-observations.json"
    data = json.loads(file.read_bytes())
    raw = manifest["raw_sha256"]
    if (data["schema_version"] != 1 or data["trial_id"] != manifest["trial_id"]
            or data["first_frame"] != manifest["first_frame"] or data["last_frame"] != manifest["last_frame"]
            or data["capture_frames"] != manifest["summary"]["frames"]
            or data["source_journal_sha256"] != raw["perception.jsonl"]):
        raise ValueError("Cone observations do not describe this recording")
    for camera, info in data["camera_streams"].items():
        if info["sha256"] != raw[f"{camera}.mjpeg"] or info["images"] != data["capture_frames"]:
            raise ValueError(f"Incorrect source camera for cone review: {camera}")
    frames = [row["frame"] for row in data["frames"]]
    if (frames != sorted(set(frames))
            or any(f < data["first_frame"] or f > data["last_frame"] for f in frames)):
        raise ValueError("Cone observations have invalid exposure numbers")
    fields = {"cone_review_url": f"./media/{relative}/{file.name}", "cone_review_sha256": digest(file)}
    catalog_path, build_path = media / "viewer-catalog.json", media / "build.json"
    catalog = json.loads(catalog_path.read_bytes())
    build = json.loads(build_path.read_bytes())
    entry, = (r for r in catalog if r["path"] == relative and r["trial_id"] == data["trial_id"])
    show, = (r for r in build["showcases"] if r["recording_path"] == relative and r["trial_id"] == data["trial_id"])
    for row in (entry, show, manifest, manifest["catalog_entry"]):
        row.update(fields)
    manifest["sha256"][file.name] = fields["cone_review_sha256"]
    write_json(manifest_path, manifest)
    write_json(catalog_path, catalog)
    write_json(build_path, build)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("name")
    cli.add_argument("--web", type=Path, default=Path(__file__).resolve().parents[1] / "web")
    args = cli.parse_args()
    index(args.web, args.name)
