"""Stage verified recordings within the GitHub Pages size limit."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen


GITHUB_REPO = "https://github.com/yafi-s/ADO"
ARCHIVE_REPO = "https://raw.githubusercontent.com/yafi-s/ADO"
RELEASE_FILMS = {
    "v0.2.0": ("long-city-drive", "ADO-Town03-Chase-1080p.mp4"),
    "v0.3.0": ("surround-city", "ADO-Surround-City-1080p.mp4"),
    "v0.4.0": ("urban-pace", "ADO-Urban-Pace-1080p.mp4"),
    "v0.5.0": ("city-life", "ADO-City-Life-1080p.mp4"),
}
LARGE_RECORDINGS = ("urban-pace", "surround-city", "long-city-drive")
SITE_LIMIT = 1_000_000_000
SITE_RESERVE = 1_000_000
# Keep at least the bit budget of the previous Pages film for the archived Urban Pace chase.
URBAN_CHASE_BUDGET = 108_000_000
CITY_LIFE_CHASE_BUDGET = 125_000_000
PAGES_ORIGIN = "https://yafi-s.github.io"
VIDEO_FIELDS = ("width", "height", "codec_name", "nb_read_frames", "pix_fmt", "r_frame_rate")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8", newline="\n")


def site_size(web):
    return sum(path.stat().st_size for path in web.rglob("*") if path.is_file())


def download(source, target, master):
    try:
        response = urlopen(source, timeout=90)
    except HTTPError as error:
        if error.code == 404:
            error.close()
            return False
        raise
    with response, target.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024**2)
    if target.stat().st_size != master["file_bytes"] or sha256(target) != master["sha256"]:
        raise ValueError("Released chase master differs from the original CARLA capture")
    return True


def verify_encode(path, master):
    result = subprocess.check_output([
        "ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=" + ",".join(VIDEO_FIELDS), "-of", "json", str(path),
    ])
    stream, = json.loads(result)["streams"]
    if any(stream.get(field) != master[field] for field in VIDEO_FIELDS):
        raise ValueError(f"Pages chase encode differs from the verified timeline or frame size: {stream}")
    return stream


def encode_for_site(source, target, master, budget):
    numerator, denominator = map(int, master["r_frame_rate"].split("/"))
    seconds = int(master["nb_read_frames"]) * denominator / numerator
    # The rate targets the video payload and leaves room for the MP4 index.
    kbps = int(budget * 8 * 0.97 / (1000 * seconds))
    ffmpeg = [
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-y", "-threads", "2", "-i", str(source),
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-passlogfile", str(target.with_suffix(".pass")),
    ]
    for _ in range(3):
        rate = ["-b:v", f"{max(1, kbps)}k"]
        subprocess.run([*ffmpeg, *rate, "-pass", "1", "-f", "null", os.devnull], check=True)
        subprocess.run([*ffmpeg, *rate, "-pass", "2", "-movflags", "+faststart", str(target)], check=True)
        actual = target.stat().st_size
        if actual <= budget:
            return verify_encode(target, master)
        kbps = int(kbps * budget / actual * 0.93)
    raise ValueError(f"Encoded chase film exceeds its {budget} byte Pages allocation")


def verify_archive_url(url, path):
    size = path.stat().st_size
    end = min(size, 32) - 1
    request = Request(url, headers={"Range": f"bytes=0-{end}", "Origin": PAGES_ORIGIN})
    with path.open("rb") as local, urlopen(request, timeout=30) as remote:
        if (remote.status != 206 or remote.headers.get("Content-Range") != f"bytes 0-{end}/{size}"
                or remote.headers.get("Access-Control-Allow-Origin") not in ("*", PAGES_ORIGIN)
                or remote.read(end + 1) != local.read(end + 1)):
            raise ValueError(f"Archive URL does not serve the capture with seekable byte ranges: {url}")


def archive_revision_url(revision, directory, name):
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pages needs the immutable checkout commit to link archived recording data")
    return f"{ARCHIVE_REPO}/{revision}/web/media/{directory}/{name}"


def catalog_runs(build):
    return [run for section in ("runs", "showcases", "development_runs", "native_replays")
            for run in build.get(section, [])]


def fit_archive_data(web, build, revision, *, limit=None, city_life=False):
    if limit is None:
        limit = SITE_LIMIT - SITE_RESERVE
    if site_size(web) <= limit:
        return
    archive_revision_url(revision, "", "")

    catalog = web / "media/viewer-catalog.json"
    records = json.loads(catalog.read_bytes())
    runs = catalog_runs(build)
    directories = (*LARGE_RECORDINGS, "city-life") if city_life else LARGE_RECORDINGS
    for directory in directories:
        folder = web / "media" / directory
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_bytes())
        checks = manifest["sha256"]
        entry = next((entry for entry in records if entry["path"] == directory), None)
        pack = folder / "replay-pack.json.gz"
        pack_digest = entry and entry.get("replay_pack_sha256")
        if (entry is None or entry.get("replay_pack_url") != f"./media/{directory}/{pack.name}"
                or not pack.is_file() or sha256(pack) != pack_digest
                or checks.get(pack.name, pack_digest) != pack_digest):
            raise ValueError(f"Missing the verified local replay pack at {folder}")
        for name, field in (("replay.json", "replay_url"), ("telemetry.json", "log_url")):
            if site_size(web) <= limit:
                return
            path = folder / name
            if not path.is_file():
                continue
            if sha256(path) != checks.get(name):
                raise ValueError(f"Archived public data differs from the capture manifest: {path}")
            local = f"./media/{directory}/{name}"
            url = archive_revision_url(revision, directory, name)
            if city_life:
                verify_archive_url(url, path)
            entry[field] = url
            for run in runs:
                if run.get("recording_path") == directory:
                    run[field] = url
            for item in build.get("evidence", []):
                if item["url"] == local:
                    item["url"] = url
            manifest.setdefault("external_artifacts", {})[name] = url
            write_json(manifest_path, manifest)
            write_json(catalog, records)
            write_json(web / "media/build.json", build)
            path.unlink()
            print(f"Linked the original {directory}/{name} at the checkout commit.")


def fit_archive_cameras(web, build, revision, limit):
    if site_size(web) <= limit:
        return
    catalog = web / "media/viewer-catalog.json"
    records = json.loads(catalog.read_bytes())
    # Keep the current and archived Urban Pace chase videos on Pages. Prefer local chase playback for older drives.
    candidates = []
    for entry in records:
        directory = entry["path"]
        if directory not in LARGE_RECORDINGS:
            continue
        for camera in entry["cameras"]:
            if camera not in {"chase", "front", "left", "right", "rear_left", "rear_right"}:
                raise ValueError(f"Unknown historical camera: {camera}")
            if (directory, camera) == ("urban-pace", "chase"):
                continue
            path = web / "media" / directory / f"{camera}.mp4"
            if path.is_file():
                candidates.append((camera == "chase", -path.stat().st_size, directory, camera, path, entry))
    runs = catalog_runs(build)
    for _, _, directory, camera, path, entry in sorted(candidates, key=lambda row: row[:4]):
        if site_size(web) <= limit:
            return
        manifest_path = path.with_name("manifest.json")
        manifest = json.loads(manifest_path.read_bytes())
        if sha256(path) != manifest["sha256"][path.name]:
            raise ValueError(f"Archived camera differs from the capture manifest: {path}")
        url = archive_revision_url(revision, directory, path.name)
        verify_archive_url(url, path)
        local = f"./media/{directory}/{path.name}"
        entry.setdefault("camera_urls", {})[camera] = url
        for run in runs:
            if run.get("recording_path") == directory:
                run.setdefault("camera_urls", {})[camera] = url
                if run.get("video_url") == local:
                    run["video_url"] = url
        for item in build.get("evidence", []):
            if item["url"] == local:
                item["url"] = url
        manifest.setdefault("external_artifacts", {})[path.name] = url
        write_json(manifest_path, manifest)
        write_json(catalog, records)
        write_json(web / "media/build.json", build)
        path.unlink()
        print(f"Linked the original {directory}/{path.name} at the checkout commit.")


def stage_chase(web, build, tag, *, budget=None, archive_revision=None, require_1080=False):
    directory, asset = RELEASE_FILMS[tag]
    folder = web / "media" / directory
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        if tag in {"v0.3.0", "v0.4.0", "v0.5.0"}:
            raise ValueError(f"Missing the featured chase capture manifest at {manifest_path}")
        return
    manifest = json.loads(manifest_path.read_bytes())
    master = manifest["master"]["chase"]
    compact = folder / "chase.mp4"
    compact_digest = manifest["sha256"]["chase.mp4"]
    if sha256(compact) != compact_digest:
        raise ValueError("Committed chase film differs from its capture manifest")
    if require_1080 and ((master["width"], master["height"]) != (1920, 1080)
                         or master["file_bytes"] >= 2_000_000_000):
        raise ValueError(f"Wrong native 1080p chase master at {manifest_path}")
    if (master["sha256"] == compact_digest
            or manifest.get("playback_chase", {}).get("source_sha256") == master["sha256"]):
        if require_1080 and (manifest["video"]["width"], manifest["video"]["height"]) != (1920, 1080):
            raise ValueError(f"Wrong staged 1080p film at {manifest_path}")
        return

    source = f"{GITHUB_REPO}/releases/download/{tag}/{asset}"
    compact_size = compact.stat().st_size
    if budget is None:
        budget = SITE_LIMIT - SITE_RESERVE - (site_size(web) - compact_size)
    with TemporaryDirectory(prefix=".ado-pages-film-", dir=web.parent) as scratch:
        original = Path(scratch) / "master.mp4"
        if not download(source, original, master):
            if require_1080:
                verify_encode(compact, master)
            print("Full-quality asset has not been published yet. Serving the verified compact film.")
            return
        if original.stat().st_size <= budget:
            playback = original
            video = {key: master[key] for key in VIDEO_FIELDS}
            kind = "release-master"
        elif budget > 0 and (require_1080 or budget >= compact_size * 1.10
                             or manifest["video"]["height"] < master["height"]):
            playback = Path(scratch) / "playback.mp4"
            video = encode_for_site(original, playback, master, budget)
            kind = "pages-encode"
        else:
            print("Serving the verified compact film within the Pages budget. The chase master is available in the release.")
            return
        digest = sha256(playback)
        playback.replace(compact)

    manifest["archive_chase"] = {
        "sha256": compact_digest, "video": manifest["video"],
        "url": (archive_revision_url(archive_revision, directory, compact.name) if archive_revision
                else f"{ARCHIVE_REPO}/{tag}/web/media/{directory}/chase.mp4"),
    }
    manifest["playback_chase"] = {"kind": kind, "source_sha256": master["sha256"], "source": source}
    manifest["sha256"]["chase.mp4"] = digest
    manifest["video"] = video
    if "camera_videos" in manifest:
        manifest["camera_videos"]["chase"] = video
    write_json(manifest_path, manifest)
    for run in build.get("showcases", []):
        if run["trial_id"] == manifest["trial_id"]:
            run.update(width=video["width"], height=video["height"])
    write_json(web / "media/build.json", build)
    total = site_size(web)
    if total > SITE_LIMIT:
        raise ValueError(f"Staged Pages site exceeds {SITE_LIMIT} bytes ({total} bytes)")
    print(f"Serving {video['width']} x {video['height']} chase footage as {kind} ({total} bytes for Pages).")


def stage_city_life(web, build, revision):
    city = web / "media/city-life/chase.mp4"
    urban = web / "media/urban-pace/chase.mp4"
    budget = max(CITY_LIFE_CHASE_BUDGET, city.stat().st_size)
    archive_budget = max(URBAN_CHASE_BUDGET, urban.stat().st_size)
    limit = SITE_LIMIT - SITE_RESERVE - (budget - city.stat().st_size) - (archive_budget - urban.stat().st_size)
    fit_archive_data(web, build, revision, limit=limit, city_life=True)
    fit_archive_cameras(web, build, revision, limit)
    if site_size(web) > limit:
        raise ValueError(f"The city films and their evidence exceed the Pages budget ({site_size(web)} bytes)")
    stage_chase(web, build, "v0.4.0", budget=archive_budget, archive_revision=revision, require_1080=True)
    stage_chase(web, build, "v0.5.0", archive_revision=revision, require_1080=True)


def prepare(web, archive_revision=None):
    build_path = web / "media/build.json"
    if not build_path.is_file():
        return
    build = json.loads(build_path.read_bytes())
    tag = build.get("release", {}).get("label")
    if tag == "v0.5.0":
        stage_city_life(web, build, archive_revision)
        return
    if tag == "v0.4.0":
        fit_archive_data(web, build, archive_revision)
    total = site_size(web)
    if total > SITE_LIMIT:
        raise ValueError(f"The committed Pages site already exceeds {SITE_LIMIT} bytes ({total} bytes)")
    if tag in RELEASE_FILMS:
        stage_chase(web, build, tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", type=Path, default=Path(__file__).resolve().parents[1] / "web")
    parser.add_argument("--archive-revision", help="Git commit serving the original recording data")
    args = parser.parse_args()
    prepare(args.web, args.archive_revision)
