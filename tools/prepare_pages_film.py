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
    "v0.6.0": ("city-progress", "ADO-City-Progress-1080p.mp4"),
}
LARGE_RECORDINGS = ("urban-pace", "surround-city", "long-city-drive")
SITE_LIMIT = 1_000_000_000
SITE_RESERVE = 1_000_000
MAX_RELEASE_ASSET_BYTES = 2 * 1024**3
# Keep at least the bit budget of the previous Pages film for the archived Urban Pace chase.
URBAN_CHASE_BUDGET = 108_000_000
CITY_LIFE_CHASE_BUDGET = 125_000_000
ARCHIVED_CITY_LIFE_CHASE_BUDGET = 133_000_000
CITY_PROGRESS_CHASE_BUDGET = 125_000_000
PAGES_ORIGIN = "https://yafi-s.github.io"
VIDEO_FIELDS = ("width", "height", "codec_name", "nb_read_frames", "pix_fmt", "r_frame_rate")
# Exact playback encodes from the completed v0.5 Pages deployment, keyed by native master SHA-256.
PUBLISHED_ENCODINGS = {
    "6fb0871f7f1e0319190c00a4c8eb222e1bda32b0481af0614c786400dbf4f640": (
        104_800_528, "f80773c7f3800ae0d30a43b0541f9bc7bbce4e67cde5815d0bcdd65dc7f1801f"),
    "2ef957161d7bae02cd5fda53e090c9157e34c1c5bec56ce7ba13c9a11beab703": (
        132_942_750, "adb6b84700b5525c93f8850187cf1da67b60c5c0efe01fa013fab022f1a0bc48"),
}


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


def restore_encode(cache_path, playback, master, budget):
    if cache_path is None:
        return None
    checksum = cache_path.with_suffix(".sha256")
    if not cache_path.is_file() or not checksum.is_file() or cache_path.stat().st_size > budget:
        return None
    try:
        if sha256(cache_path) != checksum.read_text(encoding="ascii").strip():
            return None
        video = verify_encode(cache_path, master)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    shutil.copyfile(cache_path, playback)
    return video


def published_encode(directory, playback, master, budget):
    published = PUBLISHED_ENCODINGS.get(master["sha256"])
    if published is not None and published[0] <= budget:
        size, digest = published
        url = f"{PAGES_ORIGIN}/ADO/media/{directory}/chase.mp4"
        try:
            if download(url, playback, {"file_bytes": size, "sha256": digest}):
                return verify_encode(playback, master)
        except (OSError, ValueError, subprocess.CalledProcessError):
            print(f"The pinned Pages encode for {directory} is unavailable. Using its release master.")
    return None


def store_encode(cache_path, playback, digest):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".partial")
    shutil.copyfile(playback, temporary)
    temporary.replace(cache_path)
    cache_path.with_suffix(".sha256").write_text(digest + "\n", encoding="ascii")


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


def fit_archive_data(web, build, revision, *, limit=None, directories=LARGE_RECORDINGS, verify_remote=False):
    if limit is None:
        limit = SITE_LIMIT - SITE_RESERVE
    if site_size(web) <= limit:
        return
    archive_revision_url(revision, "", "")

    catalog = web / "media/viewer-catalog.json"
    records = json.loads(catalog.read_bytes())
    runs = catalog_runs(build)
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
        replay_name = manifest.get("replay", "replay.json")
        if replay_name not in ("replay.json", "replay.json.gz"):
            raise ValueError(f"Invalid original replay filename at {folder}")
        for name, field in ((replay_name, "replay_url"), ("telemetry.json", "log_url")):
            if site_size(web) <= limit:
                return
            path = folder / name
            if not path.is_file():
                continue
            if sha256(path) != checks.get(name):
                raise ValueError(f"Archived public data differs from the capture manifest: {path}")
            local = f"./media/{directory}/{name}"
            url = archive_revision_url(revision, directory, name)
            if verify_remote:
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


def fit_archive_cameras(web, build, revision, limit, *, directories=LARGE_RECORDINGS):
    if site_size(web) <= limit:
        return
    catalog = web / "media/viewer-catalog.json"
    records = json.loads(catalog.read_bytes())
    # Keep Urban Pace and City Life chase videos on Pages. Prefer to keep City Life's onboard cameras local.
    candidates = []
    for entry in records:
        directory = entry["path"]
        if directory not in directories:
            continue
        for camera in entry["cameras"]:
            if camera not in {"chase", "front", "left", "right", "rear_left", "rear_right"}:
                raise ValueError(f"Unknown historical camera: {camera}")
            if directory in {"urban-pace", "city-life"} and camera == "chase":
                continue
            path = web / "media" / directory / f"{camera}.mp4"
            if path.is_file():
                priority = 2 if directory == "city-life" else int(camera == "chase")
                candidates.append((priority, -path.stat().st_size, directory, camera, path, entry))
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


def stage_chase(web, build, tag, *, budget=None, archive_revision=None, require_1080=False, cache_dir=None):
    directory, asset = RELEASE_FILMS[tag]
    folder = web / "media" / directory
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        if tag != "v0.2.0":
            raise ValueError(f"Missing the featured chase capture manifest at {manifest_path}")
        return
    manifest = json.loads(manifest_path.read_bytes())
    master = manifest["master"]["chase"]
    compact = folder / "chase.mp4"
    compact_digest = manifest["sha256"]["chase.mp4"]
    if sha256(compact) != compact_digest:
        raise ValueError("Committed chase film differs from its capture manifest")
    if require_1080 and ((master["width"], master["height"]) != (1920, 1080)
                         or not 0 < master["file_bytes"] < MAX_RELEASE_ASSET_BYTES
                         or not re.fullmatch(r"[0-9a-f]{64}", master["sha256"])):
        raise ValueError(f"Wrong native 1080p chase master at {manifest_path}")
    if (master["sha256"] == compact_digest
            or manifest.get("playback_chase", {}).get("source_sha256") == master["sha256"]):
        if require_1080 and (manifest["video"]["width"], manifest["video"]["height"]) != (1920, 1080):
            raise ValueError(f"Wrong staged 1080p film at {manifest_path}")
        return True

    source = f"{GITHUB_REPO}/releases/download/{tag}/{asset}"
    compact_size = compact.stat().st_size
    if budget is None:
        budget = SITE_LIMIT - SITE_RESERVE - (site_size(web) - compact_size)
    with TemporaryDirectory(prefix=".ado-pages-film-", dir=web.parent) as scratch:
        original = Path(scratch) / "master.mp4"
        playback = Path(scratch) / "playback.mp4"
        cache_path = (cache_dir / master["sha256"] / f"{budget}.mp4"
                      if cache_dir is not None and require_1080 else None)
        video = restore_encode(cache_path, playback, master, budget)
        cache_hit = video is not None
        if video is None and require_1080:
            video = published_encode(directory, playback, master, budget)
        kind = "pages-encode"
        if video is not None:
            print(f"Reused the verified 1080p Pages encode for {directory}.")
        elif not download(source, original, master):
            if require_1080:
                verify_encode(compact, master)
            print("Full-quality asset has not been published yet. Serving the verified compact film.")
            return False
        elif original.stat().st_size <= budget:
            playback = original
            video = {key: master[key] for key in VIDEO_FIELDS}
            kind = "release-master"
        elif budget > 0 and (require_1080 or budget >= compact_size * 1.10
                             or manifest["video"]["height"] < master["height"]):
            video = encode_for_site(original, playback, master, budget)
        else:
            print("Serving the verified compact film within the Pages budget. The chase master is available in the release.")
            return False
        digest = sha256(playback)
        if cache_path is not None and kind == "pages-encode" and not cache_hit:
            store_encode(cache_path, playback, digest)
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
    return True


def stage_city_life(web, build, revision, cache_dir=None):
    city = web / "media/city-life/chase.mp4"
    urban = web / "media/urban-pace/chase.mp4"
    budget = max(CITY_LIFE_CHASE_BUDGET, city.stat().st_size)
    archive_budget = max(URBAN_CHASE_BUDGET, urban.stat().st_size)
    limit = SITE_LIMIT - SITE_RESERVE - (budget - city.stat().st_size) - (archive_budget - urban.stat().st_size)
    fit_archive_data(web, build, revision, limit=limit, directories=(*LARGE_RECORDINGS, "city-life"), verify_remote=True)
    fit_archive_cameras(web, build, revision, limit)
    if site_size(web) > limit:
        raise ValueError(f"The city films and their evidence exceed the Pages budget ({site_size(web)} bytes)")
    urban_ready = stage_chase(web, build, "v0.4.0", budget=archive_budget, archive_revision=revision,
                             require_1080=True, cache_dir=cache_dir)
    life_ready = stage_chase(web, build, "v0.5.0", archive_revision=revision, require_1080=True, cache_dir=cache_dir)
    return urban_ready and life_ready


def stage_city_progress(web, build, revision, cache_dir=None):
    budgets = {
        "v0.4.0": URBAN_CHASE_BUDGET,
        "v0.5.0": ARCHIVED_CITY_LIFE_CHASE_BUDGET,
        "v0.6.0": CITY_PROGRESS_CHASE_BUDGET,
    }
    limit = SITE_LIMIT - SITE_RESERVE
    for tag, budget in budgets.items():
        directory, _ = RELEASE_FILMS[tag]
        size = (web / "media" / directory / "chase.mp4").stat().st_size
        budgets[tag] = max(budget, size)
        limit -= budgets[tag] - size
    archived = (*LARGE_RECORDINGS, "city-life")
    fit_archive_data(web, build, revision, limit=limit, directories=(*archived, "city-progress"), verify_remote=True)
    fit_archive_cameras(web, build, revision, limit, directories=archived)
    total = site_size(web)
    if total > limit:
        raise ValueError(f"The city films and their evidence exceed the Pages budget ({total} bytes)")
    staged = [stage_chase(web, build, tag, budget=budget, archive_revision=revision,
                          require_1080=True, cache_dir=cache_dir) for tag, budget in budgets.items()]
    return all(staged)


def prepare(web, archive_revision=None, cache_dir=None):
    build_path = web / "media/build.json"
    if not build_path.is_file():
        return
    build = json.loads(build_path.read_bytes())
    tag = build.get("release", {}).get("label")
    if tag == "v0.6.0":
        return stage_city_progress(web, build, archive_revision, cache_dir)
    if tag == "v0.5.0":
        return stage_city_life(web, build, archive_revision, cache_dir)
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
    parser.add_argument("--cache-dir", type=Path, help="Directory to reuse verified 1080p encodes by source hash")
    args = parser.parse_args()
    staged = prepare(args.web, args.archive_revision, args.cache_dir)
    if staged and args.cache_dir is not None and args.cache_dir.is_dir() and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf8") as output:
            output.write("cache-ready=true\n")
