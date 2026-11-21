"""Stage the release-verified chase film within the GitHub Pages size limit."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import urlopen


GITHUB_REPO = "https://github.com/yafi-s/ADO"
ARCHIVE_REPO = "https://raw.githubusercontent.com/yafi-s/ADO"
RELEASE_FILMS = {
    "v0.2.0": ("long-city-drive", "ADO-Town03-Chase-1080p.mp4"),
    "v0.3.0": ("surround-city", "ADO-Surround-City-1080p.mp4"),
}
SITE_LIMIT = 1_000_000_000
SITE_RESERVE = 1_000_000
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


def prepare(web):
    build_path = web / "media/build.json"
    if not build_path.is_file():
        return
    build = json.loads(build_path.read_bytes())
    tag = build.get("release", {}).get("label")
    total = site_size(web)
    if total > SITE_LIMIT:
        raise ValueError(f"The committed Pages site already exceeds {SITE_LIMIT} bytes ({total} bytes)")
    if tag not in RELEASE_FILMS:
        return
    directory, asset = RELEASE_FILMS[tag]
    folder = web / "media" / directory
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        if tag == "v0.3.0":
            raise ValueError(f"Missing the featured chase capture manifest at {manifest_path}")
        return
    manifest = json.loads(manifest_path.read_bytes())
    master = manifest["master"]["chase"]
    compact = folder / "chase.mp4"
    compact_digest = manifest["sha256"]["chase.mp4"]
    if sha256(compact) != compact_digest:
        raise ValueError("Committed chase film differs from its capture manifest")
    if (master["sha256"] == compact_digest
            or manifest.get("playback_chase", {}).get("source_sha256") == master["sha256"]):
        return

    source = f"{GITHUB_REPO}/releases/download/{tag}/{asset}"
    compact_size = compact.stat().st_size
    budget = SITE_LIMIT - SITE_RESERVE - (total - compact_size)
    with TemporaryDirectory(prefix=".ado-pages-film-", dir=web.parent) as scratch:
        original = Path(scratch) / "master.mp4"
        if not download(source, original, master):
            print("Full-quality asset has not been published yet. Serving the verified compact film.")
            return
        if original.stat().st_size <= budget:
            playback = original
            video = {key: master[key] for key in VIDEO_FIELDS}
            kind = "release-master"
        elif budget >= compact_size * 1.10 or manifest["video"]["height"] < master["height"]:
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
        "url": f"{ARCHIVE_REPO}/{tag}/web/media/{directory}/chase.mp4",
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
    write_json(build_path, build)
    total = site_size(web)
    if total > SITE_LIMIT:
        raise ValueError(f"Staged Pages site exceeds {SITE_LIMIT} bytes ({total} bytes)")
    print(f"Serving {video['width']} x {video['height']} chase footage as {kind} ({total} bytes for Pages).")


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1] / "web")
