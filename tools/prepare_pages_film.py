"""Install a verified full-quality release film into a staged GitHub Pages site."""
import hashlib
import json
from pathlib import Path
import shutil
from urllib.error import HTTPError
from urllib.request import urlopen


def prepare(web):
    folder = web / "media/long-city-drive"
    if not (folder / "manifest.json").is_file():
        return
    manifest = json.loads((folder / "manifest.json").read_bytes())
    master = manifest["master"]["chase"]
    if master["sha256"] == manifest["sha256"]["chase.mp4"]:
        return
    release = json.loads((web / "media/build.json").read_bytes())["release"]
    if release["label"] != "v0.2.0":
        return
    source = "https://github.com/yafi-s/ADO/releases/download/v0.2.0/ADO-Town03-Chase-1080p.mp4"
    try:
        response = urlopen(source, timeout=90)
    except HTTPError as error:
        if error.code == 404:
            print("Full-quality asset has not been published yet. Serving the verified compact film.")
            return
        raise
    target = folder / "chase-1080p.mp4"
    with response, target.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024**2)
    with target.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != master["sha256"]:
            raise ValueError("Downloaded chase master differs from the original CARLA capture")
    # Pages has a site-size budget. Serve one chase film and identify the
    # compact version in the archived build by its original digest.
    manifest["archive_chase"] = {
        "sha256": manifest["sha256"]["chase.mp4"], "video": manifest["video"],
        "url": "https://raw.githubusercontent.com/yafi-s/ADO/v0.2.0/web/media/long-city-drive/chase.mp4"
    }
    target.replace(folder / "chase.mp4")
    manifest["sha256"]["chase.mp4"] = master["sha256"]
    manifest["video"] = {key: value for key, value in master.items()
                         if key not in ("path", "file_bytes", "sha256")}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf8", newline="\n")
    build_path = web / "media/build.json"
    build = json.loads(build_path.read_bytes())
    for run in build.get("showcases", []):
        if run["trial_id"] == manifest["trial_id"]:
            run.update(width=master["width"], height=master["height"])
    build_path.write_text(json.dumps(build, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf8", newline="\n")
    print(f"Serving the verified {master['width']} x {master['height']} CARLA chase master.")


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1] / "web")
