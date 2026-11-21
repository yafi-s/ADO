r"""Play private CARLA transfer recordings in the local ADO viewer.

Run from the research checkout, using one or more completed result folders:

    python -m waymo_training.replay --results "A:\ADO-Private-Research\waymo-transfer-20261120" "A:\ADO-Private-Research\waymo-transfer-v2-20261120" --port 8766 --open-browser

Open the printed address to watch a trial. The Builds & logs page links the
archived reports, summaries, and controller traces. Stop the server with Ctrl+C.
Playback uses the archive as recorded and requires only Python's standard library.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (
    "index.html", "ado-home.css", "ado-home.js", "perception-scene.js",
    "builds.html", "builds.css", "builds.js", "build-data.js",
    "benchmarks.html", "benchmarks.css", "benchmarks.js",
    "city.html", "city.css", "city-view.js",
    "drive.html", "console.css", "console.js", "sensor.js",
    "vehicle.html", "vehicle.css", "media/town.json",
    "fonts/Barlow-Regular.ttf", "fonts/Barlow-SemiBold.ttf",
    "fonts/BarlowCondensed-SemiBold.ttf",
    "vendor/city-view/three-view.js", "vendor/model-viewer/model-viewer.min.js",
)
VEHICLE_FILES = {
    "meridian.glb": "robotaxi/vehicle/meridian.glb",
    "meridian-studio.png": "robotaxi/vehicle/meridian-studio.png",
    "waymo-ipace.glb": "waymo-ipace/waymo-ipace.glb",
    "ipace-studio.png": "waymo-ipace/studio.png",
}
CAMERAS = ("chase", "front", "left", "right", "infrared")
SCENARIO_LABELS = {"lead": "Stopped lead vehicle", "pedestrian": "Pedestrian crossing", "clear": "Clear road"}
SCENARIO_PRIORITY = {"lead": 0, "pedestrian": 1, "clear": 2}
MIME_TYPES = {
    ".glb": "model/gltf-binary",
    ".js": "text/javascript",
    ".json": "application/json",
    ".mp4": "video/mp4",
    ".ttf": "font/ttf",
}


def _plain_file(path: Path) -> Path:
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError(f"Expected a regular archive or viewer file: {path.name}")
    return path


def _read_object(path: Path) -> dict:
    value = json.loads(_plain_file(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def _hex_digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError(f"Missing or invalid {name} SHA-256 in report.json")
    return value


def _viewer_source() -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True,
            check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"commit": revision, "scope": "Viewer checkout base revision"}


def _archive_routes(root: Path, archive: str) -> tuple[dict, list, list, dict]:
    report = root / "report.json"
    report_data = _read_object(report)
    metadata = report_data["checkpoint_metadata"]
    checkpoint = _hex_digest(report_data["checkpoint_sha256"], "portable checkpoint")
    source_checkpoint = _hex_digest(metadata["source_checkpoint_sha256"], "training checkpoint")
    snapshot = _hex_digest(metadata["dataset_source_sha256"], "training feature snapshot")
    tracker = report_data.get("tracker_name", "prototype")
    version = "v2" if tracker == "calibrated-v2" else "baseline" if tracker == "prototype" else tracker
    producer = report_data["producer"]
    policy = report_data["policy"]
    report_route = f"/records/{archive}/report.json"
    routes = {report_route: report}
    evidence = [{"label": f"{version} transfer report", "url": f".{report_route}"}]
    runs = []
    for trial in report_data["trials"]:
        name = trial["path"]
        if not re.fullmatch(r"[a-z0-9_-]+", name):
            raise ValueError(f"Invalid trial folder in {root.name}")
        record_id = f"{archive}-{name}"
        trial_dir = root / name
        recording = f"{archive}/{name}"
        record_prefix = f"/records/{recording}"
        if f"{record_prefix}/summary.json" in routes:
            raise ValueError(f"Repeated trial folder in {root.name}")
        summary_path = trial_dir / "summary.json"
        summary = _read_object(summary_path)
        if summary != {key: value for key, value in trial.items() if key != "path"}:
            raise ValueError(f"Trial summary and transfer report differ for {name}")
        routes[f"{record_prefix}/summary.json"] = summary_path
        routes[f"{record_prefix}/trace.json"] = _plain_file(trial_dir / "trace.json")

        movie_prefix = f"/media/{recording}"
        replay = trial_dir / "replay"
        routes[f"{movie_prefix}/replay.json"] = _plain_file(replay / "replay.json")
        cameras = []
        for camera in CAMERAS:
            clip = replay / f"{camera}.mp4"
            poster = replay / f"{camera}.jpg"
            if clip.is_file():
                routes[f"{movie_prefix}/{clip.name}"] = _plain_file(clip)
                cameras.append(camera)
            if poster.is_file():
                routes[f"{movie_prefix}/{poster.name}"] = _plain_file(poster)
        if not {"front", "chase"}.issubset(cameras):
            raise ValueError(f"Missing front or chase recording for {name}")

        speed = summary.get("requested_handoff_speed_mps", summary.get("initial_velocity_set_once_mps"))
        scene = SCENARIO_LABELS.get(summary["scenario"], summary["scenario"])
        start = "Standstill" if speed == 0 else f"Rolling {speed:g} m/s"
        outcome = summary["outcome"].replace("_", " ")
        label = f"Waymo transfer {version} · {scene} · {start} · {outcome}"
        summary_url = f".{record_prefix}/summary.json"
        trace_url = f".{record_prefix}/trace.json"
        evidence.append({"label": f"{label} summary", "url": summary_url})
        runs.append({
            "trial_id": record_id,
            "archive_trial_id": name,
            "label": label,
            "recording_path": recording,
            "cameras": cameras,
            "case_id": summary["scenario"],
            "cohort": "research_transfer",
            "seed": summary["seed"],
            "policy": policy,
            "tracker": tracker,
            "initial_speed_mps": speed,
            "outcome": summary["outcome"],
            "duration_s": summary["duration_s"],
            "progress_m": summary["progress_m"],
            "model_progress_m": summary.get("model_progress_m"),
            "model_sha256": checkpoint,
            "source_checkpoint_sha256": source_checkpoint,
            "dataset_source_sha256": snapshot,
            "summary_url": summary_url,
            "trace_url": trace_url,
            "log_url": trace_url,
            "report_url": f".{report_route}",
        })
    if not runs:
        raise ValueError(f"No completed trials in {root.name}")
    runs.sort(key=lambda run: (run["initial_speed_mps"] == 0, SCENARIO_PRIORITY.get(run["case_id"], 3)))
    return routes, runs, evidence, producer


def _routes(results) -> tuple[dict, str]:
    routes = {f"/{path}": _plain_file(ROOT / "web" / path) for path in PUBLIC_FILES}
    routes.update({
        f"/vehicle-assets/{name}": _plain_file(ROOT / "assets" / path)
        for name, path in VEHICLE_FILES.items()
    })
    runs, evidence, producers = [], [], []
    resolved = set()
    for index, directory in enumerate(results, start=1):
        root = Path(directory).resolve(strict=True)
        if root in resolved:
            raise ValueError("Each result directory must be different")
        resolved.add(root)
        files, archive_runs, archive_evidence, producer = _archive_routes(root, f"transfer-{index}")
        routes.update(files)
        runs.extend(archive_runs)
        evidence.extend(archive_evidence)
        producers.append(producer)
    if not runs:
        raise ValueError("Provide at least one completed transfer result directory")
    runs.sort(key=lambda run: run["tracker"] != "calibrated-v2")
    catalog = [
        {"trial_id": run["trial_id"], "path": run["recording_path"], "label": run["label"],
         "cameras": run["cameras"]}
        for run in runs
    ]
    build = {
        "status": "research_transfer",
        "title": "Private Waymo-trained model transfer · CARLA",
        "description": "Non-commercial research. Recorded CARLA transfer tests with Waymo-trained "
                       "weights. The viewer replays recorded model outputs and simulator telemetry.",
        "source": _viewer_source(),
        "simulator": producers[0] if all(item == producers[0] for item in producers) else {},
        "release": None,
        "evidence": evidence,
        "checkpoints": {
            f"Waymo trained · {run['model_sha256'][:12]}": {
                "sha256": run["model_sha256"],
                "source_checkpoint_sha256": run["source_checkpoint_sha256"],
                "dataset_source_sha256": run["dataset_source_sha256"],
            }
            for run in runs
        },
        "runs": runs,
    }
    for name, document in (("build", build), ("viewer-catalog", catalog), ("catalog", catalog)):
        routes[f"/media/{name}.json"] = json.dumps(document, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return routes, catalog[0]["trial_id"]


def _byte_range(header: str, size: int) -> tuple[int, int] | None:
    match = re.fullmatch(r"bytes=(\d{0,20})-(\d{0,20})", header)
    if not match:
        return None
    first, last = match.groups()
    if first:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    elif last:
        start, end = max(0, size - int(last)), size - 1
    else:
        return None
    return (start, end) if start <= end else None


class _ReplayHandler(BaseHTTPRequestHandler):
    timeout = 10

    def do_GET(self):
        self._respond(write_body=True)

    def do_HEAD(self):
        self._respond(write_body=False)

    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def _is_local(self):
        host = self.headers.get_all("Host", [])
        port = self.server.server_port
        return (
            host in ([f"127.0.0.1:{port}"], [f"localhost:{port}"])
            and self.headers.get_all("Origin", []) in ([], [f"http://{host[0]}"])
            and self.headers.get("Sec-Fetch-Site") in (None, "none", "same-origin")
        )

    def _respond(self, write_body):
        if not self._is_local():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        request, _, query = self.path.partition("?")
        if request == "/" and not query:
            self.send_response(HTTPStatus.TEMPORARY_REDIRECT)
            self.send_header("Location", f"/?run={self.server.default_run}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        resource = self.server.routes.get("/index.html" if request == "/" else request)
        if resource is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            if isinstance(resource, bytes):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resource)))
                self.end_headers()
                if write_body:
                    self.wfile.write(resource)
            else:
                self._send_file(resource, write_body)
        except (ConnectionError, TimeoutError):
            # Scrubbing or swapping cameras can cancel an outstanding video request.
            return

    def _send_file(self, resource, write_body):
        try:
            stream = _plain_file(resource).open("rb")
        except (OSError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        with stream:
            size = os.fstat(stream.fileno()).st_size
            is_video = resource.suffix == ".mp4"
            range_header = self.headers.get("Range") if is_video else None
            selected = _byte_range(range_header, size) if range_header is not None else None
            if range_header is not None and selected is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = selected or (0, size - 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT if selected else HTTPStatus.OK)
            content_type = MIME_TYPES.get(resource.suffix) or mimetypes.guess_type(resource.name)[0]
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            if is_video:
                self.send_header("Accept-Ranges", "bytes")
            if selected:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not write_body:
                return
            if selected is None:
                shutil.copyfileobj(stream, self.wfile)
                return
            stream.seek(start)
            remaining = end - start + 1
            while remaining and (chunk := stream.read(min(remaining, 65536))):
                self.wfile.write(chunk)
                remaining -= len(chunk)


class ReplayServer(ThreadingHTTPServer):
    """Serve the fixed public assets and explicitly enumerated transfer artifacts."""

    def __init__(self, port, results):
        self.routes, self.default_run = _routes(results)
        super().__init__(("127.0.0.1", port), _ReplayHandler)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_port}/?run={self.default_run}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", nargs="+", type=Path, required=True, metavar="DIR")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        server = ReplayServer(args.port, args.results)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    with server:
        print(f"Private CARLA transfer replay\n{server.url}\nPress Ctrl+C to stop.", flush=True)
        if args.open_browser:
            webbrowser.open(server.url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
