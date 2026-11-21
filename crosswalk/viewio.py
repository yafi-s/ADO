"""Local console transport and synchronized browser replay export."""
from __future__ import annotations

import base64
from contextlib import ExitStack
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from threading import Lock, Thread
from urllib.parse import urlsplit

import cv2


CAMERAS = ("front", "chase", "infrared", "left", "right")
POSTERS = ("front", "chase", "left", "right")
FPS = 10
WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
VEHICLE_ROOT = WEB_ROOT.parent / "assets" / "robotaxi" / "vehicle"
IPACE_ROOT = WEB_ROOT.parent / "assets" / "waymo-ipace"
VEHICLE_ASSETS = {
    "/vehicle-assets/meridian.glb": "meridian.glb",
    "/vehicle-assets/meridian-studio.png": "meridian-studio.png",
    "/vehicle-assets/waymo-ipace.glb": "waymo-ipace.glb",
    "/vehicle-assets/ipace-studio.png": "studio.png",
}
DRIVE_FIELDS = {"scenario", "destination", "seed", "crossing_delay"}
OPTIONAL_DRIVE_FIELDS = {"weather", "grip"}
WEATHER_OPTIONS = {"seeded", "ClearNoon", "WetNoon", "CloudySunset", "ClearNight"}
GRIP_OPTIONS = {0.5, 0.75, 1.0}
MAX_REQUEST_BYTES = 4096


def _json(data):
    return json.dumps(data, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _jpegs(shots):
    result = {}
    for name in CAMERAS:
        image = shots[name]
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype.name != "uint8":
            raise ValueError(f"{name} camera must be an RGB uint8 image")
        success, encoded = cv2.imencode(".jpg", image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not success:
            raise ValueError(f"Could not encode the {name} camera image")
        result[name] = encoded.tobytes()
    return result


class _ConsoleHandler(SimpleHTTPRequestHandler):
    timeout = 5
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".glb": "model/gltf-binary"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self._get_or_head(write_body=True)

    def do_HEAD(self):
        self._get_or_head(write_body=False)

    def translate_path(self, path):
        asset = VEHICLE_ASSETS.get(urlsplit(path).path)
        if asset is not None:
            root = IPACE_ROOT if asset in ("waymo-ipace.glb", "studio.png") else VEHICLE_ROOT
            return str(root / asset)
        return super().translate_path(path)

    def _get_or_head(self, write_body):
        path = urlsplit(self.path).path
        live = self.server.live
        if path == "/api/frame":
            frame = live.frame
            code = HTTPStatus.OK if frame is not None else HTTPStatus.SERVICE_UNAVAILABLE
            self._send_json(code, frame or b"{}", write_body)
        elif path == "/api/options" and live.options is not None:
            self._send_json(HTTPStatus.OK, live.options, write_body)
        elif path == "/api/status" and live.options is not None:
            self._send_json(HTTPStatus.OK, live.status, write_body)
        elif (path == "/vehicle-assets" or path.startswith("/vehicle-assets/")) and path not in VEHICLE_ASSETS:
            self.send_error(HTTPStatus.NOT_FOUND)
        elif path.endswith(".mp4") and self.headers.get("Range"):
            self._serve_video_range(self.translate_path(path), write_body)
        elif write_body:
            super().do_GET()
        else:
            super().do_HEAD()

    def _send_json(self, code, body, write_body=True):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if write_body:
            self.wfile.write(body)

    def _drive_error(self, code, detail):
        self._send_json(code, _json({"error": detail}))

    def do_POST(self):
        live = self.server.live
        path = urlsplit(self.path).path
        if path not in ("/api/drive", "/api/stop") or live.options is None:
            self._drive_error(HTTPStatus.NOT_FOUND, "Drive controls are unavailable.")
            return
        if not self._same_origin():
            self._drive_error(HTTPStatus.FORBIDDEN, "Drive controls are available from the local console.")
            return
        headers = self.headers
        if headers.get_content_type() != "application/json":
            self._drive_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Send an application/json request.")
            return
        lengths = headers.get_all("Content-Length", [])
        if (headers.get("Transfer-Encoding") or len(lengths) != 1
                or not re.fullmatch(r"[0-9]{1,10}", lengths[0])):
            self._drive_error(HTTPStatus.BAD_REQUEST, "A valid Content-Length is required.")
            return
        length = int(lengths[0])
        if length > MAX_REQUEST_BYTES:
            self._drive_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Drive request is too large.")
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("The drive request body is incomplete.")
            request = json.loads(raw.decode("utf-8"))
            if path == "/api/stop":
                if request != {}:
                    raise ValueError("Send an empty JSON object to stop the simulator.")
                live.request_stop()
                reply = b'{"stopping":true}'
            else:
                if not live.queue_drive(request):
                    self._drive_error(HTTPStatus.CONFLICT, "The simulator is stopping.")
                    return
                reply = b'{"queued":true}'
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            self._drive_error(HTTPStatus.BAD_REQUEST, "Send a valid JSON object.")
        except ValueError as error:
            self._drive_error(HTTPStatus.BAD_REQUEST, str(error))
        else:
            self._send_json(HTTPStatus.ACCEPTED, reply)

    def _same_origin(self):
        port = self.server.server_port
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0] not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return False
        origins = self.headers.get_all("Origin", [])
        fetch_site = self.headers.get("Sec-Fetch-Site")
        return (not origins or origins == [f"http://{hosts[0]}"]) and fetch_site in (None, "same-origin")

    def _serve_video_range(self, path, write_body):
        try:
            video = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        with video:
            size = os.fstat(video.fileno()).st_size
            request = re.fullmatch(r"bytes=(\d{0,20})-(\d{0,20})", self.headers["Range"])
            if request and request[1]:
                start = int(request[1])
                end = min(int(request[2]), size - 1) if request[2] else size - 1
            elif request and request[2]:
                start, end = max(0, size - int(request[2])), size - 1
            else:
                start, end = 1, 0
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            remaining = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(remaining))
            self.end_headers()
            if write_body:
                video.seek(start)
                while remaining:
                    chunk = video.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)


class Live:
    """Serve the console and replace its current frame with each publish."""

    def __init__(self, port, options=None):
        self.port = port
        self.options = _json(options) if options is not None else None
        self._scenarios = self._option_ids(options, "scenarios")
        self._destinations = self._option_ids(options, "destinations")
        self._weather = self._option_ids(options, "weather", WEATHER_OPTIONS) & WEATHER_OPTIONS
        self._grip = self._option_ids(options, "grip", GRIP_OPTIONS) & GRIP_OPTIONS
        self._lock = Lock()
        self._frame = None
        self._pending = None
        self._stopping = False
        self.set_status("idle")

    @staticmethod
    def _option_ids(options, key, default=()):
        return frozenset(
            option["id"] if isinstance(option, dict) else option
            for option in (options or {}).get(key, default)
        )

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/"

    def __enter__(self):
        self.clear()
        with self._lock:
            self._pending = None
            self._stopping = False
        self.set_status("idle")
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), _ConsoleHandler)
        self.server.live = self
        self.port = self.server.server_port
        self.thread = Thread(target=self.server.serve_forever, name="crosswalk-console", daemon=True)
        self.thread.start()
        return self

    @property
    def frame(self):
        with self._lock:
            return self._frame

    @property
    def status(self):
        with self._lock:
            if self._pending is not None:
                return _json(self._status_record("queued", settings=self._pending))
            return self._status

    @staticmethod
    def _status_record(state, **details):
        return {
            "state": state, "run_id": None, "settings": None, "outcome": None, "error": None,
            **details,
        }

    def set_status(self, state, **details):
        status = _json(self._status_record(state, **details))
        with self._lock:
            self._status = status

    def queue_drive(self, request):
        if (not isinstance(request, dict) or not DRIVE_FIELDS <= request.keys()
                or request.keys() - DRIVE_FIELDS - OPTIONAL_DRIVE_FIELDS):
            raise ValueError("Supply scenario, destination, seed, and crossing_delay.")
        if not isinstance(request["scenario"], str) or request["scenario"] not in self._scenarios:
            raise ValueError("Unknown scenario.")
        if not isinstance(request["destination"], str) or request["destination"] not in self._destinations:
            raise ValueError("Unknown destination.")
        seed = request["seed"]
        if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
            raise ValueError("Seed must be an integer from 0 to 2147483647.")
        delay = request["crossing_delay"]
        if type(delay) not in (int, float) or not 0 <= delay <= 5 or not math.isfinite(delay):
            raise ValueError("Crossing delay must be a number from 0 to 5 seconds.")
        normalized = {**request, "crossing_delay": float(delay)}
        if "weather" in request:
            weather = request["weather"]
            if not isinstance(weather, str) or weather not in self._weather:
                raise ValueError("Unknown weather preset.")
        if "grip" in request:
            grip = request["grip"]
            if type(grip) not in (int, float) or grip not in self._grip:
                raise ValueError("Choose a supported grip factor.")
            normalized["grip"] = float(grip)
        with self._lock:
            if self._stopping:
                return False
            self._pending = normalized
            return True

    def request_stop(self):
        with self._lock:
            self._stopping = True
            self._pending = None

    def stop_requested(self):
        with self._lock:
            return self._stopping

    def next_request(self):
        with self._lock:
            request, self._pending = self._pending, None
            if request is not None:
                self._status = _json(self._status_record("queued", settings=request))
            return request

    def publish(self, telemetry, shots):
        cameras = {name: base64.b64encode(image).decode("ascii") for name, image in _jpegs(shots).items()}
        frame = _json({**telemetry, "cameras": cameras})
        with self._lock:
            self._frame = frame

    def clear(self):
        with self._lock:
            self._frame = None

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.clear()


class Recording:
    """Collect a telemetry record and one frame from each camera per tick."""

    def __init__(self, path):
        self.path = Path(path).resolve()

    def __enter__(self):
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg is None:
            raise RuntimeError("FFmpeg is required to record a replay")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as opening:
            self.stage = Path(opening.enter_context(tempfile.TemporaryDirectory(
                prefix=".crosswalk-replay-", dir=self.path.parent)))
            self.rows = opening.enter_context((self.stage / "replay.json").open("wb"))
            self.rows.write(b"[")
            self.streams = {
                name: opening.enter_context((self.stage / f"{name}.mjpeg").open("wb"))
                for name in CAMERAS
            }
            self.count = 0
            self.failed = False
            self.resources = opening.pop_all()
        return self

    def append(self, telemetry, shots):
        row = _json(telemetry)
        images = _jpegs(shots)
        try:
            for name, stream in self.streams.items():
                stream.write(images[name])
            if self.count:
                self.rows.write(b",")
            else:
                for name in POSTERS:
                    (self.stage / f"{name}.jpg").write_bytes(images[name])
            self.rows.write(row)
        except OSError:
            self.failed = True
            raise
        self.count += 1

    def __exit__(self, exc_type, exc, traceback):
        with self.resources:
            if exc_type is not None or self.failed or not self.count:
                return
            self.rows.write(b"]\n")
            self.rows.close()
            for stream in self.streams.values():
                stream.close()
            for name in CAMERAS:
                self._encode(name)
            self._install()

    def _encode(self, name):
        command = [
            self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
            "-threads", "1", "-f", "mjpeg", "-framerate", str(FPS),
            "-i", str(self.stage / f"{name}.mjpeg"),
            "-filter_threads", "1", "-vf", "scale=out_range=tv,pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-color_range", "tv",
            "-preset", "veryfast", "-crf", "21", "-movflags", "+faststart",
            str(self.stage / f"{name}.mp4"),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"Could not encode the {name} camera: {result.stderr.strip()}")

    def _install(self):
        files = [f"{name}.mp4" for name in CAMERAS] + [f"{name}.jpg" for name in POSTERS]
        files.append("replay.json")
        self.path.mkdir(exist_ok=True)
        backups = self.stage / "previous"
        backups.mkdir()
        for name in files:
            if (self.path / name).exists():
                shutil.copy2(self.path / name, backups / name)
        installed = []
        try:
            for name in files:
                os.replace(self.stage / name, self.path / name)
                installed.append(name)
        except OSError:
            for name in reversed(installed):
                if (backups / name).exists():
                    os.replace(backups / name, self.path / name)
                else:
                    (self.path / name).unlink()
            raise
