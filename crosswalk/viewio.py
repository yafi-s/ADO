"""Local console transport and synchronized browser replay export."""
from __future__ import annotations

import base64
from contextlib import ExitStack
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from threading import Thread
from urllib.parse import urlsplit

import cv2


CAMERAS = ("front", "chase", "infrared")
POSTERS = ("front", "chase")
FPS = 10
WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self._get_or_head(write_body=True)

    def do_HEAD(self):
        self._get_or_head(write_body=False)

    def _get_or_head(self, write_body):
        path = urlsplit(self.path).path
        if path == "/api/frame":
            frame = self.server.live.frame
            self.send_response(HTTPStatus.OK if frame is not None else HTTPStatus.SERVICE_UNAVAILABLE)
            body = frame if frame is not None else b"{}"
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if write_body:
                self.wfile.write(body)
        elif path.endswith(".mp4") and self.headers.get("Range"):
            self._serve_video_range(self.translate_path(path), write_body)
        elif write_body:
            super().do_GET()
        else:
            super().do_HEAD()

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

    def __init__(self, port):
        self.port = port
        self.frame = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/"

    def __enter__(self):
        self.clear()
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), _ConsoleHandler)
        self.server.live = self
        self.port = self.server.server_port
        self.thread = Thread(target=self.server.serve_forever, name="crosswalk-console", daemon=True)
        self.thread.start()
        return self

    def publish(self, telemetry, shots):
        cameras = {name: base64.b64encode(image).decode("ascii") for name, image in _jpegs(shots).items()}
        self.frame = _json({**telemetry, "cameras": cameras})

    def clear(self):
        self.frame = None

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
