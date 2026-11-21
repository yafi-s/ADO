"""Verify that the console receives complete frames and browser-playable recordings."""
import base64
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.parse import urljoin

import cv2
import numpy as np

from crosswalk.viewio import CAMERAS, POSTERS, Live, Recording


def shots(red):
    return {
        name: np.broadcast_to(np.array([red, 0, index * 30], np.uint8), (16, 18, 3)).copy()
        for index, name in enumerate(CAMERAS)
    } | {"lidar": None}


def poster_rgb(data):
    pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return pixels[0, 0, ::-1]


class LiveTests(unittest.TestCase):
    def test_live_city_serves_geometry_from_its_own_world(self):
        geometry = {"producer": {"map": "Town03_Opt"}, "roads": [[[1, 2, 0], [3, 2, 0]]]}
        with Live(0, map_geometry=geometry) as live:
            live.publish({"frame": 37, "map": "Town03_Opt"}, shots(0))
            with urlopen(live.url + "api/frame") as response:
                frame = json.load(response)
            with urlopen(urljoin(live.url, frame["map_url"])) as response:
                self.assertEqual(json.load(response), geometry)

    def test_latest_frame_and_idle(self):
        with Live(0) as live:
            with urlopen(live.url, timeout=5) as response:
                self.assertIn(b"Autonomous Driver Omniverse", response.read())
            endpoint = live.url + "api/frame"
            with self.assertRaises(HTTPError) as idle:
                urlopen(endpoint, timeout=5)
            self.assertEqual(idle.exception.code, 503)
            idle.exception.close()

            live.publish({"sim_time": 0.0}, shots(100))
            live.publish({"sim_time": 0.1}, shots(180))
            with urlopen(endpoint, timeout=5) as response:
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                frame = json.load(response)
            self.assertEqual(frame["sim_time"], 0.1)
            self.assertEqual(set(frame["cameras"]), set(CAMERAS))
            for index, name in enumerate(CAMERAS):
                image = frame["cameras"][name]
                np.testing.assert_allclose(poster_rgb(base64.b64decode(image)),
                                           [180, 0, index * 30], atol=2)

            live.clear()
            with self.assertRaises(HTTPError) as cleared:
                urlopen(endpoint, timeout=5)
            self.assertEqual(cleared.exception.code, 503)
            cleared.exception.close()

    def test_local_drive_requests_and_latest_status(self):
        options = {"scenarios": [{"id": "pedestrian", "label": "Pedestrian"}], "destinations": ["station"]}
        initial = {"scenario": "pedestrian", "destination": "station", "seed": 0, "crossing_delay": 0}
        final = dict(initial, seed=2**31 - 1, crossing_delay=5)
        with Live(0, options) as live:
            def get(path):
                with urlopen(live.url + path, timeout=5) as response:
                    return json.load(response)

            def submit(payload, **headers):
                request = Request(live.url + "api/drive", json.dumps(payload).encode(), {
                    "Content-Type": "application/json", **headers,
                })
                return urlopen(request, timeout=5)

            self.assertEqual(get("api/options"), options)
            self.assertEqual(get("api/status")["state"], "idle")
            with submit(initial, Origin=live.url.rstrip("/")) as response:
                self.assertEqual(response.status, 202)
                self.assertEqual(json.load(response), {"queued": True})
            with submit(final) as response:
                self.assertEqual(response.status, 202)
            live.set_status("complete", run_id=1, outcome="reached")
            self.assertEqual(get("api/status")["settings"], final)
            self.assertEqual(get("api/status")["state"], "queued")
            self.assertEqual(live.next_request(), final)
            self.assertIsNone(live.next_request())
            live.set_status("running", run_id=2, settings=final)
            self.assertEqual(get("api/status"), {
                "state": "running", "run_id": 2, "settings": final, "outcome": None, "error": None,
            })

    def test_drive_weather_and_grip_follow_option_lists(self):
        options = {
            "scenarios": ["pedestrian"], "destinations": ["station"],
            "weather": [{"id": "WetNoon", "label": "Wet midday"}], "grip": [1, 0.75],
        }
        required = {"scenario": "pedestrian", "destination": "station", "seed": 101, "crossing_delay": 1}
        with Live(0, options) as live:
            def submit(**optional):
                payload = json.dumps(required | optional).encode()
                request = Request(live.url + "api/drive", payload, {"Content-Type": "application/json"})
                return urlopen(request, timeout=5)

            with submit(weather="WetNoon", grip=1) as response:
                self.assertEqual(response.status, 202)
            for fields in (
                {"weather": "ClearNoon"}, {"weather": None}, {"weather": ["WetNoon"]},
                {"grip": 0.5}, {"grip": 0.750001}, {"grip": True}, {"grip": "0.75"},
                {"grip": float("nan")}, {"grip": float("inf")},
            ):
                with self.subTest(fields=fields), self.assertRaises(HTTPError) as rejected:
                    submit(**fields)
                with rejected.exception as response:
                    self.assertEqual(response.code, 400)
            accepted = live.next_request()
            self.assertEqual(accepted, required | {"weather": "WetNoon", "grip": 1.0})
            self.assertIsInstance(accepted["grip"], float)

        live = Live(0, {"scenarios": ["pedestrian"], "destinations": ["station"]})
        for optional in ({"weather": "ClearNight"}, {"grip": 0.5}):
            live.queue_drive(required | optional)
            self.assertEqual(live.next_request(), required | optional)

    def test_rejects_invalid_or_cross_origin_drive_submissions(self):
        options = {"scenarios": ["pedestrian"], "destinations": ["station"]}
        valid = {"scenario": "pedestrian", "destination": "station", "seed": 101, "crossing_delay": 2.5}
        with Live(0, options) as live:
            def expect_error(payload, code, **headers):
                request = Request(live.url + "api/drive", json.dumps(payload).encode(), {
                    "Content-Type": "application/json", **headers,
                })
                with self.assertRaises(HTTPError) as rejected:
                    urlopen(request, timeout=5)
                with rejected.exception as response:
                    self.assertEqual(response.code, code)
                    self.assertIn("error", json.load(response))

            for change in (
                {"scenario": "unknown"}, {"destination": ["station"]}, {"destination": "unknown"},
                {"seed": -1}, {"seed": 2**31}, {"seed": True}, {"seed": 4.0},
                {"crossing_delay": -1}, {"crossing_delay": 5.1}, {"crossing_delay": False},
                {"crossing_delay": float("nan")}, {"crossing_delay": float("inf")}, {"other": "value"},
            ):
                with self.subTest(change=change):
                    expect_error(valid | change, 400)
            expect_error(valid, 403, Origin="https://example.org")
            expect_error(valid, 403, Origin="null")
            expect_error(valid, 403, Host="example.org")
            expect_error(valid, 403, **{"Sec-Fetch-Site": "cross-site"})
            expect_error(valid, 415, **{"Content-Type": "text/plain"})
            self.assertIsNone(live.next_request())

    def test_video_byte_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "camera.mp4").write_bytes(b"0123456789")
            with patch("crosswalk.viewio.WEB_ROOT", root), Live(0) as live:
                with urlopen(Request(live.url + "camera.mp4", headers={"Range": "bytes=2-5"}),
                             timeout=5) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
                    self.assertEqual(response.read(), b"2345")

    def test_local_vehicle_assets_are_whitelisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            (assets / "meridian.glb").write_bytes(b"glTF")
            (assets / "meridian-studio.png").write_bytes(b"studio")
            (assets / "meridian.blend").write_bytes(b"source asset")
            ipace = root / "ipace"
            ipace.mkdir()
            (ipace / "waymo-ipace.glb").write_bytes(b"I-Pace")
            (ipace / "waymo-ipace.blend").write_bytes(b"source asset")
            with (patch("crosswalk.viewio.VEHICLE_ROOT", assets),
                  patch("crosswalk.viewio.IPACE_ROOT", ipace), Live(0) as live):
                base = live.url + "vehicle-assets/"
                with urlopen(base + "meridian.glb?v=1", timeout=5) as response:
                    self.assertEqual(response.headers["Content-Type"], "model/gltf-binary")
                    self.assertEqual(response.read(), b"glTF")
                with urlopen(Request(base + "meridian-studio.png", method="HEAD"), timeout=5) as response:
                    self.assertEqual(response.headers["Content-Type"], "image/png")
                    self.assertEqual(response.headers["Content-Length"], "6")
                    self.assertEqual(response.read(), b"")
                with urlopen(base + "waymo-ipace.glb", timeout=5) as response:
                    self.assertEqual(response.headers["Content-Type"], "model/gltf-binary")
                    self.assertEqual(response.read(), b"I-Pace")
                for path in ("", "meridian.blend", "waymo-ipace.blend", "%2e%2e/meridian.blend"):
                    with self.subTest(path=path), self.assertRaises(HTTPError) as missing:
                        urlopen(base + path, timeout=5)
                    with missing.exception as response:
                        self.assertEqual(response.code, 404)

    def test_stop_latches_for_local_interactive_console_only(self):
        options = {"scenarios": ["pedestrian"], "destinations": ["station"]}
        drive = {"scenario": "pedestrian", "destination": "station", "seed": 0, "crossing_delay": 0}

        def post(live, path="stop", payload=None, **headers):
            request = Request(live.url + f"api/{path}", json.dumps({} if payload is None else payload).encode(),
                              {"Content-Type": "application/json", **headers})
            return urlopen(request, timeout=5)

        with Live(0, options) as live:
            for code, kwargs in (
                (403, {"Origin": "https://example.org"}),
                (415, {"Content-Type": "text/plain"}),
                (400, {"payload": {"unexpected": True}}),
            ):
                with self.subTest(code=code), self.assertRaises(HTTPError) as rejected:
                    post(live, **kwargs)
                with rejected.exception as response:
                    self.assertEqual(response.code, code)
                self.assertFalse(live.stop_requested())
            with post(live, "drive", drive) as response:
                self.assertEqual(response.status, 202)
            with post(live, Origin=live.url.rstrip("/")) as response:
                self.assertEqual(response.status, 202)
                self.assertEqual(json.load(response), {"stopping": True})
            with self.assertRaises(HTTPError) as conflict:
                post(live, "drive", drive)
            with conflict.exception as response:
                self.assertEqual(response.code, 409)
            self.assertTrue(live.stop_requested())
            self.assertIsNone(live.next_request())

        with Live(0) as live:
            with self.assertRaises(HTTPError) as unavailable:
                post(live)
            with unavailable.exception as response:
                self.assertEqual(response.code, 404)
            self.assertFalse(live.stop_requested())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Requires FFmpeg")
class RecordingTests(unittest.TestCase):
    def test_five_synchronized_tracks(self):
        rows = [{"sim_time": step / 10, "frame": step + 17} for step in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "media"
            with Recording(target) as movie:
                for index, row in enumerate(rows):
                    movie.append(row, shots(120 + index * 30))

            self.assertEqual(json.loads((target / "replay.json").read_bytes()), rows)
            self.assertEqual(list(parent.iterdir()), [target])
            for name in CAMERAS:
                result = subprocess.run([
                    "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                    "-show_entries", "stream=codec_name,pix_fmt,avg_frame_rate,nb_read_frames",
                    "-of", "json", str(target / f"{name}.mp4"),
                ], capture_output=True, check=True)
                [stream] = json.loads(result.stdout)["streams"]
                self.assertEqual(stream, {
                    "codec_name": "h264", "pix_fmt": "yuv420p",
                    "avg_frame_rate": "10/1", "nb_read_frames": str(len(rows)),
                })
            for name in POSTERS:
                np.testing.assert_allclose(poster_rgb((target / f"{name}.jpg").read_bytes()),
                                           [120, 0, CAMERAS.index(name) * 30], atol=2)

    def test_failure_preserves_existing_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "media"
            target.mkdir()
            (target / "replay.json").write_bytes(b"previous replay")
            (target / "front.mp4").write_bytes(b"previous video")
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                with patch.object(Recording, "_encode", side_effect=RuntimeError("encoder failed")):
                    with Recording(target) as movie:
                        movie.append({"sim_time": 0}, shots(120))
            self.assertEqual((target / "replay.json").read_bytes(), b"previous replay")
            self.assertEqual((target / "front.mp4").read_bytes(), b"previous video")
            self.assertEqual(sorted(path.name for path in target.iterdir()), ["front.mp4", "replay.json"])
            self.assertEqual(list(parent.iterdir()), [target])


if __name__ == "__main__":
    unittest.main()
