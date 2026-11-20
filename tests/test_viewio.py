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

import cv2
import numpy as np

from crosswalk.viewio import CAMERAS, Live, Recording


def shots(red):
    image = np.zeros((16, 18, 3), dtype=np.uint8)
    image[:, :, 0] = red
    return {name: image for name in CAMERAS} | {"lidar": None}


def poster_rgb(data):
    pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return pixels[0, 0, ::-1]


class LiveTests(unittest.TestCase):
    def test_latest_frame_and_idle(self):
        with Live(0) as live:
            with urlopen(live.url, timeout=5) as response:
                self.assertIn(b"Drive monitor", response.read())
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
            for image in frame["cameras"].values():
                np.testing.assert_allclose(poster_rgb(base64.b64decode(image)), [180, 0, 0], atol=2)

            live.clear()
            with self.assertRaises(HTTPError) as cleared:
                urlopen(endpoint, timeout=5)
            self.assertEqual(cleared.exception.code, 503)
            cleared.exception.close()

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


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Requires FFmpeg")
class RecordingTests(unittest.TestCase):
    def test_three_synchronized_tracks(self):
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
            for name in ("front", "chase"):
                np.testing.assert_allclose(poster_rgb((target / f"{name}.jpg").read_bytes()),
                                           [120, 0, 0], atol=2)

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
