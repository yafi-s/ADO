"""Verify that the optional replay transport preserves the original evidence."""
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from crosswalk import replay_pack


def expand(payload):
    replay = payload["replay"]
    frames = replay if isinstance(replay, list) else replay["frames"]
    for start, end in payload["path_ranges"]:
        for index in range(start, end):
            frames[index]["observer_map"]["path"] = payload["shared_path"]
    return replay


class ReplayPackTests(unittest.TestCase):
    def test_streaming_preserves_dynamic_missing_and_null_routes(self):
        shared = [[1, 2], [3, 4]]
        raw = [
            {"frame": 40, "observer_map": {"ego": [0, 0, 0], "path": shared}},
            {"frame": 41, "observer_map": {"path": shared}, "tracks": [{"id": 3, "type": "🚲"}]},
            {"frame": 42, "observer_map": {"path": [[2, 3], [4, 5]]}},
            {"frame": 43, "observer_map": {}},
            {"frame": 44, "observer_map": {"path": None}},
            {"frame": 45},
            {"frame": 46, "observer_map": {"path": shared}, "note": "left" * 70},
        ]
        with TemporaryDirectory() as directory:
            source = Path(directory) / "replay.json"
            data = (" \n" * 50 + json.dumps(raw, ensure_ascii=False)).encode("utf-8")
            source.write_bytes(data)
            with patch.object(replay_pack, "CHUNK_SIZE", 32):
                packed = replay_pack.pack(source)
            self.assertEqual(packed.name, "replay-pack.json.gz")
            with gzip.open(packed, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            self.assertEqual(payload["path_ranges"], [[0, 2], [6, 7]])
            self.assertEqual(payload["shared_path"], shared)
            self.assertEqual(expand(payload), raw)
            self.assertEqual(source.read_bytes(), data)
            copy = replay_pack.pack(source, source.with_name("second.gz"))
            self.assertEqual(copy.read_bytes(), packed.read_bytes())
            with self.assertRaises(FileExistsError):
                replay_pack.pack(source)

    def test_preserves_container_metadata_and_an_empty_route(self):
        raw = {"frames": [{"frame": 1, "observer_map": {"path": []}}], "fps": 10, "cameras": ["front"]}
        with TemporaryDirectory() as directory:
            source = Path(directory) / "replay.json"
            source.write_text(json.dumps(raw), encoding="utf-8")
            with gzip.open(replay_pack.pack(source), "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            self.assertEqual(payload["path_ranges"], [[0, 1]])
            self.assertEqual(expand(payload), raw)

    def test_truncated_input_leaves_no_published_pack(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "replay.json"
            source.write_text('[{"frame":1},{"frame":', encoding="utf-8")
            with self.assertRaises(ValueError):
                replay_pack.pack(source)
            self.assertEqual(list(source.parent.iterdir()), [source])
