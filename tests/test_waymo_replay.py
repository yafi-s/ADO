from http.client import HTTPConnection
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest import TestCase

from waymo_training.replay import ReplayServer, _routes


def make_archive(root, tracker=None):
    name = "lead-seed41-start3"
    replay = root / name / "replay"
    replay.mkdir(parents=True)
    summary = {
        "scenario": "lead", "seed": 41, "outcome": "lane_departure",
        "initial_velocity_set_once_mps": 3, "duration_s": 3.4, "progress_m": 22.57,
    }
    report = {
        "policy": "Waymo-trained predictor",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_metadata": {"source_checkpoint_sha256": "b" * 64, "dataset_source_sha256": "c" * 64},
        "producer": {"server_version": "1.0", "client_version": "1.0", "map": "Town01_Opt"},
        "trials": [{"path": name, **summary}],
    }
    if tracker:
        report["tracker_name"] = tracker
    (root / "report.json").write_text(json.dumps(report))
    (root / "secret.npz").write_bytes(b"excluded weights")
    (replay.parent / "summary.json").write_text(json.dumps(summary))
    (replay.parent / "trace.json").write_text('[{"frame": 99}]')
    (replay / "replay.json").write_text('[{"frame": 99, "future_rear_xy_m": [[1, 2]]}]')
    for name in ("chase", "front", "left", "right", "infrared"):
        (replay / f"{name}.mp4").write_bytes(b"0123456789")
    (replay / "unlisted.mp4").write_bytes(b"excluded footage")
    return root


class WaymoReplayTests(TestCase):
    def setUp(self):
        with_dir = self.enterContext(TemporaryDirectory())
        self.root = Path(with_dir)
        self.baseline = make_archive(self.root / "baseline")
        self.revised = make_archive(self.root / "revised", "calibrated-v2")
        self.server = self.enterContext(ReplayServer(0, [self.baseline, self.revised]))
        thread = Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)

    def request(self, path, method="GET", headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.headers), response.read()
        finally:
            connection.close()

    def test_manifest_and_catalog_link_to_exact_archives(self):
        status, headers, body = self.request("/media/viewer-catalog.json")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        catalog = json.loads(body)
        build = json.loads(self.request("/media/build.json")[2])
        self.assertEqual(build["status"], "research_transfer")
        self.assertEqual(len(catalog), 2)
        self.assertEqual(len({entry["trial_id"] for entry in catalog}), 2)
        self.assertEqual(build["runs"][0]["tracker"], "calibrated-v2")
        for run, entry in zip(build["runs"], catalog):
            self.assertEqual(run["trial_id"], entry["trial_id"])
            self.assertEqual(set(entry["cameras"]), {"front", "chase", "left", "right", "infrared"})
            self.assertEqual(run["model_sha256"], "a" * 64)
            self.assertEqual(run["source_checkpoint_sha256"], "b" * 64)
            self.assertEqual(run["dataset_source_sha256"], "c" * 64)
            self.assertEqual(run["outcome"], "lane_departure")
            self.assertEqual(run["progress_m"], 22.57)
            self.assertEqual(json.loads(self.request(run["summary_url"][1:])[2])["progress_m"], 22.57)
            self.assertEqual(self.request(run["trace_url"][1:])[2], b'[{"frame": 99}]')
            self.assertEqual(
                self.request(f"/media/{entry['path']}/replay.json")[2],
                b'[{"frame": 99, "future_rear_xy_m": [[1, 2]]}]',
            )
        status, headers, _ = self.request("/")
        self.assertEqual(status, 307)
        self.assertEqual(headers["Location"], f"/?run={catalog[0]['trial_id']}")
        self.assertEqual(self.request(headers["Location"])[0], 200)
        self.assertEqual(self.request("/vehicle-assets/waymo-ipace.glb", "HEAD")[1]["Content-Type"],
                         "model/gltf-binary")

    def test_video_ranges_support_browser_scrubbing_and_head(self):
        path = "/media/transfer-1/lead-seed41-start3/chase.mp4"
        status, headers, body = self.request(path, headers={"Range": "bytes=2-6"})
        self.assertEqual((status, body), (206, b"23456"))
        self.assertEqual(headers["Content-Range"], "bytes 2-6/10")
        self.assertEqual(headers["Content-Type"], "video/mp4")
        status, headers, body = self.request(path, "HEAD", {"Range": "bytes=-4"})
        self.assertEqual((status, body), (206, b""))
        self.assertEqual(headers["Content-Length"], "4")
        status, headers, body = self.request(path, headers={"Range": "bytes=10-"})
        self.assertEqual((status, body), (416, b""))
        self.assertEqual(headers["Content-Range"], "bytes */10")
        self.assertEqual(self.request(path)[2], b"0123456789")

    def test_binding_and_routes_are_restricted(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        self.assertEqual(self.request("/media/build.json", headers={"Host": "outside.example"})[0], 403)
        self.assertEqual(self.request("/media/build.json", headers={"Origin": "https://outside.example"})[0], 403)
        self.assertEqual(self.request("/media/build.json", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
        for path in (
            "/records/transfer-1/secret.npz", "/logs/dev31-fusion.npz", "/media/training-drive.mp4",
            "/media/transfer-1/lead-seed41-start3/unlisted.mp4", "/records/transfer-1", "/api/frame",
            "/media/transfer-1/lead-seed41-start3/%2e%2e/report.json",
        ):
            self.assertEqual(self.request(path)[0], 404, path)
        self.assertEqual(self.request("/api/drive", "POST")[0], 501)

    def test_refuses_archive_if_trial_summary_disagrees_with_report(self):
        summary = self.baseline / "lead-seed41-start3" / "summary.json"
        data = json.loads(summary.read_text())
        data["outcome"] = "complete"
        summary.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "summary and transfer report differ"):
            _routes([self.baseline])
