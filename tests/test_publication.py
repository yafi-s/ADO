"""Exercise publication against isolated synthetic experiments, including rejected inputs."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosswalk import publish_evidence as pub
from crosswalk.experiment import Trial
from crosswalk.policy import head_columns, input_mask


class PublicationTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.sha = "a" * 40
        self.producer = dict(server_version="1.0", client_version="1.0", map="Town01_Opt")
        self.trial = Trial(101, "pedestrian", "fusion", "heldout", "primary")
        mock_schedule = patch.object(pub, "schedule", return_value=[self.trial])
        mock_schedule.start()
        self.addCleanup(mock_schedule.stop)

    def put(self, rel, data):
        pub.write_json(self.root / rel, data)

    def publish(self, commissioning=False):
        return pub.publish(self.root, "v0.1.0", self.sha, commissioning)

    def commissioning_files(self):
        for label, rel in pub.COMMISSIONING_FILES:
            if rel.endswith("verification.json"):
                data = dict(server="1.0", client="1.0", map="Carla/Maps/Town01_Opt",
                            stock={"lidar_points": 2}, retrofit={"lidar_points": 2})
            else:
                data = dict(producer=self.producer, trials=[{"seed": 11}],
                            profiles=[dict(mode="native", frames=2)])
            self.put(rel, data)

    def test_commissioning(self):
        self.commissioning_files()
        build = self.publish(True)
        self.assertEqual(build["status"], "commissioning")
        self.assertIsNone(build["release"])
        self.assertEqual((build["runs"], build["checkpoints"]), ([], {}))
        for entry, (_, rel) in zip(build["evidence"], pub.COMMISSIONING_FILES):
            self.assertEqual((self.root / "web" / entry["url"]).read_bytes(), (self.root / rel).read_bytes())

    def fixture(self):
        digests = {}
        for policy in pub.POLICIES:
            norm = np.ones((2, len(input_mask(policy))), dtype=np.float32)
            heads = [(np.ones((len(cols), 1)), np.ones(1), np.ones((1, 1)), np.ones(1))
                     for cols in head_columns(policy)]
            path = self.root / f"models/{policy}.npz"
            pub.Model(policy, norm, norm, heads).save(path)
            digests[policy] = pub.fingerprint(path)
        trial = self.trial
        plan = dict(**asdict(trial), trial_id=trial.trial_id, settings=trial.settings, replay=trial.record)
        manifest = dict(plan=[plan], stage="evaluate", training_producer=self.producer,
                        evaluation_producer=self.producer, model_sha256=digests, evaluation_id="abc",
                        map="Town01_Opt", test_seeds=[101], training_seeds=list(pub.TRAIN), dt_seconds=0.1)
        self.put("results/manifest.json", manifest)
        row = dict(**asdict(trial), trial_id=trial.trial_id, map="Town01_Opt", simulator_version="1.0",
                   policy_version=digests["fusion"], progress_m=1.25, frame=44)
        summary = dict(**row, producer=self.producer, outcome="timeout", frames=1, duration_s=0.1)
        self.put("results/episodes.json", [summary])
        trace = self.root / "results/traces.jsonl.gz"
        with gzip.open(trace, "wt", encoding="utf8") as stream:
            stream.write(json.dumps(row) + "\n")
        self.put("web/media/catalog.json", [dict(path="", trial_id=trial.trial_id, cameras=list(pub.CAMERAS),
                                                evaluation_id="abc")])
        self.put("web/media/replay.json", [dict(frame=44, run_id=trial.trial_id)])
        media = self.root / "web/media"
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        for camera in pub.CAMERAS:
            writer = cv2.VideoWriter(str(media / f"{camera}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
            writer.write(image)
            writer.release()
        for camera in pub.POSTERS:
            self.assertTrue(cv2.imwrite(str(media / f"{camera}.jpg"), image))
        return manifest, summary, row

    def test_full_and_rejected_republication(self):
        manifest, summary, row = self.fixture()
        build = self.publish()
        media = self.root / "web/media"
        self.assertEqual(build["showcases"], [])
        self.assertEqual(pub.read_json(media / "viewer-catalog.json"), pub.read_json(media / "catalog.json"))
        self.assertEqual(build["runs"][0]["recording_path"], "")
        self.assertEqual(build["runs"][0]["model_sha256"], manifest["model_sha256"]["fusion"])
        log = self.root / "web" / build["runs"][0]["log_url"]
        with gzip.open(log, "rt") as stream:
            self.assertEqual(json.loads(stream.read()), row)
        repo = "https://github.com/yafi-s/ADO"
        raw = "https://raw.githubusercontent.com/yafi-s/ADO/v0.1.0/"
        self.assertEqual(build["release"]["zip_url"], f"{repo}/releases/download/v0.1.0/ADO-v0.1.0-Windows.zip")
        self.assertEqual(build["release"]["sha256_url"], f"{repo}/releases/download/v0.1.0/SHA256SUMS.txt")
        self.assertEqual(build["release"]["url"], f"{repo}/releases/tag/v0.1.0")
        self.assertEqual(build["source"]["url"], f"{repo}/tree/{self.sha}")
        self.assertTrue(all(c["url"].startswith(raw) for c in build["checkpoints"].values()))
        self.assertTrue(all(e["url"].startswith(raw) for e in build["evidence"]))
        index = self.root / "web/media/build.json"
        published = index.read_bytes()
        for changes in ({"stage": "dev"}, {"evaluation_producer": dict(self.producer, server_version="0.9.16")},
                        {"model_sha256": dict(manifest["model_sha256"], fusion="a"*64)}):
            with self.subTest(changes=changes):
                self.put("results/manifest.json", dict(manifest, **changes))
                with self.assertRaises(ValueError):
                    self.publish()
        self.put("results/manifest.json", manifest)
        for episodes in ([summary, summary], [dict(summary, frames=2, duration_s=0.2)]):
            self.put("results/episodes.json", episodes)
            with self.assertRaises(ValueError):
                self.publish()
        self.assertEqual(index.read_bytes(), published)

    def test_separate_showcase_and_integrity(self):
        manifest, summary, _ = self.fixture()
        media = self.root / "web/media"
        dest = media / "showcase/trip"
        dest.mkdir(parents=True)
        for name in ("front.mp4", "chase.mp4", "front.jpg", "chase.jpg"):
            shutil.copyfile(media / name, dest / name)
        shutil.copyfile(dest / "chase.jpg", dest / "poster.jpg")
        key = "showcase-211-trip-fusion"
        summary.update(scenario="trip", seed=211)
        row = dict(frame=44, camera_frame=44, video_index=0, video_time=0, scenario="trip", seed=211,
                   policy="fusion", map="Town01_Opt", steer=0.1, throttle=0.4, brake=0, progress_m=1.25)
        pub.write_json(dest / "telemetry.json", [row])
        pub.write_json(dest / "replay.json", [dict(row, run_id=key, lidar=[], radar=[], tracks=[], detections=[],
                                                  observer_map={"ego": [0, 0, 0]})])
        entry = dict(path="showcase/trip", trial_id=key, benchmark_member=False, label="Showcase · Market trip",
                     manifest="./media/showcase/trip/manifest.json", cameras=["chase", "front"])
        record = dict(trial_id=key, benchmark_member=False, catalog_entry=entry, replay="replay.json",
                      summary=summary, outcome=summary["outcome"], producer=self.producer,
                      model_sha256=manifest["model_sha256"]["fusion"], frame_period_s=0.1, quality_requested="High",
                      first_frame=44, last_frame=44, poster_frame=44, video=pub.check_video(dest / "chase.mp4", 1),
                      front_video=pub.check_video(dest / "front.mp4", 1),
                      sha256={file.name: pub.fingerprint(file) for file in dest.iterdir()})
        pub.write_json(dest / "manifest.json", record)
        build = self.publish()
        self.assertEqual([r["trial_id"] for r in build["runs"]], [self.trial.trial_id])
        self.assertEqual(build["showcases"][0]["cohort"], "showcase")
        self.assertEqual([e["trial_id"] for e in pub.read_json(media / "viewer-catalog.json")],
                         [key, self.trial.trial_id])
        self.assertTrue(all((self.root / "web" / e["url"]).is_file() for e in build["evidence"]
                            if e["url"].startswith("./")))
        for changes in (dict(model_sha256="c"*64), dict(benchmark_member=True), dict(last_frame=45),
                        dict(video=dict(record["video"], width=1920))):
            with self.subTest(changes=changes):
                pub.write_json(dest / "manifest.json", dict(record, **changes))
                with self.assertRaises(ValueError):
                    self.publish()
        pub.write_json(dest / "manifest.json", record)
        (dest / "front.jpg").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "digest"):
            self.publish()
        self.assertEqual(build, pub.read_json(media / "build.json"))

    def test_development_cli(self):
        self.commissioning_files()
        _, summary, _ = self.fixture()
        media = self.root / "web/media"
        dest = media / "dev-crossing"
        dest.mkdir()
        for path in [*media.glob("*.mp4"), *media.glob("*.jpg")]:
            path.rename(dest / path.name)
        key = "dev31-crossing"
        self.put("web/media/catalog.json", [dict(trial_id=key, path="dev-crossing", cameras=list(pub.CAMERAS))])
        (self.root / "web/logs").mkdir()
        shutil.copyfile(self.root / "models/fusion.npz", self.root / "web/logs/dev31-fusion.npz")
        summary.update(seed=31, scenario="pedestrian", district="train", cohort="development",
                       model_sha256=summary["policy_version"])
        row = dict(frame=44, seed=31, scenario="pedestrian", policy="fusion", map="Town01_Opt", sim_time=0.0,
                   steer=0.1, throttle=0.4, brake=0, speed_mps=0.5, progress_m=summary["progress_m"])
        self.put("web/logs/dev31-crossing.json", dict(summary=summary, trace=[row]))
        self.put("web/media/dev-crossing/replay.json", [dict(row, run_id=key)])
        args = [sys.executable, "-m", "crosswalk.publish_evidence", "--root", str(self.root),
                "--source-commit", self.sha, "--development"]
        subprocess.run(args, check=True, capture_output=True)
        build = pub.read_json(media / "build.json")
        self.assertEqual(build["status"], "development")
        self.assertIsNone(build["release"])
        self.assertEqual([r["trial_id"] for r in build["runs"]], [key])
        self.assertEqual(build["checkpoints"]["fusion"]["sha256"], build["runs"][0]["model_sha256"])
        self.assertEqual(build["preview_video"], "./media/dev-crossing/chase.mp4")
        self.assertEqual(pub.read_json(media / "viewer-catalog.json"), pub.read_json(media / "catalog.json"))
        self.assertNotEqual(subprocess.run([*args, "--commissioning"], capture_output=True).returncode, 0)
        for changes in ({"cohort": "primary"}, {"district": "heldout"}, {"model_sha256": "0"*64}, {"frames": 2}):
            with self.subTest(changes=changes):
                self.put("web/logs/dev31-crossing.json", dict(summary=dict(summary, **changes), trace=[row]))
                with self.assertRaises(ValueError):
                    pub.publish(self.root, "v0.1.0", self.sha, development_only=True)
        self.put("web/logs/dev31-crossing.json", dict(summary=summary, trace=[row]))
        self.put("web/media/dev-crossing/replay.json", [dict(row, run_id=key, steer=-0.9)])
        with self.assertRaisesRegex(ValueError, "controls"):
            pub.publish(self.root, "v0.1.0", self.sha, development_only=True)
        self.assertEqual(build, pub.read_json(media / "build.json"))
