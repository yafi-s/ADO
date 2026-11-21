"""Exercise publication against isolated synthetic experiments, including rejected inputs."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import shutil
import struct
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

    def test_native_recording_matches_scored_interval(self):
        key = "103-trip-fusion"
        dest = self.root / "results/native-replays"
        dest.mkdir(parents=True)
        path = dest / f"{key}.rec"
        town = self.producer["map"].encode()
        with path.open("wb") as output:
            output.write(b"\x01\x00\x0e\x00CARLA_RECORDER" + bytes(8) + struct.pack("<H", len(town)) + town)
            for number, duration in ((1, .1), (2, -1.)):
                output.write(struct.pack("<BIQdd", 0, 24, number, duration, (number - 1)*.1))
                output.write(struct.pack("<BId", 20, 8, 7 + number*.1))
        meta = dict(trial_id=key, filename=path.name, sha256=pub.fingerprint(path), bytes=path.stat().st_size,
                    producer=self.producer, observed_start_frame=90, observed_stop_frame=92, start_elapsed_s=7,
                    stop_elapsed_s=7.2, recorder_summary="Frames: 2\nDuration: 0.1 seconds\n")
        pub.write_json(path.with_suffix(".json"), meta)
        episodes = {key: dict(cohort="primary", district="heldout", producer=self.producer)}
        links = pub.native_evidence(self.root, "v0.1.0", episodes, {key: [91, 92]}, .1)
        self.assertEqual(len(links), 2)
        self.assertTrue(links[0]["url"].endswith("/v0.1.0/results/native-replays/103-trip-fusion.rec"))
        with self.assertRaisesRegex(ValueError, "misses scored"):
            pub.native_evidence(self.root, "v0.1.0", episodes, {key: [90, 91]}, .1)
        for change in (dict(bytes=0), dict(sha256="0"*64), dict(observed_stop_frame=99), dict(start_elapsed_s=9)):
            pub.write_json(path.with_suffix(".json"), dict(meta, **change))
            with self.assertRaises(ValueError):
                pub.native_evidence(self.root, "v0.1.0", episodes, {key: [91, 92]}, .1)

    def test_native_film_frame_matching_and_viewer_eligibility(self):
        key = "103-trip-fusion"
        logs = self.root / "logs"
        logs.mkdir()
        film = self.root / "web/media/recorded-trip"
        film.mkdir(parents=True)
        row = dict(frame=44, seed=103, policy="fusion", trial_id=key, steer=0.1)
        with gzip.open(logs / f"{key}.jsonl.gz", "wt") as stream:
            stream.write(json.dumps(row) + "\n")
        shutil.copyfile(logs / f"{key}.jsonl.gz", film / "original-trace.jsonl.gz")
        origin = dict(trial_id=key, cohort="primary", model_sha256="d"*64)
        summary = dict(trial_id=key, policy_version=origin["model_sha256"])
        pub.write_json(film / "original-summary.json", summary)
        self.put("results/episodes.json", [summary])
        meta = self.root / f"results/native-replays/{key}.json"
        pub.write_json(meta, dict(producer=self.producer, sha256="a"*64, filename=f"{key}.rec", observed_start_frame=42))
        video = dict(width=1920, height=1080, r_frame_rate="10/1")
        build = dict(checkpoints={"fusion": {"sha256": origin["model_sha256"]}}, simulator=self.producer,
                     runs=[origin], evidence=[])
        replay = dict(row, video_index=0, video_time=0, original_frame=44, native_frame=2,
                      tracks=[], detections=[], radar=[], lidar=[], observer_map={"ego": [1, 2, 0]})
        pub.write_json(film / "replay.json", [replay])
        alignment = dict(status="pose_verified", passing_windows=1, max_ego_error_m=.01,
                         max_observer_error_m=.01, max_rotation_error_deg=.01)
        manifest = dict(trial_id=key, additional_scored_drive=False, cohort="native_replay", original_cohort="primary",
                        district="heldout", producer=self.producer, summary=summary, quality_requested="High",
                        frame_period_s=.1, frames=1, first_frame=44, last_frame=44, alignment=alignment,
                        native_recorder=dict(sha256="a"*64, metadata_sha256=pub.fingerprint(meta), filename=f"{key}.rec"))
        (film / "poster.jpg").write_bytes(b"poster")
        with patch.object(pub, "check_video", return_value=video):
            for cameras in (["chase"], ["chase", "front"]):
                for name in cameras:
                    for ext in ("mp4", "jpg"):
                        (film / f"{name}.{ext}").write_bytes(name.encode())
                manifest.update(cameras=cameras, videos={name: video for name in cameras},
                                sha256={p.name: pub.fingerprint(p) for p in film.iterdir() if p.name != "manifest.json"})
                pub.write_json(film / "manifest.json", manifest)
                entry = pub.native_film_entry(self.root, build, logs)
                if len(cameras) == 1:
                    self.assertIsNone(entry)
                else:
                    self.assertEqual(entry["origin_trial_id"], key)
                    self.assertEqual(entry["trial_id"], "native-" + key)
                    self.assertFalse(entry["benchmark_member"])
            self.assertEqual(build["runs"], [origin])
            pub.write_json(film / "replay.json", [dict(replay, steer=.9)])
            manifest["sha256"]["replay.json"] = pub.fingerprint(film / "replay.json")
            pub.write_json(film / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "scored frames"):
                pub.native_film_entry(self.root, build, logs)

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
        curated = dest.with_name("market-demo")
        shutil.copytree(dest, curated)
        curated_key = "showcase-31-market-demo-fusion"
        curated_path = "./media/showcase/market-demo"
        disclosure = dict(cohort="showcase", district="train", selection_note=pub.CURATED_NOTE)
        curated_entry = dict(entry, trial_id=curated_key, path="showcase/market-demo",
                             label="Curated Market drive · training district", manifest=f"{curated_path}/manifest.json",
                             **disclosure)
        for name in ("telemetry.json", "replay.json"):
            rows = pub.read_json(curated / name)
            rows[0]["seed"] = 31
            if name == "replay.json":
                rows[0]["run_id"] = curated_key
            pub.write_json(curated / name, rows)
        curated_manifest = dict(record, trial_id=curated_key, catalog_entry=curated_entry, **disclosure,
                                summary=dict(summary, district="train", seed=31, destination="market"),
                                sha256={name: pub.fingerprint(curated / name) for name in record["sha256"]})
        pub.write_json(curated / "manifest.json", curated_manifest)
        build = self.publish()
        self.assertEqual([r["trial_id"] for r in build["runs"]], [self.trial.trial_id])
        self.assertEqual([e["trial_id"] for e in pub.read_json(media / "viewer-catalog.json")],
                         [key, curated_key, self.trial.trial_id])
        self.assertEqual(build["showcases"][1]["selection_note"], pub.CURATED_NOTE)
        self.assertEqual(build["showcases"][1]["district"], "train")
        pub.write_json(curated / "manifest.json", dict(curated_manifest, selection_note=""))
        with self.assertRaisesRegex(ValueError, "disclosure"):
            self.publish()
        pub.write_json(curated / "manifest.json", curated_manifest)
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
        locked_catalog = pub.read_json(media / "catalog.json")
        locked_replay = pub.read_json(media / "replay.json")
        dest = media / "dev-crossing"
        dest.mkdir()
        for path in [*media.glob("*.mp4"), *media.glob("*.jpg")]:
            shutil.copyfile(path, dest / path.name)
        key = "dev31-crossing"
        self.put("web/media/catalog.json", [dict(trial_id=key, path="dev-crossing", cameras=list(pub.CAMERAS))])
        (self.root / "web/logs").mkdir()
        checkpoint = self.root / "web/logs/dev31-fusion.npz"
        earlier = pub.Model.load(self.root / "models/fusion.npz")
        earlier.mean += 1.
        earlier.save(checkpoint)
        summary.update(seed=31, scenario="pedestrian", district="train", cohort="development",
                       model_sha256=pub.fingerprint(checkpoint))
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
        self.put("web/media/dev-crossing/replay.json", [dict(row, run_id=key)])
        self.put("web/media/replay.json", locked_replay)
        self.put("web/media/catalog.json", locked_catalog)
        final = self.publish()
        self.assertEqual([r["trial_id"] for r in final["runs"]], [self.trial.trial_id])
        self.assertEqual(final["development_runs"], build["runs"])
        self.assertNotEqual(final["development_runs"][0]["model_sha256"], final["checkpoints"]["fusion"]["sha256"])
        self.assertEqual([r["trial_id"] for r in pub.read_json(media / "viewer-catalog.json")],
                         [self.trial.trial_id, key])
