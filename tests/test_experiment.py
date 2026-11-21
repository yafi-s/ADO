from collections import Counter
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from crosswalk import experiment as ex
from crosswalk.district import DISTRICTS


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.producer = {"server_version": "1.0.0", "client_version": "1.0.0", "map": "Town01_Opt"}
        replacements = (
            patch.object(ex.runtime, "CARLA_VERSION", "1.0", create=True),
            patch.object(ex.runtime, "MAP_NAME", "Town01_Opt", create=True),
            patch.object(ex.runtime, "producer_info", return_value=self.producer, create=True),
            patch.object(ex.sensors, "render_profile", return_value={"rgb": "dlss"}, create=True),
        )
        for replacement in replacements:
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_study_fingerprint_and_data_reject_legacy_or_mixed_producers(self):
        current = ex.generator_sources()
        with patch.object(ex.sensors, "render_profile", return_value={"rgb": "native"}):
            self.assertNotEqual(ex.identity(current), ex.identity(ex.generator_sources()))
        for altered in ({"server_version": "0.9.16"}, {"client_version": "1.01"}, {"map": "Town01"}):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                ex.validate_producer(dict(self.producer, **altered))
    def test_locked_design_is_paired_and_geographically_disjoint(self):
        trials = list(ex.schedule("evaluate"))
        self.assertEqual(Counter(t.cohort for t in trials),
                         {"primary": 100, "camera_ablation": 20, "supplemental": 1})
        self.assertEqual(len({t.trial_id for t in trials}), len(trials))
        by_id = {t.trial_id: t for t in trials}
        for t in trials:
            self.assertEqual(t.district, "heldout")
            if t.policy == "camera":
                self.assertEqual(t.settings, by_id[t.trial_id.replace("-camera", "-fusion")].settings)
        base = by_id["101-trip-fusion"]
        delayed = by_id["101-trip-delayed-fusion"]
        self.assertEqual(base.kind, delayed.kind)
        self.assertEqual(dict(base.settings, crossing_delay=1.0), delayed.settings)
        self.assertEqual({t.case_id for t in trials if t.record}, set(ex.VIDEO_CASES))
        trips = [t for t in trials if t.kind == "trip" and t.cohort == "primary"]
        self.assertEqual(Counter(t.settings["destination"] for t in trips), {"market": 5, "station": 5})
        demos = list(ex.schedule("collect"))
        self.assertEqual(len(demos), 48)
        self.assertEqual({t.district for t in demos}, {"train"})
        self.assertNotIn("braking", {t.kind for t in demos})
        self.assertTrue(all(a.isdisjoint(b) for a, b in zip(DISTRICTS["train"], DISTRICTS["heldout"])))

    def test_no_leak_through_subset_options(self):
        for stage, seeds in (("evaluate", [101]), ("collect", [101]), ("dev", [11]), ("dev", [31, 31])):
            with self.subTest(stage=stage, seeds=seeds), self.assertRaises(ValueError):
                list(ex.schedule(stage, seeds))
        self.assertEqual({t.seed for t in ex.schedule("dev", [32])}, {32})

    def test_dependency_hash_tracks_feature_helpers_only(self):
        with TemporaryDirectory() as folder:
            module = Path(folder) / "policy.py"
            original = "OFFSET = 4\ndef features(n): return n + OFFSET\ndef fit(): return 0\n"
            module.write_text(original)
            before = ex.dependency_code(module, ("features",))
            module.write_text(original.replace("return 0", "return 1"))
            self.assertEqual(before, ex.dependency_code(module, ("features",)))
            module.write_text(original.replace("OFFSET = 4", "OFFSET = 5"))
            self.assertNotEqual(before, ex.dependency_code(module, ("features",)))

    def test_resume_requires_intact_evidence_and_retains_trial_failure(self):
        trial = next(ex.schedule("collect", [11]))
        fake = (dict(frames=2, outcome="collision", seed=11, scenario="pedestrian",
                     policy="demonstrator", road_ids=[1], junction_ids=[]), [{"speed": 0}, {"speed": 1}],
                np.zeros((1, len(ex.FEATURE_NAMES)), dtype=np.float32), np.zeros((1, 3), dtype=np.float32))
        with TemporaryDirectory() as temp, patch.object(ex, "episode", return_value=fake) as episode:
            folder = Path(temp)
            summary, reused = ex.run_trial(object(), object(), trial, folder)
            self.assertFalse(reused)
            self.assertEqual(summary["outcome"], "collision")
            self.assertTrue(ex.run_trial(object(), object(), trial, folder)[1])
            self.assertEqual(episode.call_count, 1)
            with gzip.open(folder / "trace.jsonl.gz", "rt") as archive:
                self.assertEqual([json.loads(row)["trial_id"] for row in archive], [trial.trial_id] * 2)
            (folder / "samples.npz").write_bytes(b"incomplete write")
            self.assertFalse(ex.run_trial(object(), object(), trial, folder)[1])
            self.assertEqual(episode.call_count, 2)
            with np.load(folder / "samples.npz") as sample:
                self.assertEqual(sample["seed"].tolist(), [11])

    def test_partial_fit_dev_gate_and_full_publication(self):
        class FakeRecording:
            def __init__(self, path):
                self.path = path

            def __enter__(self):
                self.path.mkdir(parents=True, exist_ok=True)
                for name in ex.CAMERAS:
                    (self.path / f"{name}.mp4").write_bytes(b"fixture")
                for name in ex.POSTERS:
                    (self.path / f"{name}.jpg").write_bytes(b"fixture")
                (self.path / "replay.json").write_text("[]")
                return self

            def __exit__(self, *args):
                pass

        def drive(world, kind, seed, policy, model, *, district, **kwargs):
            roads, _ = DISTRICTS[district]
            summary = dict(frames=2, outcome="complete", seed=seed, policy=policy, scenario=kind,
                           progress_m=5, road_ids=[min(roads)], junction_ids=[], yield_violations=int(kind == "left"))
            return (summary, [{"t": 0}, {"t": 1}], np.zeros((1, len(ex.FEATURE_NAMES))), np.zeros((1, 3)))

        def fit(data, weights, report, *, inputs):
            weights.write_bytes(b"test model")
            ex.write_json(report, {})

        with TemporaryDirectory() as temp, patch.object(ex, "episode", side_effect=drive) as driver, \
                patch.object(ex, "train", side_effect=fit), patch.object(ex.Model, "load"), \
                patch.object(ex, "Recording", FakeRecording), patch("builtins.print"):
            root = Path(temp)
            base = root / "out/city/dataset"
            ex.collect(None, object(), base, [11])
            ex.fit_archive(root, base)
            ex.evaluate(None, root, base, "dev", [31])
            with self.assertRaisesRegex(RuntimeError, "required training seeds"):
                ex.evaluate(None, root, base, "evaluate", None)
            ex.collect(None, object(), base, ex.TRAIN[1:])
            ex.fit_archive(root, base)
            ex.evaluate(None, root, base, "evaluate", None)
            calls = driver.call_count
            ex.evaluate(None, root, base, "evaluate", None)
            self.assertEqual(driver.call_count, calls)
            self.assertEqual(len(ex.read_json(root / "web/media/episodes.json")), 121)
            self.assertEqual(len(ex.read_json(root / "results/primary.json")), 100)
            self.assertEqual(len(ex.read_json(root / "results/failures.json")), 12)
            catalog = ex.read_json(root / "web/media/catalog.json")
            self.assertEqual([r["path"] for r in catalog], ["", "occluded", "signal", "left", "trip", "trip-delayed"])
            self.assertTrue(all((root / "web/media" / r["path"] / "replay.json").exists() for r in catalog))
            manifest = ex.read_json(root / "results/manifest.json")
            self.assertEqual(manifest["evaluation_producer"], self.producer)
            self.assertEqual(manifest["map"], "Town01_Opt")
            with np.load(root / "data/training.npz") as sample:
                self.assertEqual(json.loads(str(sample["producer"])), self.producer)
            self.assertTrue((root / "web/media/trip-delayed/right.mp4").is_file())
            with gzip.open(root / "results/traces.jsonl.gz", "rt") as trace:
                self.assertEqual(sum(1 for row in trace), 242)


if __name__ == "__main__":
    unittest.main()
