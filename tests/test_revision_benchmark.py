from collections import Counter
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from crosswalk import revision_benchmark as bench


class RevisionBenchmarkTests(unittest.TestCase):
    def test_schedule_locks_geography_cases_and_paired_order(self):
        fresh = list(bench.schedule("fresh"))
        self.assertEqual(Counter(t.policy for t in fresh), {"baseline": 50, "candidate": 50})
        self.assertEqual({t.seed for t in fresh}, set(range(2111, 2116)))
        for a, b in zip(fresh[::2], fresh[1::2]):
            self.assertEqual((a.seed, a.case_id, a.settings), (b.seed, b.case_id, b.settings))
            self.assertEqual(a.district, "heldout")
            self.assertNotEqual(a.policy, b.policy)
        self.assertEqual(Counter(t.policy for t in fresh[::2]), {"baseline": 25, "candidate": 25})
        self.assertEqual(len({t.trial_id for t in bench.schedule("regression")}), 100)
        for stage in ("fresh", "regression"):
            with self.assertRaises(ValueError):
                list(bench.schedule(stage, cases=["clear"]))
        with self.assertRaises(ValueError):
            list(bench.schedule("dev", seeds=[101]))

    def test_historical_clean_endpoint_and_mismatched_layouts(self):
        history = bench.ex.read_json(bench.PROJECT / "results/primary.json")
        self.assertEqual(bench.arm_stats(history)["clean"], 70)
        candidate = [dict(r, policy="candidate") for r in history]
        report = bench.comparison("regression", candidate, history)
        self.assertTrue(report["all_pairs_verified"])
        self.assertEqual(report["paired_clean"], {"both_clean": 70, "neither_clean": 30})
        previously_clean = next(r for r in candidate if bench.report.clean(r))
        previously_clean["yield_violations"] = 1
        candidate[0]["route_id"] += "-mismatch"
        report = bench.comparison("regression", candidate, history)
        self.assertFalse(report["all_pairs_verified"])
        self.assertEqual(report["matched_pairs"], 99)
        self.assertEqual(report["arms"]["candidate"]["clean"], 69)
        self.assertEqual(report["setup_mismatches"], [[candidate[0]["seed"], candidate[0]["case_id"]]])
        self.assertEqual(report["arms"]["candidate"]["drives"], 100)

    def test_freeze_rejects_mix_or_changed_archives(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "weights"
            candidate.write_bytes(b"original")
            history = bench.ex.read_json(bench.PROJECT / "results/primary.json")
            args = (None, None, bench.PRODUCER, history[0]["render_profile"])
            folder, version, _ = bench.freeze(root, "fresh", candidate, *args)
            self.assertEqual((folder / "inputs/candidate/checkpoint.npz").read_bytes(), b"original")
            self.assertEqual(bench.freeze(root, "regression", candidate, *args)[:2], (folder, version))
            candidate.write_bytes(b"edited")
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                bench.freeze(root, "regression", candidate, *args)
            self.assertNotEqual(bench.freeze(root, "dev", candidate, *args)[1], version)
            candidate.write_bytes(b"original")
            (folder / "inputs/candidate/checkpoint.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Frozen input differs"):
                bench.freeze(root, "fresh", candidate, *args)

    def test_development_study_resumes_and_preserves_expected_denominator(self):
        trial = next(bench.schedule("dev", seeds=[31], cases=["lead"], arms=["candidate"]))
        records = bench.ex.read_json(bench.PROJECT / "results/dev/episodes.json")
        row = next(r for r in records if (r["seed"], r["case_id"]) == (trial.seed, trial.case_id))
        scored = dict(row, frames=1, policy="candidate", outcome="timeout")
        world = Mock()
        world.get_settings.return_value = SimpleNamespace(synchronous_mode=True, no_rendering_mode=False,
                fixed_delta_seconds=.1, substepping=True, max_substeps=10, max_substep_delta_time=.01)
        world.get_map.return_value.to_opendrive.return_value = "<test-road/>"
        with TemporaryDirectory() as tmp, patch.object(bench.runtime, "producer_info", return_value=bench.PRODUCER), \
                patch.object(bench, "load_model"), patch.object(bench, "episode",
                return_value=(scored, [{"frame": 10}], [], [])) as episode:
            root = Path(tmp)
            candidate = root / "candidate.npz"
            candidate.write_bytes(b"test")
            def run():
                path = bench.study(world, root, "dev", candidate, seeds=[31], cases=["lead"], arms=["candidate"])
                return bench.ex.read_json(path)
            first = run()
            self.assertEqual(run(), first)
            episode.assert_called_once()
            self.assertEqual((first["completed_drives"], first["expected_drives"]), (1, 40))
            self.assertEqual(first["arms"]["candidate"]["outcomes"], {"timeout": 1})
            self.assertFalse(first["complete"])

    def test_case_retains_failure_and_trace_then_resumes_by_marker(self):
        trial = next(bench.schedule("regression",))
        row = next(r for r in bench.ex.read_json(bench.PROJECT / "results/primary.json")
                   if (r["seed"], r["case_id"]) == (trial.seed, trial.case_id))
        row.update(frames=1, outcome="collision", collision_events=1, policy=trial.policy)
        archived = dict(row)
        row["road_ids"] = tuple(row["road_ids"])
        row["junction_ids"] = tuple(row["junction_ids"])
        trace = [{"frame": 123, "collision_count": 1}]
        with TemporaryDirectory() as tmp, patch.object(bench, "episode", return_value=(row, trace, [], [])) as drive:
            folder = Path(tmp)
            self.assertIsNone(bench.read_case(folder, "version", trial))
            initial = bench.run_case(None, trial, folder, object(), "version", row["producer"], row["render_profile"])
            self.assertEqual(initial, bench.read_case(folder, "version", trial))
            self.assertEqual(bench.comparison("regression", [initial], [archived])["matched_pairs"], 1)
            drive.assert_called_once()
            self.assertEqual(bench.read_case(folder, "version", trial)["outcome"], "collision")
            with gzip.open(folder / "trace.jsonl.gz", "rt") as stream:
                self.assertIn('"frame":123', stream.read())
            with self.assertRaisesRegex(ValueError, "Invalid saved case"):
                bench.read_case(folder, "other", trial)
            (folder / "trace.jsonl.gz").write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "Invalid saved case"):
                bench.read_case(folder, "version", trial)


if __name__ == "__main__":
    unittest.main()
