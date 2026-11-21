from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from crosswalk import range_development as rd


class RangeDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.producer = dict(map="Town01_Opt", client_version="1.0", server_version="1.0")
        self.settings = SimpleNamespace(no_rendering_mode=True, synchronous_mode=True, fixed_delta_seconds=.1)
        self.world = SimpleNamespace(get_map=lambda: SimpleNamespace(name="/Game/Carla/Maps/Town01_Opt"),
                                     get_settings=lambda: self.settings)
        self.tm = object()
        self.model = SimpleNamespace(inputs="range")
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        producer_patch = patch.object(rd.runtime, "producer_info", return_value=self.producer)
        producer_patch.start()
        self.addCleanup(producer_patch.stop)

    def test_protocol_and_session(self):
        collect, dev = list(rd.schedule(True)), list(rd.schedule(False))
        self.assertEqual([len(collect), len(dev)], [24, 12])
        self.assertEqual({t.seed for t in collect}, {11, 12, 13, 14})
        self.assertEqual({t.seed for t in dev}, {31, 32})
        self.assertEqual({t.district for t in collect + dev}, {"train"})
        self.assertEqual({t.kind for t in collect}, {"pedestrian", "lead", "occluded", "clear", "turn", "trip"})
        self.assertEqual(rd.checked_session(self.world), self.producer)
        self.settings.no_rendering_mode = False
        with self.assertRaises(ValueError):
            rd.checked_session(self.world)

    def episode(self, world, kind, seed, policy, model, *, teacher, range_only, run_id, district, **settings):
        self.assertTrue(range_only)
        collecting = policy == "demonstrator"
        self.assertIs(teacher, self.tm if collecting else None)
        self.assertIs(model, None if collecting else self.model)
        self.assertEqual(district, "train")
        rows = [dict(event_triggered=True, event_cleared=step >= 1, speed_mps=step, reference_gap_m=5.,
                     progress_m=p, frame=step) for step, p in enumerate((5., 5.1, 7.2))]
        summary = dict(frames=3, scenario=kind, seed=seed, outcome="collision", collision_events=1,
                       progress_m=7.2, route_length_m=60., road_ids=[0], junction_ids=[],
                       sensor_configuration="lidar-radar", camera_rendering={"cameras": {}})
        if collecting:
            summary["route_xy"] = [[0., 0.], [60., 0.]]
        return summary, rows, np.ones((2, len(rd.policy.RANGE_FEATURE_NAMES))), np.zeros((2, 3))

    def train(self, data, model, report, *, inputs, route_refinement, imitation_path):
        self.assertEqual(inputs, "range")
        self.assertTrue(route_refinement)
        with np.load(data, allow_pickle=False) as records:
            self.assertEqual(tuple(records["feature_names"]), rd.policy.RANGE_FEATURE_NAMES)
            self.assertEqual(records["X"].shape, (48, 108))
            self.assertEqual(len(records["route_lengths"]), 24)
            np.testing.assert_array_equal(records["route_seeds"], np.repeat([11, 12, 13, 14], 6))
        model.write_bytes(b"model")
        imitation_path.write_bytes(b"imitation")
        report.write_text("{}")

    def test_full_run_resumes_and_repairs_single_corrupt_trace(self):
        with patch.object(rd, "episode", side_effect=self.episode) as episode, \
                patch.object(rd, "train", side_effect=self.train) as train, \
                patch.object(rd.policy.Model, "load", return_value=self.model), redirect_stdout(io.StringIO()):
            archive = rd.run(self.world, self.tm, self.root)
            results = self.root / "results" / "range-development"
            self.assertEqual(len(rd.read_json(results / "episodes.json")), 12)
            self.assertIn("12 of 12 eligible", (results / "report.md").read_text(encoding="utf-8"))
            self.assertEqual(episode.call_count, 36)
            self.assertEqual(rd.run(None, None, self.root, "fit"), archive)
            rd.run(self.world, self.tm, self.root)
            self.assertEqual(episode.call_count, 36)
            trace = next((archive / "eval").rglob("trace.jsonl.gz"))
            with gzip.open(trace, "rt") as fp:
                self.assertEqual(len([json.loads(line) for line in fp]), 3)
            trace.write_bytes(b"corrupt")
            rd.run(self.world, None, self.root, stage="eval")
            self.assertEqual(episode.call_count, 37)
            self.assertEqual(train.call_count, 1)
        self.assertFalse((self.root / "results" / "primary.json").exists())

    def test_recovery_requires_observed_clearance_and_post_clearance_progress(self):
        row = dict(event_triggered=True, event_cleared=False, progress_m=5, speed_mps=0, reference_gap_m=5)
        self.assertIsNone(rd.recovery([row])["recovered"])
        self.assertFalse(rd.recovery([row, dict(row, event_cleared=True, progress_m=9)])["recovered"])
        self.assertIsNone(rd.recovery([dict(row, event_triggered=False), dict(row, event_cleared=True)])["recovered"])


if __name__ == "__main__":
    unittest.main()
