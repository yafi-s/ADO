"""Check the measured speed envelope and raw CARLA frame alignment without a server."""
import gzip
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from crosswalk.boulevard_calibration import BRAKE, DELAY_STEPS, fit


ROOT = Path(__file__).resolve().parents[1] / "results/city-maneuvers/speed"


class StoppingEnvelopeTests(unittest.TestCase):
    def test_measured_closed_loop_frames_and_speed_durations(self):
        for folder in sorted(ROOT.iterdir()):
            summary_path = folder/"summary.json"
            if not summary_path.is_file():
                continue
            with self.subTest(trial=folder.name):
                summary = json.loads(summary_path.read_bytes())
                grade = summary["grade"]
                with gzip.open(folder/"trace.jsonl.gz","rt") as stream:
                    trace = [json.loads(line) for line in stream]
                with np.load(folder/"observations.npz",allow_pickle=False) as tape:
                    np.testing.assert_array_equal(tape["frame"],[r["frame"] for r in trace])
                    self.assertEqual(len(tape["features"]),len(trace))
                speeds = np.array([r["speed_mps"] for r in trace])*3.6
                times = np.array([r["sim_time"] for r in trace])
                np.testing.assert_allclose(np.diff(times),.1)
                self.assertAlmostEqual(float(max(speeds)),grade["peak_kmh"])
                in_zone = np.array([r["posted_limit_kmh"]==90. for r in trace])
                for threshold in (40,45,50,55,60):
                    active = (speeds>threshold)&in_zone
                    longest = current = 0
                    for on in active:
                        current = current+1 if on else 0
                        longest = max(longest,current)
                    metric = grade[f"above_{threshold}_kmh_in_90_zone"]
                    self.assertAlmostEqual(.1*sum(active),metric["seconds"])
                    self.assertAlmostEqual(.1*longest,metric["longest_s"])
                if "queue" in grade:
                    tail = trace[-20:]
                    self.assertEqual(grade["queue"]["full_stop_held_2s"],
                                     all(r["speed_mps"]<=.051 and 0.<r["lead_body_gap_m"]<15.
                                         and not r["ego_ahead_of_lead"] for r in tail))
                    self.assertEqual(grade["queue"]["whole_body_min_gap_m"],
                                     min(r["lead_body_gap_m"] for r in trace))

    def test_measured_distances_and_independent_guarded_fit(self):
        data = json.loads((ROOT/"stopping-envelope.json").read_bytes())
        fitted = fit([r.copy() for r in data["trials"]])
        for field in ("delay_s", "quadratic_s2_per_m", "guard_m", "max_fit_speed_mps"):
            self.assertAlmostEqual(data[field], fitted[field])
        for trial in data["trials"]:
            with self.subTest(trial=trial["name"]):
                path = ROOT/trial["trace"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), trial["trace_sha256"])
                with gzip.open(path, "rt") as stream:
                    rows = [json.loads(line) for line in stream]
                cue = next(r for r in rows if r["frame"] == trial["cue_frame"])
                self.assertAlmostEqual(cue["speed_mps"], trial["cue_speed_mps"])
                self.assertAlmostEqual(rows[-1]["progress_m"]-cue["progress_m"], trial["stopping_distance_m"])
                self.assertLessEqual(rows[-1]["speed_mps"], data["stop_threshold_mps"])
                self.assertTrue(all(b["frame"] == a["frame"]+1 for a,b in zip(rows,rows[1:])))
                coasting = [r for r in rows if r["phase"]=="response_delay"]
                self.assertEqual(len(coasting), DELAY_STEPS)
                self.assertTrue(all(r["throttle"] == r["brake"] == 0. for r in coasting))
                if trial["brake"] == BRAKE:
                    self.assertGreater(trial["guarded_margin_m"], 0.)
        self.assertEqual(fitted["fit_targets_kmh"], [40,50,60])
        held, stress = (next(r for r in data["trials"] if r["split"] == k)
                        for k in ("held_out","grip_stress"))
        self.assertNotIn(held["target_kmh"], fitted["fit_targets_kmh"])
        self.assertNotEqual(held["start_station_m"], fitted["trials"][0]["start_station_m"])
        self.assertGreaterEqual(held["cue_speed_mps"]*3.6, 55.)
        np.testing.assert_allclose(stress["grip_readback"], np.array(held["grip_readback"])*.5)
        # A perturbed holdout must never change either coefficient or the guard.
        altered = [r.copy() for r in data["trials"]]
        next(r for r in altered if r["split"]=="held_out")["stopping_distance_m"] += 100.
        refit = fit(altered)
        self.assertEqual(fitted["quadratic_s2_per_m"], refit["quadratic_s2_per_m"])
        self.assertEqual(fitted["guard_m"], refit["guard_m"])


if __name__ == "__main__":
    unittest.main()
