"""Fixed native evidence and matched safety counterfactuals for the opt-in tool."""
import gzip
import json
import unittest

import numpy as np

from tools.maneuver_source_approach_candidate import MODEL, ROOT, SourceApproach, replay


CASE = ROOT / "results/city-maneuvers/pace/native-v07-boulevard/boulevard"


def tape(folder):
    with gzip.open(folder/"trace.jsonl.gz","rt") as stream:
        rows = [json.loads(line) for line in stream]
    with np.load(folder/"observations.npz") as archive:
        return rows, list(archive["feature_names"]), archive["features"]


class SourceApproachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.names,cls.features = tape(CASE)
        cls.index = {r["frame"]:i for i,r in enumerate(cls.rows)}

    def prime(self, frame, override=None, tactical=None):
        model = SourceApproach.load(MODEL)
        model.new_observer()
        idx = self.index[frame]
        for i in range(idx-15,idx+1):
            row = self.rows[i]
            info = dict(row["tactical_pass"])
            x = self.features[i].copy()
            if i==idx:
                for k,v in (override or {}).items():
                    x[self.names.index(k)] = v
                info.update(tactical or {})
            model.tactical_speed_limit = info.get("tactical_speed_limit_mps")
            model.posted_limit_kmh = row["posted_limit_kmh"]
            model.observe_refusal(row["frame"],info)
            model.predict(x)
        return info

    def test_refusal_and_untouched_parent_pace(self):
        result = replay(CASE)
        interval = [r for r in result if 15341<=r["frame"]<=15393]
        original_record = self.rows[self.index[15354]]
        self.assertEqual(original_record["tactical_pass"]["turn_signal"],"off")
        self.assertFalse(original_record["tactical_pass"]["committed"])
        self.assertTrue(all(r["baseline"]==r["recorded_request"]==0. for r in interval))
        board = next(r for r in interval if r["frame"]==15354)
        self.assertAlmostEqual(board["cap_mps"],3.949888,places=5)
        self.assertAlmostEqual(board["candidate"],1.502092,places=5)
        self.assertGreater(board["throttle"],0.)
        self.assertEqual(board["brake"],0.)

    def test_current_radar_can_span_two_missed_source_returns(self):
        for frame in (15355,15356):
            info = self.prime(frame)
            self.assertGreater(info["source_approach_candidate"]["cap_mps"],0.)
            for changes in ({"path_body_radar":36.}, {"radar_surface_speed":-2.}):
                result=self.prime(frame,changes)
                self.assertEqual(result["source_approach_candidate"]["cap_mps"],0.)
        for frame in (15357,15358):
            self.assertEqual(self.prime(frame)["source_approach_candidate"]["cap_mps"],0.)

    def test_near_blocked_pedestrian_and_stopped_lead(self):
        scenarios = [
            ("results/roadworks-pace/blocked",7292),
            ("results/city-maneuvers/speed/stationary-queue",5432),
            ("results/city-maneuvers/attempts/rehearsal02",8956),
            ("results/city-maneuvers/pace/native-v06-boulevard/boulevard",4368),
        ]
        for folder,frame in scenarios:
            rows,names,features = tape(ROOT/folder)
            i = next(i for i,r in enumerate(rows) if r["frame"]==frame)
            self.assertEqual(rows[i]["speed_request_mps"],0.)
            # Replace board-frame body returns with the *recorded* near object.
            # Keep the originally eligible distant tactical landmark to ensure a
            # fresh, nearer sensor can veto even an otherwise inviting approach.
            x = dict(zip(names,features[i]))
            keep = ["path_body_lidar","path_body_radar","path_body_lidar_recent","path_body_radar_recent"]
            info=self.prime(15354,{k:x[k] for k in keep})
            self.assertEqual(info["source_approach_candidate"]["cap_mps"],0.)
            self.assertEqual(info["source_approach_candidate"]["final_request_mps"],0.)
        for changes in ({"crossing_stop_gap_m":8.},{"rgb_person_height":.06},
                        {"geometry_scans":1.},{"mapped_lamp_gap_m":18.},{"camera_stop_horizon_m":0.},
                        {"oncoming_gap":25.},{"verified_forward_m":5.}):
            self.assertEqual(self.prime(15354,changes)["source_approach_candidate"]["cap_mps"],0.)
        for change in ({"source_approach_pavement":False},{"return_gap_accepted":False}):
            self.assertEqual(self.prime(15354,tactical=change)["source_approach_candidate"]["cap_mps"],0.)
        # An existing in-progress change or the required navigation merge is
        # owned by the untouched planner. This variant can only permit following.
        for change in ({"committed":True},{"navigation_merge":True},{"turn_signal":"left"}):
            self.assertNotIn("source_approach_candidate",self.prime(15354,tactical=change))


if __name__ == "__main__":
    unittest.main()
