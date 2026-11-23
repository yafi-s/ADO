"""Archive and counterfactual checks on the separate positive-pavement approach."""
import gzip
import json
import unittest
from pathlib import Path

import numpy as np

from tools.maneuver_free_source_candidate import FreeSource, MODEL, ROOT, paired


BOARD = ROOT/"results/city-maneuvers/pace/native-v07-boulevard/boulevard"


class LocalRunway(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(BOARD/"trace.jsonl.gz","rt") as stream:
            cls.board = next(r for r in map(json.loads,stream) if r["frame"]==15354)["tactical_pass"]
        with np.load(BOARD/"observations.npz") as tape:
            cls.fields = dict(zip(tape["feature_names"],tape["features"][tape["frame"]==15354][0]))

    def setUp(self):
        self.model=FreeSource.load(MODEL)
        self.model.new_observer()
        self.model.tactical_speed_limit=0.

    def allowance(self,features=None,source=None):
        return self.model.allowance(self.board|(source or {}),self.fields|(features or {}))[0]

    def test_source_path_and_current_body_clearance(self):
        self.assertAlmostEqual(self.allowance(),.9490153,places=5)
        for change in ({"return_swept_intrusion":True},{"target_front_m":2.},
                       {"source_approach_pavement":False},{"source_lateral_m":.31}):
            self.assertEqual(self.allowance(source=change),0.)
        # True swept footprint shortly after a lane-change abort, while the
        # ego centroid is still inside the nominal allowed center tolerance.
        self.assertEqual(self.allowance({"route_7m_y_m":.45},{"source_lateral_m":.24}),0.)
        for change in ({"geometry_scans":1.},{"observed_corridor_m":4.},{"verified_forward_m":4.},
                       {"path_body_lidar_recent":6.},{"crossing_stop_gap_m":8.},
                       {"mapped_lamp_gap_m":20.},{"oncoming_gap_recent":22.}):
            self.assertEqual(self.allowance(change),0.)
        self.assertIsNone(self.allowance(source={"committed":True}))
        self.assertIsNone(self.allowance(source={"navigation_merge":True}))

    def test_resolved_flank_person_only_allows_stopping_before_camera_horizon(self):
        optical=dict(rgb_person_height=.085,camera_stop_horizon_m=6.85,off_path_camera_boxes=1.,
                     observed_corridor_m=0.,observed_conflict_m=0.,confirmed_scans=0.,
                     verified_forward_m=20.,verified_cross_sight_m=6.74)
        cap=self.allowance(optical)
        self.assertGreater(cap,0.)
        self.assertLess(cap,1.1)
        self.assertLessEqual(cap,self.model._stoppable_speed(6.85-self.model.standoff))
        for change in ({"camera_stop_horizon_m":0.},{"camera_stop_horizon_m":4.6},
                       {"off_path_camera_boxes":0.},{"crossing_stop_gap_m":8.},
                       {"geometry_scans":1.},{"verified_cross_sight_m":0.}):
            self.assertEqual(self.allowance(optical|change),0.)

    def test_retained_original_and_near_obstacles(self):
        rows=[r for r in paired(BOARD) if 15341<=r["frame"]<=15393]
        self.assertEqual(len(rows),53)
        self.assertEqual(sum(r["stock"]>0. for r in rows),0)
        self.assertEqual(sum(r["request"]>0. for r in rows),32)
        self.assertGreater(next(r for r in rows if r["frame"]==15354)["throttle"],.3)
        for folder,frame in (("roadworks-pace/blocked",7292),
                             ("city-maneuvers/speed/stationary-queue",5432),
                             ("city-maneuvers/attempts/rehearsal02",8956),
                             ("city-maneuvers/pace/native-v06-boulevard/boulevard",4368)):
            with np.load(ROOT/Path("results")/folder/"observations.npz") as tape:
                saved=dict(zip(tape["feature_names"],tape["features"][tape["frame"]==frame][0]))
            near={k:saved[k] for k in ("path_body_lidar","path_body_radar","path_body_lidar_recent","path_body_radar_recent")}
            self.assertEqual(self.allowance(near),0.)


if __name__=="__main__":
    unittest.main()
