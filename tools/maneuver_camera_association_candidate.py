"""Isolated camera eligibility correction for the exact parallel-flank source experiment."""
import argparse
from collections import deque
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

if __name__=="__main__":
    sys.path.insert(0,str(Path.cwd()))
from crosswalk.behavior_v2 import BUMPER_X


flank_path=Path(__file__).with_name("maneuver_parallel_flank_candidate.py")
spec=spec_from_file_location("frozen_parallel_flank",flank_path)
flank=module_from_spec(spec)
sys.modules[spec.name]=flank
spec.loader.exec_module(flank)
parent=flank.parent
FLANK_HASH="97abcd12f7d612e98168efb642507ac774c0540bcaafb1f89c0a78c7657142d1"


class GeometricSource(parent.FreeSource):
    """Frozen allowance with only the central-summary clause removed from bounded_person."""

    def allowance(self, a, x):
        if (self.tactical_speed_limit != 0. or a.get("maneuver_phase") != "yield"
                or a.get("committed") or a.get("navigation_merge") or a.get("turn_signal")!="off"):
            return None, "ordinary_planner"
        dx = x["route_7m_x_m"]-x["route_3m_x_m"]
        dy = x["route_7m_y_m"]-x["route_3m_y_m"]
        length = np.hypot(dx,dy)
        if dx<3. or length>4.2:
            return 0., "unresolved_heading"
        sine,cosine = abs(dy)/length,dx/length
        # The native Lincoln is 4.892 m long and 1.836 m wide, including a
        # 0.0063 m longitudinal origin offset (town03-body-offsets.json).
        # Round its support outward and allow 3 m of continued heading drift.
        # Discovery guarantees at least 3.4 m native source width everywhere.
        swept = abs(a["source_lateral_m"])+.93*cosine+(2.46+3.)*sine
        if (sine>np.sin(np.deg2rad(7.)) or swept+.12>=1.70 or
                abs(a["source_lateral_m"])>.30 or sum(abs(x[k]) for k in
                ("bend_7_13","bend_13_25","bend_25_40"))>.15 or x["junction_gap"]<35.
                or a.get("return_swept_intrusion") or min(a["target_front_m"],a["target_rear_m"])<4.5):
            return 0., "lane_or_flank"
        camera = x["camera_stop_horizon_m"]
        # The normal camera/lidar association returns zero for unresolved,
        # in-path, or inadequately witnessed people. A positive finite bound
        # for an off-path person is the earliest plausible conflict station
        # after checking the projected box, foot, nearer occluders and ground.
        bounded_person = x["off_path_camera_boxes"]>=1. and self.standoff+1.<camera<59.
        if (x["geometry_scans"]<3. or not a.get("source_approach_pavement")
                or (not bounded_person and (camera<59. or x["confirmed_scans"]<3. or
                    max(x["rgb_person_height"],x["rgb_person_height_recent"])>0.))
                or x["crossing_stop_gap_m"]<60.
                or x["mapped_lamp_gap_m"]<59. or max(self.light_window,default=0.)>.25
                or x["rgb_lamp_red_probability"]>.25 or
                min(x["oncoming_gap"],x["oncoming_gap_recent"],x["oncoming_near_gap_recent"])<34.):
            return 0., "perception_hold"
        # Both corridor features come from successive real roof sweeps:
        # every 4 m cell in the actual 2.16 m-wide driving path must have
        # supported pavement, and an earlier supported solid shortens it.
        # The tactical path additionally requires two occupied near pavement
        # bands in the surveyed source lane on the current scan.
        ground = min(x["verified_forward_m"],a["pavement_reach_m"][0]-BUMPER_X)
        side = x["verified_cross_sight_m"]
        # The older observer clears its entire pavement window on a near
        # person. The separate three-scan geometry window still measures the
        # roadway and returns a conflict station for the resolved case only.
        if not bounded_person:
            ground = min(ground,x["observed_corridor_m"])
            side = min(side,x["observed_conflict_m"])
        body = min(x[k] for k in ("path_body_lidar","path_body_radar","path_body_lidar_recent",
                                  "path_body_radar_recent","long_body_gap_m"))
        body = min(body,a["source_gap_m"]-BUMPER_X,a["raw_source_gap_m"]-BUMPER_X)
        if (ground<10. or side<3. or body<=3.*self.standoff or
                (x["ego_speed_mps"]<.35 and body<=4.*self.standoff)):
            return 0., "near_or_unobserved"
        # Keep a full learned standoff inside measured pavement. Reserve three
        # before an occupied source surface. Flank sight, measured in the
        # adjoining sector, limits how far ahead the car could meet an
        # unseen crossing. Any measured crossing above retains the full hold.
        runway = min(ground-self.standoff,body-3.*self.standoff,side-1.08,
                     camera-self.standoff-.1*x["ego_speed_mps"])
        return float(min(3.,self._stoppable_speed(runway),max(0.,runway)/2.5)), "verified_source"


class CameraAssociation(flank.ParallelFlank,GeometricSource):
    """Parallel-flank admission with a nine-exposure camera attribution window."""

    def new_observer(self):
        observer=super().new_observer()
        self.legacy_exposures=deque(maxlen=9)
        self.camera_frame=None
        self.camera_witness=False
        return observer

    def record_camera(self,frame,x):
        if frame is None or self.camera_frame!=frame-1:
            self.legacy_exposures.clear()
        self.camera_frame=frame
        self.camera_witness=False
        if frame is None:
            return
        height=float(x["rgb_person_height"])
        stop=float(x["camera_stop_horizon_m"])
        finite=(x["geometry_scans"]==3. and x["off_path_camera_boxes"]>=1. and self.standoff+1.<stop<59.)
        attributed=height==0. or (height>.055 and finite)
        self.legacy_exposures.append((height,attributed))
        # The actual legacy feature averages these nine nonnegative heights.
        # A missing exposure cannot account for a positive tail of that mean.
        self.camera_witness=(len(self.legacy_exposures)==9 and
                             all(resolved for _,resolved in self.legacy_exposures) and
                             np.isclose(np.mean([h for h,_ in self.legacy_exposures]),
                                        x["rgb_person_height_recent"],atol=2.e-7,rtol=1.e-5))

    def predict(self,x):
        self.record_camera(self.scene_frame,dict(zip(self.feature_names,x,strict=True)))
        return super().predict(x)

    def allowance(self,a,x):
        cap,reason=super().allowance(a,x)
        if cap is None:
            return cap,reason
        if x["camera_stop_horizon_m"]<=0.:
            return 0.,"zero_optical_horizon"
        if x["off_path_camera_boxes"]>=1. and not (
                x["geometry_scans"]==3. and self.standoff+1.<x["camera_stop_horizon_m"]<59.):
            return 0.,"unresolved_current_box"
        if any(not resolved for _,resolved in self.legacy_exposures) or (
                not self.camera_witness and (x["rgb_person_height"]>0. or
                x["rgb_person_height_recent"]>0. or x["off_path_camera_boxes"]>=1.)):
            return 0.,"unresolved_legacy_history"
        return cap,reason


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["native"])
    parser.add_argument("--runner",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if (args.output.exists() or parent.sha(flank_path)!=FLANK_HASH or
            parent.sha(flank.parent_path)!=flank.PARENT_HASH):
        raise ValueError("Expected a new evidence folder and both frozen source candidates")
    # Reuse the original native High runner with its pinned checkout, case,
    # port, source archive, control rig and grading. Only loader selection
    # in this process changes. The selected pace head receives the true tape.
    args.suite,args.case,args.port,args.scratch="spaced","continuous",3818,None
    try:
        with patch.object(parent,"FreeSource",CameraAssociation),patch.object(parent,"__file__",__file__):
            parent.native(args)
    finally:
        if args.output.is_dir():
            hashes={}
            for p in (flank_path,flank.parent_path):
                (args.output/p.name).write_bytes(p.read_bytes())
                hashes[p.name]=parent.sha(p)
            (args.output/"camera-dependencies.json").write_text(json.dumps(hashes,indent=2),encoding="utf8")

