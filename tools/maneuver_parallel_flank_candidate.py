"""Opt-in measured parallel-flank variant of the frozen free-source experiment."""
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
from crosswalk import behavior_v2 as v2
from crosswalk.city_progress import _pavement_horizon
from crosswalk.policy import BUMPER_X, LIDAR_Z


parent_path=Path(__file__).with_name("maneuver_free_source_candidate.py")
spec=spec_from_file_location("frozen_free_source",parent_path)
parent=module_from_spec(spec)
sys.modules[spec.name]=parent
spec.loader.exec_module(parent)
PARENT_HASH="94ce39e5f3b4136b2301a1104b454af4d2ce4f0268592d5a55f5ce5e703f504f"


def flank_scan(scan,path,a):
    """Measure actual face, ground strip and both flanks in the source tangent frame."""
    if len(scan)<30 or not np.isfinite(path).all():
        return None
    delta=path[2]-path[1]
    norm=np.linalg.norm(delta)
    if not (3.<norm<4.2) or delta[0]<=0.:
        return None
    c,s=delta/norm
    if abs(s)>np.sin(np.deg2rad(7.)) or abs(a["source_lateral_m"])>.30:
        return None
    # Conservative projected support of the entire calibrated Lincoln, plus
    # continued heading drift through the next 3 m. Positive y is ego right.
    half=.93*c+5.46*abs(s)+abs(a["source_lateral_m"])
    if half+.12>=1.70:
        return None
    x,y,z=scan[:,:3].T
    z=z+LIDAR_Z
    ga,gb,gc=v2._ground_plane(x,y,z)
    height=z-ga*x-gb*y-gc
    along=c*x+s*y
    d=-s*x+c*y-(-s*path[1,0]+c*path[1,1])
    ground=abs(height)<.22
    solid=(height>.25)&(height<3.4)&((abs(x)>3.1)|(abs(y)>1.36))
    visible=(along>=-3.)&(along<14.)
    if np.any(solid&visible&(abs(d)<half+1.12)):
        return None
    # Witness a longitudinal, near-constant left face rather than a small
    # isolated upright return. Independent ground measurements must occupy
    # the metre-wide strip between that face and the swept source envelope.
    side=solid&visible&(d < -half-1.12)&(d > -5.4)
    if np.count_nonzero(side)<10:
        return None
    edge=float(np.quantile(-d[side],.08))
    face=side&(abs(-d-edge)<.22)
    centers=[]
    for j in np.unique(np.floor(along[face]).astype(int)):
        cell=face&(along>=j)&(along<j+1)
        if np.count_nonzero(cell)>=2:
            centers.append(float(np.median(along[cell])))
    if len(centers)<3 or centers[-1]-centers[0]<1.4 or max(np.diff(centers))>2.1:
        return None
    front=float(np.min(along[side]))
    far=float(np.max(along[face]))
    strip=ground&(d < -half)&(d > -half-1.)
    breaks=np.linspace(centers[0]-.35,centers[-1]+.35,3)
    if any(np.count_nonzero(strip&(along>=lo)&(along<hi))<2 for lo,hi in zip(breaks,breaks[1:])):
        return None
    # Direct, close, width-distributed ground ahead of the current footprint.
    # Missing returns in any half or in the margin withdraw the exemption.
    masks=[(d>=-half)&(d<0.),(d>=0.)&(d<=half),(d > -half-1.)&(d < -half)]
    end=4.
    for lo in range(4,24,2):
        cell=ground&(along>=lo)&(along<lo+2)
        if any(np.count_nonzero(cell&m)<2 for m in masks):
            break
        end=lo+2.
    if end<10.:
        return None
    station=along-BUMPER_X
    right_road=ground&(station>=0.)&(station<24.)&(d>=4.)&(d<=9.)
    right=min(_pavement_horizon(station,right_road),
              v2._gap(station,solid&(d>1.08)&(d<7.),support=4))
    left_road=ground&(station>=0.)&(station<24.)&(d<=-4.)&(d>=-9.)
    parked=(along>=front-.10)&(along<=far+.60)&(-d>=edge-.18)&(-d<=edge+1.95)
    left=min(_pavement_horizon(station,left_road),
             v2._gap(station,solid&(d < -1.08)&(d > -7.)&~parked,support=4))
    # A side car can hide a crossing beyond its far end. Count positive
    # outboard ground beyond that corner, or stop short of the visible end.
    outboard=ground&(d < -half-1.)&(d > -5.4)
    far_visible=all(np.count_nonzero(outboard&(along>=far+off)&(along<far+off+2.))>=2
                    for off in (0.,2.))
    return dict(edge_m=edge,margin_m=float(np.min(-d[side])-half),front_m=front,far_m=far,
                world_front_m=a["corridor_station_m"]+front,side_ground_end_m=float(end-BUMPER_X),
                right_sight_m=float(right),remaining_left_sight_m=float(left),far_visible=bool(far_visible))


class ParallelFlank(parent.FreeSource):
    def new_observer(self):
        observer=super().new_observer()
        self.flank_history=deque(maxlen=5)
        self.last_flank_frame=None
        self.confirmed_flank=None
        read=observer.observe
        def observe(obs,detections,local_route,speed,*,route_preview=None,stopline_gap=None):
            features=read(obs,detections,local_route,speed,route_preview=route_preview,stopline_gap=stopline_gap)
            a=obs.get("progress_tactical",{})
            near=(a.get("maneuver_phase") in ("keep","follow","yield") and not a.get("navigation_merge")
                  and not a.get("committed") and min(a.get("target_front_m",60.),a.get("target_rear_m",60.))<15.)
            f=obs.get("progress_frame")
            scan=flank_scan(obs["lidar"],v2._checked_route(local_route,route_preview),a) if near else None
            self.flank_witness(f,a.get("source_lane"),scan)
            return features
        observer.observe=observe
        return observer

    def flank_witness(self,frame,lane,w):
        self.confirmed_flank=None
        # A native observation is delivered every frame. A retained raw replay
        # may sample every second frame. Never carry a missing current sweep.
        if (frame is None or self.last_flank_frame is None or self.last_flank_frame[1]!=lane or
                frame-self.last_flank_frame[0] not in (1,2) or w is None):
            self.flank_history.clear()
        self.last_flank_frame=(frame,lane) if frame is not None else None
        if w is None or frame is None:
            return
        self.flank_history.append(w)
        if len(self.flank_history)<5:
            return
        side=[o["edge_m"] for o in self.flank_history]
        pos=[o["world_front_m"] for o in self.flank_history]
        # Reject lateral or longitudinal approach even when today's margin is
        # large. Compare the same physical object's world-aligned near end.
        if np.ptp(side)>.12 or np.ptp(pos)>.45 or side[0]-side[-1]>.06 or pos[0]-pos[-1]>.12:
            return
        self.confirmed_flank=dict(w,side_ground_end_m=min(o["side_ground_end_m"] for o in self.flank_history),
                                  right_sight_m=min(o["right_sight_m"] for o in self.flank_history),
                                  remaining_left_sight_m=min(o["remaining_left_sight_m"] for o in self.flank_history),
                                  far_visible=all(o["far_visible"] for o in self.flank_history))

    def allowance(self,a,x):
        original,reason=super().allowance(a,x)
        w=self.confirmed_flank
        if (original is None or w is None or a.get("target_rear_closing_mps",float("inf"))>.5
                or abs(a.get("target_front_speed_mps",float("inf")))>.7):
            return original,reason
        # A second close bumper behind the measured flank cannot disappear
        # into its exemption. Near readings must match the current direct face.
        ahead=max(0.,w["front_m"])
        behind=max(0.,-w["far_m"])
        for gap,support,seen in ((a["target_front_m"],ahead,w["far_m"]>=-1.),
                                 (a["target_rear_m"],behind,w["front_m"]<=1.)):
            if gap<4.5 and (not seen or abs(gap-support)>1.3):
                return original,reason
        sight=min(w["right_sight_m"],w["remaining_left_sight_m"],w["side_ground_end_m"])
        # Replace only the local allowance's blanket adjacent-lane checks.
        # The actual lane planner and the complete pace model see their true,
        # original features, including both occupied-lane ranges. The parent
        # allowance still checks full body, camera, crossing, signal and lead.
        guarded=dict(a,target_front_m=max(4.5,a["target_front_m"]),target_rear_m=max(4.5,a["target_rear_m"]))
        local=dict(x,observed_conflict_m=sight,verified_cross_sight_m=sight)
        alternative,_=super().allowance(guarded,local)
        advance=w["side_ground_end_m"]-self.standoff
        if not w["far_visible"]:
            advance=min(advance,w["far_m"]-BUMPER_X-self.standoff)
        cap=min(alternative or 0.,1.5,self._stoppable_speed(advance),max(0.,advance)/2.5)
        if cap<=original:
            return original,reason
        a["parallel_flank_evidence"]=dict(w,original_cap_mps=original,local_cap_mps=float(cap))
        return float(cap),"measured_parallel_flank"


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command",choices=["native"])
    cli.add_argument("--suite",choices=["spaced","board"],required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--runner",type=Path)
    cli.add_argument("--case",choices=["continuous","later-gap"],default="continuous")
    cli.add_argument("--port",type=int,default=3818)
    cli.add_argument("--scratch",type=Path)
    args=cli.parse_args()
    if args.output.exists() or parent.sha(parent_path)!=PARENT_HASH:
        raise ValueError("Expected new evidence folder and original free-source parent")
    try:
        with patch.object(parent,"FreeSource",ParallelFlank),patch.object(parent,"__file__",__file__):
            parent.native(args)
    finally:
        if args.output.is_dir():
            (args.output/parent_path.name).write_bytes(parent_path.read_bytes())
            (args.output/"flank-dependency.json").write_text(json.dumps({parent_path.name:PARENT_HASH}),encoding="utf8")
