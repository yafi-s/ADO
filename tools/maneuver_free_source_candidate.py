"""Opt-in straight source-lane following using positive local swept-pavement evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0,str(Path.cwd()))
from crosswalk import city_maneuvers as release
from crosswalk import route_maneuvers
from crosswalk.behavior_v2 import BUMPER_X


ROOT = Path(release.__file__).resolve().parents[1]
MODEL = ROOT/"models/city-maneuvers/candidate.npz"
CHECKPOINT = "12257634d76f30bdcdfb7a61b34c1cddecbc5713ee88720631bc9bb7b8af1071"
TACTICAL = "18c7543ed01beb2fce9b94abd9aac0db426df72dbf3877702e106741da059ada"
SECTOR = "3e7fff3bb8fed01e1f95829cce9e1bce529ce58d6bc4cb60eeae849c950242e0"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FreeSource(release.ManeuverModel):
    def new_observer(self):
        observer = super().new_observer()
        self.scene_frame = self.source = None
        read = observer.observe

        def observe(obs,*args,**kwargs):
            features = read(obs,*args,**kwargs)
            self.witness(obs.get("progress_frame"),obs.get("progress_tactical",{}))
            return features

        observer.observe = observe
        return observer

    def witness(self, frame, source):
        self.source = source if frame is not None and self.scene_frame==frame-1 else None
        self.scene_frame = frame

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
        bounded_person = (x["off_path_camera_boxes"]>=1. and self.standoff+1.<camera<59.
                          and max(x["rgb_person_height"],x["rgb_person_height_recent"])>0.)
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

    def predict(self, x):
        a, self.source = self.source, None
        if a is None:
            return super().predict(x)
        original = self.tactical_speed_limit
        cap, why = self.allowance(a,dict(zip(self.feature_names,x,strict=True)))
        if cap is None:
            return super().predict(x)
        self.tactical_speed_limit = cap
        try:
            act = super().predict(x)
        finally:
            self.tactical_speed_limit = original
        a["free_source_candidate"] = dict(original_cap_mps=original,cap_mps=cap,reason=why,
                                          composed_request_mps=self.last_speed_request)
        a["tactical_speed_limit_mps"] = cap
        return act


def paired(folder):
    stock, new = release.ManeuverModel.load(MODEL),FreeSource.load(MODEL)
    stock.new_observer()
    new.new_observer()
    with gzip.open(folder/"trace.jsonl.gz","rt") as rows, np.load(folder/"observations.npz") as obs:
        if tuple(obs["feature_names"])!=tuple(new.feature_names):
            raise ValueError("Paired replay needs the selected model's actual feature tape")
        for line,frame,x in zip(rows,obs["frame"],obs["features"],strict=True):
            r = json.loads(line)
            if r["frame"]!=frame:
                raise ValueError("Frame mismatch")
            a = dict(r.get("tactical_pass",{}))
            for p in (stock,new):
                p.tactical_speed_limit = a.get("tactical_speed_limit_mps")
                p.signed_speed_limit = r.get("signed_approach_cap_mps")
                p.posted_limit_kmh = r["posted_limit_kmh"]
            stock.predict(x)
            new.witness(int(frame),a)
            _,throttle,brake = new.predict(x)
            yield dict(frame=int(frame),station=r["progress_m"],speed=r["speed_mps"],recorded=r["speed_request_mps"],
                       stock=stock.last_speed_request,request=new.last_speed_request,throttle=throttle,brake=brake,
                       **a.get("free_source_candidate",{}))


def native(a):
    if sha(Path(route_maneuvers.__file__))!=TACTICAL or sha(MODEL)!=CHECKPOINT:
        raise ValueError("Unexpected release planner or selected model")
    if a.suite=="spaced" and (a.runner is None or sha(a.runner)!=SECTOR):
        raise ValueError("Expected the matched spaced_high_sector.py")
    if a.suite=="board" and a.scratch is None:
        raise ValueError("The original board runner requires --scratch")
    a.output.mkdir(parents=True,exist_ok=False)
    tool = Path(__file__)
    (a.output/tool.name).write_bytes(tool.read_bytes())
    (a.output/"candidate.json").write_text(json.dumps(dict(candidate=sha(tool),release=TACTICAL,
        checkpoint=CHECKPOINT,runner=SECTOR if a.suite=="spaced" else "city_maneuvers_pace_trials",
        args=sys.argv),indent=2),encoding="utf8")
    if a.suite=="spaced":
        # The original harness writes relative to its own file. Copy the exact
        # registered bytes to isolate the output while retaining its pin, port,
        # camera profile, population, geometry and all grades.
        harness=a.output/"spaced_high_sector.py"
        harness.write_bytes(a.runner.read_bytes())
        with patch.object(release,"ManeuverModel",FreeSource),patch.object(sys,"argv",[str(harness),a.case]):
            runpy.run_path(str(harness),run_name="__main__")
    else:
        mod="crosswalk.city_maneuvers_pace_trials"
        argv=[mod,"--attach","--port",str(a.port),"--output",str(a.output/"capture"),"--scratch",str(a.scratch),
              "--model",str(MODEL),"--cases","boulevard"]
        with patch.object(release,"ManeuverModel",FreeSource),patch.object(sys,"argv",argv):
            runpy.run_module(mod,run_name="__main__")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    sub=cli.add_subparsers(dest="command",required=True)
    p=sub.add_parser("replay")
    p.add_argument("folder",type=Path)
    p.add_argument("--range",nargs=2,type=int,required=True)
    p=sub.add_parser("native")
    p.add_argument("--suite",choices=["spaced","board"],required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--runner",type=Path)
    p.add_argument("--case",choices=["continuous","later-gap"],default="continuous")
    p.add_argument("--port",type=int,default=3818)
    p.add_argument("--scratch",type=Path)
    args=cli.parse_args()
    if args.command=="native":
        native(args)
    else:
        print(json.dumps([r for r in paired(args.folder) if args.range[0]<=r["frame"]<=args.range[1]],indent=2))
