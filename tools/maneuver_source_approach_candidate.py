"""Process-local source-lane approach experiment for the frozen city driver."""
import argparse
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np

# A separate frozen checkout can run this tool by absolute path.
if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))
from crosswalk import city_maneuvers as release
from crosswalk import route_maneuvers
from crosswalk.behavior_v2 import BUMPER_X, HORIZON


PIN = "9bd6dafa0348d37e163a3acf8de553f642bebf45"
TACTICAL_HASH = "18c7543ed01beb2fce9b94abd9aac0db426df72dbf3877702e106741da059ada"
MODEL_HASH = "12257634d76f30bdcdfb7a61b34c1cddecbc5713ee88720631bc9bb7b8af1071"
ROOT = Path(release.__file__).resolve().parents[1]
MODEL = ROOT / "models/city-maneuvers/candidate.npz"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceApproach(release.ManeuverModel):
    def new_observer(self):
        observer = super().new_observer()
        self.samples = deque(maxlen=8)
        self.radar_stations = deque(maxlen=5)
        self.last_frame = self.pending = None
        original = observer.observe

        def observe(obs, *args, **kwargs):
            features = original(obs, *args, **kwargs)
            self.observe_refusal(obs.get("progress_frame"), obs.get("progress_tactical", {}))
            return features

        observer.observe = observe
        return observer

    def observe_refusal(self, frame, info):
        self.pending = None
        lane = info.get("source_lane")
        if frame is None or self.last_frame != (frame-1, lane):
            self.samples.clear()
            self.radar_stations.clear()
        self.last_frame = (frame, lane)
        if frame is None:
            return
        fresh = (info.get("source_age_scans") == 0 and info.get("lead_kind") == "static"
                 and info.get("raw_source_gap_m",60.) < 60.)
        landmark = info["corridor_station_m"]+info["raw_source_gap_m"] if fresh else None
        self.samples.append(landmark)
        self.pending = info

    def ceiling(self, info, x):
        # Only a refused discretionary pass can create an approach. The lane
        # planner's selected path, indicators, and every commitment gate stand.
        if (self.tactical_speed_limit != 0. or info.get("maneuver_phase") != "yield"
                or info.get("committed") or info.get("navigation_merge")
                or info.get("turn_signal") != "off" or abs(info["source_lateral_m"]) > .35):
            return None, "ordinary_planner"
        observations = [s for s in self.samples if s is not None]
        laser_confirmed = (self.samples[-1] is not None and len(observations)>=6
                           and max(observations)-min(observations)<=.85)
        # While still on the source centerline, forward radar measures a
        # bumper-relative gap. Associate current, unsaturated, low-Doppler
        # returns with the same stationary world landmark as the roof scan.
        # At most two missed roof scans may be bridged. A missing current
        # radar return, pavement loss or an approaching object breaks support.
        radar = [s for s in self.radar_stations if s is not None]
        fused = (info.get("source_age_scans") in (0,1,2) and self.radar_stations[-1] is not None
                 and len(radar)>=4 and max(radar)-min(radar)<.50 and
                 sum(abs(s-radar[-1])<1.65 for s in observations)>=2 and
                 abs(info["corridor_station_m"]+info["source_gap_m"]-radar[-1])<1.65 and
                 sum(abs(x[k]) for k in ("bend_7_13","bend_13_25","bend_25_40"))<.15)
        if (not (laser_confirmed or fused) or info.get("lead_kind")!="static" or
                info["source_speed_mps"] >= .45):
            return 0., "unconfirmed_static_surface"
        if x["path_body_radar"] < HORIZON and abs(x["radar_surface_speed"]) >= .65:
            return 0., "moving_radar_surface"
        if not (info["source_approach_pavement"] and info["return_gap_accepted"]):
            return 0., "unconfirmed_source_lane"
        if (x["confirmed_scans"] < 3 or x["geometry_scans"] < 3 or x["camera_stop_horizon_m"] < 59.
                or x["mapped_lamp_gap_m"] < 59. or x["rgb_lamp_red_probability"] > .25
                or max(self.light_window,default=0.) > .25 or x["junction_gap"] < 35.
                or max(x["rgb_person_height"],x["rgb_person_height_recent"]) > 0.
                or x["crossing_stop_gap_m"] < 60. or x["verified_forward_m"] < 18.
                or min(x["oncoming_gap"],x["oncoming_gap_recent"],x["oncoming_near_gap_recent"]) < 34.):
            return 0., "perception_hold"
        free = min(info["raw_source_gap_m"], info["source_gap_m"], info["pavement_reach_m"][0],
                   x["path_body_lidar"],x["path_body_radar"],x["path_body_lidar_recent"],
                   x["path_body_radar_recent"],x["long_body_gap_m"])
        # Reserve four learned standoffs for a stationary lead, plus a fresh
        # three-standoff runway before restarting a car which has already held.
        # No forward motion is credited to the lead. Four seconds of approach
        # headway also makes the envelope taper before reaching that reserve.
        available = free - 4.*self.standoff
        if available <= 0. or (x["ego_speed_mps"] < .35 and available < 3.*self.standoff):
            return 0., "near_hold"
        cap = min(self._stoppable_speed(available), self.stopping[3]/2., available/4.)
        return float(cap), "source_approach"

    def predict(self, x):
        info, self.pending = self.pending, None
        if info is None:
            return super().predict(x)
        original_cap = self.tactical_speed_limit
        features = dict(zip(self.feature_names,x,strict=True))
        radar = features["path_body_radar"]
        stationary = 0. < radar < HORIZON and abs(features["radar_surface_speed"]) < .45
        s = info.get("corridor_station_m")
        self.radar_stations.append(s+BUMPER_X+radar if s is not None and stationary else None)
        cap, reason = self.ceiling(info,features)
        if cap is None:
            return super().predict(x)
        # Only exchange the tactical ceiling. The unmodified selected pace and
        # pedal stack processes the same synchronized observations and may stop.
        self.tactical_speed_limit = cap
        try:
            action = super().predict(x)
        finally:
            self.tactical_speed_limit = original_cap
        info["source_approach_candidate"] = dict(reason=reason,original_cap_mps=original_cap,
                                                  cap_mps=cap,final_request_mps=self.last_speed_request)
        info["tactical_speed_limit_mps"] = cap
        return action


def native(args):
    if (digest(Path(route_maneuvers.__file__)) != TACTICAL_HASH or digest(MODEL) != MODEL_HASH):
        raise RuntimeError("Pinned tactical driver or selected checkpoint differs")
    if args.output.exists():
        raise FileExistsError(args.output)
    module = "crosswalk.city_maneuvers_pace_trials" if args.suite=="pace" else "crosswalk.maneuver_trials"
    argv = [module,"--attach","--port",str(args.port),"--model",str(MODEL),"--output",str(args.output),
            "--cases",*args.cases]
    if args.suite == "pace":
        if args.scratch is None:
            raise ValueError("The matched pace runner needs --scratch for camera capture")
        argv += ["--scratch",str(args.scratch)]
    try:
        with patch.object(release, "ManeuverModel", SourceApproach), patch.object(sys,"argv",argv):
            runpy.run_module(module,run_name="__main__")
    finally:
        if args.output.is_dir():
            tool = Path(__file__)
            with (args.output/tool.name).open("xb") as out:
                out.write(tool.read_bytes())
            with (args.output/"source-approach.json").open("x",encoding="utf8") as out:
                json.dump(dict(base_pin=PIN,tactical_sha256=TACTICAL_HASH,checkpoint_sha256=MODEL_HASH,
                               candidate_sha256=digest(tool),argv=argv),out,indent=2)


def replay(folder):
    rows = [json.loads(line) for line in gzip.open(folder/"trace.jsonl.gz","rt")]
    candidate = SourceApproach.load(MODEL)
    baseline = release.ManeuverModel.load(MODEL)
    for p in (baseline,candidate):
        p.new_observer()
    result = []
    with np.load(folder/"observations.npz") as obs:
        if tuple(obs["feature_names"]) != tuple(candidate.feature_names):
            raise ValueError("Fixed-input pace replay needs the matching feature schema")
        for row,frame,x in zip(rows,obs["frame"],obs["features"],strict=True):
            assert row["frame"] == frame
            info = dict(row.get("tactical_pass",{}))
            for p in (baseline,candidate):
                p.posted_limit_kmh = row["posted_limit_kmh"]
                p.signed_speed_limit = row.get("signed_approach_cap_mps")
                p.tactical_speed_limit = info.get("tactical_speed_limit_mps")
            baseline.predict(x)
            candidate.observe_refusal(int(frame),info)
            _,throttle,brake = candidate.predict(x)
            result.append(dict(frame=int(frame),station_m=row["progress_m"],speed=row["speed_mps"],
                               recorded_request=row["speed_request_mps"],baseline=baseline.last_speed_request,
                               candidate=candidate.last_speed_request,throttle=throttle,brake=brake,
                               **info.get("source_approach_candidate",{})))
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="mode",required=True)
    audit = sub.add_parser("replay")
    audit.add_argument("folder",type=Path)
    audit.add_argument("--frames",type=int,nargs="+",required=True)
    live = sub.add_parser("native")
    live.add_argument("--suite",choices=["pace","maneuvers"],default="pace")
    live.add_argument("--cases",nargs="+",required=True)
    live.add_argument("--port",type=int,required=True)
    live.add_argument("--output",type=Path,required=True)
    live.add_argument("--scratch",type=Path)
    args = cli.parse_args()
    if args.mode == "native":
        native(args)
    else:
        data = replay(args.folder)
        print(json.dumps([r for r in data if r["frame"] in args.frames],indent=2))
