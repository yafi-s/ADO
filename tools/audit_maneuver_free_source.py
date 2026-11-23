"""Score the isolated source-lane proposal on fixed native motion and recorded raw returns."""
import argparse
import gzip
from itertools import groupby
import json
from pathlib import Path
from types import SimpleNamespace

import carla
import numpy as np

from crosswalk import behavior_v2
from crosswalk.city_progress import _profile
from crosswalk.control import Navigation
from crosswalk.policy import LIDAR_Z
from crosswalk.urban_map import urban_route
from tools import maneuver_free_source_candidate as candidate


ROOT = candidate.ROOT
BOARD = ROOT/"results/city-maneuvers/pace/native-v07-boulevard/boulevard"


def stats(values):
    return dict(zip(("min","median","max"),map(float,np.quantile(values,[0,.5,1]))))


def tape(folder):
    with gzip.open(folder/"trace.jsonl.gz","rt") as stream:
        rows=list(map(json.loads,stream))
    with np.load(folder/"observations.npz") as f:
        assert [r["frame"] for r in rows]==f["frame"].tolist()
        return rows,f["feature_names"].tolist(),f["features"]


def score(folder,held):
    rows=list(candidate.paired(folder))
    interval=[r for r in rows if held(r)]
    stopped=[r for r in interval if r["speed"]==0.]
    zero_intervals=[list(g) for k,g in groupby(interval,key=lambda q:q["request"]==0.) if k]
    return dict(frames=len(rows),max_stock_replay_abs_error_mps=max(abs(r["stock"]-r["recorded"]) for r in rows),
                held_frames=len(interval),frame_range=[interval[0]["frame"],interval[-1]["frame"]],
                positive_requests=int(sum(r["request"]>0 for r in interval)),
                positive_throttle=int(sum(r["throttle"]>0 for r in interval)),
                originally_exactly_stopped=len(stopped),positive_throttle_while_stopped=int(sum(r["throttle"]>0 for r in stopped)),
                longest_preserved_zero_request_frames=max(map(len,zero_intervals),default=0),
                cap_mps=stats([r["cap_mps"] for r in interval]),request_mps=stats([r["request"] for r in interval]),
                case_frames=[r for r in interval if r["frame"] in (15354,2610,2833,3056)],
                source_sha256={n:candidate.sha(folder/n) for n in ("trace.jsonl.gz","observations.npz")})


def halves(s,d,height):
    ground=(s>=0)&(s<24)&(abs(height)<.22)
    result=[]
    for sign in (-1,1):
        bins=np.bincount((s[ground&(sign*d>.12)&(sign*d<.98)]/4).astype(int),minlength=6)
        empty=np.flatnonzero(bins<2)
        result.append(float(4*(empty[0] if len(empty) else 6)))
    return result


def raw_score(folder,road,held):
    rows,names,features=tape(folder)
    nav=Navigation([p.transform.location for p in urban_route(carla.Map("Town03_Opt",road.read_text()),212).points])
    measurements=[]
    optical=[]
    for r,x in zip(rows,features):
        if not held({"station":r["progress_m"],"speed":r["speed_mps"]}):
            continue
        features_by_name=dict(zip(names,x))
        optical.append([features_by_name[k] for k in ("camera_stop_horizon_m","off_path_camera_boxes","crossing_stop_gap_m")])
        file=folder/"sensor-witness"/f"{r['frame']}.npz"
        if not file.exists():
            continue
        with np.load(file) as raw:
            assert raw["frame"]==r["frame"]
            pose=SimpleNamespace(location=SimpleNamespace(x=raw["ego_xyz"][0],y=raw["ego_xyz"][1]),
                                 rotation=SimpleNamespace(yaw=raw["pitch_yaw_roll"][1]))
            local=nav.read(SimpleNamespace(get_transform=lambda:pose),(3,7,13,18,25,32,40)).reshape(-1,2)
            path=np.vstack(([0.,0.],local))
            scan=raw["lidar"]
            prox=_profile(scan,path,{})
            cues=behavior_v2._range_cues(scan,raw["radar"],r["speed_mps"],path)
            cx,cy,cz=scan[:,0],scan[:,1],scan[:,2]+LIDAR_Z
            a,b,c=behavior_v2._ground_plane(cx,cy,cz)
            station,side=behavior_v2._project(cx,cy,path)
            half=halves(station,side,cz-cx*a-cy*b-c)
            measurements.append([*prox[:2],*half,cues[1],cues[2],
                                 abs(cues[1]-features_by_name["path_body_lidar"]),
                                 abs(cues[2]-features_by_name["path_body_radar"]),max(abs(local[:3].ravel()-x[:6]))])
    keys=("current_swept_runway_m","current_flank_sight_m","half_a_runway_m","half_b_runway_m",
          "raw_lidar_gap_m","raw_radar_gap_m","lidar_tape_abs_error_m","radar_tape_abs_error_m","navigation_abs_error_m")
    return dict(matched_raw_frames=len(measurements),map_sha256=candidate.sha(road),
                **{k:stats(vals) for k,vals in zip(keys,np.asarray(measurements).T)},
                **{k:stats(v) for k,v in zip(("camera_stop_horizon_m","off_path_boxes","crossing_gap_m"),np.asarray(optical).T)})


def counterfactuals():
    rows,names,features=tape(BOARD)
    index=next(i for i,r in enumerate(rows) if r["frame"]==15354)
    x=dict(zip(names,features[index]))
    a=rows[index]["tactical_pass"]
    m=candidate.FreeSource.load(candidate.MODEL)
    m.new_observer()
    m.tactical_speed_limit=0.
    out=[]
    for name,frame in (("roadworks-pace/blocked",7292),("city-maneuvers/speed/stationary-queue",5432),
                        ("city-maneuvers/attempts/rehearsal02",8956),
                        ("city-maneuvers/pace/native-v06-boulevard/boulevard",4368)):
        original,fields,obs=tape(ROOT/"results"/name)
        i=next(i for i,r in enumerate(original) if r["frame"]==frame)
        near=dict(zip(fields,obs[i]))
        replace={k:float(near[k]) for k in ("path_body_lidar","path_body_radar","path_body_lidar_recent","path_body_radar_recent")}
        cap,reason=m.allowance(a,x|replace)
        out.append(dict(native_case=name,frame=frame,recorded_request_mps=original[i]["speed_request_mps"],
                        board_substitution=replace,cap_mps=cap,reason=reason))
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaced",type=Path,required=True)
    parser.add_argument("--map",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    def stopped(q):
        return abs(q["station"]-581.841)<.003 and q["speed"]<=.01
    report=dict(scope="Fixed original ego motion and synchronized sensor evidence. No native candidate trajectory.",
                candidate_sha256=candidate.sha(Path(candidate.__file__)),checkpoint_sha256=candidate.sha(candidate.MODEL),
                frozen_tactical_sha256=candidate.sha(Path(candidate.route_maneuvers.__file__)),
                spaced_control=score(args.spaced,stopped),spaced_sensors=raw_score(args.spaced,args.map,stopped),
                original_board=score(BOARD,lambda r:15341<=r["frame"]<=15393),
                matched_near_returns_on_otherwise_eligible_board=counterfactuals())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    document=json.dumps(report,indent=2,allow_nan=False)+"\n"
    with args.output.open("x",encoding="utf8") as f:
        f.write(document)
