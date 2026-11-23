"""Single-thread CPU audit of normal NanoDet boxes and matched native source-lane returns."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import carla
import cv2
import numpy as np

from crosswalk.behavior_v2 import _ground_plane
from crosswalk.camera_geometry import CAMERAS
from crosswalk.maneuver_map import discover
from crosswalk.policy import LIDAR_Z
from crosswalk.radar_geometry import ego_returns
from crosswalk.urban_map import urban_route
from crosswalk.vision import Detector,MODEL_PATH,summarize
from tools.maneuver_source_vehicle import source_surface_vehicle


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read_rows(path):
    with (gzip.open(path,"rt") if path.suffix==".gz" else path.open()) as f:
        return [json.loads(l) for l in f if l.endswith("\n")]


def matrix(pos,pyr):
    t=carla.Transform(carla.Location(*pos),carla.Rotation(pitch=pyr[0],yaw=pyr[1],roll=pyr[2]))
    return np.asarray(t.get_matrix())


def world(points,pose):
    return points@pose[:3,:3].T+pose[:3,3]


def local(points,pose):
    return (points-pose[:3,3])@pose[:3,:3]


def project(points):
    cam=CAMERAS["front"]
    depth=points[:,0]-cam.x
    safe=np.maximum(.001,depth)
    return depth,.5+cam.fx*(points[:,1]-cam.y)/safe,.5-cam.fy*(points[:,2]-cam.z)/safe


def ground_location(box,plane):
    x1,_,x2,y2=box
    cam=CAMERAS["front"]
    cy=(.5*(x1+x2)-.5)/cam.fx
    down=(y2-.5)/cam.fy
    a,b,c=plane
    den=down+a+b*cy
    if den<=0.:
        return None
    depth=(cam.z-a*cam.x-b*cam.y-c)/den
    if not 1.<depth<90.:
        return None
    return np.array([[cam.x+depth,cam.y+cy*depth,cam.z-down*depth]])


def matched_projection(box,lane,station,laser,radar,radial,scan_pose,box_pose,gap):
    target=[]
    for points in (laser,radar):
        in_world=world(points,scan_pose)
        rgb_local=local(in_world,box_pose)
        depth,u,v=project(rgb_local)
        rel,side,height,_,valid=lane.project_scan(in_world,station)
        xmin,ymin,xmax,ymax=box
        mask=(depth>0.)&(u>=xmin)&(u<=xmax)&(v>=ymin)&(v<=ymax)&valid&(height>.25)&(height<3.2)
        target.append((mask,rel,side,rgb_local))
    plane=_ground_plane(*target[0][3].T)
    foot=ground_location(box,plane)
    foot_arc=foot_side=None
    if foot is not None:
        sr,sd,_,_,valid=lane.project_scan(world(foot,box_pose),station)
        if valid[0]:
            foot_arc,foot_side=float(sr[0]),float(sd[0])
    # Score the independently tracked forward source surface. Also report
    # all supported raw depths within the box so a nearer panel cannot be
    # silently assigned to a distant vehicle. No simulated actor lookup.
    out=dict(ground_ray_station_m=foot_arc,ground_ray_lateral_m=foot_side)
    for name,(mask,rel,side,_) in zip(("lidar","radar"),target):
        source=mask&(abs(side)<1.22)&(rel>2.)&(rel<59.)
        close=source&(abs(rel-gap)<1.65)
        out[name+"_anywhere_hits"]=int(np.sum(mask))
        out[name+"_source_hits"]=int(np.sum(source))
        out[name+"_matched_hits"]=int(np.sum(close))
        out[name+"_nearest_m"]=float(np.min(rel[source])) if np.any(source) else None
        out[name+"_station_m"]=float(np.median(rel[close])) if np.any(close) else None
        out[name+"_lateral_m"]=float(np.median(side[close])) if np.any(close) else None
        if name=="radar":
            out["radar_ground_speed_mps"]=float(np.median(radial[close])) if np.any(close) else None
    out["matched_both"]=bool(out["lidar_matched_hits"]>=2 and out["radar_matched_hits"]>=2
        and abs(out["lidar_station_m"]-out["radar_station_m"])<1.3 and
        abs(out["lidar_lateral_m"]-out["radar_lateral_m"])<1.)
    return out


def capture(folder,road,detector,windows,destination):
    src=folder/"trace.jsonl.gz"
    rows=read_rows(src if src.exists() else folder/"live.jsonl")
    first=rows[0]["frame"]
    refs={r["frame"]:r["ego_pose"] for r in read_rows(folder/"reference-native.jsonl")}
    route=[p.transform.location for p in urban_route(road,212,destination).points]
    lanes=discover(road,route)
    video=cv2.VideoCapture(str(folder/"front.mjpeg"))
    assert video.isOpened()
    tape_path=folder/"observations.npz"
    tape=names=features=None
    if tape_path.exists():
        with np.load(tape_path) as f:
            names=f["feature_names"].tolist();features=f["features"];tape=f["frame"]
    end=max(b for _,b,_ in windows)
    boxes,box_frame=[],None
    records=[]
    try:
        for i,r in enumerate(rows):
            frame=r["frame"]
            if frame>end:
                break
            assert frame==first+i and video.grab()
            cases=[case for start,stop,case in windows if start-1<=frame<=stop]
            if not cases:
                continue
            if i%2==0:
                ok,bgr=video.retrieve();assert ok
                boxes=detector.detect(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));box_frame=frame
            if box_frame is None or not any(start<=frame<=stop for start,stop,_ in windows):
                continue
            a=r["tactical_pass"]
            pose_record=refs[box_frame]
            box_pose=matrix([pose_record[k] for k in ("x","y","z")],[pose_record[k] for k in ("pitch","yaw","roll")])
            boxlist=[dict(b) for b in boxes if b["label"] in ("car","bus","truck")]
            summary=summarize(boxes)
            item=dict(frame=frame,box_frame=box_frame,age_s=(frame-box_frame)*.1,station=r["progress_m"],
                      phase=a.get("maneuver_phase"),gap=a.get("source_gap_m"),lead_speed=a.get("source_speed_mps"),
                      case=cases[-1],boxes=boxlist)
            if tape is not None:
                assert frame==tape[i]
                item["live_vehicle_summary"]=[float(features[i,names.index("rgb_vehicle_"+k)]) for k in ("confidence","height","center_offset")]
                item["jpeg_vehicle_summary"]=summary[3:6]
                item["live_body"]={k:float(features[i,names.index(k)]) for k in
                    ("path_body_lidar","path_body_radar","radar_surface_speed","body_support")}
            file=folder/"sensor-witness"/f"{frame}.npz"
            if file.exists() and a.get("source_lane"):
                with np.load(file) as f:
                    assert f["frame"]==frame
                    scan_pose=matrix(f["ego_xyz"],f["pitch_yaw_roll"])
                    laser=f["lidar"][:,:3].copy();laser[:,2]+=LIDAR_Z
                    radar=ego_returns(f["radar"],"radar")
                    az,alt=f["radar"][:,1:3].T
                    # Native forward radar's positive radial velocity is
                    # receding. Correct for ego speed along its measured ray.
                    radial=radar[:,3]+r["speed_mps"]*np.cos(az)*np.cos(alt)
                    lane=next(l for l in lanes if list(l.source_key)==a["source_lane"] and
                              l.locate(scan_pose[:2,3],scan_pose[:2,0]) is not None)
                    station=lane.locate(scan_pose[:2,3],scan_pose[:2,0])[0]
                    item["raw"]=True
                    item["source_vehicle"]=source_surface_vehicle(boxlist,f["lidar"],f["radar"],scan_pose,box_pose,
                        lane,station,a["source_gap_m"],r["speed_mps"],frame,box_frame)
                    for b in boxlist:
                        b.update(matched_projection(b["box"],lane,station,laser,radar[:,:3],radial,
                                                    scan_pose,box_pose,a["source_gap_m"]))
            records.append(item)
    finally:
        video.release()
    return records


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix",type=Path,required=True)
    p.add_argument("--moving",type=Path,required=True)
    p.add_argument("--board",type=Path,required=True)
    p.add_argument("--map",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    cv2.setNumThreads(1)
    detector=Detector();road=carla.Map("Town03_Opt",args.map.read_text())
    prefix=capture(args.prefix,road,detector,[(12037,12085,"source cones"),(12209,12275,"board and stopped sedan")],"market-loop")
    moving=capture(args.moving,road,detector,[(2365,2433,"moving lead")],"market")
    board=[dict(frame=r["frame"],phase=r["tactical_pass"]["maneuver_phase"],
                original_detector_boxes=r["detections"],source_vehicle=None) for r in read_rows(args.board/"trace.jsonl.gz")
           if 7553<=r["frame"]<=7587]
    if not board or any(q["original_detector_boxes"] for q in board):
        raise ValueError("Board control requires the original empty camera observations")
    provenance={}
    for case,pfolder in (("prefix",args.prefix),("moving",args.moving)):
        selected=prefix if case=="prefix" else moving
        provenance[case]=dict(files={n:sha(pfolder/n) for n in ("front.mjpeg","trace.jsonl.gz","observations.npz")},
                              range_witnesses={str(r["frame"]):sha(pfolder/"sensor-witness"/f"{r['frame']}.npz")
                                               for r in selected if r.get("raw")})
    result=dict(scope="Normal detector rerun on JPEG recording; rows contain no external traffic identity or score annotations. Box timestamps follow normal 5 Hz cadence, association uses the current native range snapshot and both ego poses. Ground-plane ray is an independent, sometimes biased box-bottom estimate. Only raw=true rows with a non-null source_vehicle are positive witnesses. Missing raw data is unobserved. Rigid-board control uses retained original empty detector observations.",
                detector_sha256=sha(MODEL_PATH),map_sha256=sha(args.map),helper_sha256=sha(Path(__file__).with_name("maneuver_source_vehicle.py")),
                board_trace_sha256=sha(args.board/"trace.jsonl.gz"),inputs=provenance,prefix=prefix,moving=moving,rigid_board=board)
    document=json.dumps(result,indent=2,allow_nan=False)+"\n"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf8") as f:
        f.write(document)
