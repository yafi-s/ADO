"""Replay range, sealed base features and reconstructed camera boxes on fixed recorded motion."""
import argparse
import json
import os
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np

from .city_drive import load_driver
from .vision import Detector
from .experiment import fingerprint
from .control import Navigation
from .speed_signs import SignedRouteLimits
from .urban_map import urban_route

WINDOWS={"J1932":[704.7,735.4], "boards":[525,550],"bus_exit":[882,909],
         "occluded_approach":[130,152],"moving_lead":[600,645],"boulevard":[420,500],
         "drums":[375,395],"red_approach":[299,323],"protected_left":[1025,1077],
         "delayed_pedestrian":[1256,1278],"bus_flank":[858,872]}


def boxes_from_video(stage,output):
    cv2.setNumThreads(1)
    rows=json.loads((stage/"raw/telemetry.json").read_bytes())
    stream=cv2.VideoCapture(str(stage/"raw/front.mjpeg"))
    detector=Detector()
    with output.open("x",encoding="utf8") as f:
        for i,row in enumerate(rows):
            ok,bgr=stream.read()
            if not ok:
                raise ValueError(f"Unmatched native RGB frame {row['frame']}")
            if not i%2:
                boxes=detector.detect(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            print(json.dumps([row["frame"],boxes]),file=f)
    stream.release()


def run(stage,sweeps,model_path,boxes,output):
    rows=json.loads((stage/"raw/replay.json").read_bytes())
    camera=dict(json.loads(line) for line in boxes.read_text().splitlines())
    tape=np.load(stage/"policy-observations.npz")
    base,_,_=load_driver(Path("models/city-progress/candidate.npz"))
    model,_,_=load_driver(model_path)
    base.new_observer()
    obs=model.new_observer()
    video=cv2.VideoCapture(str(stage/"raw/front.mjpeg")); cv2.setNumThreads(1)
    import carla
    installation=Path(os.environ.get("CARLA_ROOT") or ".local/carla-1.0")
    map_file=installation/"CarlaUnreal/Packs/AdditionalMaps/Content/Maps/OpenDrive/Town03_Opt.xodr"
    native=carla.Map("Town03_Opt",map_file.read_text())
    route=[w.transform.location for w in urban_route(native,rows[0]["seed"],"market").points]
    nav=Navigation(route)
    metadata=json.loads(Path("results/perception/town03-body-offsets.json").read_bytes())["vehicle.lincoln.mkz"]
    nose=metadata["offset_xyz"][0]+.5*metadata["dimensions_m"][0]
    signs=SignedRouteLimits.from_map(native,route,nav.arc,nose)
    report=[]
    for i,r in enumerate(rows):
        frame=r["frame"]
        if frame!=tape["frame"][i]:
            raise ValueError("Feature frame mismatch")
        pose=r["ego_pose"]
        yaw=np.deg2rad(pose["yaw"])
        c,s=np.cos(yaw),np.sin(yaw)
        path=(np.asarray(r["plan_xy"])-[pose["x"],pose["y"]])@np.array([[c,-s],[s,c]])
        path[0]=0.
        x=tape["features"][i]
        obs.travel+=.1*r["speed_mps"]
        ok,bgr=video.read()
        if not ok:
            raise ValueError("Missing synchronized image")
        patch=bgr[bgr.shape[0]//3:3*bgr.shape[0]//4:12,::12]
        valid_image=bool(patch.size and np.quantile(patch,.75)>24 and np.std(patch)>7)
        with np.load(sweeps/f"{frame}.npz") as scan:
            np.testing.assert_allclose(scan["ego_xy_yaw"], [pose["x"],pose["y"],pose["yaw"]],rtol=0.,atol=.002)
            transform=SimpleNamespace(location=SimpleNamespace(x=pose["x"],y=pose["y"]),
                                      rotation=SimpleNamespace(yaw=pose["yaw"]))
            far_local=nav.read(SimpleNamespace(get_transform=lambda:transform),(50.,60.,72.)).reshape(-1,2)
            full=scan["lidar"]
            limited=full[np.einsum("ij,ij->i",full[:,:3],full[:,:3])<=60**2]
            signals=dict(lidar=limited,radar=scan["radar"],long_sight_lidar=full,long_sight_preview=far_local,
                         progress_tactical=r["tactical_pass"],progress_ego_pose=[pose["x"],pose["y"],pose["yaw"]])
            extra=obs.measure(signals,camera[frame],path,r["speed_mps"],x[:-2],
                              valid_image=valid_image,contiguous=i>0 and rows[i-1]["frame"]==frame-1)
        for policy in (base,model):
            policy.posted_limit_kmh=r["posted_limit_kmh"]
            policy.signed_speed_limit=r.get("signed_approach_cap_mps")
            policy.tactical_speed_limit=r["tactical_pass"].get("tactical_speed_limit_mps")
        approach=signs.approach(nav.progress,r["speed_mps"],r["posted_limit_kmh"],
                                cruise_ceiling_mps=model.signed_preview_ceiling_mps)
        model.signed_speed_limit=approach[0] if approach else None
        base.predict(x)
        model.predict(np.r_[x[:-2],extra,x[-2:]])
        evidence=model.progress_evidence
        report.append(dict(frame=frame,station_m=r["progress_m"],phase=r["phase"],reference_gap_m=r["reference_gap_m"],ego_kmh=r["speed_mps"]*3.6,
            old_kmh=r["speed_request_mps"]*3.6,base_replay_kmh=base.last_speed_request*3.6,
            candidate_kmh=model.last_speed_request*3.6,added_this_frame_kmh=evidence["maneuver_added_mps"]*3.6,
            control=r["progress_evidence"]["binding"],camera_candidates=int(extra[4]),
            camera_stop_horizon_m=round(extra[3],3),shoulder_sight_m=round(extra[2],3),
            high_speed_sight_m=float(extra[7]),speed_regime=evidence["speed_regime"],
            binding=evidence["maneuver_binding"],crossing_gap_m=r["progress_evidence"]["crossing_gap_m"]))
    video.release()
    def summary(part):
        def stat(field): return round(float(np.median([q[field] for q in part])),3)
        if not part: return {}
        return dict(frames=len(part),old_median_requested_kmh=stat("old_kmh"),new_median_requested_kmh=stat("candidate_kmh"),
                    measured_unchanged_ego_kmh=stat("ego_kmh"),additional_request_frames=int(sum(r["added_this_frame_kmh"]>.05 for r in part)),
                    additions_with_crossing=int(sum(r["added_this_frame_kmh"]>.05 and r["crossing_gap_m"]<60. for r in part)),
                    open_road_frames=sum(r["speed_regime"]=="verified_open_road" for r in part),
                    max_requested_kmh=max(r["candidate_kmh"] for r in part),
                    median_camera_horizon_m=stat("camera_stop_horizon_m"),max_base_replay_error_kmh=max(abs(r["base_replay_kmh"]-r["old_kmh"]) for r in part))
    output.mkdir(parents=True,exist_ok=True)
    window={k:summary([q for q in report if a<=q["station_m"]<b]) for k,(a,b) in WINDOWS.items() if b>a}
    audit=dict(schema="ado-city-maneuvers-pace-replay-v1",native_capture=stage.name,
               checkpoint_sha256=fingerprint(model_path),maneuver_source_sha256=fingerprint(Path(__file__).with_name("city_maneuvers.py")),
               caveat="Fixed native motion and original base tape. Camera boxes reconstructed from compressed front RGB. Region exposure and policy responses can differ in closed loop. Camera brightness/contrast checked on each synchronized JPEG. Long-range navigation and the candidate sign preview use the native map and seed-212 route, reconstructed at archived poses.",
               map_sha256=fingerprint(map_file),native_sign_zones=signs.zones,all=summary(report),windows=window,
               protected_left={phase:summary([q for q in report if q["phase"]==phase])
                               for phase in sorted(set(q["phase"] for q in report if q["phase"].startswith("left:")))},
               deepest_yield=summary([q for q in report if q["phase"]=="left: yield to oncoming" and q["reference_gap_m"]<6.]),abrupt_fallbacks=[q for q in report if q["frame"] in (2021,2033,2045,1607,1615)])
    (output/"recorded-input.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf8")
    print(json.dumps(audit))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",type=Path,required=True)
    p.add_argument("--sweeps",type=Path,required=True)
    p.add_argument("--boxes",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",type=Path,default=Path("models/city-maneuvers/candidate.npz"))
    a=p.parse_args()
    if not a.boxes.exists():
        boxes_from_video(a.stage,a.boxes)
    run(a.stage,a.sweeps,a.model,a.boxes,a.output)

if __name__=="__main__":
    main()
