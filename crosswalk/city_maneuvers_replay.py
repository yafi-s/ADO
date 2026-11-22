"""Replay range, sealed base features and reconstructed camera boxes on fixed recorded motion."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from .city_drive import load_driver
from .city_maneuvers import ManeuverObserver
from .vision import Detector

WINDOWS={"J1932":[704.7,735.4], "boards":[525,550],"bus_exit":[882,909],
         "occluded_approach":[130,152],"moving_lead":[600,645],"boulevard":[420,500],
         "bus_flank":[858,872]}


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
            extra=obs.measure(scan,camera[frame],path,r["speed_mps"],x[:-2],
                              valid_image=valid_image,contiguous=i>0 and rows[i-1]["frame"]==frame-1)
        for policy in (base,model):
            policy.posted_limit_kmh=r["posted_limit_kmh"]
            policy.signed_speed_limit=r.get("signed_approach_cap_mps")
            policy.tactical_speed_limit=r["tactical_pass"].get("tactical_speed_limit_mps")
        base.predict(x)
        model.predict(np.r_[x[:-2],extra,x[-2:]])
        evidence=model.progress_evidence
        report.append(dict(frame=frame,station_m=r["progress_m"],ego_kmh=r["speed_mps"]*3.6,
            old_kmh=r["speed_request_mps"]*3.6,base_replay_kmh=base.last_speed_request*3.6,
            candidate_kmh=model.last_speed_request*3.6,added_this_frame_kmh=evidence["maneuver_added_mps"]*3.6,
            control=r["progress_evidence"]["binding"],camera_candidates=int(extra[4]),
            camera_stop_horizon_m=round(extra[3],3),shoulder_sight_m=round(extra[2],3),
            binding=evidence["maneuver_binding"],crossing_gap_m=r["progress_evidence"]["crossing_gap_m"]))
    video.release()
    def summary(part):
        def stat(field): return round(float(np.median([q[field] for q in part])),3)
        if not part: return {}
        return dict(frames=len(part),old_median_requested_kmh=stat("old_kmh"),new_median_requested_kmh=stat("candidate_kmh"),
                    measured_unchanged_ego_kmh=stat("ego_kmh"),additional_request_frames=int(sum(r["added_this_frame_kmh"]>.05 for r in part)),
                    additions_with_crossing=int(sum(r["added_this_frame_kmh"]>.05 and r["crossing_gap_m"]<60. for r in part)),
                    median_camera_horizon_m=stat("camera_stop_horizon_m"),max_base_replay_error_kmh=max(abs(r["base_replay_kmh"]-r["old_kmh"]) for r in part))
    output.mkdir(parents=True,exist_ok=True)
    window={k:summary([q for q in report if a<=q["station_m"]<b]) for k,(a,b) in WINDOWS.items() if b>a}
    audit=dict(schema="ado-city-maneuvers-pace-replay-v1",native_capture=stage.name,
               caveat="Fixed native motion and original base tape. Camera boxes reconstructed from compressed front RGB. Region exposure and policy responses can differ in closed loop. Camera brightness/contrast checked on each synchronized JPEG.",
               all=summary(report),windows=window,abrupt_fallbacks=[q for q in report if q["frame"] in (2021,2033,2045,1607,1615)])
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
