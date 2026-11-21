"""Recheck uncertainty withdrawals against native radar and the sealed policy-camera JPEG."""
import argparse
import hashlib
import gzip
import json
from pathlib import Path
import re
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from crosswalk.vision import Detector
from tools.score_city_interactions import box_for, camera_box, iou, rotation

def assess(stage, sweeps, sheet=None):
    cv2.setNumThreads(1)
    keys={};prev=None;first=None
    with (stage/'raw/perception.jsonl').open('rb') as f:
        for line in f:
            try:r=json.loads(line)
            except ValueError:break
            if first is None:first=r["frame"]
            e=r.get("progress_evidence",{});target=r['speed_request_mps']*3.6;extra=target-e['base_request_mps']*3.6
            if prev and prev[0]-target>=5 and prev[1]>=1 and extra<=.1 and e['binding']=='unconfirmed':
                k=r['frame'];cached=k-(k-first)%2
                keys[k]=dict(frame=k,detector_frame=cached,station_m=r['progress_m'],time_s=r['sim_time'],delta_kmh=round(prev[0]-target,3),history_scans=e['supported_scans'])
            prev=target,extra
    targets=set(keys)|{p['detector_frame'] for p in keys.values()}
    net=Detector();out={};thumbs={};digest=hashlib.sha256();carry=b'';i=first
    def sample(jpeg):
        rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        if rgb is None:raise ValueError("JPEG could not be decoded")
        patch=rgb[rgb.shape[0]//3:3*rgb.shape[0]//4:12,::12]
        if sheet:thumbs[i]=Image.fromarray(rgb).resize((360,203))
        det=net.detect(rgb)
        near=[dict(label=d['label'],confidence=round(float(d['score']),3),normalized_height=round(float(d['box'][3]-d['box'][1]),4),
                   normalized_box=[round(float(k),3) for k in d['box']]) for d in det if d['label'] in ('person','bicycle','motorcycle') and d['box'][3]-d['box'][1]>.055]
        return dict(q75=float(np.quantile(patch,.75)),std=round(float(patch.std()),3),vulnerable_candidates=near)
    with (stage/'raw/front.mjpeg').open('rb') as f:
        while chunk:=f.read(4*1024*1024):
            digest.update(chunk);buf=carry+chunk;start=0
            for sep in re.finditer(b'\xff\xd9\xff\xd8',buf):
                end=sep.start()+2
                if i in targets:out[i]=sample(buf[start:end])
                i+=1;start=end
            carry=buf[start:]
        if i in targets and carry.endswith(b'\xff\xd9'):out[i]=sample(carry)
    for k,p in keys.items():
        with np.load(sweeps/f'{k}.npz',allow_pickle=False) as d:
            radar=d['radar'];p["valid_radar_returns"]=int(np.isfinite(radar).all(axis=1).sum())
            p["finite_ego_pose"]=bool(np.isfinite(d["ego_xy_yaw"]).all())
        p["jpeg_brightness_q75"]=out[k]["q75"];p["jpeg_contrast_sd"]=out[k]["std"]
        p["jpeg_camera_quality_pass"]=p['jpeg_brightness_q75']>24 and p['jpeg_contrast_sd']>7
        p["jpeg_detector_vulnerable"]=out[p['detector_frame']]["vulnerable_candidates"]
    if (stage/"capture.json").is_file():
        capture=json.loads((stage/"capture.json").read_bytes())
        if capture["raw_sha256"]["front.mjpeg"]!=digest.hexdigest():raise ValueError("Unsealed native camera")
        cal=capture["summary"]["camera_rendering"]["surround"]["camera_mounts"]["front"]
        telemetry={r["frame"]:r for r in json.loads((stage/"raw/telemetry.json").read_bytes()) if r["frame"] in targets}
        shapes=json.loads((ROOT/"results/perception/town03-body-offsets.json").read_bytes())
        refs={}
        with gzip.open(stage/"population-reference.jsonl.gz","rt",encoding="utf8") as f:
            for row in map(json.loads,f):
                if row["frame"] in targets:refs[row["frame"]]=row["others"]
        for p in keys.values():
            frame=p["detector_frame"];pose=telemetry[frame]["ego_pose"]
            origin=tuple(pose[k] for k in ("x","y","z"));basis=rotation(pose["pitch"],pose["yaw"],pose["roll"])
            for detection in p["jpeg_detector_vulnerable"]:
                matches=[]
                for actor in refs[frame]:
                    if actor["type"] not in ("pedestrian","cyclist","motorcycle","bicycle"):continue
                    if detection["label"]=="person" and actor["type"]!="pedestrian":continue
                    body,_=box_for(actor,shapes)
                    pix=camera_box(body,origin,basis,cal) if body else None
                    if pix is None:continue
                    overlap=iou(detection["normalized_box"],pix)
                    if overlap<.1:continue
                    relative=np.array(body.center)-np.asarray(origin)
                    local=np.asarray(basis).T@relative
                    matches.append(dict(actor_id=actor["actor_id"],role=actor["role"],iou=round(overlap,3),
                                        forward_m=round(local[0],2),side_m=round(local[1],2)))
                detection["native_box_candidates"]=sorted(matches,key=lambda m:-m["iou"])
    if sheet and keys:
        grid=Image.new("RGB",(1080,229*((len(keys)+2)//3)),(26,28,30))
        pen=ImageDraw.Draw(grid)
        for j,p in enumerate(keys.values()):
            x=j%3*360;y=j//3*229
            grid.paste(thumbs[p["detector_frame"]],(x,y))
            for d in p["jpeg_detector_vulnerable"]:
                b=d["normalized_box"]
                color="#00e084" if d.get("native_box_candidates") else "#ffd64c"
                pen.rectangle((x+360*b[0],y+203*b[1],x+360*b[2],y+203*b[3]),outline=color,width=2)
            pen.text((x+3,y+205),f'f{p["frame"]}  {p["station_m"]:.1f} m  request -{p["delta_kmh"]:.1f} km/h',fill="white")
        grid.save(sheet,quality=90)
    return dict(front_sha256=digest.hexdigest(),frames=list(keys.values()),note="Radar and odometry checks use the native sweep archive. Brightness, contrast and original policy detector are rerun independently on the recorded quality-94 JPEG from the original policy camera, respecting detector cadence. JPEG compression prevents exact attribution of the raw-RGB threshold. Frame continuity and 0/0.1 s camera age follow the fixed synchronous recording. Each reproduced box lists class-compatible projected native bodies at IoU >=0.10; these are association candidates, not semantic pixel labels. Sheet colors identify matched (green) versus unmatched (yellow) reprojections.")
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',type=Path);p.add_argument('sweeps',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(assess(a.stage,a.sweeps,a.output.with_suffix(".jpg")),indent=2)+'\n',encoding='utf8')
