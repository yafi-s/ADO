"""Project independent reference bodies into each acquired front image using the native six-axis ego pose."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from score_city_interactions import Body, OFFSETS, PARKED, box_for, read_json, rotation, subtract, transform


def project(raw, output):
    telemetry_path=raw/"telemetry.json"
    poses={r["frame"]:r["ego_pose"] for r in read_json(telemetry_path) if r["camera_frame"]==r["frame"]}
    offsets=read_json(OFFSETS)
    parked=read_json(PARKED)["actors"]
    records=[]
    methods=Counter()
    with (raw/"perception.jsonl").open() as source:
        for line in source:
            f=json.loads(line)
            pose=poses[f["frame"]]
            ego=tuple(pose[k] for k in ("x","y","z"))
            world_to_ego=tuple(zip(*rotation(pose["pitch"],pose["yaw"],pose["roll"])))
            boxes={}
            for a in [*f["observer_map"]["others"],*parked]:
                if a["type"] not in ("pedestrian","car","van","truck","bus"):
                    continue
                if math.hypot(a["x"]-ego[0],a["y"]-ego[1])>102:continue
                body,method=box_for(a,offsets)
                if body is None:
                    origin=tuple(a[k] for k in ("x","y","z"))
                    body=Body(origin,rotation(0,a["yaw"],0),a["dimensions_m"],
                              (0,0,0 if a["type"]=="pedestrian" else a["dimensions_m"][1]/2),(0,0,0))
                vertices=[transform(world_to_ego,subtract(p,ego)) for p in body.vertices]
                if any(x-1.30<.5 for x,y,z in vertices):continue
                uv=[(.5+.5*y/(x-1.30),.5-(640/360)*.5*(z-1.65)/(x-1.30)) for x,y,z in vertices]
                us,vs=zip(*uv)
                b=[max(0,min(us)),max(0,min(vs)),min(1,max(us)),min(1,max(vs))]
                if b[0]<b[2] and b[1]<b[3]:
                    boxes[str(a["actor_id"])]=[round(v,7) for v in b]
                    methods[method]+=1
            records.append([f["frame"],boxes])
    if len(poses)!=len(records):
        raise ValueError("Pose and sensor journals differ in length")
    with telemetry_path.open("rb") as source:
        digest=hashlib.file_digest(source,"sha256").hexdigest()
    output.write_text(json.dumps(dict(telemetry_sha256=digest,body_sources=methods,frames=records),separators=(",",":")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args();project(a.raw,a.output)
