"""Compare matched original City Life segments and seed-matched safety controls."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

import numpy as np

from tools.score_city_interactions import Body, OFFSETS, box_for, rotation, signed_gap


def read_gzip(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        return list(map(json.loads,f))


def summarize(path, start=None,end=None):
    root=Path(path)
    summary=json.loads((root/"summary.json").read_bytes())
    trace=read_gzip(root/"trace.jsonl.gz")
    selected=[r for r in trace if start is None or start<=r["progress_m"]<end]
    keys={r["frame"] for r in selected}
    references=read_gzip(root/"reference.jsonl.gz")
    shapes=json.loads(OFFSETS.read_bytes())
    ego_shape=shapes["vehicle.lincoln.mkz"]
    pedestrian_gap=math.inf
    seen=set()
    for r in references:
        if r["frame"] not in keys:
            continue
        x,y,yaw=r["ego"]
        ego=Body([x,y,r["ego_z"]],rotation(0.,yaw,0.),ego_shape["dimensions_m"],
                 ego_shape["offset_xyz"],ego_shape["rotation"])
        for actor in r["others"]:
            if actor["type"]!="pedestrian" or math.hypot(actor["x"]-x,actor["y"]-y)>35.:
                continue
            body,_=box_for(actor,shapes)
            if body is None or body.bottom>ego.top or body.top<ego.bottom:
                continue
            seen.add(actor["actor_id"])
            pedestrian_gap=min(pedestrian_gap,signed_gap(ego.polygon,body.polygon))
    speed=np.array([r["speed_mps"] for r in selected])
    target=np.array([r["speed_request_mps"] for r in selected])
    acceleration=(speed[5:]-speed[:-5])/.5
    events=[{k:v for k,v in e.items() if k in {"kind","exposed","handled","stopped_for_event","violation",
                "minimum_gap_m","minimum_cone_clearance_m","minimum_barrier_clearance_m",
                "max_cone_displacement_m","max_barrier_displacement_m","minimum_stopline_margin_m",
                "cones_alongside","barriers_alongside"}} for e in summary["events"] if e["exposed"]]
    return dict(native_pass=summary.get("passed",summary.get("target_pass")),checkpoint=summary["model_sha256"],
        contacts=summary["collision_events"],red=summary["red_light_violations"],yield_violations=summary["yield_violations"],
        station_m=None if start is None else [start,end],frames=len(selected),seconds=round(.1*len(speed),1),
        movement_kmh=dict(median=round(float(np.median(speed)*3.6),2),peak=round(float(max(speed)*3.6),2)),
        requested_kmh=dict(median=round(float(np.median(target)*3.6),2),peak=round(float(max(target)*3.6),2)),
        seconds_below_10=round(.1*sum(speed<10./3.6),1),seconds_below_15=round(.1*sum(speed<15./3.6),1),
        worst_half_second_decel_mps2=round(float(max(0.,-min(acceleration))),2) if len(acceleration) else None,
        target_drops_over_5_kmh=int(sum(np.diff(target)*3.6< -5.)), 
        binding_counts=dict(Counter(r.get("progress_evidence",{}).get("maneuver_binding",
                          r.get("progress_evidence",{}).get("binding","unknown")) for r in selected)),
        sampled_nearby_pedestrians=len(seen),minimum_sampled_pedestrian_gap_m=round(pedestrian_gap,3) if seen else None,
        events=events)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--window",nargs=2,type=float,action="append",default=[])
    a=p.parse_args()
    comparisons=[]
    for key in a.window or [(None,None)]:
        v0=summarize(a.base,*key)
        v1=summarize(a.candidate,*key)
        comparisons.append(dict(window_m=key,v06=v0,v07=v1,median_delta_kmh=round(
            v1["movement_kmh"]["median"]-v0["movement_kmh"]["median"],2),
            time_delta_s=round(v1["seconds"]-v0["seconds"],1)))
    a.output.write_text(json.dumps(dict(schema="ado-city-maneuvers-native-pace-v1",results=comparisons),
                                    indent=2)+"\n",encoding="utf8")

if __name__=="__main__":
    main()
