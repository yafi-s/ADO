"""Reference the bus's physical ends while reviewing the speed request and reacceleration."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.score_city_interactions import box_for

def measure(stage):
    shapes=json.loads((ROOT/"results/perception/town03-body-offsets.json").read_bytes())
    ego=shapes["vehicle.lincoln.mkz"];half=ego["dimensions_m"][0]/2
    rows=[]
    with (stage/"raw/perception.jsonl").open("rb") as stream:
        for line in stream:
            try:r=json.loads(line)
            except ValueError:break
            if r['progress_m']<840:continue
            if r['progress_m']>920:break
            actors=r["observer_map"]["others"]
            bus=next((a for a in actors if a["role"]=="stopped_bus"),None)
            if bus is None:continue
            box,_=box_for(bus,shapes)
            x,y,heading=r["observer_map"]["ego"];c,s=math.cos(math.radians(heading)),math.sin(math.radians(heading))
            along=[(v[0]-x)*c+(v[1]-y)*s for v in box.vertices]
            across=[-(v[0]-x)*s+(v[1]-y)*c for v in box.vertices]
            front=half+ego["offset_xyz"][0];rear=front-2*half
            close=min(along)-front;far=max(along)-front;remaining=max(along)-rear
            e=r.get("progress_evidence",{})
            rows.append(dict(frame=r["frame"],time_s=r["sim_time"],station_m=r["progress_m"],
                             kmh=round(r["speed_mps"]*3.6,3),request=round(r["speed_request_mps"]*3.6,3),
                             nose_to_bus_rear_m=round(close,3),nose_to_bus_front_m=round(far,3),
                             rear_to_bus_front_m=round(remaining,3),side_margin_m=round(max(min(across),-max(across))-ego["dimensions_m"][2]/2,3),
                             overlap=close<=0<=remaining,phase=r["phase"],binding=e.get("binding"),budget_m=e.get("sight_m"),
                             blind_end_m=e.get("blind_end_m"),tactical_speed_cap_mps=r.get("tactical_pass",{}).get("tactical_speed_limit_mps")))
    def first(cond):
        return next((r for r in rows if cond(r)),None)
    def stats(seq):
        if not seq:return {"frames":0}
        return dict(frames=len(seq),seconds=round(.1*len(seq),1),median_kmh=round(statistics.median(v["kmh"] for v in seq),3),
                    median_request_kmh=round(statistics.median(v["request"] for v in seq),3),sub10_s=round(.1*sum(v['kmh']<10 for v in seq),1),
                    first=seq[0],last=seq[-1])
    nose_passed=first(lambda v:v['nose_to_bus_front_m']<=0);rear_passed=first(lambda v:v['rear_to_bus_front_m']<0)
    return dict(first_overlap=first(lambda v:v["overlap"]),ego_nose_at_far_end=nose_passed,ego_rear_at_far_end=rear_passed,
                before_nose_reaches_far_end=stats([r for r in rows if r["overlap"] and r["nose_to_bus_front_m"]>0]),
                nose_beyond_bus_but_bodies_still_overlap=stats([r for r in rows if r["overlap"] and r["nose_to_bus_front_m"]<=0]),
                after_both_bodies_clear_first_3s=stats([r for r in rows if rear_passed and rear_passed["time_s"]<=r["time_s"]<rear_passed["time_s"]+3]),
                after_nose_passes_first_at_15=first(lambda r:nose_passed and r["time_s"]>=nose_passed["time_s"] and r["kmh"]>=15),
                trace=[r for i,r in enumerate(rows) if i%5==0])
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(measure(a.stage),indent=2)+"\n",encoding="utf8")

