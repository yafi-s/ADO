"""Audit physical plate-to-plate overlap, lead onset and speed gates along the board site."""
import argparse
from collections import Counter
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
    car=shapes["vehicle.lincoln.mkz"]["dimensions_m"]
    rows=[]
    with (stage/"raw/perception.jsonl").open('rb') as stream:
        for line in stream:
            try:r=json.loads(line)
            except ValueError:break
            if r["progress_m"]<492:continue
            if r["progress_m"]>590:break
            refs=r["observer_map"]; x,y,yaw=refs["ego"]; yaw=math.radians(yaw)
            c,s=math.cos(yaw),math.sin(yaw)
            long=lambda v:(v[0]-x)*c+(v[1]-y)*s
            side=lambda v:-(v[0]-x)*s+(v[1]-y)*c
            boards=[];lead=None
            for a in refs['others']:
                if a['role'] not in ('workzone_barrier','lead'):continue
                b,_=box_for(a,shapes)
                verts=b.vertices
                front=min(long(v) for v in verts)-car[0]/2
                back=max(long(v) for v in verts)+car[0]/2
                lateral=max(min(side(v) for v in verts)-car[2]/2,-max(side(v) for v in verts)-car[2]/2)
                if a['role']=='lead':
                    lead=dict(forward_edge_gap_m=round(front,2),lateral_side_gap_m=round(lateral,2))
                else:
                    boards.append(dict(id=a["actor_id"],forward_edge_gap_m=round(front,2),rear_edge_gap_m=round(back,2),
                                       lateral_side_gap_m=round(lateral,2),alongside=front<=0<=back))
            e=r.get("progress_evidence",{})
            rows.append(dict(frame=r["frame"],time_s=r["sim_time"],station_m=r["progress_m"],
                             speed_kmh=round(r["speed_mps"]*3.6,3),request_kmh=round(r["speed_request_mps"]*3.6,3),
                             body_reference_gap_m=r["reference_gap_m"],lead=lead,boards=boards,
                             both_boards_behind=bool(boards) and all(b['rear_edge_gap_m']<0 for b in boards),
                             alongside=any(b["alongside"] for b in boards),phase=r["phase"],evidence=e))
    def stats(seq):
        if not seq:return dict(frames=0)
        med=lambda k:round(statistics.median(r[k] for r in seq),3)
        return dict(frames=len(seq),seconds=round(len(seq)*.1,1),first_frame=seq[0]['frame'],last_frame=seq[-1]['frame'],
                    from_s=seq[0]['time_s'],to_s=seq[-1]['time_s'],from_m=seq[0]['station_m'],to_m=seq[-1]['station_m'],
                    median_kmh=med('speed_kmh'),median_target_kmh=med('request_kmh'),
                    minimum_lead_gap_m=min(r['lead']['forward_edge_gap_m'] for r in seq if r['lead']),
                    maximum_lead_gap_m=max(r['lead']['forward_edge_gap_m'] for r in seq if r['lead']),
                    guards=dict(Counter(r['evidence'].get('binding','unavailable') for r in seq)),
                    body_lidar_radar_clear_36_frames=sum(r['evidence'].get("raw_body_memory_gap_m",0)>=35.99 for r in seq))
    lead25=lambda r:r["lead"] is not None and r["lead"]["lateral_side_gap_m"]<0 and 0<=r["lead"]["forward_edge_gap_m"]<=25
    lead35=lambda r:r["lead"] is not None and r["lead"]["lateral_side_gap_m"]<0 and 0<=r["lead"]["forward_edge_gap_m"]<=35
    pts={}
    for name,cond in [("first_lead_within_35m",lead35),("first_lead_within_25m",lead25),
                      ("first_board_overlap",lambda r:r["alongside"]),("both_boards_physically_behind",lambda r:r["both_boards_behind"]),
                      ("first_curb_target",lambda r:r["request_kmh"]<=5.42)]:
        pts[name]=next((r for r in rows if cond(r)),None)
    return dict(note="Forward lead bumper and board longitudinal footprint overlap in the ego heading frame; measured road is nearly straight at the boards. Polygons use native actor-rotation and box offsets. Rounded ego yaw/position and nominal centered ego half-length/width permit centimeter-scale differences from native world vertices.",points=pts,
                fixed_sections=[stats([r for r in rows if lo<=r["station_m"]<hi]) for lo,hi in [(510,520),(520,525),(525,530),(530,535),(535,540),(540,545),(545,550),(550,560),(560,570)]],
                alongside=stats([r for r in rows if r["alongside"]]),alongside_outside_lead_25m=stats([r for r in rows if r["alongside"] and not lead25(r)]),
                after_both_boards_before_lead_within_25m=stats([r for r in rows if r["both_boards_behind"] and r["station_m"]<pts["first_lead_within_25m"]["station_m"]]),
                source_excerpt=[r for i,r in enumerate(rows) if i%10==0 and 510<=r["station_m"]<565])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument("stage",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(measure(a.stage),indent=2)+"\n",encoding="utf8")
