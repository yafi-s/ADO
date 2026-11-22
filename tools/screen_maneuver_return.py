"""Kinematic screen of return space on the original close-spaced road.

Native participants replay their observed trajectories. This is an optimistic
feasibility screen; it cannot supply missing sensor sight or authorize a pass.
"""
import argparse
import gzip
import json
import math
from pathlib import Path

import carla
import numpy as np

from crosswalk.maneuver_map import discover
from crosswalk.urban_map import urban_route
from tools.score_city_interactions import box_for


def read(path):
    opener = gzip.open if path.suffix==".gz" else open
    with opener(path,"rt",encoding="utf8") as f:
        for l in f:
            if l.strip():
                try:yield json.loads(l)
                except json.JSONDecodeError:return


def extract(corridor, trace, refs=None):
    compact={}
    for row in read(trace):
        if 475.<=row["progress_m"]<=612.:
            t=row["tactical_pass"]
            compact[row["frame"]]=dict(frame=row["frame"],progress=row["progress_m"],speed=row["speed_mps"],
                       t=row["sim_time"],s=t.get("corridor_station_m",0.),lat=t.get("source_lateral_m",0.),
                       phase=t.get("maneuver_phase"),lead_speed=t.get("tracked_lead_speed_mps"),
                       obs_sight=t.get("pavement_reach_m",[0.] *4)[0],energy=row["speed_request_mps"])
        elif row["progress_m"]>630.:break
    shapes=json.load(open("results/perception/town03-body-offsets.json"))
    for row in read(refs or trace):
        if row["frame"]>max(compact):break
        r=compact.get(row["frame"])
        if r is None:continue
        om=row.get("observer_map",row)
        for actor in om["others"]:
            role=actor["role"]
            if role not in ("slow-traffic","lead","adjacent-rear","adjacent-front"):continue
            body,_=box_for(actor,shapes)
            poly=np.array(body.polygon)
            rel,*_=corridor.project_scan(np.c_[poly,np.full(len(poly),actor["z"])],r["s"])
            r[role]=[float(r["s"]+rel.min()),float(r["s"]+rel.max())]
    return [r for r in compact.values() if "slow-traffic" in r and "lead" in r]


def evaluate(rows, offset, target_v, decel, halfwidth_clear=.82):
    # Begin on the outbound physical path of the moving pass. Change only
    # longitudinal motion while already committed; same traffic at same clock.
    start=next(i for i,r in enumerate(rows) if r["progress"]>490. and r["phase"]=="change_left" and
               r["lead_speed"] is not None and .55<r["lead_speed"]<4.3)
    s=rows[start]["s"];v=rows[start]["speed"];dt=.1
    onset=False;votes=0;commit=None; best=-1.e4; best_at=None; bad=False
    for i in range(start,min(len(rows),start+240)):
        r=rows[i]; front=r["lead"][0]-s-2.5;rear=s-r["slow-traffic"][1]-2.5
        slow_v=1.65
        # A calibrated model would need to observe this onset from the slow
        # target's tracked bumper. Positive offsets start before passing it.
        onset |= s>=r["slow-traffic"][0]-offset
        desired=min(10.65,r["energy"]) if r["progress"]<521. else 10.65
        if onset:desired=min(desired,target_v)
        safe_front=front-max(15.,2.*v,5.*v)
        safe_rear=rear-max(12.,2.*slow_v,5.*max(0.,slow_v-v))
        slack=min(safe_front,safe_rear)
        if slack>best:
            best=slack;best_at=(r["frame"],round(v,3),round(front,3),round(rear,3))
        # Check actual target-lane motion too, never assume its vehicles yield.
        ahead=[];behind=[]
        for role in ("adjacent-front","adjacent-rear"):
            if role not in r:continue
            b,f=r[role]
            if b>s:ahead.append(b-s-2.5)
            elif f<s:behind.append(s-2.5-f)
            else:ahead.append(0.);behind.append(0.)
        target_room=min([*ahead,*behind,60.])
        if target_room<1. and s>rows[start]["s"]+15.:
            bad=True
        if commit is None:
            votes=votes+1 if slack>=0. and target_room>=1. else 0
            if votes>=5:
                L=float(np.clip(3.5+2.7*v,14.,30.))
                commit=(s,r["frame"],L,slack);min_remaining=slack
        else:
            min_remaining=min(min_remaining,slack)
            desired=min(desired,max(1.5,(front-12.5)/6.))
            if s-commit[0]>=halfwidth_clear*commit[2]+1.:
                return dict(start_frame=rows[start]["frame"],first_right=commit[1],end_frame=r["frame"],
                   front_at_end=round(front,3),rear_at_end=round(rear,3),
                   all_straddling_gap_margin_m=round(min_remaining,3),hit_target=bad,
                   can_fully_return=not bad and min_remaining>=0.,eligible_samples=True,best=round(best,3),
                   allowed_start_slack_m=round(commit[3],3),best_at=best_at)
        nxt=min(max(desired,v-dt*decel),v+dt*2.59)
        s+=(v+nxt)*.5*dt;v=nxt
        if front<3. or s>250.:break
    return dict(can_fully_return=False,eligible_samples=commit is not None,best=round(best,3),
                best_at=best_at,first_right=commit[1] if commit else None,hit_target=bad)


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument("--output",type=Path,required=True);a=cli.parse_args()
    project=Path.cwd();local=Path.home()/"AppData/Local/ADO/city-maneuvers"
    xodr=project/".local/carla-1.0/CarlaUnreal/Packs/AdditionalMaps/Content/Maps/OpenDrive/Town03_Opt.xodr"
    m=carla.Map("Town03_Opt",xodr.read_text());route=urban_route(m,212)
    c=next(c for c in discover(m,[wp.transform.location for wp in route.points]) if c.source_key==(67,-2))
    sources={"high":(local/"hero-captures/long-city-drive-9h8jac5g/raw/perception.jsonl",None),
             "nominal":(local/"full-qualification-pinned01/trace.jsonl.gz",local/"full-qualification-pinned01/reference-native.jsonl"),
             "heldout":(local/"timing-variation-registered01/trace.jsonl.gz",local/"timing-variation-registered01/reference-native.jsonl")}
    by_source={name:extract(c,*paths) for name,paths in sources.items()}
    configs=[]
    for offset in [-10.,-5.,0.,5.,10.,15.,20.,25.,30.]:
        for v in np.arange(3.,8.26,.25):
            for decel in (1.9,2.69,3.2):
                runs={k:evaluate(rows,offset,float(v),decel) for k,rows in by_source.items()}
                configs.append(dict(onset_before_lead_bumper_m=offset,sustained_cap_mps=round(float(v),3),
                                    maximum_decel_mps2=decel,runs=runs))
    n=len(configs);stat={k:dict(eligible=sum(x['runs'][k]['eligible_samples'] for x in configs),
                         fully_returnable=sum(x['runs'][k]['can_fully_return'] for x in configs)) for k in by_source}
    common=[x for x in configs if all(y['can_fully_return'] for y in x['runs'].values())]
    ordered=sorted(common,key=lambda x:min(y['all_straddling_gap_margin_m'] for y in x['runs'].values()),reverse=True)
    best_each={k:max([x for x in configs if x['runs'][k]['eligible_samples']],
                key=lambda x:x['runs'][k].get('all_straddling_gap_margin_m',-100.),default=max(configs,key=lambda x:x['runs'][k]['best'])) for k in by_source}
    data=dict(scope="Counterfactual scalar ego integration at 0.1s, same-time recorded actor boxes. Not a controller or a physical validation. Missing visibility/unknown actor actions can further restrict feasibility.",configs=n,counts=stat,common_profiles=len(common),best_common=ordered[:4],best_per_scene=best_each,
              limits=dict(source_front_floor_m=15,source_rear_floor_m=12,headway_s=2,closing_ttc_s=5,
              clear_distance="0.82 times the native 14–30m comfort blend plus 1m; full interval must maintain both original thresholds",
              candidate_acceleration_mps2=2.59,deceleration_mps2=[1.9,2.69,3.2],bumper_extent_m=2.5))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2)+'\n')
    print(n,stat,len(common))
    for k,b in best_each.items():print(k,b['onset_before_lead_bumper_m'],b['sustained_cap_mps'],b['maximum_decel_mps2'],b['runs'][k])
if __name__=="__main__":main()
