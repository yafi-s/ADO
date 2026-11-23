"""Oracle upper-bound for returning between two recorded source-lane bumpers."""
import argparse
import json
import warnings
from pathlib import Path

import carla
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from tools.screen_maneuver_return import extract
from crosswalk.maneuver_map import discover
from crosswalk.urban_map import urban_route


def solve(rows,start,e,n,decel=2.69,stride=2):
    # Equalities are linear trapezoidal odometry; velocity and lateral
    # transition extent constrain how fast a real car can vacate the lane.
    dt=.1*stride
    size=2*n+3
    ps=lambda i:i
    pv=lambda i:n+1+i
    slack=2*n+2
    A=[];b=[]
    E=[];d=[]
    def put(dst,values,rhs,terms):
        j=len(values)
        for k,v in terms:dst.append((j,k,v))
        values.append(rhs)
    def bound(rhs,*terms):put(A,b,rhs,terms)
    def eq(rhs,*terms):put(E,d,rhs,terms)
    eq(rows[start]["s"],(ps(0),1.));eq(rows[start]["speed"],(pv(0),1.))
    for i in range(1,n+1):
        eq(0.,(ps(i),1.),(ps(i-1),-1.),(pv(i),-.5*dt),(pv(i-1),-.5*dt))
        bound(dt*2.59,(pv(i),1.),(pv(i-1),-1.))
        bound(dt*decel,(pv(i),-1.),(pv(i-1),1.))
    for j in range(stride*n+1):
        i=j//stride
        weights=[(i,1.)] if j%stride==0 else [(i,.5),(i+1,.5)]
        row=rows[start+j];front=row["lead"][0]-2.5; rear=row["slow-traffic"][1]+2.5
        sp=[(ps(k),v) for k,v in weights]; vm=[(pv(k),v) for k,v in weights]
        if j>=stride*e-4:
            # Precisely the present 15m/12m, two-second headway and five-second
            # closing requirements, from first signal vote through evacuation.
            for floor,time in ((15.,0.),(0.,2.),(0.,5.)):
                bound(front-floor,*sp,*[(k,time*v) for k,v in vm],(slack,1.))
            bound(-rear-12.,*[(k,-v) for k,v in sp],(slack,1.))
            bound(-rear-2*1.65,*[(k,-v) for k,v in sp],(slack,1.))
            bound(-rear-5*1.65,*[(k,-v) for k,v in sp],*[(k,-5*v) for k,v in vm],(slack,1.))
        # Preserve target traffic longitudinal noncontact; this optimistic
        # screen knows all future trajectories (no hidden rear acceleration).
        for role in ("adjacent-front","adjacent-rear"):
            if role not in row:continue
            B,F=row[role]
            if B>rows[start]["s"]+2.5:
                bound(B-3.5,*sp)
            elif F<rows[start]["s"]-2.5:
                bound(-F-3.5,*[(k,-v) for k,v in sp])
    bound(-12.48,(ps(n),-1.),(ps(e),1.))
    bound(-(1.+.82*3.5),(ps(n),-1.),(ps(e),1.),(pv(e),.82*2.7))
    def matrix(lst,count):
        i,j,v=zip(*lst)
        return coo_matrix((v,(i,j)),shape=(count,size)).tocsr()
    obj=np.zeros(size);obj[slack]=-1.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res=linprog(obj,A_ub=matrix(A,len(b)),b_ub=b,A_eq=matrix(E,len(d)),b_eq=d,
            bounds=[(None,None)]*(n+1)+[(0.,10.635)]*(n+1)+[(-80.,80.)],method="highs",
            options={"threads":1})
    if not res.success:return None
    return dict(slack=round(float(res.x[slack]),3),entry_frame=rows[start+stride*e]["frame"],finish_frame=rows[start+stride*n]["frame"],
      start_frame=rows[start]["frame"],duration_s=round(dt*(n-e),2),entry_s=round(float(res.x[ps(e)]),3),
      entry_v=round(float(res.x[pv(e)]),3),end_s=round(float(res.x[ps(n)]),3),end_v=round(float(res.x[pv(n)]),3),
      profile=[[rows[start+stride*i]["frame"],round(float(res.x[ps(i)]),3),round(float(res.x[pv(i)]),3)] for i in range(n+1)])


def main():
    cli=argparse.ArgumentParser();cli.add_argument("--output",type=Path,required=True);args=cli.parse_args()
    p=Path.home()/"AppData/Local/ADO/city-maneuvers"
    sources={"high":(p/"hero-captures/long-city-drive-9h8jac5g/raw/perception.jsonl",None),
             "heldout":(p/"timing-variation-registered01/trace.jsonl.gz",p/"timing-variation-registered01/reference-native.jsonl"),
             "nominal":(p/"full-qualification-pinned01/trace.jsonl.gz",p/"full-qualification-pinned01/reference-native.jsonl")}
    x=Path(".local/carla-1.0/CarlaUnreal/Packs/AdditionalMaps/Content/Maps/OpenDrive/Town03_Opt.xodr")
    m=carla.Map("Town03_Opt",x.read_text());c=next(c for c in discover(m,[w.transform.location for w in urban_route(m,212).points]) if c.source_key==(67,-2))
    summary={}
    for name,paths in sources.items():
        rows=extract(c,*paths)
        start=next(i for i,r in enumerate(rows) if r["progress"]>490. and r["phase"]=="change_left" and r["lead_speed"] is not None and .55<r["lead_speed"]<4.3)
        best=None;count=0;tot=0
        for e in range(18,75,2):
            for duration in range(10,42,3):
                n=e+duration
                if start+n*2>=len(rows):continue
                a=solve(rows,start,e,n)
                if a is not None:
                    tot+=1;count+=a['slack']>=0.
                    if best is None or a["slack"]>best["slack"]:best=a
        frames={r["frame"]:i for i,r in enumerate(rows)}
        base_entry=frames[best["entry_frame"]]-start
        base_end=frames[best["finish_frame"]]-start
        for e in range(base_entry-5,base_entry+6):
            for n in range(base_end-6,base_end+7):
                if n<=e or start+n>=len(rows):continue
                a=solve(rows,start,e,n,stride=1)
                if a is not None:
                    tot+=1;count+=a['slack']>=0.
                    if a["slack"]>best["slack"]:best=a
        summary[name]=dict(bound_profiles=tot,nonnegative=count,best=best)
        print(name,tot,count,{k:v for k,v in best.items() if k!='profile'},flush=True)
    args.output.write_text(json.dumps(dict(scope='Actor-omniscient upper-bound on original fixed-time trajectories; not a deployed policy, no future perception or lateral control guarantee. Initial state at actual moving-pass commitment; no target-lane cooperation assumed beyond recorded motion. Original sedan remains stationary in stringent transfer until physically observed moving. Sampled entry/finish grid in 0.4/0.6s increments plus a 0.1s local refinement around the strongest candidate. Profile fulfills five 10Hz acceptance gates before entry then all forward/rear limits through .82 of blend length plus 1m.',results=summary),indent=2)+'\n')
if __name__=='__main__':main()
