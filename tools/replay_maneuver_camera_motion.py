"""CPU longitudinal sensitivity on recorded observations; no new sensor viewpoint."""
import argparse
import gzip
import json
from collections import deque
from pathlib import Path
import numpy as np

from crosswalk.city_maneuvers import ManeuverModel
from tools.check_maneuver_camera_predicate import build


DT=.1


def tape(path):
    with gzip.open(path,"rt") as stream:
        return list(map(json.loads,stream))


def state(v,acc,gas,brake,g1,b1,g2):
    # Native transitions match speed, current pedal and its recent changes.
    return np.array([v,acc,gas,brake,g1,b1,g2])/np.array([.8,1.3,.18,.20,.21,.25,.29])


class NativeResponse:
    def __init__(self,paths):
        inputs=[];labels=[];braking=[]
        for p in paths:
            rows=tape(p)
            for i in range(2,len(rows)-1):
                prev,r,nxt=rows[i-1:i+2]
                if nxt["frame"]!=r["frame"]+1 or r["frame"]!=prev["frame"]+1:continue
                v=r["speed_mps"]
                if v>7.:continue
                if v<.015 and r["throttle"]<.02 and i%16:continue
                if .35<v<4. and nxt["speed_mps"]>.09 and r["brake"]>.15 and r["throttle"]==0.:
                    # Fit the rolling brake response before the floor at zero
                    # censors the observed deceleration.
                    braking.append((v-nxt["speed_mps"])/DT/r["brake"])
                a=(v-prev["speed_mps"])/DT
                inputs.append(state(v,a,r["throttle"],r["brake"],prev["throttle"],prev["brake"],rows[i-2]["throttle"]))
                labels.append((nxt["speed_mps"]-v)/DT)
        self.x=np.asarray(inputs);self.y=np.asarray(labels)
        self.brake_gain=float(np.median(braking)); self.brake_quantiles=np.quantile(braking,[.1,.5,.9]).tolist()

    def accel(self,v,acc,gas,brake,g1,b1,g2):
        if gas<=.005 and brake>.15:
            return -min(v/DT,self.brake_gain*brake)
        if v<.01 and gas<.012:return -v/DT
        dx=self.x-state(v,acc,gas,brake,g1,b1,g2)
        ds=(dx*dx).sum(axis=1)
        near=np.argpartition(ds,12)[:12]
        w=1./(.025+ds[near])**1.5
        return float(np.sum(w*self.y[near])/sum(w))


def replay(folder,response,correct,debit,anchored):
    rows=tape(folder/"trace.jsonl.gz");obs=np.load(folder/"observations.npz");names=list(map(str,obs["feature_names"]))
    corrected=build();m=ManeuverModel.load(folder/"checkpoint.npz");m.new_observer()
    start=next(i for i,r in enumerate(rows) if r["progress_m"]>=581.5)
    s=rows[start]["progress_m"];v=rows[start]["speed_mps"];a=(v-rows[start-1]["speed_mps"])/DT
    g1=rows[start-1]["throttle"];b1=rows[start-1]["brake"];g2=rows[start-2]["throttle"]
    initial=s; maxdv=[];out=[];unresolved=deque(maxlen=9)
    simulated_speeds=deque([rows[j]["speed_mps"] for j in range(start-8,start)],maxlen=8)
    for i,(r,observed) in enumerate(zip(rows,obs["features"],strict=True)):
        if i==len(rows)-1:break
        t=dict(r['tactical_pass']);old=t.get('free_source_candidate',{}); cap=t.get('tactical_speed_limit_mps')
        f=dict(zip(names,observed)); x=observed.copy()
        c=f["camera_stop_horizon_m"];count=f["off_path_camera_boxes"];cur=f["rgb_person_height"];recent=f["rgb_person_height_recent"]
        unresolved.append(bool(cur>0. and not(cur>.055 and count>=1. and m.standoff+1.<c<59.)))
        m.posted_limit_kmh=r["posted_limit_kmh"];m.signed_speed_limit=r.get("signed_approach_cap_mps");m.tactical_speed_limit=cap
        if i>=start:
            f["ego_speed_mps"]=v;f["ego_acceleration"]=float(np.clip((v-simulated_speeds[0])/.8,-12.,8.));simulated_speeds.append(v)
            advance=max(0.,s-r["progress_m"]) if debit else 0.
            if advance:
                # A new pose cannot see farther on the recorded rays. Charge
                # extra travel against each already observed forward budget.
                for name in ("verified_forward_m","observed_corridor_m","path_body_radar","path_body_lidar",
                             "path_body_radar_recent","path_body_lidar_recent","long_body_gap_m","junction_gap"):
                    f[name]=max(0.,f[name]-advance)
                if c<59.:f["camera_stop_horizon_m"]=max(0.,c-advance)
                for k in ("source_gap_m","raw_source_gap_m","target_front_m"):
                    if t.get(k,60.)<60.:t[k]=max(0.,t[k]-advance)
                if "pavement_reach_m" in t:
                    t["pavement_reach_m"]=[max(0.,h-advance) if j in (0,2) else h for j,h in enumerate(t["pavement_reach_m"])]
                # Preserve additional side/flank sight only up to the closest
                # existing longitudinal evidence; do not extrapolate a side car.
                for k in ("verified_cross_sight_m","observed_conflict_m"):
                    f[k]=max(0.,f[k]-advance)
            for k in f:x[names.index(k)]=f[k]
            if correct and old and old.get('original_cap_mps')==0.:
                m.tactical_speed_limit=0.
                # Recheck the full local allowance, using only the corrected
                # optical predicate. Recorded parallel-flank exemptions may
                # not be extended to a different pose.
                if (cur==0. and recent==0. and count>=1. and m.standoff+1.<c<59.) or (debit and cap is not None and cap>0.):
                    cand,_=corrected(m,t,f)
                    if cand is not None:cap=cand
                if c==0. or (recent>0. and any(unresolved)):cap=0.
            m.tactical_speed_limit=cap
        control=m.predict(x)
        if i<start:continue
        gas,brake=float(control[1]),float(control[2])
        acc=response.accel(v,a,gas,brake,g1,b1,g2)
        if anchored:
            obs_v=r["speed_mps"];obs_a=(obs_v-rows[i-1]["speed_mps"])/DT
            native=response.accel(obs_v,obs_a,r["throttle"],r["brake"],rows[i-1]["throttle"],rows[i-1]["brake"],rows[i-2]["throttle"])
            acc+=(rows[i+1]["speed_mps"]-obs_v)/DT-native
        if gas<=.005 and brake>.15:acc=min(0.,acc)
        next_v=max(0.,v+DT*acc);effective_a=(next_v-v)/DT
        s+=.5*(v+next_v)*DT
        out.append((s,next_v,effective_a,abs(effective_a-a)/DT,m.last_speed_request,gas,brake))
        maxdv.append(abs(next_v-rows[i+1]["speed_mps"]))
        a,v=effective_a,next_v;g2,g1,b1=g1,gas,brake
    y=np.asarray(out);stopped=y[:,1]<.05
    spans=[];streak=0
    for flag in stopped:
        if flag:streak+=1
        else:spans.append(streak);streak=0
    spans.append(streak)
    # This local range is measured at the actual terminal pose. Absolute
    # route travel to the scripted source encounter is a scoring comparison.
    threshold=rows[-1]['progress_m']+24.913894-.0-20.
    meeting=np.flatnonzero(y[:,0]>=threshold)
    return dict(start_frame=rows[start]['frame'],frames=len(out),start_station_m=initial,end_station_m=round(s,3),
        travel_m=round(s-initial,3),actual_travel_m=round(rows[-1]['progress_m']-initial,3),
        terminal_extra_vs_recorded_m=round(s-rows[-1]['progress_m'],3),first_possible_20m_projection_s=round((meeting[0]+1)*DT,1) if len(meeting) else None,
        peak_mps=round(float(y[:,1].max()),3),median_mps=round(float(np.median(y[:,1])),3),
        longest_stationary_s=round(max(spans)*DT,1),peak_accel_mps2=round(float(y[:,2].max()),3),peak_brake_mps2=round(float(y[:,2].min()),3),
        p95_abs_jerk_mps3=round(float(np.quantile(y[:,3],.95)),3),mean_abs_recorded_speed_difference_mps=round(float(np.mean(maxdv)),3))


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--folder',type=Path,required=True);cli.add_argument('--output',type=Path,required=True);args=cli.parse_args()
    root=Path('results/city-maneuvers')
    train=list((root/'passing/road65-pinned-r2').glob('*/trace.jsonl.gz'))+[root/'qualification/full-market/trace.jsonl.gz',root/'pace/native-v07-boulevard/boulevard/trace.jsonl.gz',root/'pace/native-v06-boulevard/boulevard/trace.jsonl.gz']
    response=NativeResponse(train)
    result=dict(scope='CPU longitudinal response only, matched held-out native controls for validation. Changing speeds and acceleration feed the unmodified parent controller each tick. Camera, occupancy and other driver positions do not re-render or react to a new pose. A possible crossing of the 20m station is not a physical encounter. The debit arm consumes extra travel from the recorded road, optical, body and flank view; the unadjusted arm repeatedly receives observations made at the actual mostly stationary pose.',fit=dict(native_transitions=len(response.y),uncensored_brake_gain_p10_p50_p90=response.brake_quantiles,input_files=[str(p) for p in train],method='12 nearest speed, acceleration and three consecutive pedal states; weighted mean measured next acceleration, 10 Hz trapezoidal odometry; jerk from consecutive accelerations. Training excludes this trial.'),runs={})
    for tag,correct,debit,anchored in [('unchanged_held_out',False,False,False),('anchored_unchanged_check',False,False,True),('corrected_optimistic',True,False,False),('corrected_debited',True,True,False),('corrected_debited_anchored',True,True,True)]:
        result['runs'][tag]=replay(args.folder,response,correct,debit,anchored)
        print(tag,result['runs'][tag],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
