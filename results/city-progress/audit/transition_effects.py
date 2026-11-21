"""Compare the same road stations around candidate request withdrawals; association only."""
import argparse
import json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_bytes())

def trace(folder):
    path=folder/"raw/telemetry.json" if (folder/"capture.json").exists() else folder/"telemetry.json"
    rr=read(path)
    v=np.array([r["speed_mps"] for r in rr])
    return dict(seconds=np.array([r["sim_time"] for r in rr]),s=np.array([r["progress_m"] for r in rr]),
                v=v*3.6,a=np.r_[np.full(5,np.nan), (v[5:]-v[:-5])/.5],
                brake=np.array([r["brake"] for r in rr]),throttle=np.array([r["throttle"] for r in rr]))

def measure(base, new, audited, old):
    o,c=trace(base),trace(new)
    drops=[d for d in audited["abrupt_request_drops"]["entries"] if d['returned_to_inherited']]
    def metric(x,m):
        return dict(seconds=round(.1*int(m.sum()),2),sub15_s=round(.1*int((m&(x["v"]<15)).sum()),2),
                    pedal_brake_s=round(.1*int((m&(x['brake']>.05)).sum()),2),
                    half_s_hard_braking_s=round(.1*int((m&(x['a'] < -3)).sum()),2))
    def compare(mo,mc):
        a,b=metric(o,mo),metric(c,mc)
        return dict(v05=a,v06=b,delta={k:round(b[k]-a[k],3) for k in a})
    # For fixed candidate forward time intervals, project the endpoints onto the
    # baseline station axis. A contact/release may happen at a different time.
    whole={}
    road_windows=[]
    for horizon in (1.5,5):
        mo=np.zeros(len(o["s"]),bool);mc=np.zeros(len(c["s"]),bool)
        detail=[]
        for d in drops:
            i=d["index"];end=min(i+round(horizon*10),len(c["s"]))
            lo=c['s'][i]
            hi=c['s'][end] if end<len(c['s']) else c['s'][-1]+.001
            old_mask=(o['s']>=lo)&(o['s']<hi)
            fresh=np.zeros(len(c['s']),bool);fresh[i:end]=True
            mo|=old_mask;mc|=fresh
            detail.append(dict(frame=d["frame"],from_m=lo,to_m=hi,binding=d["new_binding"],
                               drop_kmh=d["drop_kmh"],max_applied_brake=round(float(np.max(c["brake"][i:end])),3),
                               peak_decel_mps2=d['after_1_5s_peak_decel_mps2'],**compare(old_mask,fresh)))
        whole[str(horizon)]=dict(**compare(mo,mc),per_retraction=detail,
                                brake_applied_after_switch_count=sum(x["max_applied_brake"]>.05 for x in detail),
                                slower_fixed_station_windows=sum(x["delta"]["seconds"]>.099 for x in detail),
                                faster_fixed_station_windows=sum(x["delta"]["seconds"]<-.099 for x in detail))
    for a,b in zip(old["station_bins"],audited["station_bins"],strict=True):
        lo,hi=a['from_m'],a['to_m']
        hits=[d["frame"] for d in drops if lo<=d["station_m"]<hi]
        road_windows.append(dict(from_m=lo,to_m=hi,v05_s=a['seconds'],v06_s=b['seconds'],
                                 time_saved_s=round(a['seconds']-b['seconds'],2),retraction_frames=hits,
                                 sub15_delta_s=round(b['below_kmh_s']['15']-a['below_kmh_s']['15'],2)))
    def totals(r):
        gains=sum(max(0,w['time_saved_s']) for w in r);loss=sum(max(0,-w['time_saved_s']) for w in r)
        return dict(bins=len(r),faster_bins=sum(w['time_saved_s']>.001 for w in r),slower_bins=sum(w['time_saved_s']<-.001 for w in r),
                    same_bins=sum(abs(w['time_saved_s'])<=.001 for w in r),gross_time_saved_s=round(gains,2),
                    slower_bin_offset_s=round(loss,2),net_time_saved_s=round(gains-loss,2),
                    slower_bin_offset_share_of_gross_pct=round(100*loss/gains,2) if gains else None)
    # Further slice the same post-switch physical locations by the locked primary windows.
    local={}
    for label,regions in {"physical_bus":[(858,872)],"bus_recovery":[(872,882)],
                          "six_near_straights":[(j["from_m"],j["to_m"]) for j in old["junctions"] if j["id"] in old["near_straight_junctions"]["junction_ids"]]}.items():
        wanted=lambda x:np.logical_or.reduce([(x['s']>=lo)&(x['s']<hi) for lo,hi in regions])
        linked=lambda x:np.logical_or.reduce([(x['s']>=w['from_m'])&(x['s']<w['to_m']) for w in whole["5"]["per_retraction"]]) if drops else np.zeros(len(x["s"]),bool)
        local[label]=dict(full=compare(wanted(o),wanted(c)),within_next_5s_station_overlap=compare(wanted(o)&linked(o),wanted(c)&linked(c)))
    return dict(note="Candidate 1.5s and 5s forward windows after each >=5km/h one-tick withdrawal from above the inherited request back to it. Matched baseline measurements use the candidate window's spatial endpoints. Unions deduplicate overlapping windows. Later events and differing traffic timing can affect both arms. Gross savings/offsets use all fixed 25m bins; they cannot estimate a hypothetical drive without safety guards.",
                direct_retraction_count=len(drops),following_windows=whole,fixed_bins=road_windows,all_bins=totals(road_windows),
                bins_containing_retraction=totals([w for w in road_windows if w['retraction_frames']]),primary_regions=local)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('new',type=Path);p.add_argument('audit',type=Path)
    p.add_argument('old',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(measure(a.base,a.new,read(a.audit),read(a.old)),indent=2)+'\n',encoding='utf8')
