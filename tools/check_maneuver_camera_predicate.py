"""One fixed-motion sensitivity test for the optional camera cross-check."""
import argparse
from collections import Counter,deque
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import numpy as np
from crosswalk.city_maneuvers import ManeuverModel
from tools import maneuver_free_source_candidate as frozen


def build():
    text=textwrap.dedent(inspect.getsource(frozen.FreeSource.allowance))
    before='and max(x["rgb_person_height"],x["rgb_person_height_recent"])>0.)'
    if text.count(before)!=1:
        raise ValueError("Unexpected frozen bounded-person predicate")
    changed=text.replace(before,')')
    scope=dict(vars(frozen))
    exec(compile(changed,"<one bounded-person predicate removed>","exec"),scope)
    return scope["allowance"]


def window(rows):
    def streak(condition):
        groups=[]
        for r in rows:
            if condition(r):
                if not groups or groups[-1][-1]["frame"]!=r["frame"]-1:
                    groups.append([])
                groups[-1].append(r)
        if not groups:return dict(frames=0)
        g=max(groups,key=len)
        return dict(frames=len(g),first_frame=g[0]["frame"],last_frame=g[-1]["frame"])
    return dict(frames=len(rows),positive_before=sum(r["old_request"]>0 for r in rows),
        positive_literal=sum(r["literal_request"]>0 for r in rows), positive_after=sum(r["new_request"]>0 for r in rows),
        positive_throttle_before=sum(r["old_throttle"]>0 for r in rows),
        positive_throttle_after=sum(r["new_throttle"]>0 for r in rows),
        longest_zero_before=streak(lambda r:r["old_request"]<=0.),
        longest_zero_literal=streak(lambda r:r["literal_request"]<=0.),
        longest_zero_after=streak(lambda r:r["new_request"]<=0.),
        changed_gate_frames=sum(r["admitted"] for r in rows),
        changed_quantiles=[round(float(x),5) for x in np.quantile(
            [r["new_request"] for r in rows if r["admitted"]],[0,.5,1])] if any(r["admitted"] for r in rows) else [],
        zero_reason_counts=dict(Counter(r["reason"] for r in rows if r["new_request"]<=0.)),
        last=rows[-1] if rows else None)


def run(folder,output):
    modified=build()
    ref=ManeuverModel.load(folder/"checkpoint.npz");alt=ManeuverModel.load(folder/"checkpoint.npz")
    ref.new_observer();alt.new_observer();literal=ManeuverModel.load(folder/"checkpoint.npz");literal.new_observer()
    captured=list(map(json.loads,gzip.open(folder/"trace.jsonl.gz","rt")));ds=np.load(folder/"observations.npz")
    assert [r['frame'] for r in captured]==list(ds['frame'])
    rows=[];maxerr=0.;legacy_history=deque(maxlen=9)
    samples=None
    for rec,x in zip(captured,ds['features'],strict=True):
        a=rec['tactical_pass'];f=dict(zip(ref.feature_names,x));fr=rec['frame']
        info=a.get('free_source_candidate',{}); cap=a.get('tactical_speed_limit_mps')
        c=f['camera_stop_horizon_m'];n=f['off_path_camera_boxes'];cur=f['rgb_person_height'];recent=f['rgb_person_height_recent']
        finite=n>=1 and ref.standoff+1.<c<59.
        # A current central scalar can only be attributed if it meets the
        # same height cutoff as every box verified by camera_horizon.
        legacy_history.append(bool(cur>0. and not (cur>.055 and finite)))
        reason=info.get('reason','ordinary');newcap=cap;admitted=False
        for m in (ref,alt,literal):
            m.posted_limit_kmh=rec['posted_limit_kmh']
            m.signed_speed_limit=rec.get('signed_approach_cap_mps')
            m.tactical_speed_limit=cap
        if (reason=='perception_hold' and info.get('original_cap_mps')==0.
                and finite and cur==0. and recent==0. and not any(legacy_history)):
            alt.tactical_speed_limit=0.
            proposed,gate=modified(alt,dict(a),f)
            if proposed is not None and proposed>0.:
                newcap,reason,admitted=proposed,'geometry_only_off_path',True
                if samples is None:samples=(dict(a),dict(f))
            else:reason=gate
        literal_cap=newcap
        if info and c==0.:
            newcap,reason=0.,'zero_optical_horizon'
        elif info and recent>0. and any(legacy_history):
            newcap,reason=0.,'unresolved_legacy_history'
        literal.tactical_speed_limit=literal_cap;literal.predict(x)
        ref_control=ref.predict(x);alt.tactical_speed_limit=newcap;alt_control=alt.predict(x)
        maxerr=max(maxerr,abs(ref.last_speed_request-rec['speed_request_mps']))
        rows.append(dict(frame=fr,station=rec['progress_m'],speed=rec['speed_mps'],admitted=admitted,reason=reason,
                         old_request=float(ref.last_speed_request),literal_request=float(literal.last_speed_request),new_request=float(alt.last_speed_request),
                         old_throttle=float(ref_control[1]),new_throttle=float(alt_control[1]),
                         old_cap=cap,new_cap=newcap))
    assert maxerr<.0001
    held=[r for r in rows if r['station']>=581.5 and r['speed']<.1]
    tail=rows[-150:]
    # Independently challenge the actual allowance on a newly eligible frame.
    a,f=samples;alt.tactical_speed_limit=0.
    tests={}
    for name,change,changea in [
        ('finite_offpath_zero_legacy',{},{}),
        ('missing_box',{'off_path_camera_boxes':0},{}),
        ('zero_horizon',{'camera_stop_horizon_m':0},{}),
        ('recent_uncategorized',{'camera_stop_horizon_m':60.,'off_path_camera_boxes':0,'rgb_person_height_recent':.08},{}),
        ('crossing',{'crossing_stop_gap_m':3.5},{}),('red',{'rgb_lamp_red_probability':.9},{}),
        ('near_body',{'path_body_radar':3.8},{}),('straddling',{}, {'source_lateral_m':-1.35}),
        ('flank_intrusion',{}, {'return_swept_intrusion':True}),('stale_geometry',{'geometry_scans':1},{}),
        ('no_source_pavement',{}, {'source_approach_pavement':False})]:
        rate,why=modified(alt,dict(a,**changea),dict(f,**change))
        tests[name]=dict(cap=rate,reason=why)
    assert tests['finite_offpath_zero_legacy']['cap']>0 and all(v['cap']==0 for k,v in tests.items() if k!='finite_offpath_zero_legacy')
    result=dict(scope='Fixed native features of the rejected parallel-flank drive, including original recorded lateral decisions and all recorded caps. This comparison admits only previously denied finite geometric off-path boxes with BOTH scalar person summaries zero. Existing recent unattributed legacy evidence is conservatively withheld. No steering, physical translation, or native release is established.',max_stock_replay_error_mps=round(maxerr,8),source_trace_sha256=hashlib.sha256((folder/'trace.jsonl.gz').read_bytes()).hexdigest(),source_approach_sha256=frozen.sha(Path(frozen.__file__)),
                held_window=window(held),last_fifteen_seconds=window(tail),controls=tests)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print('max replay error',maxerr)
    for k in ('held_window','last_fifteen_seconds'):
        print(k,{n:v for n,v in result[k].items() if n!='last'})
    print('last',rows[-1])
if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--folder',type=Path,required=True);cli.add_argument('--output',type=Path,required=True);a=cli.parse_args();run(a.folder,a.output)
