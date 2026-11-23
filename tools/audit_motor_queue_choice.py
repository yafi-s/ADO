"""Compare stationary-vehicle source choices with archived controller inputs."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from tools.maneuver_motor_queue_candidate import MotorQueue

ROOT=Path(__file__).resolve().parents[1]


def run(args):
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    control=Path.home()/"AppData/Local/ADO/city-maneuvers/market-loop-flow-9145/prefix-continuous"
    data=ROOT/"results/city-maneuvers/passing/source-vehicle-association/9145-measured.json"
    evidence=json.loads(data.read_bytes())
    rows=[json.loads(l) for l in gzip.open(control/"trace.jsonl.gz","rt")]
    donors={r['frame']:r.get('source_vehicle') for r in evidence['prefix']}
    intent=None
    if args.model:
        spec=importlib.util.spec_from_file_location('crosswalk.queue_research_head',args.source)
        code=importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name]=code
        spec.loader.exec_module(code)
        intent=code.StationaryIntent(args.model)
    q=MotorQueue(); decisions=[]; first=None
    for r in rows:
        f=r['frame']; t=r['tactical_pass']
        if not 12208<=f<=12272:continue
        v=None if t['source_speed_mps']==20. else t['source_speed_mps']
        q.update(f,t['corridor_station_m'],t['source_gap_m'],t['raw_source_gap_m'],v,
                 t['source_age_scans'],r['speed_mps'],donors.get(f))
        feat=q.preference_features(f,t['source_gap_m'],r['speed_mps'],v,t['projected_lane_gain_m'])
        q.learned_preference(intent,feat)
        if q.anchor is not None and first is None:first=f
        accepted=q.choose(t['fitted_choice'])
        decisions.append(dict(frame=f,station_m=r['progress_m'],recorded_fitted=t['fitted_choice'],
                         recorded_phase=t['maneuver_phase'],candidate_preference=q.preference,
                         prob_follow=q.probability_follow,learned_pass_gain_m=q.intent_gain_m,leaf_index=q.intent_leaf,features=feat,raw_fitted_choice_after_guard=accepted,
                         source_mode=q.status,witness=donors.get(f)))
    departure=[r for r in rows if 12340<=r['frame']<=12472]
    faster=[r for r in departure if r['tactical_pass']['source_age_scans']==0
        and r['tactical_pass']['source_speed_mps']>max(4.3,r['speed_mps']+.65)
        and r['tactical_pass']['source_gap_m']>=12.]
    candidates=[r for r in faster if r['tactical_pass']['return_gap_accepted']]
    report=dict(scope='Causal replay through the first actual outbound commitment. Source-vehicle association uses only independently matched native LiDAR/radar and reconstructed 5 Hz image detections; missing sweeps provide no new witness. Later motion/clock event outcomes cannot be inferred from original poses.',
        head='engineering comparator only' if intent is None else 'fitted stationary-intent head',
        native_source=sha(control/'trace.jsonl.gz'),association_sha256=sha(data),
        intent_sha256=sha(args.model) if args.model else None,first_confirmed=first,
        first_signal=next(r for r in decisions if r['frame']==12242),first_commit=decisions[-1],
        after_confirmation_actions=sorted({r['raw_fitted_choice_after_guard'] for r in decisions if r['frame']>=first}),
        decisions=decisions,departing_diagnosis=dict(
            faster_forward_surface_frames=len(faster),earliest_faster_forward_surface=faster[0]['frame'] if faster else None,
            accepted_right_gaps_at_these_frames=len(candidates),
            all_return_gap_permitted_frames=sum(bool(r['tactical_pass']['return_gap_accepted']) for r in rows
                if 12272<=r['frame']<=12472 and r['tactical_pass']['source_lateral_m']<-2.8)))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(report['first_confirmed'],report['first_commit']['raw_fitted_choice_after_guard'],report['departing_diagnosis'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path);p.add_argument('--source',type=Path);run(p.parse_args())
