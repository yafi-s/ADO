"""Score frozen vehicle intent on sensor histories before matching native actor references."""
import gzip
import hashlib
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np

from crosswalk.stationary_intent import MODEL, StationaryIntent
from tools.audit_maneuver_queue_controls import front_reference


FOLDERS = (
    "speed/final-primary-queue", "speed/final-long-queue",
    "trials/candidate/departing", "trials/candidate/slow", "trials/candidate/blocked",
    "passing/road65-pinned-r2/clear", "passing/road65-pinned-r2/occupied", "passing/road65-pinned-r2/rebrake",
)
ROOT = Path("results/city-maneuvers")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def audit(folder, model):
    trace=folder/"trace.jsonl.gz"
    rows=list(map(json.loads,gzip.open(trace,"rt")))
    reference=folder/"reference.jsonl.gz"
    refs=list(map(json.loads,gzip.open(reference,"rt"))) if reference.is_file() else [None]*len(rows)
    # Original recordings predate the new camera association. Use only original
    # measured telemetry to compute the decision, then test correspondence
    # against independently withheld car geometry to score usable exposures.
    history=deque(maxlen=8)
    last=None
    start=0
    wait=0.
    groups={}
    for row,ref in zip(rows,refs,strict=True):
        t=row["tactical_pass"]
        frame=row["frame"]
        if (t.get("source_age_scans")!=0 or abs(t["source_lateral_m"])>=.55 or
                t.get("raw_source_gap_m") is None or t["raw_source_gap_m"]>=59.):
            continue
        gap=t["raw_source_gap_m"]
        world_station=t["corridor_station_m"]+gap
        if last is None or frame-last[0]>2 or abs(world_station-last[1])>2.:
            history.clear()
            start=frame
            wait=0.
        observed=.1*(frame-start+1)
        speed=t["source_speed_mps"]
        history.append((frame,speed))
        near=list(history)[-3:]
        earlier=list(history)[:-3] or near
        current=float(np.median([v for _,v in near]))
        previous=float(np.median([v for _,v in earlier]))
        elapsed=.1*(np.mean([f for f,_ in near])-np.mean([f for f,_ in earlier]))
        acceleration=(current-previous)/max(.22,elapsed)
        wait=(wait+.1 if speed<.5 and row["speed_mps"]<.5 and gap<=15. else 0.)
        answer,p_follow=model.predict((gap,row["speed_mps"],current,previous,acceleration,
                                       wait,observed,t["projected_lane_gain_m"]))
        last=(frame,world_station)
        if ref and ref["frame"]!=frame:
            raise ValueError("Reference and control frame mismatch")
        matches=[source for source in front_reference(row,ref) if abs(source[2]-(gap-2.5))<=2.]
        if len(matches)!=1:
            continue
        _,v,physical_gap=matches[0]
        labels=[]
        if v<.25:
            labels.append("stationary")
            if physical_gap<15. and row["speed_mps"]<.5:
                labels.append("settled_close")
        if .8<=v<=3.5:
            labels.append("moving_slow")
        if v>5.:
            labels.append("moving_above_5mps")
        for label in labels:
            group=groups.setdefault(label,Counter())
            group[answer]+=1
    return dict(folder=folder.as_posix(),seed=rows[0]["seed"],sha256=digest(trace),
                reference_sha256=digest(reference) if reference.is_file() else None,frames=len(rows),responses=groups)


def main():
    model=StationaryIntent()
    report=dict(schema="ado-stationary-intent-native-probe-v1",checkpoint_sha256=digest(MODEL),
        inference_source_sha256=digest(Path("crosswalk/stationary_intent.py")),
        method="Frame-level, retrospective counterfactual. Build the eight features from the archived source lane range/motion/gain history only, compute the intent action, then select fresh origin-lane measurements that match the withheld physical source actor within 2m. Camera/Doppler identity gate assumed for the matched subset. The historic actions did not use this head. Moving above 5m/s is not a departure label. The two queues share seed 316, the slow-traffic trials share seed 212, and road65 controls share seed 7401.",
        exclusions="No 9145 evaluation exposure was read by this evaluation script or used in fitting. No chronological actor observations became synthetic training samples.",
        captures=[audit(ROOT/f,model) for f in FOLDERS])
    path=ROOT/"passing/queue-preference-research/stationary-intent-native-probe.json"
    path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    for r in report["captures"]:
        print(r["folder"],r["responses"])


if __name__=="__main__":
    main()
