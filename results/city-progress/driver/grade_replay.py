"""Replay the candidate's grade-only decision on retained native R3 model inputs and actual range packets."""
import argparse
from collections import deque
import gzip
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from crosswalk import behavior_v2 as v2
from crosswalk.city_drive import load_driver
from crosswalk.city_grade import supported_grade
from crosswalk.experiment import fingerprint


def compare(folder, checkpoint):
    with gzip.open(folder/"trace.jsonl.gz","rt") as stream:
        rows = [json.loads(line) for line in stream]
    with np.load(folder/"observations.npz",allow_pickle=False) as array:
        features = array["features"]
        names = list(array["feature_names"])
    baseline, _, _ = load_driver(folder.parent/"candidate.npz")
    candidate, _, _ = load_driver(checkpoint)
    baseline.new_observer()
    candidate.new_observer()
    history = deque(maxlen=9)
    changes = []
    flagged = 0
    count = 0
    prior = None
    with zipfile.ZipFile(folder/"sensor-sweeps.zip") as archive:
        prefixes = {int(name.split("/")[0].split("-")[-1]):name.split("/")[0]
                    for name in archive.namelist() if name.endswith("/lidar.npy")}
        for row, x in zip(rows,features):
            frame = row["frame"]
            with_sensor = frame in prefixes
            measured = 0.
            distance = min(x[names.index("path_body_lidar")],x[names.index("path_body_radar")])
            if with_sensor:
                count += 1
                if x[names.index("confirmed_scans")] > 0 and distance < 32.:
                    prefix = prefixes[frame]
                    scan = [np.load(io.BytesIO(archive.read(f"{prefix}/{name}.npy"))) for name in ("lidar","radar")]
                    # Native fixture follows a straight route. The measured 3/7/13 m
                    # navigation is retained. Extend its final tangent for this replay.
                    measured = supported_grade(*scan,v2._checked_route(x[:6],None),float(x[6]))
            else:
                history.clear()
            history.append((measured,float(distance)))
            prior3 = list(history)[-3:]
            permit = (min(g for g,_ in prior3) if len(history)==9 and all(g>0 or d>=24 for g,d in history) else 0.)
            for model in (baseline,candidate):
                model.posted_limit_kmh=row["posted_limit_kmh"]
                model.signed_speed_limit=row.get("signed_approach_cap_mps")
                model.tactical_speed_limit=row.get("tactical_pass",{}).get("tactical_speed_limit_mps")
            baseline.predict(x)
            candidate.predict(np.r_[x[:-2],np.array([measured,permit],dtype=x.dtype),x[-2:]])
            delta = candidate.last_speed_request-baseline.last_speed_request
            tick = dict(frame=frame,time_s=row["sim_time"],raw_lidar_m=float(x[names.index("path_body_lidar")]),
                        raw_radar_m=float(x[names.index("path_body_radar")]),grade_extent_m=measured,
                        confirmed_grade_extent_m=permit,old_request_kmh=3.6*baseline.last_speed_request,
                        new_request_kmh=3.6*candidate.last_speed_request)
            if frame==520 and folder.name=="candidate-clear":
                prior=tick
            if permit>0 and candidate.progress_evidence["grade_added_request_mps"]>0:
                flagged+=1
            if abs(delta)>.02:
                changes.append(tick)
    return dict(episode=folder.as_posix(),sensor_sha256=fingerprint(folder/"sensor-sweeps.zip"),frames=len(rows),
                retained_ranges=count,direct_grade_additions=flagged,largest_added_mps=max(
                    ((r["new_request_kmh"]-r["old_request_kmh"])/3.6 for r in changes),default=0.),
                frame_520=prior,changed_requests=changes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--episodes",type=Path,nargs="+",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    results=dict(schema="grade-proposal-recorded-trace-v1",checkpoint_sha256=fingerprint(a.model),
                 interpretation="Identical native R3 observations and geometry. Changes are requested speed at recorded R3 poses; achieved speed and timing require separate driving trials.",
                 comparisons=[compare(folder,a.model) for folder in a.episodes])
    a.out.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
    print([(c["episode"],c["direct_grade_additions"],len(c["changed_requests"]),c["frame_520"]) for c in results["comparisons"]])


if __name__=="__main__":
    main()
