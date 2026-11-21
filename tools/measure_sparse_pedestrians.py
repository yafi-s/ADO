"""Evaluate supplemental sparse-depth readings on a native camera / range tape."""
import argparse
from collections import Counter
from itertools import islice
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosswalk.sparse_pedestrian_study import SparsePedestrianStudy


def measure(raw, sweeps, limit=None):
    cv2.setNumThreads(1)
    observers = {mode: SparsePedestrianStudy() for mode in ("registered", "radar_azimuth_plus_12deg")}
    readings = {mode: [] for mode in observers}
    counts = Counter()
    with (raw / "perception.jsonl").open() as journal:
        for i,line in enumerate(islice(journal,limit)):
            f = json.loads(line)
            boxes = f["camera_detections"].get("front")
            counts["frames"] += 1
            if boxes is None:
                continue
            counts["front_acquisitions"] += 1
            counts["eligible_boxes"] += sum(b["label"] == "person" and b["score"] >= .60
                                            and b["range_supported"] is False for b in boxes)
            if not any(b["label"] == "person" and b["range_supported"] is False for b in boxes):
                continue
            with np.load(sweeps / f"{f['frame']}.npz") as r:
                scan, radar, pose = r["lidar"], r["radar"], r["ego_xy_yaw"]
            for mode, observer in observers.items():
                samples = radar
                if mode != "registered":
                    samples = radar.copy()
                    samples[:, 1] += math.radians(12)
                proposed = observer.observe(boxes, scan, samples, pose, i*.1)
                # Read actor identities only after independent sensors have emitted their answers.
                actors = [a for a in f["observer_map"]["others"] if a["type"] == "pedestrian"]
                pairs = sorted((math.dist(p["world_xy"], [a["x"],a["y"]]), j, k)
                               for j,p in enumerate(proposed) for k,a in enumerate(actors))
                used, matches = set(), {}
                for d,j,k in pairs:
                    if j not in matches and k not in used and d<=1.5:
                        matches[j]=dict(actor_id=actors[k]["actor_id"],role=actors[k]["role"],error_m=round(d,3))
                        used.add(k)
                for j,p in enumerate(proposed):
                    readings[mode].append(dict(frame=f["frame"],time_s=f["sim_time"],**p,match=matches.get(j)))
    return dict(schema=1,gate_m=1.5,counts=counts,readings=readings)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("raw","sweeps","output"):
        p.add_argument(name,type=Path)
    p.add_argument("--limit",type=int)
    args=p.parse_args()
    args.output.write_text(json.dumps(measure(args.raw,args.sweeps,args.limit),indent=2)+"\n")
