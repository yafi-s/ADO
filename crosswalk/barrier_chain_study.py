"""Re-evaluate the connected-chain detector on archived native sensor exposures."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from .barrier import detect
from .barrier_study import read_rgb
from .surround_vision import SurroundDetector
from .tracking import Tracker


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def witness(row, cloud):
    """Scorer-only hit counts within the separately placed posts and suspended-span corridor."""
    if row["condition"] != "chain":
        return None
    ends = np.array([p["origin_xyz_m"] for p in row["scorer_bodies"]])
    axis = ends[1, :2] - ends[0, :2]
    delta = cloud[:, :2] - ends[0, :2]
    fraction = delta @ axis / (axis @ axis)
    cross_distance = np.linalg.norm(delta - fraction[:, None]*axis, axis=1)
    z = cloud[:, 2] + 2.30 - ends[0, 2]
    above = (z > .20) & (z < .86)
    posts = [int(np.count_nonzero(above & (np.linalg.norm(cloud[:, :2] - pos[:2],axis=1) < .17)))
             for pos in ends]
    middle = above & (fraction >= .17) & (fraction <= .83) & (cross_distance < .115)
    return {"posts": posts, "interior_corridor_hits": int(sum(middle))}


def evaluate(archive_path):
    detector = SurroundDetector()
    tally, rows, seen = defaultdict(Counter), [], defaultdict(set)
    totals = Counter()
    with ZipFile(archive_path) as archive:
        for row in json.loads(archive.read("manifest.json")):
            n, case = row["name"], row["condition"]
            target = row["target_xy_m"]
            rgb = read_rgb(archive.read(n+".jpg"))
            with np.load(io.BytesIO(archive.read(n+".npz"))) as packet:
                cloud = packet["lidar"].astype(np.float32)
            detections = detector.detect(rgb) + detect(rgb, cloud, "front", verify_posts=False)
            tracks = Tracker(surround=True).update(cloud,np.empty((0,4)),{"front":detections},(0.,0.,0.),0.)
            expected = {"chain": "chain", "barricade": "striped_board"}.get(case)
            predicted = []
            matched = False
            for t in tracks:
                if t["label"] != "barrier" or not t["cameras"]:
                    continue
                err = float(np.hypot(t["x"]-target[0], t["y"]-target[1])) if target else None
                correct = bool(expected and expected == t.get("barrier_subtype") and err is not None and err <= .65
                               and not matched)
                predicted.append(dict(subtype=t.get("barrier_subtype"), score=round(t["subtype_confidence"],4),
                    center_error_m=None if err is None else round(err,4), laser_hits=t["lidar_hits"],
                    chain_support=t.get("chain_support"), matched=correct))
                matched |= correct
                tally[case]["fresh_predictions"] += 1
                if not correct:
                    tally[case]["unmatched_predictions"] += 1
                    totals["unmatched_predictions"] += 1
            tally[case]["exposures"] += 1
            if expected:
                tally[case]["matched"] += int(matched)
                tally[case]["missed"] += int(not matched)
                if matched:
                    seen[case].add((row["road"],row["station"],row["distance_m"]))
            rows.append(dict(name=n, condition=case, target_subtype=expected, measurements=predicted,
                             reference_range=witness(row,cloud)))
    for case, counts in tally.items():
        for field in ("fresh_predictions", "unmatched_predictions", "matched", "missed"):
            counts.setdefault(field, 0)
        counts["distinct_matched_site_distance_pairs"] = len(seen[case])
    source = Path(__file__).with_name("barrier_chain.py")
    canonical = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return dict(schema="ado-chain-recognition-v1",archive_sha256=checksum(archive_path),
                detector_source_sha256=checksum(source), detector_source_lf_sha256=canonical,
                exposures=len(rows),by_condition={k:dict(v) for k,v in tally.items()},
                unmatched_predictions=totals["unmatched_predictions"],frames=rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    cv2.setNumThreads(2)
    report=evaluate(a.archive)
    a.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report["by_condition"],indent=2))
