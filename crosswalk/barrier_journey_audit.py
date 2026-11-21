"""Score a completed, continuous Town03 perception journal against its independent board poses."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def evaluate(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    actors, reference, predictions = defaultdict(list), Counter(), Counter()
    unmatched, examples = [], []
    digest = hashlib.sha256()
    count = 0
    subtype_counts = Counter()
    with opener(path, "rb") as stream:
        for line in stream:
            row = json.loads(line)
            digest.update(line)
            count += 1
            observer = row["observer_map"]
            ex, ey, yaw = observer["ego"][:3]
            angle = math.radians(yaw)
            targets = []
            for actor in observer["others"]:
                if actor["type"] != "barrier":
                    continue
                dx, dy = actor["x"] - ex, actor["y"] - ey
                targets.append((str(actor["actor_id"]), dx*math.cos(angle) + dy*math.sin(angle),
                                -dx*math.sin(angle) + dy*math.cos(angle)))
            for key, x, y in targets:
                if math.hypot(x, y) <= 20.:
                    reference[key] += 1
            fresh = [t for t in row["tracks"] if t["label"] == "barrier" and t["cameras"] and
                     "rgb" in t["sources"] and "lidar" in t["sources"] and t["lidar_hits"] >= 3]
            pairs = sorted((math.hypot(t["x"] - x, t["y"] - y), i, key, x, y)
                           for i, t in enumerate(fresh) for key, x, y in targets)
            used, occupied = set(), set()
            for error, i, key, x, y in pairs:
                if error > .65 or fresh[i].get("barrier_subtype") != "striped_board" or i in used or key in occupied:
                    continue
                t = fresh[i]
                used.add(i)
                occupied.add(key)
                actors[key].append(dict(frame=row["frame"], sim_time_s=row["sim_time"],
                                 distance_m=round(math.hypot(x, y), 3), error_m=round(error, 4),
                                 track_id=t["id"], cameras=t["cameras"], rgb_score=round(t["confidence"], 4),
                                 laser_hits=t["lidar_hits"], support=t["sources"], subtype=t["barrier_subtype"]))
            subtype_counts.update(t.get("barrier_subtype","unspecified") for t in fresh)
            predictions["fresh"] += len(fresh)
            predictions["matched"] += len(used)
            if len(used) < len(fresh):
                unmatched.append({"frame": row["frame"], "unmatched_ids": [t["id"] for i, t in enumerate(fresh)
                                                                         if i not in used]})
    for key in reference:
        if actors[key]:
            examples.append(dict(actor_id=key, **min(actors[key], key=lambda t: abs(t["distance_m"]-8.))))
    return dict(schema="ado-journey-barrier-audit-v1", uncompressed_journal_sha256=digest.hexdigest(),
                journal_frames=count, max_match_error_m=.65, reference="two striped scenario barricades in observer_map",
                fresh_predictions_by_subtype=dict(subtype_counts),
                fresh_predictions=predictions["fresh"],
                matched_fresh_predictions=predictions["matched"], unmatched_fresh_predictions=unmatched,
                actors=[dict(actor_id=key, actor_frames_within_20m=reference[key],
                             matched_fresh_observations=len(actors[key]),
                             matched_fresh_within_20m=sum(t["distance_m"]<=20. for t in actors[key]),
                             median_center_error_m=median(t["error_m"] for t in actors[key]) if actors[key] else None,
                             first_observation=actors[key][0] if actors[key] else None,
                             last_observation=actors[key][-1] if actors[key] else None,
                             observed_in=sorted({v for t in actors[key] for v in t["cameras"]}),
                             tracker_ids=sorted({t["track_id"] for t in actors[key]})) for key in sorted(reference)],
                examples=examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.journal)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
