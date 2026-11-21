"""Report which mapped light heads could be read from actual onboard camera frames."""
import argparse
from collections import Counter
import json
from pathlib import Path

from crosswalk.replay_pack import _array_items


def score(frames):
    counts = Counter()
    confusion = Counter()
    sources = Counter()
    junction = Counter()
    for row in frames:
        truth = row.get("observer_map", {}).get("signal_reference", {})
        for head in row.get("signals", []):
            counts["mapped_head_frames"] += 1
            if head["state"] == "unknown":
                counts["unknown"] += 1
                continue
            if head["source"] != "camera":
                raise ValueError("Colored head without camera evidence")
            state = truth.get(head["map_id"])
            if not state:
                counts["unscored"] += 1
                continue
            observed = head["state"]
            confusion[state, observed] += 1
            counts["scored_camera_readings"] += 1
            counts["correct"] += state == observed
            counts["held_from_previous_frame"] += head.get("age_s",0) > 0
            sources[head["view"]] += 1
            if head["map_id"] == "2024":
                junction[f"{state}->{observed}"] += 1
    return dict(counts=counts, by_camera=sources, target_junction=junction,
                confusion={f"{a}->{b}":v for (a,b),v in sorted(confusion.items())},
                units="head-frames; correlated views and nearby frames are not independent",
                denominator="All mapped heads within 60 m, including out-of-view, occluded and back-facing heads",
                source="RGB measurements compared to CARLA reference after control")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("replay",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    with args.replay.open(encoding="utf8") as stream:
        data=score(_array_items(stream))
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(data,indent=2)+"\n",encoding="utf8")
    print(data["counts"])
