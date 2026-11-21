"""Grade the physical traffic and recorded pace of one continuous Town03 drive."""
import argparse
from collections import Counter, deque
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from crosswalk.replay_pack import _array_items


def moving_nearby(history):
    """Require a continuous measured trajectory for each moving road user."""
    now = history[-1]
    counts = Counter()
    if len(history) < 6:
        return None
    elapsed = now[0] - history[0][0]
    if not math.isclose(elapsed, 0.5, abs_tol=0.002):
        raise ValueError("Traffic window must contain half a second of observations")
    for actor_id, (kind, x, y) in now[2].items():
        if math.dist((x, y), now[1]) > 50:
            continue
        if any(actor_id not in observations or observations[actor_id][0] != kind
               for _, _, observations in history):
            continue
        positions = [actors[actor_id][1:] for _, _, actors in history]
        limit = 2.5 if kind == "vehicle" else 0.65  # max plausible displacement per 0.1 s
        if any(math.dist(p, q) > limit for p, q in zip(positions, positions[1:])):
            continue
        speed = math.dist(positions[0], positions[-1]) / elapsed
        if speed >= (1.0 if kind == "vehicle" else 0.5):
            counts[kind] += 1
    return [counts["vehicle"], counts["pedestrian"]]


def measure(folder):
    manifest = json.loads((folder / "manifest.json").read_bytes())
    path = folder / "replay.json"
    opener = open
    if not path.exists():
        path = folder / "replay.json.gz"
        opener = gzip.open
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != manifest["sha256"][path.name]:
        raise ValueError("Replay differs from the recording manifest")
    summary = manifest["summary"]
    cone = next((e for e in summary["events"] if e["kind"] == "roadworks"), None)
    samples, windows = [], {"alongside": [], "after": []}
    history, density = deque(maxlen=6), []
    cone_review = {}
    if manifest.get("cone_review_sha256"):
        source = folder / "cone-observations.json"
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["cone_review_sha256"]:
                raise ValueError("Cone review differs from the recording manifest")
        cone_review = {r["frame"]: r["cone_observations"] for r in json.loads(source.read_bytes())["frames"]}
    cone_sample, mixed_sample = None, None
    cone_rank, mixed_rank = (0, 0), (0, 0)
    names = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "vehicle": "vehicle",
             "van": "vehicle", "motorcycle": "vehicle", "pedestrian": "pedestrian"}
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for index, row in enumerate(_array_items(stream)):
            if row["frame"] != manifest["first_frame"] + index:
                raise ValueError("Nonconsecutive native exposures")
            t = row["sim_time"]
            scene = row["observer_map"]
            actors = {a["actor_id"]: (names[a["type"]], a["x"], a["y"])
                      for a in scene["others"] if a["type"] in names}
            history.append((t, scene["ego"][:2], actors))
            nearby = moving_nearby(history)
            if nearby is not None:
                density.append(nearby)
            speed = row["speed_mps"] * 3.6
            request = row.get("speed_request_mps")
            request = None if request is None else round(request * 3.6, 2)
            sample = [round(t, 2), row["frame"], round(row["progress_m"], 2), round(speed, 2),
                      request, row.get("posted_limit_kmh"), *(nearby or [None, None])]
            if index % 10 == 0 or row["frame"] == manifest["last_frame"]:
                samples.append(sample)
            ranged = row.get("cone_observations") or cone_review.get(row["frame"], [])
            front = [c for c in ranged if c.get("camera") == "front" and c.get("lidar_hits", 0) > 0]
            rank = (len(front), sum(c["lidar_hits"] for c in front))
            if rank > cone_rank:
                cone_rank, cone_sample = rank, sample
            if nearby and min(nearby) > 0:
                rank = (min(nearby), sum(nearby))
                if rank > mixed_rank:
                    mixed_rank, mixed_sample = rank, sample
            if cone:
                offset = row["progress_m"] - cone["station_m"]
                if 0 <= offset < 40:
                    windows["alongside" if offset < 20 else "after"].append((speed, request))
    if not samples or samples[-1][1] != manifest["last_frame"] or len(density) != summary["frames"] - 5:
        raise ValueError("Incomplete traffic or speed timeline")
    pace = {name: {"seconds": round(len(rows) * 0.1, 2),
                   "median_kmh": round(median(s for s, _ in rows), 2),
                   "minimum_kmh": round(min(s for s, _ in rows), 2),
                   "median_requested_kmh": round(median(r for _, r in rows), 2)
                   if all(r is not None for _, r in rows) else None}
            for name, rows in windows.items() if rows}
    for sample in (cone_sample, mixed_sample):
        if sample is not None and sample not in samples:
            samples.append(sample)
    samples.sort(key=lambda s: s[0])
    return {"trial": manifest["trial_id"], "label": manifest["catalog_entry"]["label"],
            "frames": summary["frames"], "source_sha256": digest,
            "nearby": {"radius_m": 50, "eligible_frames": len(density),
                       "moving_car_and_ped_frames": sum(v > 0 and p > 0 for v, p in density),
                       "two_each_frames": sum(v >= 2 and p >= 2 for v, p in density),
                       "peak_moving_vehicles": max(v for v, _ in density),
                       "peak_moving_pedestrians": max(p for _, p in density)},
            "roadworks": pace, "cones_at_s": cone_sample[0] if cone_sample else None,
            "mixed_at_s": mixed_sample[0] if mixed_sample else None, "samples": samples}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("recordings", nargs="+", type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    data = {"columns": ["seconds", "frame", "route_m", "speed_kmh", "target_kmh",
                        "limit_kmh", "moving_vehicles", "moving_pedestrians"],
            "drives": [measure(folder) for folder in args.recordings]}
    args.output.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n",
                           encoding="utf8", newline="\n")
