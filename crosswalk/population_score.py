"""Score simultaneous street activity using native research references after capture."""
from collections import Counter, deque
import gzip
from io import TextIOWrapper
import json
import math


RADII = (35, 50, 65, 100)
SPEED_THRESHOLDS = {"car": 1., "pedestrian": .5}
WINDOW = 5  # 0.5 simulated seconds at 10 Hz


def score_population(source, reference_path):
    histories, actors = {}, {}
    counters = {radius: Counter(dict(scored_frames=0, moving_car_frames=0, moving_pedestrian_frames=0,
                                     both_frames=0, two_of_each_frames=0)) for radius in RADII}
    peaks = {radius: dict(cars=0, pedestrians=0, simultaneous_frame=None, simultaneous_cars=0,
                          simultaneous_pedestrians=0) for radius in RADII}
    first = last = None
    with (source.open(encoding="utf-8") as stream, reference_path.open("wb") as target,
          gzip.GzipFile(filename="", fileobj=target, mode="wb", compresslevel=5, mtime=0) as zipped,
          TextIOWrapper(zipped, encoding="utf-8", newline="\n") as output):
        for line in stream:
            sample = json.loads(line)
            frame, time = sample["frame"], sample["sim_time"]
            if first is None:
                first = frame
            if last is not None and frame != last + 1:
                raise ValueError("Population audit requires a contiguous frame sequence")
            last = frame
            observer = sample["observer_map"]
            ego = observer["ego"][:2]
            counts = {r: Counter() for r in RADII}
            present = set()
            row = dict(frame=frame, sim_time=time, progress_m=sample["progress_m"], ego=observer["ego"],
                       others=observer["others"], signal_reference=observer.get("signal_reference"))
            output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            for other in row["others"]:
                kind = {"car": "car", "van": "car", "bus": "car", "pedestrian": "pedestrian"}.get(other["type"])
                if kind is None:
                    continue
                key, xy = other["actor_id"], (other["x"], other["y"])
                present.add(key)
                series = histories.setdefault(key, deque(maxlen=WINDOW+1))
                actor = actors.setdefault(key, dict(actor_id=key, kind=kind, blueprint=other["blueprint"],
                                                    role=other["role"], frames=0, moving_frames=0,
                                                    observed_distance_m=0., max_step_m=0.))
                actor["frames"] += 1
                if series:
                    step = math.dist(series[-1][1], xy)
                    actor["max_step_m"] = max(actor["max_step_m"], step)
                    if step > 3.:
                        series.clear()
                    else:
                        actor["observed_distance_m"] += step
                series.append((time, xy))
                if len(series) <= WINDOW:
                    continue
                old_time, old_xy = series[0]
                if math.dist(old_xy, xy) < SPEED_THRESHOLDS[kind] * (time - old_time):
                    continue
                actor["moving_frames"] += 1
                distance = math.dist(ego, xy)
                for radius in RADII:
                    if distance <= radius:
                        counts[radius][kind] += 1
            for key in histories.keys() - present:
                del histories[key]
            if frame - first < WINDOW:
                continue
            for radius, c in counts.items():
                tally, peak = counters[radius], peaks[radius]
                cars, peds = c["car"], c["pedestrian"]
                tally["scored_frames"] += 1
                tally["moving_car_frames"] += cars > 0
                tally["moving_pedestrian_frames"] += peds > 0
                tally["both_frames"] += min(cars, peds) >= 1
                tally["two_of_each_frames"] += min(cars, peds) >= 2
                peak["cars"] = max(peak.get("cars", 0), cars)
                peak["pedestrians"] = max(peak.get("pedestrians", 0), peds)
                # Select a single same-frame pair, prioritizing variety before total.
                rank = (min(cars, peds), cars + peds)
                if rank > tuple(peak.get("rank", (-1, -1))):
                    peak.update(rank=rank, simultaneous_frame=frame, simultaneous_cars=cars,
                                simultaneous_pedestrians=peds)
    if last is None:
        raise ValueError("Empty population reference")
    for peak in peaks.values():
        peak.pop("rank", None)
    for actor in actors.values():
        for field in ("observed_distance_m", "max_step_m"):
            actor[field] = round(actor[field], 3)
    return dict(schema="ado-city-population-score-v1", first_frame=first, last_frame=last,
                speed_threshold_mps=SPEED_THRESHOLDS, motion_window_s=.5, capture_frames=last-first+1,
                source="native actor centre displacement across six consecutive 10 Hz frames",
                includes="staged and background road users; cyclists excluded",
                radii_m={str(r): {**counters[r], "peak": peaks[r]} for r in RADII},
                actors=sorted(actors.values(), key=lambda a: a["actor_id"]))
