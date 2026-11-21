"""Score the High city cyclist and false bicycle labels using the frozen native and rider rows."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def reference(frame):
    ex, ey, yaw = frame["observer_map"]["ego"]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    rows = []
    for actor in frame["observer_map"]["others"]:
        if actor["type"] not in ("cyclist", "pedestrian"):
            continue
        dx, dy = actor["x"] - ex, actor["y"] - ey
        rows.append(
            dict(
                id=actor["actor_id"],
                role=actor["role"],
                type=actor["type"],
                x=dx * c + dy * s,
                y=-dx * s + dy * c,
                world=(actor["x"], actor["y"]),
            )
        )
    return rows


def nearest(pred, refs):
    return min(
        ((math.hypot(pred["x"] - a["x"], pred["y"] - a["y"]), a["id"]) for a in refs),
        default=(float("inf"), None),
    )


def match_row(t, d, actor):
    info = t.get("cycle_association", {})
    out = dict(
        id=t["id"],
        label=t["label"],
        error_m=round(d, 4),
        forward_error_m=round(t["x"] - actor["x"], 4),
        right_error_m=round(t["y"] - actor["y"], 4),
    )
    if "classification" in t:
        out.update(source_id=t.get("sourceId"), status=t["status"], classification=t["classification"])
    elif "lidar_hits" in t:
        out.update(
            lidar_hits=t["lidar_hits"],
            coasting=t["coasting"],
            sources=t["sources"],
            sensors=t.get("radar_sensors", []),
            box_views=t.get("cameras", []),
        )
        if info:
            out.update(
                range_source=info["range_source"],
                association_age_s=round(info["age_s"], 4),
                pair_hits=info["surface_hits"],
                exclusive_hits=info["bicycle_exclusive_hits"],
            )
    return out


def paths(rows, key):
    track_ids = set()
    runs = []
    continuous_changes = []
    duplicate = 0
    errs = []
    for row in rows:
        candidates = sorted(
            (t for t in row[key] if t["label"] == "bicycle"), key=lambda t: t["error_m"]
        )
        duplicate += int(len(row[key]) > 1)
        if not candidates:
            continue
        t = candidates[0]
        fid = row["frame"]
        track_ids.add(t["id"])
        errs.append(t["error_m"])
        if runs and fid == runs[-1]["last"] + 1 and t["id"] == runs[-1]["id"]:
            runs[-1]["last"] = fid
        else:
            if runs and fid == runs[-1]["last"] + 1:
                continuous_changes.append(fid)
            runs.append(dict(id=t["id"], first=fid, last=fid))
    spatial_runs = []
    spatial_switches = []
    label_switches = []
    prior = None
    for row in rows:
        if not row[key]:
            prior = None
            continue
        nearest_vru = min(row[key], key=lambda t: t["error_m"])
        fid = row["frame"]
        ident = nearest_vru["id"]
        label = nearest_vru["label"]
        consecutive = prior is not None and fid == prior[0] + 1
        if consecutive and ident != prior[1]:
            spatial_switches.append(fid)
        if consecutive and label != prior[2]:
            label_switches.append(fid)
        if consecutive and ident == prior[1]:
            spatial_runs[-1]["last"] = fid
        else:
            spatial_runs.append(dict(id=ident, first=fid, last=fid))
        prior = fid, ident, label
    return dict(
        frames=len(rows),
        bicycle_frames=len(errs),
        person_only_frames=sum(
            bool(r[key]) and all(t["label"] == "person" for t in r[key]) for r in rows
        ),
        neither_person_nor_bicycle=sum(not r[key] for r in rows),
        duplicate_vru_frames=duplicate,
        duplicate_same_source_person_bicycle_frames=sum(
            len(r[key]) > 1
            and {v["label"] for v in r[key]} == {"person", "bicycle"}
            and len({v.get("source_id", v["id"]) for v in r[key]}) == 1
            for r in rows
        ),
        spatial_vru_intervals=spatial_runs,
        spatial_id_switch_frames=spatial_switches,
        spatial_label_switch_frames=label_switches,
        ids=sorted(track_ids),
        direct_id_switches=continuous_changes,
        bicycle_intervals=runs,
        bicycle_error_median_m=round(median(errs), 4) if errs else None,
        bicycle_error_max_m=round(max(errs), 4) if errs else None,
    )


def score(journal, display_file, detail):
    rider = {}
    if display_file:
        with display_file.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rider[r["frame"]] = r
    rows = []
    false_walker = []
    unattached = []
    obs_other = []
    previous = {}
    h = hashlib.sha256()
    frame_count = 0
    first = last = None
    with journal.open("rb") as f:
        for line in f:
            frame = json.loads(line)
            h.update(line)
            frame_count += 1
            fid = frame["frame"]
            if last is not None and fid != last + 1:
                raise ValueError(f"Exposure discontinuity at {fid}")
            first = fid if first is None else first
            last = fid
            refs = reference(frame)
            cyclists = [a for a in refs if a["type"] == "cyclist"]
            target = next((a for a in cyclists if math.hypot(a["x"], a["y"]) <= 35), None)
            row = dict(frame=fid, time_s=frame["sim_time"], native=[], display=[])
            wr = rider.get(fid)
            if target and display_file and wr is None:
                raise ValueError(f"Missing display frame {fid}")
            speed = None
            for a in cyclists:
                if a["id"] in previous:
                    x, y, tm = previous[a["id"]]
                    dt = frame["sim_time"] - tm
                    speed = math.hypot(a["world"][0] - x, a["world"][1] - y) / dt
                previous[a["id"]] = (*a["world"], frame["sim_time"])
            if target:
                row.update(
                    reference_id=target["id"],
                    reference_xy=[round(target["x"], 3), round(target["y"], 3)],
                    reference_speed_mps=round(speed, 3) if speed is not None else None,
                )
            for layer, tracks in (
                ("native", frame["tracks"]),
                ("display", wr["display"] if wr else []),
            ):
                for t in tracks:
                    if t["label"] not in ("person", "bicycle"):
                        continue
                    distance, actor_id = nearest(t, refs)
                    if distance <= 3 and target and actor_id == target["id"]:
                        row[layer].append(match_row(t, distance, target))
                    elif t["label"] == "bicycle":
                        if display_file and wr is None:
                            raise ValueError(f"Missing display evaluation on bicycle frame {fid}")
                        ref = next((a for a in refs if a["id"] == actor_id), None)
                        if distance <= 3 and ref and ref["type"] == "pedestrian":
                            false_walker.append(
                                dict(
                                    frame=fid,
                                    time_s=frame["sim_time"],
                                    layer=layer,
                                    reference_id=actor_id,
                                    reference_role=ref["role"],
                                    **match_row(t, distance, ref),
                                )
                            )
                        elif not cyclists or distance > 3:
                            unattached.append(dict(frame=fid, layer=layer, id=t["id"]))
                if target and wr:
                    for a in wr["reference"]:
                        if a["type"] != "pedestrian" or math.hypot(a["x"], a["y"]) > 35:
                            continue
                        match = a["tracker_match" if layer == "native" else "display_match"]
                        obs_other.append(
                            (a["id"], a["role"], layer, match["label"] if match else "miss")
                        )
            if target:
                rows.append(row)
    moving = [
        r for r in rows if r["reference_speed_mps"] is not None and r["reference_speed_mps"] > 0.3
    ]
    metrics = {}
    summary = {}
    for key in ("native", "display") if display_file else ("native",):
        metrics[key] = dict(nearby=paths(rows, key), moving=paths(moving, key))
        classes = Counter()
        grouped = {}
        for row in rows:
            for t in row[key]:
                if t["label"] != "bicycle":
                    continue
                if key == "native":
                    category = t.get("range_source", "ordinary bicycle track")
                    category += (
                        " fresh" if t.get("association_age_s", int(t["coasting"])) == 0 else " carried"
                    )
                else:
                    category = (t.get("classification") or "retained display class") + " " + t["status"]
                classes[category] += 1
                grouped.setdefault(category, []).append(t)
        if key == "native":
            for kind in (
                "bicycle-exclusive lidar fresh",
                "shared rider/bicycle lidar fresh",
                "bicycle-exclusive lidar carried",
                "shared rider/bicycle lidar carried",
                "ordinary bicycle track fresh",
                "ordinary bicycle track carried",
            ):
                classes.setdefault(kind, 0)
        metrics[key]["matched_bicycle_provenance"] = classes
        metrics[key]["error_by_provenance"] = {
            k: dict(
                frames=len(ts),
                median_error_m=round(median(t["error_m"] for t in ts), 4),
                max_error_m=max(t["error_m"] for t in ts),
                median_forward_error_m=round(median(t["forward_error_m"] for t in ts), 4),
                median_right_error_m=round(median(t["right_error_m"] for t in ts), 4),
            )
            for k, ts in grouped.items()
        }
    for aid, role, layer, label in obs_other:
        summary.setdefault(f"{aid} {role}", {}).setdefault(layer, Counter())[label] += 1
    body = dict(
        nearby_cyclist_frames=rows,
        false_bicycle_at_reference_pedestrian=false_walker,
        bicycle_with_no_vru_within_3m=unattached,
    )
    capture = json.loads((journal.parent.parent / "capture.json").read_text(encoding="utf-8"))
    if (first, last, h.hexdigest()) != (
        capture["first_frame"],
        capture["last_frame"],
        capture["raw_sha256"]["perception.jsonl"],
    ):
        raise ValueError("Capture manifest and scored journal differ")
    if detail:
        with detail.open("wb") as binary, gzip.GzipFile(fileobj=binary, mode="wb", mtime=0) as z:
            z.write(json.dumps(body, separators=(",", ":")).encode())
    return dict(
        capture_id=capture["capture_id"],
        capture_manifest_verified=True,
        seed=frame["seed"],
        source_sha256=h.hexdigest(),
        source_frames=frame_count,
        first_frame=first,
        last_frame=last,
        reference_ids=sorted({r["reference_id"] for r in rows}),
        nearby_range_m=35,
        match_radius_m=3,
        moving_threshold_mps=0.3,
        principal_cyclist=metrics,
        nearby_pedestrian_matches_during_encounter=summary,
        false_bicycle_frames_at_pedestrian={
            k: len({r["frame"] for r in false_walker if r["layer"] == k}) for k in metrics
        },
        bicycle_frames_no_vru_within_3m={
            k: len({r["frame"] for r in unattached if r["layer"] == k}) for k in metrics
        },
        rider_input_sha256=hashlib.sha256(display_file.read_bytes()).hexdigest()
        if display_file
        else None,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("journal", type=Path)
    p.add_argument("summary", type=Path)
    p.add_argument("--rider", type=Path)
    p.add_argument("--detail", type=Path)
    a = p.parse_args()
    a.summary.write_text(json.dumps(score(a.journal, a.rider, a.detail), indent=2) + "\n")
