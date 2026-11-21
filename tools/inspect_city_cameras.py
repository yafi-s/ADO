"""Extract synchronized, detector-independent visibility spot checks from a city camera film."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def projection(actor, mount):
    yaw = math.radians(mount["yaw"])
    dx, dy = actor["x"] - mount["x"], actor["y"] - mount["y"]
    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    if forward <= 0:
        return None
    lateral = dy * math.cos(yaw) - dx * math.sin(yaw)
    u = .5 + .5 * lateral / (forward * math.tan(math.radians(mount["hfov"]) / 2))
    return u if 0 <= u <= 1 else None


def select_frames(scored, mounts):
    # Each actor can supply at most one frame per sector. Select by proximity to 15 m before
    # accessing the detector or display fields. Three distinct actors at most per sector.
    best = {}
    with scored.open(encoding="utf-8") as rows:
        for index, line in enumerate(rows):
            frame = json.loads(line)
            for actor in frame["tracker"]["actors"]:
                cameras = [(abs(u-.5), name, u) for name, mount in mounts.items()
                           if (u := projection(actor, mount)) is not None]
                if not cameras:
                    continue
                _, camera, u = min(cameras)
                key = actor["sector"], actor["id"]
                rank = abs(actor["range_m"] - 15), index
                if key not in best or rank < best[key][0]:
                    best[key] = (rank, dict(frame=frame["frame"], index=index, camera=camera,
                                            sim_time=frame["sim_time"], actor_id=actor["id"], role=actor["role"],
                                            kind=actor["type"], sector=actor["sector"], u=round(u, 3),
                                            range_m=actor["range_m"]))
    chosen = []
    for sector in ("front", "right", "rear", "left"):
        ranked = sorted((rank, actor, data) for (s, actor), (rank, data) in best.items() if s == sector)
        chosen.extend(data for _, _, data in ranked[:3])
    return chosen


def requested_frames(scored, mounts, probes):
    keys = [probe.split(":") for probe in probes]
    if any(len(key) != 3 for key in keys):
        raise ValueError("Use FRAME:ACTOR_ID_OR_ROLE:CAMERA for each probe")
    wanted = {int(key[0]) for key in keys}
    selected = {}
    with scored.open(encoding="utf-8") as rows:
        for index, line in enumerate(rows):
            frame = json.loads(line)
            if frame["frame"] not in wanted:
                continue
            actors = frame["tracker"]["actors"] + frame.get("parked", {}).get("tracker", [])
            for number, token, camera in keys:
                if int(number) != frame["frame"]:
                    continue
                candidates = [a for a in actors if str(a["id"]) == token or a["role"] == token]
                if len(candidates) != 1 or camera not in mounts:
                    raise ValueError(f"Missing or ambiguous actor or camera for {number}:{token}:{camera}")
                a = candidates[0]
                u = projection(a, mounts[camera])
                if u is None:
                    raise ValueError(f"Actor center is outside the selected camera at {number}:{token}:{camera}")
                selected[number,token,camera] = dict(frame=frame["frame"], index=index, camera=camera,
                    sim_time=frame["sim_time"], actor_id=a["id"], role=a["role"], kind=a["type"],
                    sector=a["sector"], u=round(u,3), range_m=a["range_m"])
    return [selected[tuple(key)] for key in keys]


def film_frames(source, indices):
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open {source}")
        images = {}
        next_index = 0
        for index in sorted(set(indices)):
            if source.suffix == ".mjpeg":
                # Raw MJPEG has no seek index. Some backends report success after
                # seeking but return the next sequential image instead.
                while next_index < index:
                    if not cap.grab():
                        raise ValueError(f"End of {source} before frame {index}")
                    next_index += 1
            elif not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise RuntimeError(f"Could not seek {source} to frame {index}")
            ok, image = cap.read()
            if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
                raise ValueError(f"Could not verify frame {index} of {source}")
            images[index] = image
            next_index = index + 1
        return images
    finally:
        cap.release()


def render(directory, scored, rig, destination, probes=()):
    cv2.setNumThreads(1)
    mounts = json.loads(rig.read_text(encoding="utf-8"))["cameras"] if rig else {
        "front": dict(x=1.3, y=0, yaw=0, hfov=90)}
    media = {name: next((p for ext in (".mp4", ".mjpeg") if (p := directory/(name+ext)).is_file()), None)
             for name in mounts}
    available = {k: v for k, v in mounts.items() if media[k]}
    chosen = requested_frames(scored, available, probes) if probes else select_frames(scored, available)
    if not chosen:
        raise ValueError("No actor center falls within the available camera bearings")
    panels = {}
    for camera in sorted({r["camera"] for r in chosen}):
        sample = [r for r in chosen if r["camera"] == camera]
        frames = film_frames(media[camera], [r["index"] for r in sample])
        for r in sample:
            image = cv2.resize(frames[r["index"]], (480, 270), interpolation=cv2.INTER_AREA)
            panel = cv2.copyMakeBorder(image, 0, 50, 0, 0, cv2.BORDER_CONSTANT, value=(16, 16, 16))
            labels = (f"{r['sector']} | {camera} | frame {r['frame']} | {r['sim_time']:.1f} s",
                      f"{r['kind']} {r['actor_id']} | {r['range_m']:.1f} m | expected at {r['u']:.0%} across")
            for line, label in enumerate(labels):
                cv2.putText(panel, label, (8, 290+21*line), cv2.FONT_HERSHEY_SIMPLEX, .55,
                            (255, 255, 255), 1, cv2.LINE_AA)
            panels[r["frame"], r["camera"], r["actor_id"]] = panel
    # Three columns, grouped by sector, with blank cells where fewer than three actors qualified.
    sheet = []
    blank = np.zeros((320, 480, 3), dtype=np.uint8)
    columns = 2 if probes else 3
    groups = ([chosen[i:i+columns] for i in range(0,len(chosen),columns)] if probes else
              [[r for r in chosen if r["sector"] == sector] for sector in ("front","right","rear","left")])
    for group in groups:
        row = [panels[r["frame"],r["camera"],r["actor_id"]] for r in group]
        if row:
            sheet.append(np.hstack(row + [blank]*(columns-len(row))))
    if not cv2.imwrite(str(destination), np.vstack(sheet)):
        raise OSError(f"Could not save {destination}")
    return chosen


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, help="Recording directory containing synchronized policy-camera films")
    cli.add_argument("scored", type=Path, help="Frame-level JSONL generated with audit_city_perception.mjs --frames")
    cli.add_argument("destination", type=Path, help="Output image file")
    cli.add_argument("--rig", type=Path, help="Surround evidence frame.json containing camera calibration; omit for v0.2")
    cli.add_argument("--probe", action="append", default=[], help="Explicit FRAME:ACTOR_ID_OR_ROLE:CAMERA; repeat for a contact sheet")
    args = cli.parse_args()
    print(json.dumps(render(args.directory, args.scored, args.rig, args.destination, args.probe), indent=2))
