"""Extract the measured same-station frame from a distinct native city film, for visual comparison."""
import argparse
import json
from pathlib import Path

from crosswalk.experiment import fingerprint, write_json


def locate(raw, station):
    selected = None
    latest_front = None
    with (raw / "perception.jsonl").open("rb") as stream:
        for index, line in enumerate(stream):
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            if row["camera_inference_age_s"].get("front") == 0.:
                latest_front = dict(frame=row["frame"], detections=row["camera_detections"].get("front", []))
            if selected is None or abs(row["progress_m"]-station) < abs(selected[1]["progress_m"]-station):
                selected = index, row, latest_front
            if row["progress_m"] >= station+4.:
                return selected
    raise ValueError(f"Capture has not reached station {station:g} m")


def camera_frame(path, wanted):
    count = 0
    buffer = b""
    with path.open("rb") as source:
        while chunk := source.read(1024*1024):
            buffer += chunk
            while (end := buffer.find(b"\xff\xd9")) >= 0:
                jpg, buffer = buffer[:end+2], buffer[end+2:]
                if not jpg.startswith(b"\xff\xd8"):
                    raise RuntimeError("Invalid JPEG stream boundary")
                if count == wanted:
                    return jpg
                count += 1
    raise ValueError("Requested native frame is not available")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--station", type=float, default=433.55)
    a = p.parse_args()
    raw = a.stage / "raw"
    i, row, last_front = locate(raw, a.station)
    jpg = camera_frame(raw / "front.mjpeg", i)
    a.output.mkdir(parents=True, exist_ok=True)
    target = a.output / "front.jpg"
    target.write_bytes(jpg)
    source = json.loads((a.stage / "request.json").read_bytes())
    # The frame's object list belongs to the recorded surround recognition rig.
    # This is independent from the original three-camera policy's feature trace.
    info = dict(capture=a.stage.name, requested_station_m=a.station, video_index=i,
                jpeg_sha256=fingerprint(target), phase=row["phase"], frame=row["frame"], time_s=row["sim_time"],
                station_m=row["progress_m"], speed_kmh=round(row["speed_mps"]*3.6,2),
                request_kmh=round(row["speed_request_mps"]*3.6,2), posted_kmh=row["posted_limit_kmh"],
                commanded_brake=row["brake"], commanded_throttle=row["throttle"], policy=row["policy"],
                front_inference_age_s=row["camera_inference_age_s"].get("front"), latest_front_inference=last_front,
                nearby_forward_tracks=[{k:t[k] for k in ("id","label","x","y") if k in t}
                                       for t in row.get("tracks", []) if 0 < t.get("x", 0) < 30],
                model_sha256=source["policy"]["sha256"],
                quality=source["quality"], population_profile=source.get("population_profile", "standard"))
    write_json(a.output / "frame.json", info)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
