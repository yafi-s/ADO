"""Compare native front exposures on a common elapsed clock from each matched station."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


SECTIONS = (("through", 650., 688., "Straight through junction 82"),
            ("through", 701., 731., "Straight through junction 1932"),
            ("bus", 855., 882., "Stopped bus and return to pace"),
            ("bus-crossing", 862., 878., "A pedestrian crosses from behind the bus"))


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def section(folder, case, start, end):
    with gzip.open(folder / case / "trace.jsonl.gz", "rt", encoding="utf-8") as stream:
        trace = [json.loads(line) for line in stream]
    first = next(i for i, row in enumerate(trace) if row["progress_m"] >= start)
    last = next(i for i, row in enumerate(trace) if row["progress_m"] >= end)
    file = folder / case / "front.mp4"
    video = cv2.VideoCapture(str(file))
    if not video.isOpened() or int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(trace):
        raise ValueError(f"Could not synchronize {file}")
    video.set(cv2.CAP_PROP_POS_FRAMES, first)
    images = []
    try:
        for _ in range(first, last + 1):
            ok, image = video.read()
            if not ok:
                raise ValueError(f"Missing native exposure in {file}")
            images.append(image)
    finally:
        video.release()
    return images, trace[first:last+1], fingerprint(file)


def label(image, text, xy, size=.63, color=(224, 230, 234)):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def render(before, after, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", "1280x470", "-framerate", "10", "-i", "-", "-c:v", "libx264", "-threads", "2",
               "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18", "-movflags", "+faststart", str(output)]
    chapters = []
    total = 0
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for case, start, end, title in SECTIONS:
            arms = [section(folder, case, start, end) for folder in (before, after)]
            count = max(len(images) for images, _, _ in arms)
            chapters.append(dict(title=title, station_m=[start, end], start_frame=total, frames=count,
                arms=[dict(start_native_frame=rows[0]["frame"], end_native_frame=rows[-1]["frame"],
                           elapsed_s=round(rows[-1]["sim_time"] - rows[0]["sim_time"], 1),
                           source_film_sha256=sha) for _, rows, sha in arms]))
            for i in range(count):
                screen = np.full((470, 1280, 3), (28, 26, 24), dtype=np.uint8)
                label(screen, title, (16, 25), .76)
                label(screen, f"{i / 10:.1f} s from {start:.0f} m", (992, 25))
                for index, (images, rows, _) in enumerate(arms):
                    pos = min(i, len(rows)-1)
                    r = rows[pos]
                    x = 640 * index
                    screen[39:399, x:x+640] = images[pos]
                    name = ("0.5  City Life", "0.6  City Progress")[index]
                    label(screen, name, (x+15, 427), .70)
                    if i >= len(rows)-1:
                        status = f"Exited at {(len(rows)-1)/10:.1f} s | final exposure"
                    else:
                        status = f'{r["speed_mps"] * 3.6:.1f} km/h | requested {r["speed_request_mps"] * 3.6:.1f} km/h'
                    label(screen, status, (x+15, 454), .58)
                    label(screen, f'{r["progress_m"]:.1f} m', (x+518, 427), .62)
                encoder.stdin.write(screen.tobytes())
                total += 1
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Comparison film encoding failed")
    info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=r_frame_rate,nb_read_frames", "-of", "json", str(output)]))
    stream, = info["streams"]
    if stream["r_frame_rate"] != "10/1" or int(stream["nb_read_frames"]) != total:
        raise ValueError("Comparison film frame clock differs from its scored data")
    output.with_suffix(".json").write_text(json.dumps(dict(fps=10, total_frames=total, chapters=chapters,
        sha256=fingerprint(output)), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    render(a.before, a.after, a.output)
