"""Render synchronized native front cameras with their own frame-aligned speed readouts."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def read_run(folder):
    with gzip.open(folder / "trace.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    first_cone = next(e["station_m"] for e in rows[0]["events"] if e["kind"] == "roadworks")
    start = max(0, next(i for i,r in enumerate(rows) if r["progress_m"] >= first_cone) - 20)
    video = cv2.VideoCapture(str(folder / "front.mjpeg"), cv2.CAP_FFMPEG)
    if not video.isOpened():
        raise RuntimeError(f"Unavailable native footage in {folder}")
    for _ in range(start):
        if not video.grab():
            raise RuntimeError("Missing native frame before synchronization")
    return video, rows[start:], first_cone


def title(image, text, xy, size=.59, color=(244, 244, 244)):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    cv2.setNumThreads(1)
    sources = [read_run(folder) for folder in (a.before, a.after)]
    width, height = (int(sources[0][0].get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = max(min(len(rows), next(i for i, r in enumerate(rows)
                      if r["progress_m"] >= station+24.) + 10) for _, rows, station in sources)
    final_frames = {}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","bgr24",
           "-s",f"{width*2}x{height+78}","-r","10","-i","-","-an","-c:v","libx264","-threads","2",
           "-preset","medium","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(a.output)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
        for index in range(frame_count):
            panels = []
            for (capture, rows, station), name in zip(sources, ("v0.4 city pace", "Clearance-aware city pace")):
                finished = index >= len(rows)
                if finished:
                    frame = final_frames[name]
                else:
                    ok, frame = capture.read()
                    if not ok or frame.shape[:2] != (height, width):
                        raise RuntimeError("Native frame count or geometry differs")
                    final_frames[name] = frame
                canvas = np.full((height+78, width, 3), (28, 24, 21), dtype=np.uint8)
                canvas[33:height+33] = frame
                row = rows[min(index, len(rows)-1)]
                title(canvas, name, (12,23), .64)
                along = row["progress_m"] - station
                speed = row["speed_mps"] * 3.6
                title(canvas, "Finished" if finished else f"{speed:4.1f} km/h", (12,height+63), .76, (166,238,209))
                title(canvas, f"{along:+.1f} m from first cone", (width//3,height+62), .53)
                panels.append(canvas)
            encoder.stdin.write(np.concatenate(panels, axis=1).tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Encoding failed")
    for source, _, _ in sources:
        source.release()
    print(f"{frame_count} frames at native 10 Hz")


if __name__ == "__main__":
    main()
