"""Run both front optical scales on identical sealed MJPEG exposures, with bounded CPU inference."""
import argparse
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosswalk.surround_vision import SurroundDetector


def replay(source, destination, stride):
    cv2.setNumThreads(1)
    detector = SurroundDetector()
    camera = cv2.VideoCapture(str(source / "front.mjpeg"), cv2.CAP_FFMPEG)
    if not camera.isOpened():
        raise ValueError("Missing front MJPEG stream")
    try:
        with (source / "perception.jsonl").open() as journal, destination.open("x") as output:
            for index, line in enumerate(journal):
                if not camera.grab():
                    raise ValueError(f"Missing front exposure at {index}")
                frame = json.loads(line)
                if index % stride or frame["camera_inference_age_s"]["front"] != 0:
                    continue
                ok, bgr = camera.retrieve()
                if not ok:
                    raise ValueError(f"Could not decode exposure at {index}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                base = detector.detect(rgb)
                zoom = detector.horizon_boxes(rgb)
                detections = detector.merge_horizon(base, zoom)
                output.write(json.dumps(dict(frame=frame["frame"], boxes=detections)) + "\n")
            if camera.grab():
                raise ValueError("Front stream exceeds sensor journal")
    finally:
        camera.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    replay(args.source, args.destination, args.stride)
