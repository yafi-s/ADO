"""Run both front optical scales on identical sealed MJPEG exposures, with bounded CPU inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosswalk.surround_vision import SurroundDetector, horizon_region


def verify_native_crops(source, sweeps, destination, stride):
    cv2.setNumThreads(1)
    detector=SurroundDetector()
    tested=[]
    with (source/"perception.jsonl").open() as journal:
        for index,line in enumerate(journal):
            f=json.loads(line)
            if f["camera_inference_age_s"]["front"]!=0:
                continue
            expected=f["camera_detections"]["front"]
            added=[b for b in expected if b.get("inference_layout")=="front-horizon-2x"]
            # Also include the first observed addition even if it falls between fixed samples.
            if index%stride and (not added or any(row["added_boxes"] for row in tested)):
                continue
            with np.load(sweeps/f"{f['frame']}.npz") as data:
                w,h=map(int,data["front_size_wh"])
                rgb=np.zeros((h,w,3),np.uint8)
                rect=tuple(data["horizon_xyxy"])
                if rect!=horizon_region(rgb):
                    raise ValueError(f"Crop registration mismatch at {f['frame']}")
                left,top,right,bottom=rect
                crop=data["horizon_rgb"]
                rgb[top:bottom,left:right]=crop
                crop_hash=hashlib.sha256(crop.tobytes()).hexdigest()
            baseline=[b for b in expected if b.get("inference_layout")=="single"]
            new=detector.merge_horizon(baseline,detector.horizon_boxes(rgb))[len(baseline):]
            select=lambda entries:[(b["label"],b["box"],b["score"]) for b in entries]
            if select(new)!=select(added):
                raise ValueError(f"Native horizon inference differs at {f['frame']}")
            tested.append(dict(frame=f["frame"],front_px=[w,h],added_boxes=len(added),crop_sha256=crop_hash))
    if not tested:
        raise ValueError("No native optical crops verified")
    with destination.open("x",encoding="utf8",newline="\n") as f:
        json.dump(dict(exposures=len(tested),added_boxes=sum(r["added_boxes"] for r in tested),
                       sampling_stride=stride,all_boxes_reproduced_exactly=True,frames=tested),f,indent=2)
        f.write("\n")


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
    parser.add_argument("--native-sweeps", type=Path, help="Verify sampled native HD crops against recorded boxes")
    args = parser.parse_args()
    if args.native_sweeps:
        verify_native_crops(args.source,args.native_sweeps,args.destination,args.stride)
    else:
        replay(args.source, args.destination, args.stride)
