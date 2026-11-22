"""Check camera fallback, parked obstacle and crossing controls against archived native range and front RGB."""
import gzip
import io
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from .behavior_v2 import _checked_route
from .city_drive import load_driver
from .experiment import fingerprint
from .vision import Detector

ROOT=Path("results/city-progress/junction-transfer")
EPISODES=("j1221-r4-first/candidate-clear","j1221-r4-first/candidate-crossing",
          "j1221-r4-first/candidate-stops-short","hill-r4-first/parked")


def score(folder,detector):
    model,_,_=load_driver(Path("models/city-maneuvers/candidate.npz"))
    observer=model.new_observer()
    with gzip.open(folder/"trace.jsonl.gz","rt") as f:
        rows=[json.loads(l) for l in f]
    samples=[]
    with ZipFile(folder/"sensor-sweeps.zip") as z,np.load(folder/"observations.npz") as data:
        paths={int(n.split("/")[0].split("-")[-1]):n.split("/")[0] for n in z.namelist() if n.endswith("lidar.npy")}
        last=None
        for i,(r,x) in enumerate(zip(rows,data["features"],strict=True)):
            frame=r["frame"]
            if frame not in paths:
                model.last_speed_request=r["speed_request_mps"]
                last=None
                continue
            prefix=paths[frame]
            packet={name:np.load(io.BytesIO(z.read(f"{prefix}/{name}.npy"))) for name in ("lidar","radar")}
            image=cv2.imdecode(np.frombuffer(z.read(f"{prefix}/front.jpg"),np.uint8),cv2.IMREAD_COLOR)
            if i%2==0 or last is None:
                boxes=detector.detect(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            patch=image[image.shape[0]//3:3*image.shape[0]//4:12,::12]
            observer.travel+=.1*r["speed_mps"]
            extra=observer.measure(packet,boxes,_checked_route(x[:6],None),r["speed_mps"],x[:-2],
                                   valid_image=bool(patch.size and np.quantile(patch,.75)>24 and np.std(patch)>7),
                                   contiguous=last==frame-1)
            model.posted_limit_kmh=r["posted_limit_kmh"]
            model.signed_speed_limit=r.get("signed_approach_cap_mps")
            # Isolate the effect of this exposure and geometry history from
            # feedback due to a preceding different command on the same poses.
            model.last_speed_request=rows[i-1]["speed_request_mps"] if i else None
            model.predict(np.r_[x[:-2],extra,x[-2:]])
            last=frame
            samples.append(dict(frame=frame,station_m=r["progress_m"],old_kmh=r["speed_request_mps"]*3.6,
                new_kmh=model.last_speed_request*3.6,extra_kmh=model.progress_evidence["maneuver_added_mps"]*3.6,
                old_binding=r["progress_evidence"]["binding"],camera_horizon_m=extra[3],
                camera_candidates=int(extra[4]),crossing_gap_m=r["progress_evidence"]["crossing_gap_m"],
                raw_body_m=r["progress_evidence"]["raw_body_memory_gap_m"]))
    def window(items):
        return dict(frames=len(items),direct_additions=sum(float(r["extra_kmh"])>.05 for r in items),
                    max_extra_kmh=max((r["extra_kmh"] for r in items),default=0.))
    return dict(case=str(folder.relative_to(ROOT)).replace("\\","/"),ranges_sha256=fingerprint(folder/"sensor-sweeps.zip"),
                all=window(samples),crossing=window([r for r in samples if r["crossing_gap_m"]<60.]),
                close_body=window([r for r in samples if r["raw_body_m"]<10.]),
                focal_frames=[r for r in samples if r["frame"] in range(1109,1116)] if "candidate-clear" in str(folder) else [])


def main():
    cv2.setNumThreads(1)
    net=Detector()
    results=[score(ROOT/e,net) for e in EPISODES if (ROOT/e/"sensor-sweeps.zip").is_file()]
    output=Path("results/city-maneuvers/pace/junction-controls.json")
    output.write_text(json.dumps(dict(schema="ado-city-maneuvers-junction-controls-v1",
        checkpoint_sha256=fingerprint(Path("models/city-maneuvers/candidate.npz")),
        source_sha256=fingerprint(Path(__file__).with_name("city_maneuvers.py")),
        method="Original grade-model feature tape and camera/range exposures. The recorded straight route extends its last measured tangent. Each request uses the native previous request. Detector reruns on archived JPEG. Same-pose requests only.",cases=results),indent=2)+"\n",encoding="utf8")
    print([(r["case"],r["crossing"],r["close_body"]) for r in results])

if __name__=="__main__":
    main()
