"""Encode every synchronized native front frame against its measured speed trace."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from crosswalk.experiment import fingerprint, write_json


def render(trial, front, output, title):
    cv2.setNumThreads(1)
    with gzip.open(trial/"trace.jsonl.gz","rt") as stream:
        rows=[json.loads(line) for line in stream]
    s=json.loads((trial/"summary.json").read_bytes())
    fps=10
    if any(r["frame"]!=rows[0]["frame"]+i or abs(r["sim_time"]-i/fps)>1.e-5 for i,r in enumerate(rows)):
        raise ValueError("Frame and simulation clocks differ")
    if len(list(front.glob("*.png")))!=len(rows):
        raise ValueError("Incomplete synchronized front-camera archive")
    first=cv2.imread(str(front/f'{rows[0]["frame"]}.png'))
    h,w=first.shape[:2]
    footer=96
    output.parent.mkdir(parents=True,exist_ok=True)
    part=output.with_suffix(".partial.mp4")
    args=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","bgr24",
          "-s",f"{w}x{h+footer}","-r",str(fps),"-i","pipe:0","-an","-c:v","libx264","-preset","fast",
          "-crf","20","-maxrate","1600k","-bufsize","3200k","-threads","2","-pix_fmt","yuv420p",
          "-movflags","+faststart",str(part)]
    native_hash=hashlib.sha256()
    manifest=[]
    def words(screen,caption,x,y,scale,color=(240,240,240),bold=1):
        cv2.putText(screen,caption,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,bold,cv2.LINE_AA)
    process=subprocess.Popen(args,stdin=subprocess.PIPE)
    try:
        for i,r in enumerate(rows):
            path=front/f'{r["frame"]}.png'
            raw=path.read_bytes()
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.shape[:2]!=(h,w):
                raise ValueError(f"Missing or changed native image dimensions at exposure {r['frame']}")
            native_hash.update(r["frame"].to_bytes(8,"big"))
            native_hash.update(hashlib.sha256(raw).digest())
            screen=np.full((h+footer,w,3),(28,25,21),np.uint8)
            screen[:h]=image
            words(screen,title,14,h+19,.46,(231,218,190))
            words(screen,f'{r["sim_time"]:.1f} s',w-78,h+19,.47)
            for col,label,value in ((14,"MEASURED SPEED",3.6*r["speed_mps"]),
                                    (w//3+10,"REQUESTED",3.6*r["speed_request_mps"]),
                                    (2*w//3+12,"POSTED LIMIT",r["posted_limit_kmh"])):
                words(screen,label,col,h+46,.37,(198,195,188))
                words(screen,f"{value:.1f} km/h" if label!="POSTED LIMIT" else f"{value:.0f} km/h",
                      col,h+77,.70,(247,248,250),2 if col==14 else 1)
            process.stdin.write(screen.tobytes())
            manifest.append(dict(index=i,native_frame=r["frame"],sim_time_s=r["sim_time"],
                source_png_sha256=hashlib.sha256(raw).hexdigest(),actual_kmh=round(3.6*r["speed_mps"],4),
                requested_kmh=round(3.6*r["speed_request_mps"],4),posted_kmh=r["posted_limit_kmh"],
                progress_m=r["progress_m"],lead_gap_m=r.get("lead_body_gap_m")))
    finally:
        process.stdin.close()
        rc=process.wait(timeout=45)
    if rc:
        raise RuntimeError(f"Native front film encoder exited {rc}")
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
        "-show_entries","stream=nb_read_frames,avg_frame_rate,width,height,duration","-of","json",str(part)]))["streams"][0]
    if int(probe["nb_read_frames"])!=len(rows) or probe["avg_frame_rate"]!="10/1" or part.stat().st_size>15_000_000:
        raise RuntimeError(f"Encoded video failed frame count, rate or size check: {probe}")
    part.replace(output)
    record=dict(schema="ado-native-speed-film-v1",source_trial=trial.as_posix(),native_map=s["producer"]["map"],
                camera="native policy front RGB",native_wh=[w,h],encoded_wh=[w,h+footer],
                edit="Full contiguous run, one captured 10 Hz native exposure per encoded frame, measured footer below native image.",
                model_sha256=s["model_sha256"],implementation=s["implementation"],scenario=s["route_id"],
                trace_sha256=fingerprint(trial/"trace.jsonl.gz"),summary_sha256=fingerprint(trial/"summary.json"),
                source_frames_sha256=native_hash.hexdigest(),frame_hash_definition="SHA256 of ordered (8-byte big-endian native frame number, 32-byte SHA256 of source PNG) pairs",
                movie_sha256=fingerprint(output),movie_bytes=output.stat().st_size,ffprobe=probe,frames=manifest)
    write_json(output.with_suffix(".json"),record)
    print(output,len(rows),record["movie_bytes"],record["movie_sha256"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("trial","front","output"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--title",required=True)
    a=p.parse_args()
    render(a.trial,a.front,a.output,a.title)


if __name__=="__main__":
    main()
