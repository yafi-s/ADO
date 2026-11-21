"""Produce review sheets from native camera JPEGs and the audit's locked frames."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from PIL import Image, ImageDraw

def render(stage, audit, output):
    r=json.loads(audit.read_bytes())
    capture=json.loads((stage/"capture.json").read_bytes())
    data=json.loads((stage/"raw/telemetry.json").read_bytes())
    frames={v["frame"]:v for v in data}
    groups={}
    buf=[(v["middle_frame"],f'{v["from_m"]}-{v["to_m"]} m') for v in r["station_bins"] if v["frames"]]
    for i in range(0,len(buf),12): groups[f"route-{i//12+1}"]=buf[i:i+12]
    releases=[(v["frame"]+round(t*10),f'{v["event"]} +{t}s') for v in r["releases"] if v["mark"]=="cleared_s" for t in [0,2,5]]
    for i in range(0,len(releases),12): groups[f"releases-{i//12+1}"]=releases[i:i+12]
    groups["junctions"]=[(v["middle_frame"],f'Jct {v["id"]} {v["direction"]}') for v in r["junctions"] if v["frames"]]
    peaks=r["lateral_acceleration_estimate"]
    groups["comfort"]=[(peaks["all_junctions"]["peak_frame"],"Peak junction yaw/speed"),
                       (r["comfort"]["peak_jerk_half_s"]["frame"],"Peak longitudinal jerk")]
    for label,m in [("Approach yield line",1040.84),("End-road flank",1379.15),("End-road recovery",1386.1),
                    ("Sidewalk approach",697.29),("Unmatched building box",894.4)]:
        hit=min(data,key=lambda row:abs(row["progress_m"]-m))
        groups["comfort"].append((hit["frame"],label))
    if data[-1]["progress_m"]>575:
        groups["boards"]=[]
        for m in [522,526,530.084,532.164,534.148,536.822,540.111,543.8,546.814,550.48,560,573]:
            hit=min(data,key=lambda row:abs(row["progress_m"]-m))
            groups["boards"].append((hit["frame"],f'{hit["progress_m"]:.1f} m'))
    if r["grade"]["outcome"]=="complete":
        near=r["clearance_below_four_m"]
        groups["close"]=[(v["closest"]["frame"],k) for k,v in near.items()]
        groups["close"].append((r["comfort"]["peak_jerk_half_s"]["frame"],"peak 0.5s jerk"))
        for lo,hi in [(855,861),(861,867),(867,873),(873,879),(525,531),(532,538)]:
            rr=[v for v in data if lo<=v["progress_m"]<hi]
            if rr:groups["close"].append((rr[len(rr)//2]["frame"],f"{lo}-{hi} m"))
    output.mkdir(parents=True,exist_ok=True)
    for camera in ["front","chase"]:
        subset=groups if camera=="front" else {k:v for k,v in groups.items() if k in {"junctions","close","boards","comfort"}}
        wanted={f for seq in subset.values() for f,_ in seq if f in frames}
        warmup=capture["warmup_frames"] if camera=="chase" else 0
        index={f-capture["first_frame"]+warmup:f for f in wanted}
        decoded={}
        fn=f"{camera}.mjpeg"
        # Hash once while indexing. Keep at most one partial JPEG between reads,
        # avoiding a multi-gigabyte mapped chase-camera file on the shared PC.
        digest=hashlib.sha256()
        carry=b""
        i=0
        def pick(payload, position):
            if position in index:
                with Image.open(BytesIO(payload)) as img:
                    decoded[index[position]]=img.convert("RGB").resize((320,180))
        with (stage/"raw"/fn).open("rb") as src:
            while chunk:=src.read(4*1024*1024):
                digest.update(chunk)
                data=carry+chunk
                start=0
                for sep in re.finditer(b"\xff\xd9\xff\xd8",data):
                    end=sep.start()+2
                    if i in index:pick(data[start:end],i)
                    i+=1
                    start=end
                carry=data[start:]
                if len(carry)>32*1024*1024:
                    raise ValueError("Invalid native JPEG stream")
            if not carry.startswith(b"\xff\xd8") or not carry.endswith(b"\xff\xd9"):
                raise ValueError("Incomplete last native JPEG")
            pick(carry,i)
        if digest.hexdigest()!=capture["raw_sha256"][fn] or i+1!=capture.get(f"{camera}_source_frames",capture["summary"]["frames"]):
            raise ValueError("Native camera digest or exposure count mismatch: "+fn)
        for name, seq in subset.items():
            seq=[(f,txt) for f,txt in seq if f in frames]
            canvas=Image.new("RGB",(3*320,204*((len(seq)+2)//3)),(19,24,26)); draw=ImageDraw.Draw(canvas)
            for i,(f,txt) in enumerate(seq):
                x,y=i%3*320,i//3*204
                canvas.paste(decoded[f],(x,y)); row=frames[f]
                draw.text((x+3,y+182),f'{txt} | f{f} | {row["sim_time"]:.1f}s | {row["speed_mps"]*3.6:.1f} km/h',fill="white")
            canvas.save(output/f"{camera}-{name}.jpg",quality=85)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",type=Path);p.add_argument("audit",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();render(a.stage,a.audit,a.output)
