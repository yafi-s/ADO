"""Check camera-reported green heads against independent native states on through approaches.

The camera classifications are from the display observer. Matching a live forward
image to the registered head and simulator color does not give truth to the driver.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import numpy as np


def assess(stage, audit):
    with (stage / "capture.json").open() as f:
        capture=json.load(f)
    fixed=json.loads(audit.read_bytes())
    roads=[j for j in fixed["junctions"] if j["direction"]=="straight"]
    out={str(j["id"]):{"entry_m":j["from_m"],"exit_m":j["to_m"],
                        "approach":[],"inside":[]} for j in roads}
    tape_path=stage/"policy-observations.npz"
    controller=None
    if tape_path.exists():
        with np.load(tape_path,allow_pickle=False) as tape:
            names=tape["feature_names"].tolist()
            controller=(tape["frame"],tape["features"][:,names.index("mapped_lamp_gap_m")],
                        tape["features"][:,names.index("rgb_lamp_red_probability")])
    with (stage/"raw/perception.jsonl").open(encoding="utf8") as stream:
        for i,line in enumerate(stream):
            row=json.loads(line)
            if row["frame"]!=capture["first_frame"]+i:raise ValueError("Frame mismatch")
            station=row["progress_m"]; x,y,yaw=row["observer_map"]["ego"]
            natives=row["observer_map"]["signal_reference"]
            for j in roads:
                if not j["from_m"]-18<=station<j["to_m"]:continue
                part="inside" if station>=j["from_m"] else "approach"
                heads=[]
                for lamp in row["signals"]:
                    dx,dy=lamp["x"]-x,lamp["y"]-y
                    angle=math.radians(yaw)
                    ahead=dx*math.cos(angle)+dy*math.sin(angle)
                    lateral=-dx*math.sin(angle)+dy*math.cos(angle)
                    heading=abs((lamp["governs_yaw"]-yaw+180)%360-180)
                    if (lamp.get("source")=="camera" and lamp.get("view")=="front" and lamp.get("age_s")==0
                            and lamp.get("confidence",0)>=.9 and lamp["state"]=="green"
                            and natives.get(lamp["map_id"])=="green" and heading<=20
                            and 0<ahead<=45 and abs(lateral)<=12):
                        heads.append(lamp["map_id"])
                out[str(j["id"])][part].append(dict(frame=row["frame"],time_s=row["sim_time"],kmh=row["speed_mps"]*3.6,
                                                      request=row["speed_request_mps"]*3.6,ids=sorted(set(heads)),
                                                      driver_gap=None if controller is None else float(controller[1][i]),
                                                      driver_red=None if controller is None else float(controller[2][i])))
            if controller is not None and controller[0][i]!=row["frame"]:
                raise ValueError("Policy features do not align with visual exposure")
    def summarize(cells):
        hits=[c for c in cells if c["ids"]]
        runs=[]
        for c in hits:
            if not runs or runs[-1][-1]["frame"]+1!=c["frame"]:runs.append([])
            runs[-1].append(c)
        median=lambda field:round(statistics.median(c[field] for c in hits),3) if hits else None
        governing=[c for c in hits if c["driver_gap"] is not None and c["driver_gap"]<59]
        return dict(samples=len(cells),fresh_confirmed_green_samples=len(hits),matched_seconds=round(.1*len(hits),1),
                    median_kmh=median("kmh"),median_requested_kmh=median("request"),
                    simultaneously_driver_governing_samples=len(governing) if controller is not None else None,
                    red_range_on_simultaneous_governing=[round(min(c["driver_red"] for c in governing),3),
                                                         round(max(c["driver_red"] for c in governing),3)] if governing else None,
                    map_ids=sorted({h for c in hits for h in c["ids"]}),
                    stretches=[dict(first_frame=r[0]["frame"],last_frame=r[-1]["frame"],
                                    from_s=r[0]["time_s"],to_s=r[-1]["time_s"]) for r in runs])
    for dest in out.values():
        for group in ("approach","inside"):
            dest[group]=summarize(dest[group])
    return dict(capture=capture["capture_id"],criteria="Fresh front camera, confidence >=0.9, mapped head and native ID both green, governs heading within 20 degrees, 0-45 m forward, <=12 m lateral. Approach is 18 m before fixed entry. Mid-junction views can lose the lamp overhead. No exact lane-ID association in this journal.",junctions=out)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",type=Path);p.add_argument("audit",type=Path)
    p.add_argument("output",type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(assess(a.stage,a.audit),indent=2)+"\n",encoding="utf8")
