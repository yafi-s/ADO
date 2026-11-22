"""Translate actual bracketing roof sweeps and retain the current optical stop bound."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from tools.maneuver_parallel_flank_candidate import ParallelFlank,flank_scan,parent,v2,LIDAR_Z


def compute(folder):
    frame=2777
    with gzip.open(folder/"trace.jsonl.gz","rt") as stream:
        row=next(r for r in map(json.loads,stream) if r["frame"]==frame)
    with np.load(folder/"observations.npz") as tape:
        x=dict(zip(tape["feature_names"],tape["features"][tape["frame"]==frame][0]))
    a=row["tactical_pass"]
    path=np.vstack(([0.,0.],[[x[f"route_{k}m_x_m"],x[f"route_{k}m_y_m"]] for k in (3,7,13)]))
    # The focal native segment is straight. The measured navigation fixes
    # its tangent and offset. The flank geometry uses its first 3/7 m points.
    h=(path[-1]-path[-2])/np.linalg.norm(path[-1]-path[-2])
    path=np.vstack((path,*[path[-1]+d*h for d in (5.,12.,19.,27.)]))
    scans={}
    hashes={}
    for fr in range(2764,2779,2):
        p=folder/"sensor-witness"/f"{fr}.npz"
        hashes[str(fr)]=parent.sha(p)
        with np.load(p) as saved:
            assert saved["frame"]==fr
            scans[fr]=saved["lidar"]
    sweep=[]
    for dx in range(9):
        model=ParallelFlank.load(parent.MODEL)
        model.new_observer()
        model.tactical_speed_limit=0.
        b=dict(a,corridor_station_m=a["corridor_station_m"]+dx,target_front_m=max(0.,a["target_front_m"]-dx),
               raw_source_gap_m=a["raw_source_gap_m"]-dx,source_gap_m=a["source_gap_m"]-dx)
        features=dict(x,verified_cross_sight_m=x["verified_cross_sight_m"]-dx,
                      camera_stop_horizon_m=max(0.,x["camera_stop_horizon_m"]-dx))
        for k in ("path_body_lidar","path_body_radar","path_body_lidar_recent","path_body_radar_recent","long_body_gap_m"):
            features[k]=max(0.,x[k]-dx)
        closest=[]
        margins=[]
        for fr,original in scans.items():
            cloud=original.copy()
            cloud[:,:2]-=dx*h
            w=flank_scan(cloud,path,b)
            model.flank_witness(fr,b["source_lane"],w)
            if fr not in (2776,2778):
                continue
            sx,sy,sz=original[:,:3].T
            z=sz+LIDAR_Z
            ga,gb,gc=v2._ground_plane(sx,sy,z)
            height=z-sx*ga-sy*gb-gc
            side=(height>.28)&(height<3.2)&(sy<-.93)&(sy>-5.6)&(sx>7.)&(sx<15.)
            ex=np.maximum(abs(sx[side]-dx)-2.46,0.)
            ey=np.maximum(abs(sy[side])-.93,0.)
            closest.append(float(np.min(np.hypot(ex,ey))))
            margins.append(float(np.min(ey)))
        original=parent.FreeSource.allowance(model,b,features)[0]
        cap,why=model.allowance(b,features)
        alternative,conditional_why=model.allowance(b,features|{"camera_stop_horizon_m":8.})
        sweep.append(dict(translation_m=dx,bracket_min_body_to_return_m=min(closest),
                          bracket_min_lateral_margin_m=min(margins),five_sweep_flank_supported=model.confirmed_flank is not None,
                          translated_camera_station_m=float(features["camera_stop_horizon_m"]),
                          frozen_source_cap_mps=original,alternate_cap_mps=cap,reason=why,
                          hypothetical_refreshed_camera_at_8m_cap_mps=alternative,hypothetical_reason=conditional_why))
    return dict(scope="Frozen geometry translated along the native source. No new optical view, actual future sweep or vehicle motion.",
                focal_frame=frame,raw_focal_saved=(folder/"sensor-witness"/f"{frame}.npz").exists(),
                bracketing_raw=[2776,2778],history_raw_sha256=hashes,
                inputs={k:parent.sha(folder/k) for k in ("trace.jsonl.gz","observations.npz")},
                candidate_sha256=parent.sha(Path(__file__).with_name("maneuver_parallel_flank_candidate.py")),
                parent_sha256=parent.sha(Path(__file__).with_name("maneuver_free_source_candidate.py")),
                measured_camera_station_m=float(x["camera_stop_horizon_m"]),source_surface_m=a["source_gap_m"],
                target_front_m=a["target_front_m"],bounds="Camera range decreases with translation. Separate final column substitutes a hypothetical fresh 8 m camera bound; it is never used by the driver.",
                sweep=sweep)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spaced",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=json.dumps(compute(a.spaced),indent=2,allow_nan=False)+"\n"
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf8") as f:
        f.write(result)
