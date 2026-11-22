"""Fit near and upcoming body curvature using a previous native city drive."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from . import behavior_v2, city_maneuvers
from .city_maneuvers import BEND_LENGTHS, ManeuverModel, VERSION


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def samples(rows):
    station=np.array([r["progress_m"] for r in rows])
    yaw=np.unwrap(np.deg2rad([r["ego_pose"]["yaw"] for r in rows]))
    # Resample by travelled metres. A long stop contributes no repeated votes.
    keep=np.r_[True,np.diff(station)>.001]
    unique=np.maximum.accumulate(station[keep])
    grid=np.arange(0.,np.max(station)-23.,1.)
    angles=np.interp(np.arange(0.,np.max(station),1.),unique,yaw[keep])
    # Body yaw change over 3m suppresses suspension jitter. Quantile targets
    # include most of each window and avoid fitting isolated heading glitches.
    local=np.abs(angles[3:]-angles[:-3])/3.
    X=[];Y=[]
    for start in grid:
        idx=int(np.searchsorted(np.maximum.accumulate(station),start))
        path=np.asarray(rows[idx]["plan_xy"],dtype=float)
        bends=np.abs(behavior_v2._map_cues(path,None)[:3])/BEND_LENGTHS
        X.append([1.,*bends])
        i=int(start)
        Y.append([np.quantile(local[i:i+7],.85),np.quantile(local[i+7:i+20],.85)])
    return np.asarray(X),np.asarray(Y)


def fit(x,y):
    # Nonnegative quantile regression: higher curvature in a navigation
    # segment may only increase the estimated lateral demand.
    coefficient=np.zeros(x.shape[1])
    for _ in range(250):
        previous=coefficient.copy()
        for j,column in enumerate(x.T):
            supported=column>1.e-6
            residual=y-x@coefficient+column*coefficient[j]
            ratios=residual[supported]/column[supported]
            weights=column[supported]
            order=np.argsort(ratios)
            quantile=np.searchsorted(np.cumsum(weights[order]),.8*np.sum(weights))
            coefficient[j]=max(0.,ratios[order[quantile]])
        if np.max(abs(coefficient-previous))<1.e-7:
            break
    return coefficient

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drive",type=Path,default=Path("web/media/city-life/replay.json"))
    p.add_argument("--base",type=Path,default=Path("models/city-progress/candidate.npz"))
    p.add_argument("--stops",type=Path,default=Path("results/city-maneuvers/speed/stopping-envelope.json"))
    p.add_argument("--output",type=Path,default=Path("models/city-maneuvers"))
    args=p.parse_args()
    rows=json.loads(args.drive.read_bytes())
    if any(r["collision_count"] for r in rows) or len(rows)<1000:
        raise ValueError("Expected a complete, collision-free native city drive")
    x,y=samples(rows)
    # Last 25% ordered route distance is withheld and receives no coefficient fit.
    n=int(len(x)*.75)
    curve=np.asarray([fit(x[:n],y[:n,k]) for k in range(2)])
    braking=json.loads(args.stops.read_bytes())
    stopping=[braking[k] for k in ("delay_s","quadratic_s2_per_m","guard_m","max_fit_speed_mps")]
    for trial in braking["trials"]:
        margin=trial.get("guarded_margin_m")
        if margin is not None and margin<0.:
            raise ValueError("Native guarded stopping distance is too short")
    overlap=[]
    for trial in braking["trials"]:
        if trial["split"] not in ("fit","held_out","grip_stress"):
            continue
        path=args.stops.parent/trial["trace"]
        if digest(path)!=trial["trace_sha256"]:
            raise ValueError("Native braking trace differs from its calibration record")
        with gzip.open(path,"rt",encoding="utf8") as stream:
            rows=[json.loads(line) for line in stream]
        final=rows[-1]["progress_m"]
        margins=[stopping[1]*r["speed_mps"]**2+stopping[2]-(final-r["progress_m"])
                 for r in rows if r["phase"]=="brake" and 7.75<=r["speed_mps"]<=11.5]
        if len(margins)<5 or min(margins)<0.:
            raise ValueError("Native residual stopping distances do not support urban handback")
        overlap.append(dict(trial=trial["name"],frames=len(margins),min_remaining_margin_m=min(margins)))
    model=ManeuverModel(args.base.read_bytes(),curve,stopping)
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"candidate.npz"
    model.save(path)
    np.testing.assert_array_equal(ManeuverModel.load(path).curve,curve)
    error=x[n:]@curve.T-y[n:]
    report=dict(schema=VERSION,revision=1,checkpoint_sha256=digest(path),base_sha256=digest(args.base),
                inputs={str(q).replace("\\","/"):digest(q) for q in (args.drive,args.base,args.stops,Path("crosswalk/city_open_road.py"),Path("crosswalk/speed_signs.py"),
                    Path("crosswalk/city_maneuvers.py"),Path("crosswalk/city_maneuvers_fit.py"))},
                camera_and_range="At runtime only current and preceding native images, range scans, navigation and odometry.",
                fit="Nonnegative 0.8 quantile of measured absolute yaw per route metre. Upcoming travel is a training label only.",
                near_window_m=[0,7],upcoming_window_m=[7,20],samples=len(x),fit_metres=n,holdout_metres=len(x)-n,
                holdout_median_absolute_error_per_m=np.median(abs(error),axis=0).tolist(),
                holdout_coverage=(error>=0.).mean(axis=0).tolist(), coefficients=curve.tolist(),
                urban_request_ceiling_kmh=39.6,max_operating_request_kmh=3.6*min(stopping[-1],60./3.6),
                high_speed_stopping=stopping,urban_handback_validation=overlap,
                stopping_calibration_tested_to_mps=model.stopping[-1])
    (args.output/"fit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report))
if __name__=="__main__":
    main()
