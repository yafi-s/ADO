"""Short-horizon WOMD speed prior. Inference uses only NumPy and standard Python.

One input is constructed at the current observation. +x is forward and +y left.
Tracks contain estimated ground velocities resolved in that frame. The static
navigation-lane heading change is optional. For simulator research only.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

HORIZONS = (1, 2, 3)
FEATURE_NAMES = (
    "speed_mps","recent_acceleration","previous_acceleration",
    "lead_present","lead_forward_m","lead_offset_m","lead_relative_vx",
    "vru_present","vru_forward_m","vru_offset_m","vru_vx","vru_vy","vru_cyclist",
    "lane_valid","lane_bend_rad",
)
MODEL_VERSION = "womd-speed-ridge-v1"


@dataclass(frozen=True)
class ActorTrack:
    kind: str
    x: float
    y: float
    vx: float
    vy: float


def make_features(speed_history, tracks=(), lane_bend=0., lane_valid=False):
    history = np.asarray(speed_history, dtype=float)
    if history.shape != (3,) or not np.isfinite(history).all() or np.any(history < 0):
        raise ValueError("expected three nonnegative finite ego speeds at -1, -.5, 0 seconds")
    speed = history[-1]
    use_lane = bool(lane_valid) and bool(np.isfinite(lane_bend))
    bend = float(np.clip(lane_bend, -1.5, 1.5)) if use_lane else 0.
    lead, vru = None, None
    best_lead, best_vru = float("inf"), float("inf")
    for a in tracks:
        if not np.isfinite([a.x,a.y,a.vx,a.vy]).all() or not 0 < a.x <= 80 or abs(a.y)>15:
            continue
        # Local lane center within 15 m, then extend its terminal tangent.
        lateral = .5*bend/15*min(a.x,15)**2 + bend*max(a.x-15,0)
        offset = a.y - lateral
        if a.kind == "vehicle" and abs(offset)<2.25 and a.x < best_lead:
            lead, best_lead = (a,offset), a.x
        if a.kind in ("pedestrian","cyclist") and a.x <= 50 and abs(offset) < 12:
            # Rank nearby road users by expected lateral offset when ego reaches their x.
            dt = np.clip(a.x/max(speed-a.vx,1.),0.,3.)
            priority = a.x + 4*min(abs(offset),abs(offset + a.vy*dt))
            if priority < best_vru:
                vru, best_vru = (a,offset), priority
    result = np.zeros(len(FEATURE_NAMES),np.float32)
    result[:3] = [speed,2*(history[2]-history[1]),2*(history[1]-history[0])]
    if lead:
        a, offset = lead
        result[3:7] = [1,a.x,offset,a.vx-speed]
    if vru:
        a, offset = vru
        result[7:13] = [1,a.x,offset,a.vx,a.vy,a.kind=="cyclist"]
    result[13:] = [use_lane,bend]
    return result


def _basis(features, mode):
    x=np.atleast_2d(features).astype(np.float64)
    if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("invalid motion features")
    speed=np.clip(x[:,0],0,50)
    a=np.clip(x[:,1],-8,6)
    a0=np.clip(x[:,2],-8,6)
    hist=[np.ones_like(speed), speed/15, np.minimum(speed,2)/2, np.minimum(speed,8)/8,
          np.exp(-speed), a, a0, np.maximum(a,0), np.maximum(a0,0), a*np.minimum(speed,3)/3,
          (a-a0)*np.minimum(speed,1), np.clip(a,-.3,.3), np.clip(a0,-.3,.3),
          np.minimum(a,0)*np.minimum(speed/6,1), np.maximum(a,0)*np.minimum(speed/6,1)]
    if mode == "history":
        return np.column_stack(hist)
    if mode != "interaction":
        raise ValueError("unrecognized basis")
    lead,d,lat,rv=x[:,3:7].T
    vru,vx_pos,vy_pos,vx,vy,cyclist=x[:,7:13].T
    lane,bend=x[:,13:].T
    rv=np.clip(rv,-30,30)
    headway=d/np.maximum(speed,1)
    closing=np.clip(-rv,0,30)
    near=np.exp(-d/20)
    time=np.clip(vx_pos/np.maximum(speed-vx,1),0,3)
    clearance=np.minimum(abs(vy_pos),abs(vy_pos+vy*time))
    vru_risk=np.exp(-vx_pos/20-clearance/2.5)
    context=[lead,lead*near,lead*rv/10,lead*rv*near/5,lead*closing/8,
             lead*closing/np.maximum(d,4),lead*np.exp(-headway/2),
             lead*np.clip((speed+rv)/10,0,3)*near,lead*near*np.minimum(speed/5,1),
             lead*near*abs(lat)/2,lead*near*np.maximum(a,0),
             vru,vru*vru_risk,vru*vru_risk*speed/10,vru*np.exp(-vx_pos/20),
             vru*cyclist*vru_risk, vru*np.clip(-np.sign(vy_pos)*vy,-6,6)*vru_risk,
             lane,lane*abs(bend),lane*abs(bend)*speed/10]
    return np.column_stack(hist+context)


class MotionSpeedModel:
    def __init__(self, mode, weight):
        self.mode=mode
        self.weight=np.asarray(weight,dtype=float)
        size=_basis(np.zeros(len(FEATURE_NAMES)),mode).shape[1]
        if self.weight.shape!=(size,len(HORIZONS)) or not np.isfinite(self.weight).all():
            raise ValueError("invalid speed model coefficients")

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            if (str(data["version"])!=MODEL_VERSION or tuple(data["features"])!=FEATURE_NAMES
                    or tuple(data["horizons"])!=HORIZONS):
                raise ValueError("unsupported motion speed model")
            return cls(str(data["mode"]),data["weight"])

    def save(self,path):
        with Path(path).open("wb") as fh:
            np.savez_compressed(fh,version=MODEL_VERSION,mode=self.mode,weight=self.weight,
                                features=FEATURE_NAMES,horizons=HORIZONS)

    def predict(self,features):
        x=np.asarray(features)
        pred=np.maximum(0,np.atleast_2d(x)[:,0,None] + _basis(x,self.mode)@self.weight)
        return pred[0] if x.ndim==1 else pred


def _fit(x,y,mode,alpha):
    z=_basis(x,mode)
    y=y-x[:,:1]
    # Scale on fitting data only. The intercept has separate centering below.
    mean=z[:,1:].mean(axis=0)
    scale=np.maximum(z[:,1:].std(axis=0),.05)
    inputs=(z[:,1:]-mean)/scale
    # Robust IRLS fits a smooth absolute error around the current speed.
    weight=np.zeros((z.shape[1],3))
    matrix=np.column_stack([z[:,0],inputs])
    penalty=np.eye(z.shape[1])*alpha
    penalty[0,0]=0
    for h in range(3):
        sample=np.ones(len(z))
        for _ in range(6):
            wt=matrix.T*sample
            coef=np.linalg.solve(wt@matrix+penalty, wt@y[:,h])
            sample=1/np.maximum(0.25,abs(y[:,h]-matrix@coef))
        weight[1:,h]=coef[1:]/scale
        weight[0,h]=coef[0]-mean@weight[1:,h]
    return MotionSpeedModel(mode,weight)


def _load(data):
    manifest=json.loads((data/"manifest.json").read_text())
    splits={"training":[],"validation":[]}
    seen=set()
    for receipt in manifest["shards"]:
        path=data/receipt["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt["features_sha256"]:
            raise ValueError("feature hash mismatch")
        with np.load(path,allow_pickle=False) as batch:
            ids=batch["scenario_id"]
            for i in ids:
                if i in seen:
                    raise ValueError("repeated scenario ID in training or evaluation")
                seen.add(i)
            x,y=batch["x"],batch["y"]
            if (x.shape!=(len(ids),len(FEATURE_NAMES)) or y.shape!=(len(ids),3) or
                    not np.isfinite(x).all() or not np.isfinite(y).all()):
                raise ValueError("invalid feature batch")
            splits[receipt["job"]["split"]].append((ids,x,y))
    return {s:tuple(np.concatenate(c) for c in zip(*batches,strict=True)) for s,batches in splits.items()}


def _tune(ids,x,y,mode):
    # All tuning scenarios are from the official training split.
    dev=np.array([int.from_bytes(hashlib.sha256(("speed-inner-v1/"+i).encode()).digest()[:8],"big")%5==0
                  for i in ids])
    if min(dev.sum(),(~dev).sum())<4:
        raise ValueError("too few official training scenarios")
    alphas=(1., 10., 50., 150.)
    scores=[]
    for alpha in alphas:
        model=_fit(x[~dev],y[~dev],mode,alpha)
        scores.append(float(np.abs(model.predict(x[dev])-y[dev]).mean()))
    alpha=alphas[int(np.argmin(scores))]
    return _fit(x,y,mode,alpha),dict(alpha=alpha,inner_mae_mps=scores,grid=alphas,
                                      inner_training=int((~dev).sum()),inner_dev=int(dev.sum()))


def _paired(delta):
    rng=np.random.default_rng(8131)
    means=[np.mean(rng.choice(delta,len(delta),replace=True)) for _ in range(2000)]
    return dict(mean=float(np.mean(delta)),ci95=np.percentile(means,[2.5,97.5]).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    splits=_load(args.data)
    ids,x,y=splits["training"]
    _,xv,yv=splits["validation"]
    models,tuning={},{}
    for mode in ("history","interaction"):
        models[mode],tuning[mode]=_tune(ids,x,y,mode)
    pred={m:model.predict(xv) for m,model in models.items()}
    pred["constant_velocity"]=np.repeat(xv[:,:1],3,axis=1)
    pred["constant_acceleration"]=np.maximum(0,xv[:,:1]+np.asarray(HORIZONS)*(xv[:,1:2]+xv[:,2:3])/2)
    errors={m:np.abs(yp-yv) for m,yp in pred.items()}
    masks={"all":np.ones(len(xv),bool),"lead":xv[:,3]>0,"vru":xv[:,7]>0,
           "near_stop":xv[:,0]<.5,"moving":xv[:,0]>=.5,"lane_valid":xv[:,13]>0,
           "logged_deceleration_gt_1mps":yv[:,-1]<xv[:,0]-1}
    metrics={name:dict(scenarios=int(mask.sum()),
                       mae_mps_by_horizon={m:e[mask].mean(axis=0).tolist() for m,e in errors.items()},
                       mean_mae_mps={m:float(e[mask].mean()) for m,e in errors.items()})
             for name,mask in masks.items() if mask.any()}
    differences={m:_paired((errors["interaction"]-e).mean(axis=1))
                 for m,e in errors.items() if m!="interaction"}
    root=args.out
    root.mkdir(parents=True,exist_ok=True)
    path=root/"motion-speed.npz"
    models["interaction"].save(path)
    if not np.array_equal(MotionSpeedModel.load(path).predict(xv),pred["interaction"]):
        raise ValueError("portable round-trip prediction mismatch")
    manifest=json.loads((args.data/"manifest.json").read_text())
    results=dict(dataset="Waymo Open Motion Dataset v1.3.1",scope="logged official validation SDC speed",
                 train_scenarios=len(x),validation_scenarios=len(xv),horizons_s=HORIZONS,
                 tuning=tuning,metrics=metrics,paired_scenario_mae_delta_mps=differences,
                 assumptions=["Waymo actor tracks are offline annotations; sensor coverage and noise differ.",
                              "Current lane is inferred from static map and current pose; route intent is unknown.",
                              "One official current timestamp per scenario. Logged future appears only in labels.",
                              "Scenario-bootstrap intervals do not account for possible overlapping source windows.",
                              "Validation uses a bounded shard sample and measures logged prediction, not driving safety."],
                 source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("motion_extract.py","interaction.py")},
                 model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 data_manifest_sha256=hashlib.sha256((args.data/"manifest.json").read_bytes()).hexdigest(),
                 reserved_download_bytes=manifest["reserved_bytes"],
                 objects=[dict(**r["job"],prefix_sha256=r["prefix_sha256"],verified_bytes=r["bytes"])
                          for r in manifest["shards"]])
    (root/"evaluation.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n",encoding="utf8")
    agreement=Path(__file__).with_name("WAYMO_DATASET_AGREEMENT.txt")
    shutil.copyfile(agreement,root/agreement.name)
    (root/"NOTICE.md").write_text(
        "# Waymo Open Motion Dataset speed research model\n\n"
        "This model, associated materials and downstream use or modification are governed by the "
        "Waymo Dataset License Agreement for Non-Commercial Use, included alongside this artifact. "
        "The source is the Waymo Open Motion Dataset v1.3.1.\n\n"
        "```bibtex\n@misc{waymo_open_dataset,\n  title = {Waymo Open Dataset: An autonomous driving dataset},\n"
        "  website = {\\url{https://www.waymo.com/open}},\n  year = {2019-2025}\n}\n```\n",encoding="utf8")
    print(json.dumps(dict(training=len(x),validation=len(xv),mae=metrics["all"],paired=differences)),flush=True)


if __name__=="__main__":
    main()
