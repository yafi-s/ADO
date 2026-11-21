"""Fit a camera color classifier on paired CARLA signal crops; hold out whole poles."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .signal_observer import COLOR_NAMES, housing_features


def fit(X, y):
    mean = X.mean(0)
    scale = np.maximum(.4, X.std(0))
    z = np.column_stack((np.clip((X-mean)/scale, -8, 8), np.ones(len(X))))
    labels = np.eye(len(COLOR_NAMES))[y]
    def objective(w):
        mat=w.reshape(z.shape[1],-1)
        pred=z @ mat
        pred-=pred.max(1,keepdims=True)
        exp=np.exp(pred)
        pr=exp/exp.sum(1,keepdims=True)
        err=-np.log(np.maximum(pr[np.arange(len(y)),y],1e-12)).sum()
        grad=z.T @ (pr-labels)
        grad[:-1]+=1.5*mat[:-1]
        return err+.75*np.sum(mat[:-1]**2), grad.ravel()
    result=minimize(objective,np.zeros((z.shape[1],4)).ravel(),jac=True,method="L-BFGS-B",options={"maxiter":600})
    if not result.success: raise RuntimeError(result.message)
    coef=result.x.reshape(z.shape[1],4)
    return dict(mean=mean,scale=scale,weights=coef[:-1],intercept=coef[-1],threshold=np.array(.85))


def evaluate(params, X, y):
    logits=np.clip((X-params["mean"])/params["scale"],-8,8) @ params["weights"] + params["intercept"]
    prob=np.exp(logits-logits.max(1,keepdims=True))
    prob/=prob.sum(1,keepdims=True)
    pred=prob.argmax(1)
    keep=(prob.max(1)>=params["threshold"]) & (pred!=0)
    return dict(crops=len(y), emitted=int(keep.sum()),correct=int((keep & (pred==y)).sum()),
                abstained=int((~keep).sum()), off_emitted=int((keep & (y==0)).sum()),
                confusion=np.bincount((4*y+pred)[keep],minlength=16).reshape(4,4).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    Xs,ys,group=[],[],[]
    for path in args.inputs:
        with np.load(path,allow_pickle=False) as d:
            Xs.extend(housing_features(c) for c in d["crops"])
            ys.extend(COLOR_NAMES.index(str(k)) for k in d["states"])
            group.extend(d["pole"])
    X=np.asarray(Xs); y=np.asarray(ys); group=np.asarray(group)
    test_ids=sorted(set(group))[1::3];test=np.isin(group,test_ids)
    params=fit(X[~test],y[~test])
    results=evaluate(params,X[test],y[test])
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/"colors.npz",**params)
    report=dict(classes=COLOR_NAMES, threshold=float(params["threshold"]), train_poles=sorted(set(group[~test])),
                heldout_poles=test_ids,heldout=results, input_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs])
    (args.output/"evaluation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report))


if __name__=="__main__":main()
