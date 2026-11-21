"""Town03 learned speed planner on causal on-board measurements and map navigation."""
import argparse
from collections import deque
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from . import behavior_v2 as v2
from .lamp import LampExpert

VERSION = "ado-city-policy-v1"
FEATURE_NAMES = (*v2.FEATURE_NAMES, "rgb_lamp_red_probability", "mapped_lamp_gap_m")
INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
APPROACH = ("junction_gap", "bend_7_13", "bend_13_25", "bend_25_40", "oncoming_gap",
            "oncoming_speed", "oncoming_near_gap", "oncoming_gap_recent", "oncoming_near_gap_recent",
            "rgb_lamp_red_probability", "mapped_lamp_gap_m")
CTX_INDEX = [INDEX[name] for name in APPROACH]


def clearance(x, radar_weight):
    """Learn the relative depth weight on a radar surface closer than the roof scan."""
    x = np.asarray(x)
    laser = x[..., INDEX["path_body_lidar"]]
    radar = x[..., INDEX["path_body_radar"]]
    return laser + radar_weight * (np.minimum(laser, radar) - laser)


def context(x, light_window):
    recent = x[INDEX["rgb_lamp_red_probability"]]
    light_window.append(recent)
    return np.r_[x[CTX_INDEX], np.mean(light_window), np.max(light_window)]


class CityModel:
    inputs = "fusion"
    lamp_expert = None

    def __init__(self, distances, requests, radar_weight, forest, steering, feedback, classifier):
        self.distances = np.asarray(distances)
        self.requests = np.asarray(requests)
        self.radar_weight = float(radar_weight)
        self.forest = forest
        self.steering = np.asarray(steering)
        self.feedback = np.asarray(feedback)
        self.sensor_lamp_classifier = classifier
        self.light_window = deque(maxlen=12)
        self.last_speed_request = None
        if (self.distances.ndim != 1 or not len(self.distances) or self.requests.shape != self.distances.shape
                or not np.isfinite(self.distances).all() or not np.isfinite(self.requests).all()
                or np.any(np.diff(self.distances) <= 0.) or np.any(np.diff(self.requests) < -1.e-6)
                or self.requests.min() < 0. or self.requests.max() > 8. or not 0. <= self.radar_weight <= 1.):
            raise ValueError("invalid city clearance response")

    def new_observer(self):
        self.light_window.clear()
        self.last_speed_request = None
        return v2.Observer()

    def _plan(self, x):
        x = np.asarray(x)
        if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
            raise ValueError("expected 40 measured city navigation and sensor values")
        body = np.interp(clearance(x,self.radar_weight), self.distances, self.requests)
        approach = context(x,self.light_window)
        # The frozen v2 container validates portable forest links. Split indices
        # are local to the city's map, Doppler and visible-lamp features.
        starts, splits, thresholds, left, right, values = self.forest
        nodes = starts.copy()
        active = left[nodes] >= 0
        while active.any():
            n = nodes[active]
            nodes[active] = np.where(approach[splits[n]] <= thresholds[n], left[n], right[n])
            active = left[nodes] >= 0
        return float(min(body, values[nodes].mean()))

    def predict(self,x):
        requested = self._plan(x)
        self.last_speed_request = float(requested)
        nav = np.asarray(x)[v2.STEER_IDX]
        steer = float(np.clip(v2._steering_basis(nav) @ self.steering, -1., 1.))
        pedal = float(np.clip(v2._feedback_features(requested, nav[0]) @ self.feedback, -1., 1.))
        return steer, max(0.,pedal), max(0.,-pedal)

    def save(self, path):
        names = ("starts","splits","thresholds","left","right","values")
        fields = dict(version=VERSION, feature_names=FEATURE_NAMES, distances=self.distances,
                      requests=self.requests, radar_weight=self.radar_weight, steering=self.steering,
                      feedback=self.feedback, **dict(zip(names,self.forest)))
        fields.update(self.sensor_lamp_classifier.arrays("lamp_"))
        with Path(path).open("wb") as f:
            np.savez_compressed(f,**fields)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if str(d["version"].item()) != VERSION or tuple(d["feature_names"]) != FEATURE_NAMES:
                raise ValueError("incompatible city model")
            forest = tuple(d[n] for n in ("starts","splits","thresholds","left","right","values"))
            v2.BehaviorModel(*forest,d["steering"],d["feedback"],None)
            if np.any(forest[1][forest[3]>=0] >= len(CTX_INDEX)+2):
                raise ValueError("invalid city context split")
            return cls(d["distances"],d["requests"],d["radar_weight"],forest,
                       d["steering"],d["feedback"],LampExpert.from_arrays(d,"lamp_"))


def town_records(path):
    result=[]
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            with np.load(io.BytesIO(archive.read(name)),allow_pickle=False) as d:
                if tuple(d["feature_names"]) != v2.FEATURE_NAMES:
                    raise ValueError(f"incompatible sensor columns in {name}")
                X = np.column_stack((d["X"],np.zeros(len(d["X"])),np.full(len(d["X"]),60.)))
                event = d["scenario"].astype("U30")
                # Relabel the same recorded actor clearance under the *city*
                # instructor's stopping-distance objective, only during fitting.
                goal = np.minimum(5.2,np.sqrt(3.6*np.maximum(0.,d["expert_gap_m"]-4.)))
                approach = np.minimum(5.2,d["teacher_speed"])
                if event[0]=="left":
                    approach = np.minimum(approach,goal)
                result.append(dict(X=X,kind=event,goal=goal,approach=approach,seed=int(d["seed"][0]),
                                   city=False,name=name,trim=0))
    return result


def city_record(path):
    path=Path(path)
    with np.load(path,allow_pickle=False) as d:
        if tuple(d["feature_names"])!=FEATURE_NAMES:
            raise ValueError("incompatible city observation schema")
        X,goal,frame=d["X"],d["teacher_speed"],d["frame"]
        seed=int(d["seed"].flat[0])
    with gzip.open(path.with_name("trace.jsonl.gz"),"rt",encoding="utf-8") as f:
        rows=[json.loads(line) for line in f]
    if not np.array_equal(frame,[r["frame"] for r in rows]):
        raise ValueError("city observation and instructor trace alignment differs")
    event=np.array([r.get("active_event") or "clear" for r in rows])
    # The instructor can ignore an actor once its scripted encounter clears.
    # Last three seconds of a collision-terminated demonstration carry
    # contradictory clear-road labels; exclude that tail from both fit heads.
    summary=json.loads(path.with_name("summary.json").read_bytes())
    trim=min(30,len(X)-1) if summary["outcome"]=="collision" else 0
    stop=len(X)-trim
    gap=np.array([r["reference_gap_m"] for r in rows])
    # Non-obstacle context learns free cruise/turn speed from unobstructed
    # rows and learns red/yield from their actual demonstrated speed requests.
    approach=goal.copy()
    physical=~np.isin(event,["signal","left"])
    approach[physical & (gap<18.)]=np.nan
    return dict(X=X[:stop],kind=event[:stop],goal=goal[:stop],approach=approach[:stop],
                seed=seed,city=True,name=path.parent.name,trim=trim)


def fit(records,base):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.isotonic import IsotonicRegression
    body_x,body_y,body_w=[],[],[]
    ctx_x,ctx_y,ctx_w=[],[],[]
    for r in records:
        X=r["X"]
        goal=r["goal"]
        approaches=r["approach"]
        if (X.shape!=(len(goal),len(FEATURE_NAMES)) or not np.isfinite(X).all()
                or not np.isfinite(goal).all() or np.any(goal<0.) or np.any(goal>8.)):
            raise ValueError("invalid city training input")
        physical=~np.isin(r["kind"],["signal","left"])
        # Far points teach unconstrained speed. A bend, signal or oncoming
        # vehicle has its own observed approach component.
        body=goal.copy()
        body[goal>=3.5]=np.where(X[goal>=3.5,INDEX["path_body_lidar"]]>16.,5.2,goal[goal>=3.5])
        body_x.append(X[physical])
        body_y.append(body[physical])
        body_w.append(np.where(body[physical]<1.,5.,1.)*(2. if r["city"] else 1.))
        window=deque(maxlen=12)
        ctxt=np.asarray([context(x,window) for x in X])
        valid=np.isfinite(approaches)
        ctx_x.append(ctxt[valid])
        ctx_y.append(approaches[valid])
        ctx_w.append(np.where(approaches[valid]<1.,3.,1.)*(5. if r["city"] else 1.))
    bx=np.vstack(body_x)
    by,bw=np.concatenate(body_y),np.concatenate(body_w)
    # Fit sensor weighting and all speed knots directly from paired
    # observations and teacher requests. Monotonic interpolation preserves a
    # learned standstill request at smaller previously unobserved clearances.
    best=None
    for weight in np.linspace(0.,1.,9):
        distance=clearance(bx,weight)
        fitted=IsotonicRegression(y_min=0.,out_of_bounds="clip").fit(distance,by,sample_weight=bw)
        squared=np.average((fitted.predict(distance)-by)**2,weights=bw)
        if best is None or squared<best[0]:
            best=(squared,float(weight),fitted)
    _,weight,iso=best
    ctx=ExtraTreesRegressor(n_estimators=32,max_depth=10,min_samples_leaf=3,
                            max_features=1.,random_state=112073,n_jobs=1)
    ctx.fit(np.vstack(ctx_x),np.concatenate(ctx_y),sample_weight=np.concatenate(ctx_w))
    return CityModel(iso.X_thresholds_,iso.y_thresholds_,weight,v2._forest_arrays(ctx),
                     base.steering,base.feedback,base.lamp_expert)


def score(model, records):
    pred=[]; truth=[]; source=[]
    for rec in records:
        model.new_observer()
        pred.extend(model._plan(x) for x in rec["X"])
        truth.extend(np.minimum(rec["goal"],np.nan_to_num(rec["approach"],nan=8.)))
        source.extend(rec["kind"])
    pred,truth=np.asarray(pred),np.asarray(truth)
    result={}
    for label,mask in (("all",np.ones(len(pred),dtype=bool)),("instructor_stop",truth<.35),
                       ("instructor_cruise",truth>3.5)):
        if mask.any():
            result[label]=dict(rows=int(mask.sum()),mae_mps=float(np.mean(abs(pred[mask]-truth[mask]))),
                               median_request_mps=float(np.median(pred[mask])))
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--town01",type=Path,default=Path("data/behavior-v2/lessons.zip"))
    cli.add_argument("--city",type=Path,nargs="+",required=True)
    cli.add_argument("--base",type=Path,default=Path("models/behavior-v2/candidate.npz"))
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--report",type=Path,required=True)
    args=cli.parse_args()
    town=town_records(args.town01)
    city=[city_record(path) for path in args.city]
    base=v2.BehaviorModel.load(args.base)
    hold=[rec for rec in town if rec["seed"]==48]
    train=[rec for rec in town if rec["seed"]!=48]
    # When a second city drive exists it is withheld as a whole. One drive
    # permits only a held-city-episode transfer check trained on Town01.
    cv_city=city[-1:]
    model=fit(train+city[:-1],base)
    hold_report=dict(town01_seed48=score(model,hold),withheld_city=score(model,cv_city))
    model=fit(town+city,base)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    model.save(args.out)
    check=CityModel.load(args.out)
    for x in city[0]["X"][:20]:
        np.testing.assert_allclose(check.predict(x),model.predict(x),atol=1.e-7)
    files=[args.town01,args.base,Path(__file__),*args.city,
           *[p.with_name("trace.jsonl.gz") for p in args.city],*[p.with_name("summary.json") for p in args.city]]
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    import platform, scipy, sklearn
    report=dict(version=VERSION,model_sha256=digest(args.out),model_bytes=args.out.stat().st_size,
                dependencies=dict(python=platform.python_version(),numpy=np.__version__,
                                  scipy=scipy.__version__,sklearn=sklearn.__version__),
                holdout_before_refit=hold_report,validation_city_episodes=[r["name"] for r in cv_city],
                town01_rows=sum(len(r["X"]) for r in town),city_fit=[
                    dict(episode=r["name"],seed=r["seed"],rows=len(r["X"]),excluded_collision_tail=r["trim"]) for r in city],
                clearance_weight=model.radar_weight,clearance_response_mps={str(g):float(np.interp(g,model.distances,model.requests)) for g in (0,1,2,3,4,5,6,8,10,15,25,36)},
                sources=[dict(name=str(p).replace("\\","/"),sha256=digest(p)) for p in files])
    args.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report))


if __name__=="__main__":
    main()
