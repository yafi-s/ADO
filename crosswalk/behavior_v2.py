"""Causal route-relative CARLA observations and compact learned driving model.

All geometry uses ego-local metres (+x forward, +y right). Demo labels and
scenario names are deliberately absent from the observation interface.
"""

from collections import deque
from pathlib import Path

import numpy as np

from .policy import BUMPER_X, FEATURE_NAMES as V1_NAMES, LIDAR_Z, MAX_RANGE, NAV_NAMES
from .policy import RADAR_X, RADAR_Z, _returns
from .vision import summarize


VERSION = "ado-behavior-v2"
PREVIEW_DISTANCES = (18., 25., 32., 40.)
HORIZON = 36.
RANGE_CUES = ("path_core_lidar", "path_body_lidar", "path_body_radar", "left_edge_lidar",
              "right_edge_lidar", "radar_surface_speed", "oncoming_gap", "oncoming_speed",
              "oncoming_near_gap", "body_support")
CAMERA_CUES = tuple(f"rgb_{group}_{name}" for group in ("person", "vehicle")
                    for name in ("confidence", "height", "center_offset"))
MAP_CUES = ("bend_7_13", "bend_13_25", "bend_25_40", "junction_gap")
STATIC_NAMES = NAV_NAMES + ("ego_speed_mps",) + MAP_CUES + RANGE_CUES + CAMERA_CUES
MEMORY_CUES = ("path_core_lidar", "path_body_lidar", "path_body_radar", "right_edge_lidar",
               "oncoming_gap", "oncoming_near_gap", "rgb_person_height", "rgb_person_center_offset")
FEATURE_NAMES = STATIC_NAMES + tuple(f"{name}_recent" for name in MEMORY_CUES) + (
    "ego_acceleration", "gap_rate", "pedestrian_side_rate")
_STATIC_INDEX = {name: i for i, name in enumerate(STATIC_NAMES)}
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
_V1_INDEX = {name: i for i, name in enumerate(V1_NAMES)}
_HISTORY_IDX = np.array([_STATIC_INDEX[name] for name in MEMORY_CUES])


def _checked_route(local_route, preview):
    close = np.asarray(local_route, dtype=float).reshape(3, 2)
    if preview is None:
        tangent = (close[-1] - close[-2]) / max(.01, np.linalg.norm(close[-1] - close[-2]))
        far = close[-1] + (np.array(PREVIEW_DISTANCES)[:, None] - 13.) * tangent
    else:
        far = np.asarray(preview, dtype=float).reshape(4, 2)
    path = np.vstack(([0., 0.], close, far))
    if not np.isfinite(path).all():
        raise ValueError("nonfinite navigation points")
    return path


def _map_cues(path, gap):
    vectors = np.diff(path, axis=0)
    yaw = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return [yaw[2]-yaw[1], yaw[4]-yaw[2], yaw[6]-yaw[4],
            np.clip(60. if gap is None else float(gap), -3., 60.)]


def _ground_plane(x, y, z):
    """Estimate road height from the lower surface of road-sized spatial cells."""
    keep = (x > 1.) & (x < 27.) & (np.abs(y) < 5.) & (np.abs(z) < .9)
    if keep.sum() < 20:
        return np.zeros(3)
    cx, cy, cz = x[keep], y[keep], z[keep]
    ix = np.floor((cx-1.)/2.).astype(int)
    iy = np.floor((cy+5.)/2.).astype(int)
    cell = ix*5 + iy
    floor = np.full(65, np.inf)
    np.minimum.at(floor, cell, cz)
    present = np.flatnonzero(np.bincount(cell, minlength=65) >= 3)
    if len(present) < 8:
        return np.zeros(3)
    design = np.column_stack((present//5 * 2. + 2., present%5 * 2. - 4., np.ones(len(present))))
    fit = np.linalg.lstsq(design, floor[present], rcond=None)[0]
    # Road support can disappear inside a parked vehicle. Refit on the
    # dominant surface and bound extrapolation outside the measured region.
    residual = floor[present] - design @ fit
    inlier = np.abs(residual - np.median(residual)) < .16
    if inlier.sum() >= 6:
        fit = np.linalg.lstsq(design[inlier], floor[present[inlier]], rcond=None)[0]
    return np.clip(fit, (-.09, -.09, -.42), (.09, .09, .42))


def _project(x, y, path):
    """Nearest point on the planned polyline, signed cross-track and arc station."""
    best = np.full(len(x), np.inf)
    station, lateral = np.zeros(len(x)), np.zeros(len(x))
    start = 0.
    for origin, delta in zip(path, np.diff(path, axis=0)):
        length = float(np.linalg.norm(delta))
        if length < .02:
            continue
        c, s = delta/length
        dx, dy = x-origin[0], y-origin[1]
        along = dx*c + dy*s
        offset = dy*c - dx*s
        clipped = np.clip(along, 0., length)
        cost = (along-clipped)**2 + offset**2
        improve = cost < best
        station[improve] = start + clipped[improve]
        # A sample near a polyline endpoint needs Euclidean distance, too.
        lateral[improve] = np.copysign(np.sqrt(cost[improve]), offset[improve])
        best = np.minimum(best, cost)
        start += length
    return station - BUMPER_X, lateral


def _gap(gaps, mask, *, support=1):
    values = gaps[mask & (gaps >= 0) & (gaps < HORIZON)]
    if len(values) < support:
        return HORIZON
    if support == 1:
        return float(values.min())
    # Adjacent samples within 45 cm give independent support; a single beam
    # from the road or a tree need not become the nearest solid surface.
    values = np.sort(values)
    supported = np.flatnonzero(values[support-1:] - values[:len(values)-support+1] <= .45)
    return float(values[supported[0]]) if len(supported) else HORIZON


def _range_cues(lidar, radar, speed, path):
    cloud = _returns(lidar)
    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2] + LIDAR_Z
    ground = _ground_plane(x, y, z)
    def above_road(px, py, pz):
        height = pz - (ground[0]*px + ground[1]*py + ground[2])
        return (px >= -.5) & (px < HORIZON+BUMPER_X) & (np.abs(py) < 22.) & (height > .28) & (height < 2.5)
    keep = above_road(x, y, z)
    along, side = _project(x[keep], y[keep], path)
    core = _gap(along, np.abs(side) < .74, support=2)
    body = _gap(along, np.abs(side) < 1.08, support=2)
    left = _gap(along, (side > -2.5) & (side <= -1.08), support=2)
    right = _gap(along, (side < 2.5) & (side >= 1.08), support=2)
    strength = np.log1p(np.count_nonzero((np.abs(side)<1.08) & (np.abs(along-body)<.65)))

    velocity, az, alt, depth = _returns(radar).T
    horizontal = depth*np.cos(alt)
    rx, ry = horizontal*np.cos(az)+RADAR_X, horizontal*np.sin(az)
    rz = depth*np.sin(alt)+RADAR_Z
    keep = above_road(rx, ry, rz)
    rx, ry, velocity = rx[keep], ry[keep], velocity[keep]
    radial_speed = velocity + speed*np.cos(az[keep])*np.cos(alt[keep])
    ra, rs = _project(rx, ry, path)
    rbody = _gap(ra, np.abs(rs) < 1.08)
    near = (np.abs(ra-rbody) < .75) & (np.abs(rs)<1.08) & (ra < HORIZON)
    surface = float(np.median(radial_speed[near])) if near.any() else 0.
    # Observed Doppler is negative for closing traffic in CARLA 1.0. Separate
    # vehicles moving toward ego from a static facade approached by ego.
    opposing = (radial_speed < -1.6) & (ry < 1.5) & (ry > -12.)
    opposing_gap = _gap(rx-BUMPER_X, opposing)
    near_opp = opposing & (np.abs(rx-BUMPER_X-opposing_gap)<2.)
    opposing_speed = float(np.median(radial_speed[near_opp])) if near_opp.any() else 0.
    crossing = _gap(rx-BUMPER_X, opposing & (np.abs(ry)<5.))
    return [core, body, rbody, left, right, surface, opposing_gap, opposing_speed, crossing, strength]


class Observer:
    """Use a fresh instance for each continuous 10 Hz trajectory."""
    def __init__(self):
        self.history = deque(maxlen=8)

    def _record(self, static):
        static = np.asarray(static, np.float32)
        if static.shape != (len(STATIC_NAMES),) or not np.isfinite(static).all():
            raise ValueError("invalid behavior observation")
        old = self.history[0] if self.history else static
        dt = max(.1, len(self.history)*.1)
        recent = np.mean([row[_HISTORY_IDX] for row in (*self.history, static)], axis=0)
        def rate(name):
            idx = _STATIC_INDEX[name]
            return (static[idx]-old[idx])/dt
        extra = [np.clip(rate("ego_speed_mps"), -12., 8.), np.clip(rate("path_body_lidar"), -12., 12.),
                 np.clip(rate("rgb_person_center_offset"), -2., 2.)]
        self.history.append(static)
        return np.concatenate((static, recent, extra)).astype(np.float32)

    def observe(self, obs, detections, local_route, speed, *, route_preview=None, stopline_gap=None):
        path = _checked_route(local_route, route_preview)
        vector = [*path[1:4].ravel(), float(speed), *_map_cues(path, stopline_gap),
                  *_range_cues(obs["lidar"], obs["radar"], speed, path), *summarize(detections)]
        return self._record(vector)

def _legacy_row(raw, observer, preview=None, junction=None):
    """Old recordings retain ego-aligned gap, grid Doppler and camera boxes.
    Reconstruct only those measurements; new sweeps require fresh sensor data.
    """
    def v(name):
        return raw[_V1_INDEX[name]]
    route = [v(name) for name in NAV_NAMES]
    path = _checked_route(route, preview)
    speed = v("ego_speed_mps")
    lidar, radar = (min(HORIZON, v(name)) for name in ("lidar_gap_m", "radar_gap_m"))
    opposing = []
    for lo, hi in ((0,4),(4,8),(8,14),(14,22),(22,36)):
        for lane in ("left", "far_left", "center"):
            tail = f"{lo}_{hi}m_{lane}"
            if v("bev_radar_density_"+tail) <= 0:
                continue
            ground_speed = 15.*v("bev_radar_radial_velocity_"+tail) + speed
            if ground_speed < -1.6:
                opposing.append(((lo+hi)/2., ground_speed, lane))
    opp_gap, opp_speed, _ = min(opposing, default=(HORIZON, 0., ""))
    near_opp = min((a for a, _, lane in opposing if lane != "far_left"), default=HORIZON)
    lidar_support = np.log1p(min(64.,np.expm1(v("lidar_hits_log")))) if lidar < HORIZON else 0.
    ranges = [lidar, lidar, radar, min(HORIZON,v("left_lidar_gap_m")), min(HORIZON,v("right_lidar_gap_m")),
              v("radar_velocity_mps")+speed if radar < HORIZON else 0., opp_gap, opp_speed, near_opp, lidar_support]
    return observer._record([*route, speed, *_map_cues(path, junction), *ranges, *[v(n) for n in CAMERA_CUES]])


def _preview(route, progress, close):
    delta = np.diff(route, axis=0)
    arc = np.r_[0., np.cumsum(np.linalg.norm(delta, axis=1))]
    stations = progress[:,None] + np.array([3.,7.,*PREVIEW_DISTANCES])
    points = np.stack([np.interp(stations,arc,axis) for axis in route.T],axis=-1)
    last = delta[-1] / max(.001,np.linalg.norm(delta[-1]))
    points += np.maximum(stations-arc[-1],0.)[:,:,None] * last
    world = points[:,1] - points[:,0]
    local = close[:,2:4] - close[:,:2]
    angle = np.arctan2(local[:,1],local[:,0]) - np.arctan2(world[:,1],world[:,0])
    dx, dy = (points[:,2:] - points[:,0:1]).transpose(2,0,1)
    c, s = np.cos(angle)[:,None], np.sin(angle)[:,None]
    return np.stack((c*dx-s*dy+close[:,0:1], s*dx+c*dy+close[:,1:2]),axis=-1)


def read_legacy_labels(records, trace_dir):
    """Verify each localized teacher trajectory before joining its oracle labels."""
    import gzip
    import json
    ids, speed = records["episode"], records["X"][:,_V1_INDEX["ego_speed_mps"]]
    gaps, progress = np.empty(len(ids)), np.empty(len(ids))
    for episode in dict.fromkeys(ids):
        directory = Path(trace_dir)/episode
        info = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        if info["district"] != "train" or info["policy"] != "demonstrator":
            raise ValueError(f"expected train-district demonstration: {episode}")
        with gzip.open(directory/"trace.jsonl.gz","rt",encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream][:-1]
        mask = ids == episode
        if len(rows) != mask.sum() or not np.allclose(speed[mask], [r["speed_mps"] for r in rows], atol=.00051):
            raise ValueError(f"misaligned teacher observation: {episode}")
        gaps[mask] = [r["reference_gap_m"] for r in rows]
        progress[mask] = [r["progress_m"] for r in rows]
    return dict(expert_gap_m=gaps.astype("f"),progress_m=progress.astype("f"),episode=ids)


def archived_junctions(records, labels, xodr, trace_dir):
    """Read the saved OpenDRIVE locally; this never connects to a simulator."""
    import carla
    import hashlib
    import json
    from .world import _scene_routes
    raw = Path(xodr).read_bytes()
    road_map = carla.Map("Town01_Opt", raw.decode("utf-8-sig"))
    routes = np.split(records["route_xy"], np.cumsum(records["route_lengths"])[:-1])
    ids = records["episode"]
    result = np.full(len(ids), 60., dtype=np.float32)
    for name, route in zip(dict.fromkeys(ids), routes, strict=True):
        info = json.loads((Path(trace_dir)/name/"summary.json").read_text(encoding="utf-8"))
        # A nearest lane query can select the other road at a connector
        # endpoint. Replay the static route sampler and require every XY match.
        kind = info["scenario"]
        direction = "straight" if kind == "signal" else info["turn_direction"]
        for points, _, _ in _scene_routes(road_map, info["seed"], kind, "train", info["destination"], direction):
            points = points[:len(route)]
            xy = np.array([(p.transform.location.x, p.transform.location.y) for p in points])
            if xy.shape == route.shape and np.allclose(xy,route,rtol=0.,atol=.003):
                break
        else:
            raise ValueError(f"archived route differs from supplied OpenDRIVE geometry: {name}")
        junctions = {point.junction_id for point in points if point.is_junction}
        if junctions != set(info["junction_ids"]):
            raise ValueError(f"static map junctions disagree with archived training route {name}")
        arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(route,axis=0),axis=1))]
        is_junction = np.array([point.is_junction for point in points])
        starts = np.flatnonzero(is_junction & ~np.r_[False,is_junction[:-1]])
        ends = np.flatnonzero(is_junction & ~np.r_[is_junction[1:],False])+1
        idx = np.flatnonzero(ids==name)
        progress = labels["progress_m"][idx]
        for start, end in reversed(list(zip(starts, ends))):
            if end>=len(arc):
                continue
            applies = progress<=arc[end]
            result[idx[applies]] = np.clip(arc[start]-progress[applies]-BUMPER_X,-2.,60.)
    return dict(junction_gap=result, xodr_sha256=np.array(hashlib.sha256(raw).hexdigest()))


def legacy_data(records, labels):
    ids, rows, y = (records[k] for k in ("episode","X","y"))
    if tuple(records["feature_names"]) != V1_NAMES or not np.array_equal(ids,labels["episode"]):
        raise ValueError("incompatible legacy demonstration labels or feature names")
    lengths, xy = records["route_lengths"], records["route_xy"]
    routes = np.split(xy,np.cumsum(lengths)[:-1])
    episodes = list(dict.fromkeys(ids))
    if len(routes)!=len(episodes):
        raise ValueError("missing demonstration route geometry")
    X = np.empty((len(ids),len(FEATURE_NAMES)),np.float32)
    nav = [_V1_INDEX[name] for name in NAV_NAMES]
    scenarios = np.empty(len(ids),dtype="U24")
    for key, route in zip(episodes,routes):
        indices = np.flatnonzero(ids==key)
        far = _preview(route,labels["progress_m"][indices],rows[np.ix_(indices,nav)])
        observer = Observer()
        junction = labels["junction_gap"] if "junction_gap" in labels else None
        for i, preview in zip(indices,far):
            X[i] = _legacy_row(rows[i],observer,preview,None if junction is None else junction[i])
        scenarios[indices] = key.split("-")[1]
    return dict(X=X,y=y,scenario=scenarios,episode_id=ids,seed=records["seed"],
                expert_gap_m=labels["expert_gap_m"],progress_m=labels["progress_m"]), routes


# Speed is intentionally absent from the planner's instantaneous feature set.
# A cruise demonstration at 6 m/s and a vehicle at rest on that same empty
# road must receive the same speed request. The downstream learned feedback
# controller sees actual speed, and relative velocity is ego-compensated.
PLANNER_NAMES = MAP_CUES + RANGE_CUES + CAMERA_CUES + tuple(f"{n}_recent" for n in MEMORY_CUES) + (
    "gap_rate", "pedestrian_side_rate")
PLAN_IDX = np.array([_INDEX[n] for n in PLANNER_NAMES])
STEER_IDX = np.array([_INDEX[n] for n in ("ego_speed_mps",*NAV_NAMES)])


def _feedback_features(request, speed):
    error = np.asarray(request) - np.asarray(speed)
    # Continuous signed speed error basis. The coefficients (including drag
    # compensation and hold effort) are fitted from actual teacher actuators.
    short, long = np.tanh(error/.45), np.tanh(error/1.8)
    return np.stack([np.tanh(speed/1.5), np.maximum(0.,short), np.maximum(0.,long),
                     np.minimum(0.,short), np.minimum(0.,long),
                     np.exp(-(speed/.55)**2)*np.exp(-(request/.40)**2)],axis=-1)


class BehaviorModel:
    inputs = "fusion"
    new_observer = Observer

    def __init__(self, starts, splits, thresholds, left, right, values, steering, feedback, lamp_expert):
        self.starts = np.asarray(starts)
        self.splits, self.thresholds = np.asarray(splits), np.asarray(thresholds)
        self.left, self.right, self.values = np.asarray(left), np.asarray(right), np.asarray(values)
        self.steering = np.asarray(steering)
        self.feedback = np.asarray(feedback)
        self.lamp_expert = lamp_expert
        n = len(self.splits)
        if (self.starts.ndim != 1 or not len(self.starts) or self.starts[0]!=0
                or not np.all(np.diff(self.starts)>0) or np.any(self.starts>=n)
                or any(a.dtype.kind not in "iu" for a in (self.starts,self.splits,self.left,self.right))
                or any(a.shape!=(n,) for a in (self.splits,self.thresholds,self.left,self.right,self.values))
                or self.steering.shape!=(3,) or self.feedback.shape!=(6,)
                or not all(np.isfinite(a).all() for a in (self.steering,self.feedback,self.thresholds,self.values))):
            raise ValueError("invalid behavior model arrays")
        branch = self.left>=0
        end = np.repeat(np.r_[self.starts[1:],n],np.diff(np.r_[self.starts,n]))
        if (np.any(self.splits[branch]<0) or np.any(self.splits[branch]>=len(PLANNER_NAMES))
                or any(np.any((a[branch]<=np.flatnonzero(branch)) | (a[branch]>=end[branch]))
                       for a in (self.left,self.right)) or np.any(self.right[~branch]>=0)):
            raise ValueError("invalid planner tree")

    def _plan(self, x):
        columns = x[PLAN_IDX]
        node = self.starts.copy()
        active = self.left[node]>=0
        while active.any():
            current=node[active]
            node[active]=np.where(columns[self.splits[current]]<=self.thresholds[current],
                                  self.left[current],self.right[current])
            active=self.left[node]>=0
        return float(self.values[node].mean())

    def predict(self,x):
        x=np.asarray(x,dtype=np.float32)
        if x.shape!=(len(FEATURE_NAMES),) or not np.isfinite(x).all():
            raise ValueError("expected one finite behavior observation")
        speed=x[_INDEX["ego_speed_mps"]]
        steer=float(np.clip(_steering_basis(x[STEER_IDX]) @ self.steering,-1.,1.))
        plan=self._plan(x)
        pedal=float(np.clip(_feedback_features(plan,speed) @ self.feedback,-1.,1.))
        return steer,max(0.,pedal),max(0.,-pedal)

    def save(self,path):
        from .policy import ACTION_NAMES
        params=dict(version=VERSION,feature_names=FEATURE_NAMES,action_names=ACTION_NAMES,
                    planner_names=PLANNER_NAMES,**{k:getattr(self,k) for k in (
                        "starts","splits","thresholds","left","right","values","steering","feedback")})
        if self.lamp_expert is not None:
            params.update(self.lamp_expert.arrays("lamp_"))
        with Path(path).open("wb") as f:
            np.savez_compressed(f,**params)

    @classmethod
    def load(cls,path):
        from .lamp import LampExpert
        from .policy import ACTION_NAMES
        with np.load(path,allow_pickle=False) as d:
            if (str(d["version"].item())!=VERSION or tuple(d["feature_names"])!=FEATURE_NAMES
                    or tuple(d["action_names"])!=ACTION_NAMES or tuple(d["planner_names"])!=PLANNER_NAMES):
                raise ValueError("incompatible behavior policy schema")
            return cls(*(d[k] for k in ("starts","splits","thresholds","left","right","values","steering","feedback")),
                       LampExpert.from_arrays(d,"lamp_") if "lamp_version" in d else None)


def _steering_basis(nav):
    return np.tanh(np.asarray(nav)[..., [2,4,6]] / [1.6,3.2,5.6])


def _fit_steering(X, y, routes):
    from .steering_search import rollout, route_bank
    # Positive steering gains on three lookaheads enforce left/right symmetry
    # and preserve corrective steering outside the demonstration pose range.
    w1=np.zeros((7,3),dtype="f")
    w1[[2,4,6],np.arange(3)]=1./np.array([1.6,3.2,5.6])
    params=(w1,np.zeros(3,dtype="f"))
    norm=(np.zeros(7,dtype="f"),np.ones(7,dtype="f"))
    bank=route_bank(routes,2026)
    # The original policy failed at very low turn speeds. Fit across several
    # actuation speeds and headings on the observed training road geometry.
    rng=np.random.default_rng(82026)
    bank=(bank[0],rng.uniform(2.3,5.8,len(bank[0])).astype("f"),bank[2],bank[3])
    a=np.linspace(.06,1.,8)
    b=np.linspace(.06,.64,6)
    c=np.linspace(0.,.27,4)
    candidates=np.array([[near,mid,far,0.] for near in a for mid in b for far in c],dtype="f")
    nav=X[:,STEER_IDX]
    use=(nav[:,0]>.8) & (np.abs(y[:,0])>.025)
    basis=_steering_basis(nav[use])
    imitation=np.mean(np.abs(basis @ candidates[:,:3].T - y[use,0:1]),axis=0)
    # Bound array sizes in each rollout, including when fitted on CPU.
    cost=np.concatenate([rollout(bank,params,norm,part)["objective"]
                         for part in np.array_split(candidates,8)])
    index=int(np.argmin(cost+.35*imitation))
    return candidates[index,:3],dict(rollout_objective_m=float(cost[index]),
                                      moving_turn_steer_mae=float(imitation[index]),candidates=len(candidates))

def _targets(records):
    """Produce privileged speed supervision, then discard oracle inputs.

    Project near-future in-lane occupancy back to the current bumper on a 0.7s
    horizon. Catching a crossing user just before lane entry matters. For lead
    departures the measured motion of the training actor releases the demand.
    """
    X=records["X"]
    gaps=np.asarray(records["expert_gap_m"],dtype=float)
    progress=np.asarray(records["progress_m"],dtype=float)
    speed=X[:,_INDEX["ego_speed_mps"]]
    if not np.isfinite(gaps).all() or not np.isfinite(progress).all() or np.any(gaps<0):
        raise ValueError("invalid aligned teacher geometry")
    cap=np.full(len(X),6.)
    supplied=records.get("teacher_speed")
    for key in dict.fromkeys(records["episode_id"]):
        idx=np.flatnonzero(records["episode_id"]==key)
        kind=str(records["scenario"][idx[0]])
        local=np.arange(len(idx))
        gap=gaps[idx].copy()
        if kind=="signal":
            # The frozen learned lamp expert owns signal color/approach response.
            gap.fill(MAX_RANGE)
        movement=np.zeros(len(idx))
        if kind in ("lead","braking"):
            before=np.maximum(0,local-3)
            after=np.minimum(len(idx)-1,local+3)
            change=(gap[after]+progress[idx[after]])-(gap[before]+progress[idx[before]])
            movement=np.clip(change/np.maximum(.1,(after-before)*.1),0.,9.)
        if kind in ("pedestrian","occluded","trip","journey"):
            for step in range(1,8):
                j=np.minimum(local+step,len(idx)-1)
                gap=np.minimum(gap,gaps[idx[j]]+np.maximum(0.,progress[idx[j]]-progress[idx]))
        if supplied is not None and kind!="signal":
            cap[idx]=np.clip(supplied[idx],0.,6.)
        else:
            # Curvature observed on these historical training routes is also
            # available at inference. Use the later recorded instructor's
            # junction cruise cap on comparable turns.
            turning=np.abs(X[idx,_INDEX["bend_7_13"]])>.16
            turning|=np.abs(X[idx,_INDEX["bend_13_25"]])>.34
            cap[idx[turning]]=3.7
        if kind!="signal":
            cap[idx]=np.minimum(cap[idx],np.sqrt(np.maximum(0.,4.5*(gap-2.8))+movement**2))
    return cap.astype(np.float32)


def _forest_arrays(forest):
    offsets=np.cumsum([0]+[tree.tree_.node_count for tree in forest.estimators_])
    split,threshold,left,right,value=[],[],[],[],[]
    for start,est in zip(offsets,forest.estimators_):
        tree=est.tree_
        split.append(tree.feature)
        threshold.append(tree.threshold)
        left.append(np.where(tree.children_left<0,-1,start+tree.children_left))
        right.append(np.where(tree.children_right<0,-1,start+tree.children_right))
        value.append(tree.value[:,0,0])
    return (offsets[:-1].astype("i"),np.concatenate(split).astype("i"),np.concatenate(threshold).astype("f"),
            np.concatenate(left).astype("i"),np.concatenate(right).astype("i"),np.concatenate(value).astype("f"))


def _fit_feedback(records, plans):
    designs,labels,weights=[],[],[]
    for rec,plan in zip(records,plans):
        X,y=rec["X"],rec["y"]
        speed=X[:,_INDEX["ego_speed_mps"]]
        action=y[:,1]-y[:,2]
        w=np.where((plan<.6)&(speed<.5),2.,1.)
        w=np.where((speed>1.)&(action<-.3),5.,w)
        w=np.where((plan-speed>1.)&(speed<2.)&(action>.25),4.,w)
        # Keep historical red phases out of ordinary cruise feedback. Native
        # signalled junction stops have their own learned controller.
        w=np.where(rec["scenario"]=="signal",0.,w)
        designs.append(_feedback_features(plan,speed))
        labels.append(action)
        weights.append(w)
    x=np.vstack(designs)
    y=np.concatenate(labels)
    w=np.concatenate(weights)
    # One-dimensional shape constraints keep the inverse dynamics monotone for
    # error magnitudes where the demonstrator provided few examples.
    from scipy.optimize import lsq_linear
    weighted=x*np.sqrt(w[:,None])
    fit=lsq_linear(weighted,y*np.sqrt(w),bounds=([0.,0.,0.,0.,0.,-1.],[.25,1.2,1.2,1.2,1.2,0.]))
    return fit.x.astype("f")


def fit(records, routes, base):
    from sklearn.ensemble import ExtraTreesRegressor
    from .policy import Model
    plans=[_targets(r) for r in records]
    frames=[r["X"] for r in records]
    labels=[r["y"] for r in records]
    X=np.vstack(frames)
    plan=np.concatenate(plans)
    if X.shape!=(len(plan),len(FEATURE_NAMES)) or not np.isfinite(X).all():
        raise ValueError("invalid training observations")
    weights=[]
    for r,t in zip(records,plans):
        # Preserve independent raw sensor samples when joining a larger bank
        # whose recorded range columns approximate the historical sensor rig.
        weight=np.where(t<1.,3.,np.where(t<4.5,1.7,1.))
        if r.get("legacy",False):
            weight*=.45
            # Old left episodes have no projected junction boundary unless
            # supplied by the static map exporter. Their teacher's stop line
            # alone cannot reconstruct that navigation input.
            weight[(r["scenario"]=="left")&(r["X"][:,_INDEX["junction_gap"]]>=59.9)]*=.1
        weights.append(weight)
    forest=ExtraTreesRegressor(n_estimators=40,max_depth=12,min_samples_leaf=4,
                               max_features=.75,random_state=20261120,n_jobs=1)
    forest.fit(X[:,PLAN_IDX],plan,sample_weight=np.concatenate(weights))
    steer,steer_report=_fit_steering(X,np.vstack(labels),routes)
    feedback=_fit_feedback(records,plans)
    model=BehaviorModel(*_forest_arrays(forest),steer,feedback,Model.load(base).lamp_expert)
    predictions=forest.predict(X[:,PLAN_IDX])
    return model,dict(frames=len(X),episodes=sum(len(set(r["episode_id"])) for r in records),
                      seeds=sorted(set(int(s) for r in records for s in r["seed"])),
                      speed_plan_training_mae_mps=float(np.mean(np.abs(predictions-plan))),
                      steering=steer_report,steer_weights=steer.tolist(),feedback_weights=feedback.tolist(),
                      planner="40 extremely randomized regression trees predicting a speed request",
                      supervision="Causal observations, train-district instructor speed and actuator labels, "
                                  "oracle clearance with a 0.7-second pedestrian anticipation horizon",
                      simulator_validation="separate closed-loop evaluation required")


def main():
    import argparse
    import hashlib
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy",type=Path,default=Path("data/training.npz"))
    src=parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--legacy-traces",type=Path)
    src.add_argument("--legacy-labels",type=Path)
    parser.add_argument("--export-labels",type=Path,help="write aligned archived teacher labels and exit")
    parser.add_argument("--xodr",type=Path,help="local original map for archived training route junctions")
    parser.add_argument("--supplement",nargs="*",type=Path,default=[])
    parser.add_argument("--base",type=Path,default=Path("models/fusion.npz"))
    parser.add_argument("--out",type=Path)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    with np.load(args.legacy,allow_pickle=False) as src:
        if args.legacy_labels:
            with np.load(args.legacy_labels,allow_pickle=False) as d:
                labels=dict(d)
        else:
            labels=read_legacy_labels(src,args.legacy_traces)
            if args.xodr:
                labels.update(archived_junctions(src, labels, args.xodr, args.legacy_traces))
        if args.export_labels:
            args.export_labels.parent.mkdir(parents=True,exist_ok=True)
            with args.export_labels.open("wb") as f:
                np.savez_compressed(f,**labels)
            return
        if args.out is None:
            parser.error("--out is required for fitting")
        legacy,routes=legacy_data(src,labels)
    legacy["legacy"]=True
    records=[legacy]
    for path in args.supplement:
        with np.load(path,allow_pickle=False) as d:
            if tuple(d["feature_names"])!=FEATURE_NAMES:
                raise ValueError(f"incompatible supplementary training measurements: {path}")
            if "district" in d and not np.all(d["district"]=="train"):
                raise ValueError(f"expected train-district samples: {path}")
            rec={k:d[k] for k in ("X","y","seed","scenario","episode_id","expert_gap_m","progress_m")}
            if "teacher_speed" in d:
                rec["teacher_speed"]=d["teacher_speed"]
            if "route_xy" in d and "route_lengths" not in d and len(set(rec["episode_id"]))==1:
                routes.append(d["route_xy"])
            records.append(rec)
    model,report=fit(records,routes,args.base)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    model.save(args.out)
    # Verify that the portable tree traversal matches the reloaded artifact.
    loaded=BehaviorModel.load(args.out)
    for rec in records:
        for x in rec["X"][::max(1,len(rec["X"])//7)]:
            np.testing.assert_allclose(loaded.predict(x),model.predict(x),atol=1e-6)
    files=[args.legacy,args.base,*args.supplement]
    if args.legacy_labels:
        files.append(args.legacy_labels)
    report["input_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report["model_sha256"]=hashlib.sha256(args.out.read_bytes()).hexdigest()
    if args.report:
        args.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
