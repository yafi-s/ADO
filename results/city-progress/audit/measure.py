"""Compare all station bins, native road-user bodies and releases of two City Life drives.

Run with system Python, numpy, shapely and Pillow. The evaluator uses future ego
positions and actor truth only after the completed drive. No production imports
other than the existing native bounding-box geometry functions.
"""
import argparse
from collections import Counter, deque
import gzip
import hashlib
import json
import math
from itertools import islice
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import LineString, Polygon

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.score_city_interactions import KINDS, Body, box_for, rotation, signed_gap

HERE = Path(__file__).resolve().parent
DT = 0.1
SECTIONS = [(420.16,501.65), (734,770), (770,800), (800,825), (825,855), (855,890),
            (150,250), (420,500), (600,645), (1080,1180), (1365,1415),
            (858,872), (872,882), (882,892)]
VULNERABLE = {"pedestrian", "cyclist", "bicycle", "motorcycle"}


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rounded(value):
    return round(float(value), 3)


def actor_box(actor, shapes):
    body, method = box_for(actor, shapes)
    return actor, body, Polygon(body.polygon) if body is not None else None, method


def occupancy(rows, references):
    shapes = read(ROOT / "results/perception/town03-body-offsets.json")
    shape = shapes["vehicle.lincoln.mkz"]
    half_width = shape["dimensions_m"][2] / 2
    nose = shape["dimensions_m"][0] / 2 + shape["offset_xyz"][0]
    height = shape["dimensions_m"][1]
    parked = [actor_box(a, shapes) for a in read(ROOT / "results/perception/town03-static-vehicles.json")["actors"]]
    # 1 m sampling in route station on the actual driven trajectory, including the bus bypass.
    station = np.array([r["progress_m"] for r in rows])
    keep = np.r_[True, np.diff(np.maximum.accumulate(station)) > .0001]
    support = station[keep]
    coords = np.asarray([[r["ego_pose"][k] for k in ("x","y","z")] for r in rows])[keep]
    flags = []
    minima = {}
    unknown = Counter()

    # Four future native exposures bound cross-traffic checks to 2 s of history.
    references = iter(references)
    lookahead = deque(islice(references, 21))
    for r in rows:
        if not lookahead:
            raise ValueError("Missing actor reference")
        ref = lookahead[0]
        if r["frame"] != ref["frame"] or abs(r["progress_m"]-ref["progress_m"]) > .01:
            raise ValueError("Actor exposure or station mismatch")
        q = np.minimum(station[-1], r["progress_m"] + nose + np.arange(26))
        xyz = np.array([np.interp(q, support, coords[:,i]) for i in range(3)]).T
        route = LineString(xyz[:,:2])
        pos = r["ego_pose"]
        x, y, z = (pos[k] for k in ("x","y","z"))
        ego = Body((x,y,z), rotation(pos["pitch"],pos["yaw"],pos["roll"]), shape["dimensions_m"],
                   shape["offset_xyz"], shape["rotation"])
        hits, side_vru, physical = [], [], []
        native = [actor_box(a, shapes) for a in ref["others"] if a["type"] in KINDS and (x-a["x"])**2+(y-a["y"])**2<38**2]
        for a, box, polygon, method in [*native, *parked]:
            if (x-a["x"])**2 + (y-a["y"])**2 > 38**2:
                continue
            if box is None:
                unknown[str(a.get("blueprint"))] += 1
                continue
            kind, role = a["type"], a["role"]
            # Height at nearest swept route location removes the overhead deck.
            idx = int(np.argmin((xyz[:,0]-box.center[0])**2 + (xyz[:,1]-box.center[1])**2))
            route_z = xyz[idx,2]
            if box.bottom <= route_z+height+.15 and box.top >= route_z-.15:
                distance = route.distance(polygon)
                if distance <= half_width:
                    hits.append(role)
                if distance <= half_width+1.5 and kind in VULNERABLE:
                    side_vru.append(role)
            if (min(ego.top, box.top) < max(ego.bottom, box.bottom)
                    or math.dist(ego.center[:2],box.center[:2]) - ego.radius - box.radius > 4):
                continue
            gap = signed_gap(ego.polygon,box.polygon)
            if gap > 4:
                continue
            aid = str(a["actor_id"])
            group = ("parked" if role=="parked-map-vehicle" else "background" if role=="background" else "staged")+"/"+kind
            physical.append((gap, role, aid))
            rec = minima.setdefault(group, dict(min_m=None, closest={}, at_or_below_1m_samples=0,
                                                at_or_below_half_m_samples=0, overlap_samples=0, actors={},
                                                moving_minimum={},inside_1m_max_ego_kmh=None))
            for key, lim in [("at_or_below_1m_samples",1), ("at_or_below_half_m_samples",.5), ("overlap_samples",0)]:
                rec[key] += int(gap <= lim)
            if rec["min_m"] is None or gap < rec["min_m"]:
                rec["min_m"] = gap
                rec["closest"] = dict(frame=r["frame"], time_s=r["sim_time"], station_m=r["progress_m"],
                                      actor_id=aid, role=role, ego_kmh=rounded(r["speed_mps"]*3.6), method=method)
            if aid not in rec["actors"] or rec["actors"][aid]["gap_m"] > gap:
                rec["actors"][aid] = dict(gap_m=gap,frame=r["frame"],time_s=r["sim_time"],role=role)
            kmh = r["speed_mps"]*3.6
            if gap<=1 and (rec["inside_1m_max_ego_kmh"] is None or kmh>rec["inside_1m_max_ego_kmh"]):
                rec["inside_1m_max_ego_kmh"] = rounded(kmh)
            for limit in (5,10,15,20):
                key = str(limit)
                if kmh>=limit and gap<rec["moving_minimum"].get(key,{}).get("gap_m",math.inf):
                    rec["moving_minimum"][key] = dict(gap_m=rounded(gap),frame=r["frame"],ego_kmh=rounded(kmh))
        later_hits = set()
        for offset in (5,10,15,20):
            if offset >= len(lookahead):
                continue
            for a in lookahead[offset]["others"]:
                if a["type"] not in KINDS or (x-a["x"])**2+(y-a["y"])**2 > 38**2:
                    continue
                _, future, outline, _ = actor_box(a, shapes)
                if future is None or route.distance(outline) > half_width:
                    continue
                idx = int(np.argmin((xyz[:,0]-future.center[0])**2 + (xyz[:,1]-future.center[1])**2))
                if future.bottom <= xyz[idx,2]+height+.15 and future.top >= xyz[idx,2]-.15:
                    later_hits.add(a["role"])
        flags.append(dict(later_path=sorted(later_hits),path=sorted(set(hits)),vulnerable=sorted(set(side_vru)),
                          closest=sorted(physical)[:1], full=bool(q[-1]-q[0]>24.95)))
        lookahead.popleft()
        extra = next(references, None)
        if extra is not None:
            lookahead.append(extra)
    if lookahead:
        raise ValueError("Actor reference exceeds telemetry")
    for d in minima.values():
        d["min_m"] = rounded(d["min_m"])
        for a in d["actors"].values():
            a["gap_m"] = rounded(a["gap_m"])
    return flags, minima, dict(unknown)


def audit(folder, reference):
    published = (folder/"manifest.json").exists()
    manifest = read(folder/("manifest.json" if published else "capture.json"))
    tele = folder/("telemetry.json" if published else "raw/telemetry.json")
    s = manifest["summary"]
    rows = read(tele)
    if len(rows)!=s["frames"] or not np.allclose(np.diff([r["sim_time"] for r in rows]), DT, atol=.0001):
        raise ValueError("Telemetry frame cadence/count mismatch")
    expected = manifest["sha256"]["telemetry.json"] if published else manifest["raw_sha256"]["telemetry.json"]
    if expected != sha(tele) or manifest["population_reference_sha256"] != sha(reference):
        raise ValueError("Source digest mismatch")
    if published and sha(folder/manifest["replay"])!=manifest["sha256"][manifest["replay"]]:
        raise ValueError("Replay digest mismatch")
    with gzip.open(reference,"rt",encoding="utf8") as stream:
        flags, minima, unknown = occupancy(rows, (json.loads(l) for l in stream))
    time = np.array([r["sim_time"] for r in rows])
    station = np.array([r["progress_m"] for r in rows])
    speed = np.array([r["speed_mps"]*3.6 for r in rows])
    yaw = np.rad2deg(np.unwrap(np.deg2rad([r["ego_pose"]["yaw"] for r in rows])))
    target = np.array([r["speed_request_mps"]*3.6 for r in rows])
    posted = np.array([r["posted_limit_kmh"] for r in rows])
    coast = np.array([r["brake"]<.05 and r["throttle"]<.05 for r in rows]) & (speed>=1)
    accel = np.r_[np.nan, np.diff(speed)/(3.6*DT)]
    accel5 = np.r_[np.full(5,np.nan), (speed[5:]-speed[:-5])/(3.6*5*DT)]
    jerk = np.r_[np.full(5,np.nan), (accel5[5:]-accel5[:-5])/(5*DT)]
    full = np.array([f["full"] for f in flags])
    path_hit = np.array([bool(f["path"]) for f in flags])
    vru_hit = np.array([bool(f["vulnerable"]) for f in flags])
    later_hit = np.array([bool(f["later_path"]) for f in flags])
    no_actor = full & ~path_hit & ~vru_hit
    phases = np.array([r["phase"] for r in rows])
    dt = lambda mask: rounded(mask.sum()*DT)

    def stats(mask):
        ix = np.flatnonzero(mask)
        if not len(ix):
            return {"frames":0,"seconds":0}
        limits = posted[mask]
        v = speed[mask]; t = target[mask]
        return dict(frames=len(ix), seconds=rounded(len(ix)*DT), from_s=time[ix[0]],to_s=time[ix[-1]],
                    start_frame=rows[ix[0]]["frame"],middle_frame=rows[ix[len(ix)//2]]["frame"],
                    p10_kmh=rounded(np.quantile(v,.1)),median_kmh=rounded(np.median(v)),
                    peak_kmh=rounded(v.max()),median_request_kmh=rounded(np.median(t)),
                    p10_request_kmh=rounded(np.quantile(t,.1)), mean_positive_lag_kmh=rounded(np.maximum(t-v,0).mean()),
                    heading_span_deg=rounded(np.ptp(yaw[mask])),heading_change_deg=rounded(yaw[ix[-1]]-yaw[ix[0]]),
                    coasting_seconds=dt(mask & coast),below_kmh_s={str(k):dt(mask & (speed<k)) for k in [5,10,15,20]},
                    posted_kmh=sorted(set(limits)),path_actor_s=dt(mask & path_hit),
                    later_path_actor_2s=dt(mask & later_hit),near_path_vru_s=dt(mask & vru_hit),no_path_actor_recorded_s=dt(mask & no_actor),
                    short_horizon_s=dt(mask & ~full), phase_counts=dict(Counter(phases[mask])))
    def window(lo,hi, detailed=False):
        mask=(station>=lo)&(station<hi)
        d=dict(from_m=lo,to_m=hi,**stats(mask))
        if detailed:
            d["no_path_actor_recorded"]=stats(mask&no_actor)
            d["path_actor_or_near_vru"]=stats(mask&(path_hit|vru_hit))
            d["path_roles"] = dict(Counter(a for i in np.flatnonzero(mask) for a in flags[i]["path"]))
            d["later_roles"] = dict(Counter(a for i in np.flatnonzero(mask) for a in flags[i]["later_path"]))
            d["vulnerable_roles"] = dict(Counter(a for i in np.flatnonzero(mask) for a in flags[i]["vulnerable"]))
        return d
    # Lock the baseline geographic windows when evaluating a later drive, even if incomplete.
    if expected == "477458dfe62392e3111bb3a96f2db5a431d395617ca3ad69bc6663fbd76dfbb7":
        length = s["route_length_m"]
        boundaries = [(j["junction_id"],j["direction"],j["entry_m"],j["exit_m"]) for j in s["junction_traversals"]]
    else:
        baseline = read(HERE/"v05.json")
        if baseline["source"]["telemetry_sha256"] != "477458dfe62392e3111bb3a96f2db5a431d395617ca3ad69bc6663fbd76dfbb7":
            raise ValueError("Missing sealed v0.5 geography")
        length = baseline["grade"]["route_length_m"]
        boundaries = [(j["id"],j["direction"],j["from_m"],j["to_m"]) for j in baseline["junctions"]]
    bins = [window(a,a+25,True) for a in range(0, math.ceil(length),25)]
    junctions = [dict(id=jid, direction=direction, **window(lo,hi,True)) for jid,direction,lo,hi in boundaries]
    # These six physical junctions had <=5 degrees measured heading span in
    # the sealed v0.5 tape. Preserve their stations even if the new steering differs.
    near_straight_ids = {103,82,238,861,1820,1901}
    true_straights = np.zeros(len(rows), dtype=bool)
    for jid, _, lo, hi in boundaries:
        if jid in near_straight_ids:
            true_straights |= (station>=lo)&(station<hi)
    straight_stats = stats(true_straights)
    straight_stats["junction_ids"] = sorted(near_straight_ids)
    straight_stats["no_path_actor_recorded"] = stats(true_straights & no_actor)
    straight_stats["no_current_or_later_path_actor_or_near_vru"] = stats(true_straights & no_actor & ~later_hit)
    releases=[]
    event_moments={}
    for e in s["events"]:
        marks={}
        for mark,ts in e.items():
            if not mark.endswith("_s") or not isinstance(ts,(int,float)) or not 0 <= ts <= time[-1]+.05:
                continue
            i=int(np.argmin(abs(time-ts)))
            if abs(time[i]-ts)<.051:
                marks[mark]=dict(time_s=time[i],frame=rows[i]["frame"],station_m=station[i],
                                 speed_kmh=rounded(speed[i]),request_kmh=rounded(target[i]),
                                 reference_gap_m=rows[i]["reference_gap_m"],phase=rows[i]["phase"])
        event_moments[e["kind"]]=marks
        for mark in ["cleared_s","queue_departed_green_s","restarted_s"]:
            if mark not in e: continue
            ts=e[mark]; i=int(np.argmin(abs(time-ts))); end=int(np.argmin(abs(time-(ts+5))))
            if abs(time[i]-ts)>.051: raise ValueError("Release missing from exposure clock")
            release=dict(event=e["kind"],mark=mark,time_s=ts,station=station[i],frame=rows[i]["frame"],
                         speed_kmh=rounded(speed[i]),request_kmh=rounded(target[i]),gain_5s_kmh=rounded(speed[end]-speed[i]),
                         first_threshold_s={},windows={})
            for k in [5,10,15,20]:
                hits=np.flatnonzero((time>=ts)&(time<=ts+10)&(speed>=k))
                release["first_threshold_s"][str(k)]=rounded(time[hits[0]]-ts) if len(hits) else None
            for lo,hi in [(0,1),(1,3),(3,5)]:
                release["windows"][f"{lo}-{hi}s"]=stats((time>=ts+lo-.001)&(time<ts+hi-.001))
            releases.append(release)
    ends=np.flatnonzero(np.diff(np.r_[False,speed<15,False])).reshape(-1,2)
    spells=[dict(from_m=station[a],to_m=station[b-1],**stats((time>=time[a])&(time<=time[b-1])))
            for a,b in ends if b-a>=20]
    worst = lambda x: dict(value=rounded(x[np.nanargmax(abs(x))]),time_s=time[np.nanargmax(abs(x))],
                           frame=rows[np.nanargmax(abs(x))]["frame"],station_m=station[np.nanargmax(abs(x))])
    r=dict(source=dict(capture=manifest["capture_id"],policy=s["policy"],policy_sha256=manifest["policy"]["sha256"],
                       map=manifest["producer"]["map"],weather=s["weather"],route=s["route_id"],
                       fixed_delta_seconds=manifest["world_settings"]["fixed_delta_seconds"],
                       sensor_configuration=s["sensor_configuration"],seed=s["seed"],crossing_delay_s=s["crossing_delay_s"],
                       population_profile=manifest.get("population_profile"),quality=manifest.get("quality_requested"),
                       telemetry_sha256=expected,population_sha256=sha(reference),raw_replay_sha256=manifest["raw_sha256"]["replay.json"]),
           grade=dict(capture_passed=manifest.get("passed"),capture_failure_reasons=manifest.get("failure_reasons"),
                      participant_contacts=s["background_traffic_activity"].get("participant_contacts"),
                      **{k:s[k] for k in ["outcome","duration_s","progress_m","route_length_m","collision_events","red_light_violations",
                                          "yield_violations","turns_completed","moving_within_1m_frames","max_lateral_error_m"]}),
           events=s["events"],event_moments=event_moments,whole=stats(np.ones(len(rows),bool)),station_bins=bins,junctions=junctions,
           near_straight_junctions=straight_stats,
           sections=[window(a,b,True) for a,b in SECTIONS],releases=releases,under_15_spells=spells,
           comfort=dict(max_accel_half_s=rounded(np.nanmax(accel5)),max_decel_half_s=rounded(-np.nanmin(accel5)),
                        raw_max_decel=rounded(-np.nanmin(accel)),raw_max_accel=rounded(np.nanmax(accel)),
                        moving_p95_abs_half_s_accel=rounded(np.nanquantile(abs(accel5[speed>1]),.95)),
                        half_s_hard_braking_seconds=dt(accel5<-3),raw_hard_braking_seconds=dt(accel<-3),
                        peak_jerk_half_s=worst(jerk),p95_abs_jerk=rounded(np.nanquantile(abs(jerk),.95)),
                        overspeed_posted_plus_1_seconds=dt(speed>posted+1),over_40_seconds=dt(speed>40)),
           clearance_below_four_m=minima,unresolved_actor_body_samples=unknown)
    return r, rows, flags


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("drive",type=Path);p.add_argument("reference",type=Path)
    p.add_argument("output",type=Path); args=p.parse_args()
    result,_,_=audit(args.drive,args.reference)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf8")

