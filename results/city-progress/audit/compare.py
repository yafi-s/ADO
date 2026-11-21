"""Compare registered station/release windows and grade the physical controls."""
import argparse
import json
from pathlib import Path

GEO_METRICS = ("seconds", "median_kmh", "p10_kmh", "median_request_kmh", "coasting_seconds",
               "no_path_actor_recorded_s", "path_actor_s", "later_path_actor_2s", "near_path_vru_s")
MATCH_FIELDS = ("map", "route", "weather", "seed", "crossing_delay_s", "fixed_delta_seconds",
                "population_profile", "quality", "sensor_configuration")


def read(p):
    return json.loads(p.read_bytes())


def comparison(old, new, old_boxes, new_boxes):
    def pair(a, b):
        delta = round(b-a,3) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else None
        return dict(v05=a,candidate=b,delta=delta)
    def window(a,b):
        if (a["from_m"],a["to_m"])!=(b["from_m"],b["to_m"]):
            raise ValueError("Unmatched station windows")
        return dict(from_m=a["from_m"],to_m=a["to_m"],**{k:pair(a.get(k),b.get(k)) for k in GEO_METRICS})
    def group(name):
        return [window(a,b) for a,b in zip(old[name],new[name],strict=True)]
    source={k:pair(old["source"][k],new["source"][k]) for k in MATCH_FIELDS}
    old_events={e["kind"]:e for e in old["events"]};new_events={e["kind"]:e for e in new["events"]}
    safety={}
    reasons=[]
    if new["grade"]["outcome"]!="complete": reasons.append("route incomplete")
    if new["grade"]["capture_passed"] is False: reasons.extend(new["grade"]["capture_failure_reasons"])
    if new["grade"]["participant_contacts"]:reasons.append("background participant contact")
    for k in ("collision_events","red_light_violations","yield_violations"):
        if new["grade"][k]: reasons.append(k)
    for k in sorted(set(old_events)|set(new_events)):
        a,b=old_events.get(k,{}),new_events.get(k,{})
        row={p:pair(a.get(p),b.get(p)) for p in ["station_m","exposed","passed","handled","violation",
                                                  "stopped_for_event","minimum_gap_m","minimum_cone_clearance_m",
                                                  "minimum_barrier_clearance_m","minimum_bus_clearance_m"] if p in a or p in b}
        if not all(b.get(key) for key in ["exposed","passed","handled"]) or b.get("violation"):
            reasons.append(k+" incomplete or violated")
        safety[k]=row
    if new["grade"]["turns_completed"]!=old["grade"]["turns_completed"]: reasons.append("junction count mismatch")
    pairs={}
    for name in sorted(set(old_boxes["ego_clearance"])|set(new_boxes["ego_clearance"])):
        a=old_boxes["ego_clearance"].get(name,{}); b=new_boxes["ego_clearance"].get(name,{})
        am=a.get("closest_overlapping_elevation") or {}; bm=b.get("closest_overlapping_elevation") or {}
        ct={k:pair(a.get("at_or_below_m",{}).get(k,0), b.get("at_or_below_m",{}).get(k,0))
            for k in ["0.0","0.25","0.5","1.0","2.0"]}
        pairs[name]=dict(min_gap_m=pair(am.get("ground_projection_gap_m"),bm.get("ground_projection_gap_m")),
                         ego_kmh_at_closest=pair(round(am["ego_speed_mps"]*3.6,2) if am else None,
                                                 round(bm["ego_speed_mps"]*3.6,2) if bm else None),
                         frame=pair(am.get("frame"),bm.get("frame")),actor_exposure_counts=ct)
        if ct["0.0"]["candidate"]:reasons.append(name+" sampled body overlap")
    releases=[]
    release_new={(a["event"],a["mark"]):a for a in new["releases"]}
    for a in old["releases"]:
        b=release_new.get((a["event"],a["mark"]),{})
        releases.append(dict(event=a["event"],mark=a["mark"],speed_kmh=pair(a["speed_kmh"],b.get("speed_kmh")),
                             request_kmh=pair(a["request_kmh"],b.get("request_kmh")),
                             change_over_5s_kmh=pair(a["gain_5s_kmh"],b.get("gain_5s_kmh")),
                             first_threshold_s={k:pair(v,b.get("first_threshold_s",{}).get(k))
                                                for k,v in a["first_threshold_s"].items()}))
    old_lat, new_lat = (r["lateral_acceleration_estimate"] for r in (old,new))
    def compare_lat(a,b):
        return {k:pair(a.get(k),b.get(k)) for k in ("windows","peak_mps2","p95_mps2")}
    lateral = {k:compare_lat(old_lat[k],new_lat[k]) for k in ("whole","all_junctions","left_and_right","near_straights")}
    lateral["by_junction"] = {k:compare_lat(v,new_lat["by_junction"].get(k,{})) for k,v in old_lat["by_junction"].items()}
    return dict(source=source,source_mismatches=[k for k,v in source.items() if v["v05"]!=v["candidate"]],
                required_grade_failures=reasons,
                route=dict(total_s=pair(old["grade"]["duration_s"],new["grade"]["duration_s"]),
                           distance_m=pair(old["grade"]["progress_m"],new["grade"]["progress_m"]),
                           peak_kmh=pair(old["whole"]["peak_kmh"],new["whole"]["peak_kmh"]),
                           slow_s={k:pair(v,new["whole"]["below_kmh_s"][k]) for k,v in old["whole"]["below_kmh_s"].items()}),
                near_straight_junctions={k:pair(old["near_straight_junctions"].get(k),new["near_straight_junctions"].get(k))
                                         for k in GEO_METRICS},
                station_bins=group("station_bins"),sections=group("sections"),
                junctions=[dict(id=a["id"],direction=a["direction"],**window(a,b))
                           for a,b in zip(old["junctions"],new["junctions"],strict=True)],
                releases=releases,physical_clearances=pairs,events=safety,lateral_acceleration_estimate=lateral,
                comfort={k:pair(v,new["comfort"][k]) for k,v in old["comfort"].items() if isinstance(v,(float,int))})

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("old",type=Path);p.add_argument("new",type=Path)
    p.add_argument("old_boxes",type=Path);p.add_argument("new_boxes",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();r=comparison(read(a.old),read(a.new),read(a.old_boxes),read(a.new_boxes))
    a.output.write_text(json.dumps(r,indent=2)+"\n",encoding="utf8")
