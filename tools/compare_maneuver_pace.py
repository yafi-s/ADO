"""Compare matched original City Life segments and seed-matched safety controls."""
import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np

from tools.score_city_interactions import Body, OFFSETS, box_for, rotation, signed_gap


def read_gzip(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        return list(map(json.loads,f))


def summarize(path, start=None,end=None):
    root=Path(path)
    summary=json.loads((root/"summary.json").read_bytes())
    trace=read_gzip(root/"trace.jsonl.gz")
    selected=[r for r in trace if start is None or start<=r["progress_m"]<end]
    keys={r["frame"] for r in selected}
    references=read_gzip(root/"reference.jsonl.gz")
    shapes=json.loads(OFFSETS.read_bytes())
    ego_shape=shapes["vehicle.lincoln.mkz"]
    pedestrian_gap=math.inf
    seen=set()
    for r in references:
        if r["frame"] not in keys:
            continue
        x,y,yaw=r["ego"]
        ego=Body([x,y,r["ego_z"]],rotation(0.,yaw,0.),ego_shape["dimensions_m"],
                 ego_shape["offset_xyz"],ego_shape["rotation"])
        for actor in r["others"]:
            if actor["type"]!="pedestrian" or math.hypot(actor["x"]-x,actor["y"]-y)>35.:
                continue
            body,_=box_for(actor,shapes)
            if body is None or body.bottom>ego.top or body.top<ego.bottom:
                continue
            seen.add(actor["actor_id"])
            pedestrian_gap=min(pedestrian_gap,signed_gap(ego.polygon,body.polygon))
    speed=np.array([r["speed_mps"] for r in selected])
    target=np.array([r["speed_request_mps"] for r in selected])
    acceleration=(speed[5:]-speed[:-5])/.5
    yaw=np.unwrap(np.radians([r["ego"][2] for r in references if r["frame"] in keys]))
    lateral=np.abs((yaw[5:]-yaw[:-5])/.5)*.5*(speed[5:]+speed[:-5])
    events=[{k:v for k,v in e.items() if k in {"kind","exposed","handled","stopped_for_event","violation",
                "minimum_gap_m","minimum_cone_clearance_m","minimum_barrier_clearance_m",
                "max_cone_displacement_m","max_barrier_displacement_m","minimum_stopline_margin_m",
                "cones_alongside","barriers_alongside"}} for e in summary["events"] if e["exposed"]]
    held=[r for r in selected if r.get("tactical_pass",{}).get("tactical_speed_limit_mps")==0.]
    stopped=next((r for r in held if r["speed_mps"]<.05),None)
    wait=None
    if stopped:
        t=stopped["tactical_pass"]
        wait=dict(frame=stopped["frame"],station_m=stopped["progress_m"],phase=t.get("maneuver_phase"),
                  source_lead_m=t.get("raw_source_gap_m"),target_rear_m=t.get("target_rear_m"),
                  target_rear_closing_mps=t.get("target_rear_closing_mps"))
    return dict(native_pass=summary.get("passed",summary.get("target_pass")),checkpoint=summary["model_sha256"],
        tactical_zero_cap_s=round(.1*len(held),1),standing_wait=wait,
        contacts=summary["collision_events"],red=summary["red_light_violations"],yield_violations=summary["yield_violations"],
        station_m=None if start is None else [start,end],frames=len(selected),seconds=round(.1*len(speed),1),
        movement_kmh=dict(median=round(float(np.median(speed)*3.6),2),peak=round(float(max(speed)*3.6),2)),
        requested_kmh=dict(median=round(float(np.median(target)*3.6),2),peak=round(float(max(target)*3.6),2)),
        seconds_below_10=round(.1*sum(speed<10./3.6),1),seconds_below_15=round(.1*sum(speed<15./3.6),1),
        worst_half_second_decel_mps2=round(float(max(0.,-min(acceleration))),2) if len(acceleration) else None,
        p95_half_second_lateral_mps2=round(float(np.quantile(lateral,.95)),3) if len(lateral) else None,
        peak_half_second_lateral_mps2=round(float(max(lateral)),3) if len(lateral) else None,
        target_drops_over_5_kmh=int(sum(np.diff(target)*3.6< -5.)), 
        binding_counts=dict(Counter(r.get("progress_evidence",{}).get("maneuver_binding",
                          r.get("progress_evidence",{}).get("binding","unknown")) for r in selected)),
        sampled_nearby_pedestrians=len(seen),minimum_sampled_pedestrian_gap_m=round(pedestrian_gap,3) if seen else None,
        events=events)


def lead_witness(folder):
    """Use native map lane assignments and measured body extents around the original lead."""
    import carla
    shape=json.loads(OFFSETS.read_bytes())["vehicle.lincoln.mkz"]
    xodr=Path(os.environ.get("CARLA_ROOT") or ".local/carla-1.0")/"CarlaUnreal/Packs/AdditionalMaps/Content/Maps/OpenDrive/Town03_Opt.xodr"
    road=carla.Map("Town03_Opt",xodr.read_text())
    trace={r["frame"]:r for r in read_gzip(folder/"trace.jsonl.gz")}
    nearby=[]
    changes=[]
    last=None
    for ref in read_gzip(folder/"reference.jsonl.gz"):
        row=trace[ref["frame"]]
        if row["progress_m"]<530.:
            continue
        x,y,yaw=ref["ego"]
        origin=carla.Location(x=x,y=y,z=ref["ego_z"])
        wp=road.get_waypoint(origin)
        if last is not None and wp.road_id==last.road_id and wp.lane_id!=last.lane_id:
            for direction,neighbor,mark in (("left",last.get_left_lane(),carla.LaneChange.Left),
                                            ("right",last.get_right_lane(),carla.LaneChange.Right)):
                if neighbor and neighbor.lane_id==wp.lane_id:
                    changes.append(dict(frame=ref["frame"],station_m=row["progress_m"],direction=direction,
                        from_lane=last.lane_id,to_lane=wp.lane_id,road=wp.road_id,
                        same_direction=wp.lane_id*last.lane_id>0,
                        marking_allowed=bool(int(last.lane_change)&int(mark))))
                    break
        last=wp
        actor=next((a for a in ref["others"] if a["role"]=="lead"),None)
        if actor is None:
            continue
        actor_position=carla.Location(x=actor["x"],y=actor["y"],z=actor["z"])
        awp=road.get_waypoint(actor_position)
        if abs(actor["x"]-x)+abs(actor["y"]-y)>80.:
            continue
        ego=Body([x,y,ref["ego_z"]],rotation(0.,yaw,0.),shape["dimensions_m"],shape["offset_xyz"],shape["rotation"])
        lead,_=box_for(actor,{})
        if lead is None:
            continue
        forward=awp.transform.get_forward_vector()
        project=lambda pts:[px*forward.x+py*forward.y for px,py in pts]
        e,a=project(ego.polygon),project(lead.polygon)
        gap=signed_gap(ego.polygon,lead.polygon)
        alongside=min(e)<=max(a) and min(a)<=max(e)
        other_driving_lane=(wp.road_id==awp.road_id and wp.lane_id!=awp.lane_id and wp.lane_id*awp.lane_id>0)
        nearby.append(dict(frame=ref["frame"],body_gap_m=gap,
            passed_rear_clear=other_driving_lane and min(e)>max(a),
            adjacent_overlap=alongside and other_driving_lane,same_lane=wp.road_id==awp.road_id and wp.lane_id==awp.lane_id))
    overlap=[r for r in nearby if r["adjacent_overlap"]]
    complete=[r for r in nearby if overlap and r["frame"]>overlap[0]["frame"] and r["passed_rear_clear"]]
    pair=None
    if complete:
        out=next((c for c in changes if c["frame"]<overlap[-1]["frame"] and c["marking_allowed"] and c["same_direction"]),None)
        back=next((c for c in changes if out and c["frame"]>=complete[0]["frame"] and
                   c["marking_allowed"] and c["from_lane"]==out["to_lane"] and c["to_lane"]==out["from_lane"]),None)
        if back:
            pair=[out,back]
    return dict(close_reference_frames=len(nearby),minimum_sampled_lead_body_gap_m=round(min((r["body_gap_m"] for r in nearby),default=60.),3),
        adjacent_longitudinal_overlap_frames=len(overlap),physical_rear_clear_observed=bool(complete),
        legal_out_and_return_observed=pair is not None,legal_lane_changes=pair,all_observed_center_lane_changes=changes)


def publication(root):
    sections=[("J1932 bend","bend",(704.7,735.4),None),("Original roadside drums","boulevard",(375.,395.),"roadworks"),
              ("Original roadside boards","boulevard",(525.,550.),"barrier"),
              ("Open boulevard","boulevard",(420.,500.),None),
              ("Stopped lead approach","boulevard",(550.,583.),"lead")]
    result=[]
    pending=[]
    for name,case,interval,kind in sections:
        if not all((root/f"native-{arm}-{case}"/case/"summary.json").is_file() for arm in ("v06","v07")):
            pending.append(name)
            continue
        pair=dict(name=name,station_m=list(interval))
        for arm in ("v06","v07"):
            folder=root/f"native-{arm}-{case}"/case
            score=summarize(folder,*interval)
            event=next((e for e in score["events"] if e["kind"]==kind),{})
            info=dict(median_kmh=score["movement_kmh"]["median"],peak_kmh=score["movement_kmh"]["peak"],
                median_request_kmh=score["requested_kmh"]["median"],seconds=score["seconds"],
                seconds_below_15=score["seconds_below_15"],seconds_below_10=score["seconds_below_10"],
                target_drops_over_5_kmh=score["target_drops_over_5_kmh"],
                worst_half_second_decel_mps2=score["worst_half_second_decel_mps2"],
                p95_half_second_lateral_mps2=score["p95_half_second_lateral_mps2"],
                peak_half_second_lateral_mps2=score["peak_half_second_lateral_mps2"],
                minimum_sampled_pedestrian_gap_m=score["minimum_sampled_pedestrian_gap_m"],
                sampled_nearby_pedestrians=score["sampled_nearby_pedestrians"],
                minimum_prop_clearance_m=event.get("minimum_barrier_clearance_m",event.get("minimum_cone_clearance_m")),
                physical_props_passed=event.get("barriers_alongside",event.get("cones_alongside")),
                minimum_lead_event_gap_m=event.get("minimum_gap_m") if kind=="lead" else None,
                stopped_for_lead=event.get("stopped_for_event") if kind=="lead" else None,
                native_segment_pass=score["native_pass"],tactical_zero_cap_s=score["tactical_zero_cap_s"],
                standing_wait=score["standing_wait"],contacts=score["contacts"],red_violations=score["red"],yield_violations=score["yield_violations"],
                evidence=(folder/"summary.json").as_posix(),camera=(folder/"front.mp4").as_posix(),
                checkpoint_sha256=score["checkpoint"])
            if kind=="lead":
                info["physical_lead"]=lead_witness(folder)
            pair[arm]=info
        pair.update(median_delta_kmh=round(pair["v07"]["median_kmh"]-pair["v06"]["median_kmh"],2),
                    time_delta_s=round(pair["v07"]["seconds"]-pair["v06"]["seconds"],1))
        if kind=="barrier":
            components=[]
            for lo,hi in ((525,530),(530,535),(535,538),(538,545),(545,550)):
                part=dict(station_m=[lo,hi])
                for arm in ("v06","v07"):
                    score=summarize(root/f"native-{arm}-{case}"/case,lo,hi)
                    part[arm]=dict(median_kmh=score["movement_kmh"]["median"],seconds=score["seconds"],
                        requested_median_kmh=score["requested_kmh"]["median"],tactical_zero_cap_s=score["tactical_zero_cap_s"])
                components.append(part)
            pair["station_components"]=components
            pair["components_method"]="Post-hoc fixed-station decomposition of the primary 525–550m window. All native frames included in each arm; primary result includes all tactical waiting."
        result.append(pair)
    controls=[]
    for case,frozen in (("hidden","occluded"),("left","left"),("red","signal")):
        before=root.parents[1]/"city-progress"/"trials"/"matched-candidate-v4"/frozen/"summary.json"
        after=root/"native-v07-safety"/case/"summary.json"
        if not after.is_file():
            continue
        old,new=(json.loads(path.read_bytes()) for path in (before,after))
        for key in ("case","seed","start_m","end_m","target_event"):
            if old[key]!=new[key]:
                raise ValueError(f"Unmatched safety cohort for {case}: {key}")
        control=dict(case=case,seed=new["seed"])
        for arm,score,path in (("v06",old,before),("v07",new,after)):
            event=next(e for e in score["events"] if e["kind"]==score["target_event"])
            control[arm]=dict(passed=score["target_pass"],exposed=event["exposed"],handled=event["handled"],
                stopped_for_event=event["stopped_for_event"],event_minimum_gap_m=event["minimum_gap_m"],
                minimum_sampled_target_body_gap_m=score["nearest_target_footprint"]["gap_m"],
                contacts=score["collision_events"],red_violations=score["red_light_violations"],
                yield_violations=score["yield_violations"],checkpoint_sha256=score["model_sha256"],
                evidence=path.as_posix())
        controls.append(control)
    return dict(schema="ado-city-maneuvers-native-pace-comparison-v1",case_count=len(result),
        pending_native_pairs=pending,negative_controls=controls,method="Original City Life seed 212, identical geometry, actors, approach and rig within each pair. Bend uses primary driving cameras and lidar; boulevard runs use the five-camera rig. Props use the native event scorer. Pedestrian gaps use 10Hz sampled physical footprints with a level ego reconstruction. Whole-arm violations exposed per window.",cases=result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publication",type=Path,help="Summarize all available registered native pairs")
    p.add_argument("--base",type=Path)
    p.add_argument("--candidate",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--window",nargs=2,type=float,action="append",default=[])
    a=p.parse_args()
    if a.publication:
        a.output.write_text(json.dumps(publication(a.publication),indent=2)+"\n",encoding="utf8")
        return
    if a.base is None or a.candidate is None:
        p.error("--base and --candidate are required when comparing two trials")
    comparisons=[]
    for key in a.window or [(None,None)]:
        v0=summarize(a.base,*key)
        v1=summarize(a.candidate,*key)
        comparisons.append(dict(window_m=key,v06=v0,v07=v1,median_delta_kmh=round(
            v1["movement_kmh"]["median"]-v0["movement_kmh"]["median"],2),
            time_delta_s=round(v1["seconds"]-v0["seconds"],1)))
    a.output.write_text(json.dumps(dict(schema="ado-city-maneuvers-native-pace-v1",results=comparisons),
                                    indent=2)+"\n",encoding="utf8")

if __name__=="__main__":
    main()
