"""Opt-in research profile for the pinned Market rehearsal. No simulator is used by audit."""
import argparse
from collections import deque
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import zipfile

import carla
import numpy as np

if __name__=="__main__" and sys.argv[1:2]==["native"]:
    # The section runner archives the current directory. Use those exact sources for imports.
    sys.path.insert(0,str(Path.cwd()))

from crosswalk import route_maneuvers as frozen
from crosswalk.city_maneuvers import ManeuverModel
from crosswalk.maneuver_traffic import rear_projection
from crosswalk.urban_map import urban_route


ROOT = Path(__file__).resolve().parents[1]
STORE = Path(os.environ["LOCALAPPDATA"]) / "ADO/city-maneuvers"
MODEL = ROOT / "models/city-maneuvers/candidate.npz"
MAP = ROOT / ".local/carla-1.0/CarlaUnreal/Packs/AdditionalMaps/Content/Maps/OpenDrive/Town03_Opt.xodr"
PIN = "9bd6dafa0348d37e163a3acf8de553f642bebf45"
RUNNER_HASH = "f17377440e2a122e52257303fcbd3c6ef96bc7764b9be5e5d324c0f1319a6bcd"
STEP = .5  # metres of native station
DT = .125  # same discretization as the frozen rear forecast
REAR_ACCEL = 4.25  # exceeds 3.92 m/s² in the development tape's half-second position estimate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Plan:
    """A scalar speed envelope along both native lane centerlines and their continuous blends."""
    def __init__(self, policy, corridor, station, speed, out_start, out_length, return_start,
                 return_length, source_reach, next_front, next_speed, observed_curvature=0.):
        self.station = station
        self.return_start = return_start
        self.return_length = return_length
        # Sample before and beyond the current target so its first curvature has a centered stencil.
        last = max(station+3.,min(corridor.end,max(station+12.,return_start+return_length+9.)))
        stations = np.arange(station-3.,last+STEP,STEP)
        alpha = frozen.blend(stations, out_start, out_length) * (1.-frozen.blend(stations, return_start, return_length))
        xy = corridor.goals(stations, alpha)
        left, right = xy[3:-3]-xy[:-6], xy[6:]-xy[3:-3]
        norm = np.linalg.norm
        curvature = np.abs(2.*(left[:,0]*right[:,1]-left[:,1]*right[:,0])) / np.maximum(
            1.e-6, norm(left,axis=1)*norm(right,axis=1)*norm(left+right,axis=1))
        self.stations = stations[3:-3]
        self.curvature = curvature
        self.lateral_budget = float(policy.coefficients[1])
        self.recovery = float(policy.coefficients[2])
        self.release = min(1.5, self.recovery)
        self.brake = min(1.5, .5/float(policy.stopping[0]))  # softer than calibrated stopping
        tracking = np.where(self.stations <= station+2.6+speed*.5, observed_curvature, 0.)
        self.cap = np.sqrt(self.lateral_budget / np.maximum(np.maximum(curvature,tracking),.001))
        # A measured next bumper or end of return-lane visibility bounds the approach to a
        # possible queue. A newly seen bumper is treated as stationary until its speed is known.
        obstacle = min(source_reach, next_front)
        available = np.maximum(0., obstacle - (self.stations-station) - policy.standoff)
        a,b,slack,tested = policy.stopping
        distance = np.maximum(0.,available-slack)
        stopping = np.minimum(tested, 2.*distance/(b+np.sqrt(b*b+4.*a*distance)))
        # Evaluate at prospective return initiation. Sight beyond the bumper is never credited.
        idx = int(np.argmin(abs(self.stations-return_start)))
        # A horizon ending before the planned return conveys missing information, not a stop sign.
        # Use the original return headway's low speed until updated scans justify more.
        return_pace = max(3., float(stopping[idx]))
        if next_front < 60.:
            headway = max(3., next_speed+max(0.,next_front-max(0.,return_start-station)-20.)/6.)
            return_pace = min(return_pace, headway)
        # Constrain speed on approach and through the return. Geometry and stock emergency
        # constraints can still be lower than this anticipatory visibility/headway target.
        self.cap = np.minimum(self.cap, np.sqrt(return_pace**2+2.*self.brake*np.maximum(0.,return_start-self.stations)))
        # Backward-pass through all upcoming curvature maxima. Use station distance (shorter
        # than blend arc on straight lanes) and discount the native braking reaction distance.
        cap2 = self.cap**2
        cap2 = np.minimum.accumulate((cap2+2.*self.brake*self.stations)[::-1])[::-1]-2.*self.brake*self.stations
        self.cap = np.sqrt(np.maximum(0.,cap2))
        self.delay = float(b)
        self.request = self.at(station,speed)

    def at(self, station, speed):
        # Consider every point traversed during the actuator lag, including bends behind
        # the far edge of that interval. A lookahead sampled only at its end can skip a peak.
        ahead = station+self.delay*speed
        inside = (self.stations>station)&(self.stations<ahead)
        return min(float(np.interp(station,self.stations,self.cap)),
                   float(np.interp(ahead,self.stations,self.cap)),
                   float(np.min(self.cap[inside],initial=math.inf)))


def forecast(plan, *, speed, rear, rear_speed, rear_seen, target_front, target_speed, exit_station,
             stock_cruise, stock_tactical=None):
    """Model the proposed slower traversal until the whole vehicle can vacate the adjacent lane."""
    sx, vr, sr = 0., max(0.,rear_speed), 0.
    vx, front_v, front = speed, max(0.,target_speed), target_front
    margin = front_margin = math.inf
    elapsed = None
    for step in range(1,161):
        desired = min(stock_cruise, plan.at(plan.station+sx,vx))
        if stock_tactical is not None:
            desired = min(desired,stock_tactical)
        # Lower speeds are pessimistic for rear closure. Never count on acceleration faster
        # than either the measured recovery or the frozen projection's +0.18 per step.
        next_v = min(desired, vx+min(.18,DT*plan.release))
        sx += .5*DT*(vx+next_v)
        vx = next_v
        next_r = min(max(13.,rear_speed),vr+DT*REAR_ACCEL)
        sr += .5*DT*(vr+next_r)
        vr = next_r
        margin = min(margin,rear-2.6+sx-sr-max(3.,1.+1.1*max(0.,vr-vx)))
        next_front_v = max(0.,front_v-.2)
        front += .5*DT*(front_v+next_front_v)
        front_v = next_front_v
        front_margin = min(front_margin,front-sx-2.6-max(3.,1.3*max(0.,vx-front_v)))
        if plan.station+sx >= exit_station:
            elapsed = step*DT
            break
    return dict(time_to_vacate_s=elapsed, rear_margin_m=float(margin) if rear_seen else None,
                target_front_margin_m=float(front_margin) if target_front<60. else None,
                allowed=bool(elapsed is not None and (not rear_seen or margin>0.) and
                             (target_front>=60. or front_margin>0.)))


class ComfortManeuvers(frozen.RouteManeuvers):
    checkpoint = MODEL

    def __init__(self, corridors, model=None):
        super().__init__(corridors,model)
        self.policy = ManeuverModel.load(self.checkpoint)

    def _reset(self):
        super()._reset()
        self.poses = deque(maxlen=6)
        self.observed_curvature = 0.

    def observe_pose(self, x, y, yaw):
        self.poses.append((x,y,math.radians(yaw)))
        if len(self.poses)==6:
            path = np.asarray(self.poses)
            travel = np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).sum()
            delta = path[-1,2]-path[0,2]
            self.observed_curvature = abs(math.atan2(math.sin(delta),math.cos(delta)))/travel if travel>.5 else 0.

    def _measure(self, pose, scan, station, speed, lateral=0.):
        measures = super()._measure(pose,scan,station,speed,lateral)
        self.position = station
        self.measures = measures
        self.observe_pose(pose.location.x,pose.location.y,pose.rotation.yaw)
        return measures

    def projected(self, **args):
        original = rear_projection(**args)
        p = self.policy
        s = self.position
        x, _, _, sight, _, _, _, _ = self.measures
        gap, lv, front, _, _, _, speed = x
        start = self.start if self.start is not None else s
        out_length = self.travel_length if self.start is not None else min(24.,max(12.,(gap-3.)*1.05))
        lead = self.lead_station if self.lead_station is not None else s+gap
        lead_v = self.lead_speed if self.lead_speed is not None else lv
        new_front = gap if s+gap>lead+12. and gap<60. else 60.
        new_speed = max(0., self.surfaces[0].speed or 0.)
        returning = self.return_start is not None
        # The release planner requires five agreeing scans and room past the moving lead.
        # Forecast the lead moving during the frozen traversal, including the signal checks.
        if returning:
            inbound = self.return_start
            length = self.return_length
        else:
            # Use the full projection horizon for a moving lead. A slower profile cannot
            # assume that the lead stops advancing after the release's shorter forecast.
            lead_at_return = lead+lead_v*20.
            inbound = max(s, start+out_length, lead_at_return+19.)+.5*speed
            length = float(np.clip(3.5+2.7*speed,14.,30.))
        plan = Plan(p,self.corridor,s,speed,start,out_length,inbound,length,sight[0],new_front,new_speed,self.observed_curvature)
        exit_station = max(s+args["transition_length"] if args["phase"]=="ongoing" else s,
                           inbound+length+2.6)
        # Respect the slower cruise assumptions of the release under limited sight. Apply
        # that lower speed through the whole horizon, including the phase before the turn.
        cruise = min(9.5,max(7.5,speed+3.)) if args["phase"]=="entry" else max(3.,speed)
        if args["source_sight"] < max(0.,inbound-s)+.85*length or args["source_ahead"]<55.:
            cruise = min(cruise, max(3.5,.72*cruise))
        if args["source_ahead"]<55.:
            cruise = min(cruise,max(2.75,(args["source_ahead"]-19.)/6.))
        second = forecast(plan,speed=speed,rear=args["rear"],rear_speed=args["rear_speed"],
                          rear_seen=args["rear_seen"],target_front=front,target_speed=args["target_speed"],
                          exit_station=exit_station,stock_cruise=cruise,stock_tactical=self.return_ceiling)
        if exit_station > self.corridor.end:
            second.update(allowed=False, corridor_exhausted=True)
        if args["rear_seen"] and not returning and sight[0]<exit_station-s:
            # A positively measured rear may catch us while we wait for a return opening
            # outside the current pavement sightline. Do not admit discretionary slowing
            # on the assumption that a fresh return measurement will arrive on schedule.
            second.update(allowed=False, clear_station_beyond_source_sight=True)
        self.check = dict(original=original, slower=second, suggested_limit_mps=plan.request,
                          future_return_station_m=inbound, clear_station_m=exit_station)
        if self.start is None:
            # Only a veto at admission. Once committed, a tightened forecast must never make
            # the parent relax a forward ceiling or delay an otherwise safe chance to exit.
            return {**original,"allowed":original["allowed"] and second["allowed"]}
        return original

    def preview(self, ego, lidar, defaults, speed, lookaheads=frozen.LOOK):
        self.check = None
        with patch.object(frozen,"rear_projection",self.projected):
            points = super().preview(ego,lidar,defaults,speed,lookaheads)
        if self.check is not None:
            check = self.check
            apply = (self.info["committed"] or self.phase=="signal_left") and all(
                check[key]["allowed"] for key in ("original","slower"))
            check["applied"] = bool(apply)
            check["frozen_tactical_limit_mps"] = self.speed_limit
            if apply:
                self.speed_limit = min(check["suggested_limit_mps"],self.speed_limit if self.speed_limit is not None else math.inf)
                self.info["tactical_speed_limit_mps"] = self.speed_limit
            self.info["comfort_candidate"] = check
        return points

def entry_probe(planner, station, speed, gap, rear, reach, rear_speed=0., lead_speed=0.):
    planner.position = station
    planner.measures = ((gap,lead_speed,60.,rear,0.,max(0.,rear_speed-speed),speed),
                        None,None,[reach,60.,60.,60.],None,None,None,None)
    planner.projected(speed=speed, rear=rear, rear_speed=rear_speed, rear_seen=True, gap=gap,
                      lead_speed=lead_speed, source_sight=reach, target_front=60., target_speed=0.,
                      source_ahead=60., phase="entry", transition_length=min(24.,max(12.,(gap-3.)*1.05)))
    return planner.check


def read_rows(path):
    op = gzip.open if path.suffix==".gz" else open
    with op(path,"rt") as stream:
        return [json.loads(line) for line in stream]


def archive_native_rear(rows):
    points = np.asarray([[r["sim_time"],o["x"],o["y"]] for r in rows
                         for o in r.get("observer_map",r)["others"] if o["role"]=="adjacent-rear"])
    # Report trailing half-second chord speed, and changes in that smooth speed over another
    # half second. Native positions in these reference tapes were rounded to centimetres.
    velocity = np.linalg.norm(points[5:,1:]-points[:-5,1:],axis=1)/(points[5:,0]-points[:-5,0])
    accel = (velocity[5:]-velocity[:-5])/(points[10:,0]-points[5:-5,0])
    moved = np.flatnonzero(velocity>.5)
    return dict(peak_speed_mps=float(max(velocity)),peak_positive_accel_mps2=float(max(accel)),
                net_displacement_m=float(np.linalg.norm(points[-1,1:]-points[0,1:])),
                first_half_second_speed_over_half_mps_after_s=float(points[moved[0]+5,0]-points[0,0]) if len(moved) else None,
                first_frame=rows[0]["frame"],last_frame=rows[-1]["frame"])


def audit(output):
    source = STORE/"integrated-section01"
    trace_file,pose_file = source/"trace.jsonl.gz",source/"reference-native.jsonl"
    trace,poses = read_rows(trace_file),read_rows(pose_file)
    by_frame = {r["frame"]:r for r in trace}
    pose = {r["frame"]:r["observer_map"]["ego"] for r in poses}
    entry = next(r for r in trace if r["tactical_pass"]["maneuver_phase"]=="change_left")
    f = entry["frame"]
    body = next(r for r in trace if r["frame"]>=f and r["tactical_pass"].get("target_lane_vacated"))
    road_map = carla.Map("Town03_Opt",MAP.read_text())
    corridor = next(c for c in frozen.discover(road_map,[w.transform.location for w in urban_route(
        road_map,212,"market").points]) if list(c.source_key)==entry["tactical_pass"]["source_lane"])
    planner = ComfortManeuvers([corridor])
    planner.corridor = corridor
    probes = []
    for frame in range(f-5,f+1):
        r = by_frame[frame]
        ap = r["tactical_pass"]
        planner.observe_pose(*pose[frame])
        checks = []
        for vr in (0.,6.,13.):
            check = entry_probe(planner,ap["corridor_station_m"],r["speed_mps"],ap["source_gap_m"],ap["target_rear_m"],
                                ap["pavement_reach_m"][0],vr,ap["source_speed_mps"])
            checks.append(dict(initial_rear_mps=vr,**check))
        probes.append(dict(frame=frame, native_forecast=ap["rear_occupancy_forecast"], probes=checks))

    # Hold the native, measured ego stations and speeds fixed; change only the rear driver's
    # trajectory. This isolates the stock forecast horizon error from candidate slowing.
    recorded_margin = math.inf
    first_negative = None
    first = entry["tactical_pass"]
    obs_rear_stations = {}
    for witness in poses:
        if f <= witness["frame"] <= body["frame"]:
            actor = next(a for a in witness["observer_map"]["others"] if a["role"]=="adjacent-rear")
            angle = math.radians(actor["yaw"])
            located = corridor.locate(np.array([actor["x"],actor["y"]]),np.array([math.cos(angle),math.sin(angle)]))
            if located is not None:
                obs_rear_stations[witness["frame"]] = located[0]
    for frame in range(f,body["frame"]+1):
        r = by_frame[frame]
        t = r["sim_time"]-entry["sim_time"]
        accel_time = min(t,13./3.6)
        rear_travel = .5*3.6*accel_time**2+13.*(t-accel_time)
        rear_v = min(13.,3.6*t)
        sx = r["tactical_pass"]["corridor_station_m"]-first["corridor_station_m"]
        margin = first["target_rear_m"]-2.6+sx-rear_travel-max(3.,1.+1.1*max(0.,rear_v-r["speed_mps"]))
        recorded_margin = min(recorded_margin,margin)
        if first_negative is None and margin<=0.:
            first_negative = frame

    # Optimistic alternative source-lane starts before the cone. The parked rear stays put,
    # all target-side evidence is free, 60 m source sight is available, and source speed is
    # stationary. A start this close to cones is not necessarily geometrically drivable.
    s0 = first["corridor_station_m"]
    cone = s0+first["source_gap_m"]
    rear_station = s0-first["target_rear_m"]
    best = {}
    planner.observed_curvature = 0.
    for s in np.r_[np.arange(s0,cone-2.6,.5),cone-2.6]:
        for v in np.arange(0.,11.01,.5):
            probe = entry_probe(planner,s,v,cone-s,s-rear_station,60.)
            m = probe["slower"]["rear_margin_m"]
            if not best or m>best["rear_margin_m"]:
                best = dict(station_m=float(s), speed_mps=float(v), rear_margin_m=m, gate=probe)
    # Isolate proposed ceilings on the recorded forced path, regardless of the entry veto.
    # This is a diagnostic of the planner's requests, never a simulated physical outcome.
    planner._reset()
    planner.corridor = corridor
    turn_start = None
    started = first["corridor_station_m"]
    out_length = min(24.,max(12.,(first["source_gap_m"]-3.)*1.05))
    conditional = []
    for frame in range(f-5,body["frame"]+1):
        r = by_frame[frame]
        a = r["tactical_pass"]
        planner.observe_pose(*pose[frame])
        if frame<f:
            continue
        s,speed = a["corridor_station_m"],r["speed_mps"]
        if a["return_blend_length_m"] is not None and turn_start is None:
            turn_start = s
        lead = a["predicted_lead_station_m"]
        length = a["return_blend_length_m"] or float(np.clip(3.5+2.7*speed,14.,30.))
        dest = turn_start if turn_start is not None else max(
            s,started+out_length,lead+20.*a["tracked_lead_speed_mps"]+19.)+.5*speed
        new_front = a["source_gap_m"] if a["source_gap_m"]<60. and s+a["source_gap_m"]>lead+12. else 60.
        plan = Plan(planner.policy,corridor,s,speed,started,out_length,dest,length,a["pavement_reach_m"][0],
                    new_front,a["source_speed_mps"] if a["source_age_scans"]==0 and a["source_speed_mps"]!=20. else 0.,
                    planner.observed_curvature)
        conditional.append(dict(frame=frame,stock_request_mps=r["speed_request_mps"],actual_mps=speed,
                                proposed_ceiling_mps=plan.request,stock_request_clamped_mps=min(plan.request,r["speed_request_mps"])))
    uncooperative = ROOT/"results/city-maneuvers/attempts/uncooperative-before01"
    comp = read_rows(uncooperative/"reference.jsonl.gz")
    unc_grade = json.loads((uncooperative/"summary.json").read_bytes())["case_score"]
    observed_motion = archive_native_rear([r for r in poses if f<=r["frame"]<=body["frame"]])
    observed_motion.update(matched_native_stations=len(obs_rear_stations),
        min_center_station_gap_m=min(by_frame[i]["tactical_pass"]["corridor_station_m"]-s for i,s in obs_rear_stations.items()),
        rear_native_station_travel_m=obs_rear_stations[body["frame"]]-obs_rear_stations[f])
    intervals = {}
    for window in [(13924,13937),(13955,13965),(13966,13970)]:
        ticks = [r for r in conditional if window[0]<=r["frame"]<=window[1]]
        intervals[f"{window[0]}-{window[1]}"] = {k: {"min":min(r[k] for r in ticks),"max":max(r[k] for r in ticks)}
                                         for k in ticks[0] if k!="frame"}
    result = dict(scope="CPU-only archive analysis and native-map path preview. No candidate vehicle trajectory or paired comfort gain.",
        pin=PIN, source_sha256={str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p.name):digest(p)
                      for p in [Path(__file__),MODEL,MAP,trace_file,pose_file]},
        checkpoint=dict(lateral_mps2=planner.policy.coefficients[1], recovery_mps2=planner.policy.coefficients[2],
                        native_stopping=planner.policy.stopping.tolist(),standoff_m=planner.policy.standoff),
        entry_gates=probes,
        stock_fixed_native_ego_and_hypothetical_rear=dict(start_frame=f, vacated_frame=body["frame"],
              native_occupancy_s=round(body["sim_time"]-entry["sim_time"],2),initial_rear_station_gap_m=first["target_rear_m"],
              original_projected=first["rear_occupancy_forecast"],assumed_initial_rear_mps=0.,assumed_max_accel_mps2=3.6,
              assumed_max_speed_mps=13.,first_nonpositive_frame=first_negative,minimum_margin_m=recorded_margin),
        candidate_rear_projection=dict(acceleration_mps2=REAR_ACCEL,speed_ceiling_mps="max(13, measured rear speed)",
            limit="Design envelope checked against these captures. Other simulated trajectories can exceed it."),
        observed_continuous_rear_during_occupancy=observed_motion,
        observed_uncooperative_comparator=dict(reference_sha256=digest(uncooperative/"reference.jsonl.gz"),
            grade_sha256=digest(uncooperative/"summary.json"),reference_motion=archive_native_rear(comp),
            grade_failures=unc_grade["failures"],closest_body_gap=unc_grade["closest_moving_adjacent_vehicle"]),
        prospective_delayed_starts=dict(min_station_m=s0,max_station_m=cone-2.6,station_step_m=.5,
            speed_range_mps=[0.,11.],speed_step_mps=.5,optimistic_source_reach_m=60.,rear_station_assumed_m=rear_station,
            best_sample=best, continuous_fixture_rear_release="native road s=81, after stationary cones at native s=70,71,73",
            interpretation="Sampled states are hypothetical. None are observations of a delayed native run."),
        forced_path_ranges=intervals,forced_path_frames=conditional,
        verdict="Profile vetoes every archived initial invitation. Conditional requests are diagnostic only; no defensible measured improvement.")
    output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print("Recorded-ego rear stress",recorded_margin,"best delayed gate",best["rear_margin_m"],"report",output)


def native(output_id):
    if not output_id or Path(output_id).name!=output_id or output_id in (".",".."):
        raise ValueError("Expected a new trial name")
    runner = STORE/"section_rehearsal.py"
    if Path(frozen.__file__).resolve().parents[1]!=Path.cwd().resolve():
        raise RuntimeError("Launch from the clean release worktree so imported and archived sources agree")
    if digest(runner)!=RUNNER_HASH:
        raise RuntimeError("Matched section runner changed")
    archive_path = STORE/output_id/"source.zip"
    checkpoint = Path.cwd()/"models/city-maneuvers/candidate.npz"
    class CandidateArchive(zipfile.ZipFile):
        def __init__(self, file, mode="r", *args, **kwargs):
            super().__init__(file,mode,*args,**kwargs)
            if mode=="x" and Path(file).resolve()==archive_path.resolve():
                self.write(__file__,"tools/maneuver_comfort_candidate.py")
                self.writestr("comfort-override.json",json.dumps(dict(pin=PIN,runner_sha256=digest(runner),
                    override_sha256=digest(Path(__file__)),class_name="ComfortManeuvers",checkpoint_sha256=digest(checkpoint))))
    argv = [str(runner),output_id,"278","625","continuous"]
    with patch.object(sys,"argv",argv), patch.object(zipfile,"ZipFile",CandidateArchive), patch.object(
            frozen,"RouteManeuvers",ComfortManeuvers), patch.object(ComfortManeuvers,"checkpoint",checkpoint):
        runpy.run_path(str(runner),run_name="__main__")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("audit").add_argument("--output",type=Path,
        default=ROOT/"results/city-maneuvers/passing/comfort/offline.json")
    sub.add_parser("native").add_argument("output_id")
    args = parser.parse_args()
    if args.command=="audit":
        audit(args.output)
    else:
        native(args.output_id)
