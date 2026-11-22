"""Road-65 transfer using the normal city episode, sensor rig and checkpoint."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import os
from tempfile import mkdtemp
import traceback

import carla
import cv2
import numpy as np

from .city_boulevard import ClearBoulevard
from .city_drive import LoggedDriver, load_driver
from .city_progress_bus import body_gap, footprint
from .drive import episode
from .experiment import PROJECT, fingerprint, write_json, write_trace
from .maneuver_choice import MODEL
from .runtime import producer_info, simulator
from .showcase import encode_and_check
from .world import Scenario, _raised


class TransferRoad(ClearBoulevard):
    road_id, lane_id = 65, 3
    _drive_to = Scenario._drive_to

    def __init__(self, world, *args, case, **kwargs):
        # Use a fresh native road and the normal ego/rig; no fixture data
        # or corridor coordinates are passed to the decision model.
        self.stack = ExitStack()
        self.world, self._map, self.case = world, world.get_map(), case
        self.rows, self.tick = [], 0
        self.owned_actor_ids = []
        self.was_alongside = False
        try:
            existing = world.get_actors()
            if existing.filter("vehicle.*") or existing.filter("walker.*"):
                raise RuntimeError("Transfer requires an empty traffic world")
            self.stack.callback(world.set_weather,world.get_weather())
            world.set_weather(carla.WeatherParameters.ClearNoon)
            points = [self._map.get_waypoint_xodr(65,3,float(s)) for s in range(210,51,-1)]
            if any(p is None or p.is_junction or p.lane_type!=carla.LaneType.Driving for p in points):
                raise ValueError("Unexpected transfer route")
            self._set_route([p.transform.location for p in points])
            self.width = min(p.lane_width for p in points)
            self.ego = self._vehicle(3,210,"vehicle.lincoln.mkz","hero")
            self.hazard = None if case=="no-lead" else self._vehicle(3,180,"vehicle.tesla.model3","source-lead")
            self.traffic = self._vehicle(2,208,"vehicle.mini.cooper","adjacent") if case in ("occupied","rebrake") else None
        except BaseException:
            self.close()
            raise

    def _vehicle(self, lane, station, blueprint, name):
        point = self._map.get_waypoint_xodr(65,lane,float(station))
        bp = self.world.get_blueprint_library().find(blueprint)
        bp.set_attribute("role_name",name)
        actor = self.world.spawn_actor(bp,_raised(point.transform,.35))
        self.stack.callback(actor.destroy)
        self.owned_actor_ids.append(actor.id)
        actor.apply_control(carla.VehicleControl(brake=1.))
        return actor

    def before_tick(self):
        # Only the other drivers can use these prescribed speeds & timing.
        self.tick += 1
        lead_speed = 0. if self.case=="rebrake" and self.tick>58 else 1.55
        for actor, lane, speed in ((self.hazard,3,lead_speed),(self.traffic,2,9. if self.case=="occupied" and self.tick>35 else 0.)):
            if actor is None:
                continue
            wp = self._map.get_waypoint(actor.get_location())
            goal = self._map.get_waypoint_xodr(65,lane,max(8.,wp.s-8.))
            self._drive_to(actor,goal.transform.location,speed)

    def metrics(self):
        state = super().metrics()
        state.update(route_id="Town03 road 65 lane +3, from s=210 to 52",road_ids=[65],phase="transfer")
        wp = self._map.get_waypoint(self.ego.get_location())
        source = self._map.get_waypoint_xodr(65,3,wp.s)
        t = source.transform
        p = t.location
        f, r = t.get_forward_vector(), t.get_right_vector()
        def envelope(actor):
            poly = footprint(actor)
            delta = poly - [p.x,p.y]
            along, side = delta@[f.x,f.y],delta@[r.x,r.y]
            return poly,along.min(),along.max(),side.min(),side.max()
        ego,eb,ef,el,er = envelope(self.ego)
        observation = dict(frame=self.world.get_snapshot().frame,ego_lane=[wp.road_id,wp.lane_id],
                           native_s=wp.s,ego_speed_mps=self.ego.get_velocity().length(),source_offset_m=(el+er)/2,
                           whole_ego_in_source=bool(el >= -source.lane_width/2 and er <= source.lane_width/2))
        if self.hazard is not None:
            lead,lr,lf,_,_ = envelope(self.hazard)
            alongside = ef>=lr and eb<=lf
            self.was_alongside |= alongside and wp.lane_id==2
            observation.update(lead_speed_mps=self.hazard.get_velocity().length(),
                               lead_gap_m=lr-ef,lead_rear_clear_m=eb-lf,
                               lead_body_gap_m=body_gap(ego,lead),alongside=bool(alongside and wp.lane_id==2))
            observation["returned_ahead"] = bool(self.was_alongside and eb>lf+2. and observation["whole_ego_in_source"]
                                                  and abs(observation["source_offset_m"])<.55)
        if self.traffic is not None:
            other,back,nose,_,_ = envelope(self.traffic)
            observation.update(adjacent_bumper_gap_m=back-ef,adjacent_body_gap_m=body_gap(ego,other),
                               adjacent_overlaps_longitudinally=bool(ef>=back and eb<=nose))
        self.rows.append(observation)
        return state

    def finished(self):
        if not self.rows:
            return False
        if self.case=="rebrake":
            tail = self.rows[-18:]
            return self.tick>90 and all(r["lead_speed_mps"]<.1 and r["ego_speed_mps"]<.2 for r in tail)
        return bool((len(self.rows)>=12 and all(r.get("returned_ahead") for r in self.rows[-12:])
                     and self.rows[-1]["native_s"]<119.)
                    or (self.case=="no-lead" and self.rows[-1]["native_s"]<135.))


class FrontVideo:
    primary_rig_only = True
    def __init__(self, stream, folder, scenes):
        self.stream, self.folder, self.scenes = stream, folder, scenes
        self.saved = set()
    def append(self, row, obs):
        ok, jpeg = cv2.imencode(".jpg",obs["front"][:,:,::-1])
        if not ok:
            raise RuntimeError("Missing front RGB exposure")
        self.stream.write(jpeg.tobytes())
        tactical = row["tactical_pass"]
        phase = tactical["maneuver_phase"]
        if phase=="yield" and tactical.get("yielding_follow_ceiling_mps",0.)>.5:
            phase = "rolling-yield"
        if phase not in self.saved:
            t = self.scenes[0].ego.get_transform()
            self.folder.mkdir(exist_ok=True)
            np.savez_compressed(self.folder/f"{row['frame']}-{phase}.npz",lidar=obs["lidar"],radar=obs["radar"],
                                ego_pose=[t.location.x,t.location.y,t.location.z,t.rotation.yaw,t.rotation.pitch,t.rotation.roll],
                                frame=row["frame"],speed_mps=row["speed_mps"])
            self.saved.add(phase)


def score(case, rows, summary):
    commits = [r for r in rows if r["tactical_pass"]["committed"]]
    waits = [r for r in rows if r["tactical_pass"]["maneuver_phase"]=="yield"]
    moving_alongside = [r for r in rows if r.get("alongside") and r["lead_speed_mps"]>.7]
    returned = next((r for r in rows if r.get("returned_ahead")),None)
    phases = {r["tactical_pass"]["maneuver_phase"] for r in rows}
    lights = {r["tactical_pass"].get("previous_signal_bits") for r in rows}
    stopped = [r for r in waits if r.get("adjacent_overlaps_longitudinally") and r["speed_mps"]<.35 and r["throttle"]==0.]
    bounded_follow = [r for r in waits if r.get("adjacent_overlaps_longitudinally")
                      and r.get("lead_speed_mps",0.)>.7 and .4<r["speed_mps"]<=r["lead_speed_mps"]+.6]
    result = dict(physical_rolling_yield_frames=len(bounded_follow),committed_frames=len(commits),waiting_frames=len(waits),held_while_physically_occupied_frames=len(stopped),
                  moving_alongside_frames=len(moving_alongside),first_actual_rejoin_frame=returned and returned["frame"],
                  active_signal_bits_seen=sorted(lights),first_commit_frame=commits[0]["frame"] if commits else None,
                  lead_speed_at_commit_mps=commits[0].get("lead_speed_mps") if commits else None,
                  adjacent_bumper_gap_at_commit_m=commits[0].get("adjacent_bumper_gap_m") if commits else None,
                  minimum_lead_body_clearance_m=min((r["lead_body_gap_m"] for r in rows if "lead_body_gap_m" in r),default=None),
                  all_committed_footprints_legal=all(r["passing_footprint_in_legal_lanes"] for r in commits),
                  maximum_source_offset_m=max(abs(r["source_offset_m"]) for r in rows))
    clean = summary["outcome"] in ("restarted","complete") and not summary["collision_events"] and result["all_committed_footprints_legal"]
    if case=="no-lead":
        clean &= not commits and result["maximum_source_offset_m"]<.65 and summary["progress_m"]>70.
    elif case=="rebrake":
        final = rows[-15:]
        clean &= bool(not commits and bounded_follow and len(final)==15 and
                      all(r["speed_mps"]<.2 and r["lead_speed_mps"]<.15 and r["throttle"]==0. for r in final)
                      and result["minimum_lead_body_clearance_m"]>2.5 and result["maximum_source_offset_m"]<.65)
    else:
        clean &= bool(returned and moving_alongside and {"change_left","change_right","rejoined"}<=phases
                      and {int(carla.VehicleLightState.LeftBlinker),int(carla.VehicleLightState.RightBlinker)}<=lights
                      and result["minimum_lead_body_clearance_m"]>.25 and result["lead_speed_at_commit_mps"]>.65)
        if case=="occupied":
            clean &= bool((stopped or bounded_follow) and commits and waits[0]["frame"] < commits[0]["frame"]
                          and result["adjacent_bumper_gap_at_commit_m"]>15.)
    return dict(passed=bool(clean),**result)


def run(world, args, case):
    folder = args.output / case
    folder.mkdir()
    driver,_,schema = load_driver(args.model)
    if not getattr(driver,"uses_route_maneuvers",False):
        raise ValueError("Transfer requires the new driver")
    scenes = []
    def create(*p,**kw):
        scene = TransferRoad(*p,case=case,**kw)
        scenes.append(scene)
        return scene
    scratch = Path(os.environ.get("LOCALAPPDATA",str(PROJECT/"out")))/"ADO"/"city-maneuvers"
    scratch.mkdir(parents=True,exist_ok=True)
    raw = Path(mkdtemp(prefix=f"transfer65-{case}-",dir=scratch))/"front.mjpeg"
    with raw.open("xb") as stream:
        summary,rows,_,_ = episode(world,"boulevard",7401,schema,LoggedDriver(driver),max_steps=480,
                                   scene_factory=create,tactical=True,weather="ClearNoon",recording=FrontVideo(stream,folder/"sensor-evidence",scenes),
                                   change_request=lambda:scenes[0].finished())
    assert len(rows)==len(scenes[0].rows) and all(r["frame"]==truth["frame"] for r,truth in zip(rows,scenes[0].rows))
    for r,truth in zip(rows,scenes[0].rows):
        r.update(truth)
    write_trace(folder/"trace.jsonl.gz",rows)
    summary.update(case=case,producer=producer_info(world),score=score(case,rows,summary),
                   raw_front_sha256=fingerprint(raw))
    write_json(folder/"summary.json",summary)
    encode_and_check(raw,folder/"front.mp4",(640,360),0,len(rows))
    # CARLA's actor cache advances with a snapshot. Check just the IDs this
    # test spawned, leaving all other actors untouched.
    for _ in range(3):
        world.tick()
        if not set(scenes[0].owned_actor_ids).intersection(a.id for a in world.get_actors().filter("vehicle.*")):
            break
    else:
        raise RuntimeError("The transfer's vehicle destruction was not acknowledged")
    print(case,summary["outcome"],summary["score"],flush=True)
    return summary["score"]["passed"]


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--model",type=Path,default=PROJECT/"models/city-maneuvers/candidate.npz")
    cli.add_argument("--port",type=int,default=3718)
    cli.add_argument("--attach",action="store_true")
    cli.add_argument("--cases",nargs="+",choices=("clear","occupied","no-lead","rebrake"),default=("clear","occupied","no-lead","rebrake"))
    a=cli.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    src=Path(__file__).parent
    files=list(src.glob("*.py"))
    sources = {p.name:fingerprint(p) for p in files}
    sources["maneuver_transfer.py"] = fingerprint(Path(__file__))
    (a.output/"executed-transfer.py").write_bytes(Path(__file__).read_bytes())
    write_json(a.output/"protocol.json",dict(road=65,source_lane=3,initial_ego_s=210,
        initial_lead_s=180,lead_speed_mps=1.55,occupied_adjacent_s=208,adjacent_release_tick=36,
        checks="moving lead at commitment and alongside, return with body clear, native signal readback, legal envelopes",
        rebrake_starts_tick=59,cases=a.cases,seed=7401,model_sha256=fingerprint(a.model),tactical_sha256=fingerprint(MODEL),
        source_sha256=sources))
    passed=[]
    with simulator(a.port,a.attach,"High",map_name="Town03_Opt") as world:
        for case in a.cases:
            # Destroyed actors disappear from CARLA's cached actor list on
            # the next world snapshot. Never remove unrelated world actors.
            world.tick()
            try:
                passed.append(run(world,a,case))
            except Exception:
                write_json(a.output/(case+"-error.json"),dict(traceback=traceback.format_exc()))
                passed.append(False)
        world.tick()
    return all(passed)


if __name__=="__main__":
    raise SystemExit(0 if main() else 1)
