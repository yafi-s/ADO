"""Paired Town03 tactical passing trials using real CARLA physics and range sensors.

The same fitted city steering, longitudinal and tactical models drive every
variant. A nearby autonomous NPC changes only the available adjacent gap.
Camera and mapped-light inputs are empty on this un-signalled test segment.
Scenario truth is accessed only by independent scoring and the NPC drivers.
"""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import carla
import numpy as np

from .city_drive import load_driver
from .control import Navigation, vehicle_speed
from .overtake import TacticalPass, PassingCorridor, MODEL
from .runtime import simulator, producer_info
from .sensors import Rig
from .world import _raised

LOOK = (3.,7.,13.,18.,25.,32.,40.)


def _spawn(world, road_map, stack, blueprint, lane, station, *, road=4):
    wp = road_map.get_waypoint_xodr(road,lane,station)
    actor = world.spawn_actor(world.get_blueprint_library().find(blueprint), _raised(wp.transform,.35))
    stack.callback(actor.destroy)
    actor.apply_control(carla.VehicleControl(brake=1.))
    return actor


def _npc(actor, speed):
    if actor is not None:
        error = speed - vehicle_speed(actor)
        actor.apply_control(carla.VehicleControl(throttle=float(np.clip(.46*error,0.,.85)),
                                                 brake=1. if speed==0. else float(np.clip(-.45*error,0.,1.))))


def _footprint(actor, corridor):
    verts = actor.bounding_box.get_world_vertices(actor.get_transform())
    arc, side = corridor.coordinates(np.array([(v.x,v.y) for v in verts]))
    return float(arc.min()), float(arc.max()), float(side.min()), float(side.max())


def _score(rows, case):
    committed = [r for r in rows if r['tactical'].get('committed')]
    waiting = [r for r in rows if r['tactical'].get('maneuver')=='yield']
    alongside = [r for r in rows if r.get('alongside_lead')]
    clear = next((r for r in rows if r.get('physically_overtaken')), None)
    result = dict(case=case, frames=len(rows), collisions=rows[-1]['collisions'],
                  yield_s=round(.1*len(waiting),2),
                  first_yield_s=waiting[0]['time_s'] if waiting else None,
                  first_commit_s=committed[0]['time_s'] if committed else None,
                  clearance_at_commit_m=committed[0].get('adjacent_bumper_gap_m') if committed else None,
                  first_alongside_s=alongside[0]['time_s'] if alongside else None,
                  completed_pass_s=clear['time_s'] if clear else None,
                  lead_travel_m=clear['lead_travel_m'] if clear else None,
                  lead_speed_during_pass_mps=min((r['lead_speed_mps'] for r in alongside),default=None),
                  minimum_passing_clearance_m=min((r['lead_lateral_clearance_m'] for r in alongside),default=None),
                  minimum_adjacent_gap_while_yielding_m=min((abs(r['adjacent_bumper_gap_m']) for r in waiting
                                                              if 'adjacent_bumper_gap_m' in r),default=None),
                  max_yield_lateral_error_m=max((abs(r['ego_offset_m']) for r in waiting),default=None),
                  ego_travel_m=rows[-1]['ego_travel_m'], maximum_source_displacement_m=max(abs(r['ego_offset_m']) for r in rows))
    ok = not result['collisions'] and all(r['in_legal_lanes'] for r in rows)
    if case=='no-lead':
        ok &= not committed and result['ego_travel_m']>=45 and result['maximum_source_displacement_m'] < .8
    elif case == 'departing-lead':
        awaiting_release = [r for r in waiting if r['time_s'] < 4.]
        moving = next((r for r in rows if r['time_s'] >= 4. and r['lead_speed_mps'] > 3.), None)
        restart = next((r for r in rows if moving and r['time_s'] >= moving['time_s']
                        and r['speed_mps'] > 1. and r['throttle'] > 0.), None)
        premature = [r for r in rows if awaiting_release and
                     awaiting_release[0]['time_s'] <= r['time_s'] < 4. and r['throttle'] > 0.]
        result.update(lead_accelerated_s=moving and moving['time_s'], resumed_following_s=restart and restart['time_s'],
                      premature_throttle_frames=len(premature), minimum_lead_gap_m=min(r['lead_bumper_gap_m'] for r in rows))
        ok &= bool(awaiting_release and moving and restart and not premature and not committed
                   and result['ego_travel_m'] >= 30. and result['maximum_source_displacement_m'] < .8)
    else:
        ok &= bool(clear and alongside and result['minimum_passing_clearance_m'] > .25 and result['lead_travel_m'] > 3.
                   and result['lead_speed_during_pass_mps'] > .4)
    if case=='occupied':
        ok &= bool(waiting and committed and result['first_yield_s'] < result['first_commit_s']
                   and result['yield_s'] >= .6 and result['max_yield_lateral_error_m'] < .75
                   and result['clearance_at_commit_m'] > 15.)
        if waiting and committed and 'tactical_speed_limit_mps' in waiting[0]['tactical']:
            held = [r for r in rows if waiting[0]['time_s'] <= r['time_s'] < committed[0]['time_s']]
            stopped = next((i for i,r in enumerate(held) if r['speed_mps'] <= .05), len(held))
            result.update(hold_throttle_frames=sum(r['throttle'] > 0. for r in held),
                          peak_speed_after_stop_mps=max((r['speed_mps'] for r in held[stopped:]), default=None),
                          unheld_frames_before_commit=sum(r['tactical']['tactical_speed_limit_mps'] != 0. for r in held))
            ok &= bool(stopped < len(held) and result['peak_speed_after_stop_mps'] < .1
                       and not result['hold_throttle_frames'] and not result['unheld_frames_before_commit'])
    if case=='clear':
        ok &= not waiting
    result['passed'] = bool(ok)
    return result


def evaluate(world, case, model_path, folder, record=False, *,
             source_lane=(4, 3), ego_s=86., corridor=None, native_rgb=False):
    model, _, _ = load_driver(model_path)
    observer = model.new_observer()
    road_map = world.get_map()
    if corridor is None:
        corridor = PassingCorridor.from_map(road_map, source_lane)
    planner = TacticalPass(corridor, measured_hold=hasattr(model, 'tactical_speed_limit'),
                           confirmed_release=getattr(model, 'confirmed_tactical_release', False))
    road, lane = source_lane
    direction = -1. if lane > 0 else 1.
    target = road_map.get_waypoint_xodr(road, lane, ego_s).get_left_lane().lane_id
    with ExitStack() as stack:
        ego = _spawn(world,road_map,stack,'vehicle.lincoln.mkz',lane,ego_s,road=road)
        lead = _spawn(world,road_map,stack,'vehicle.tesla.model3',lane,ego_s+direction*24.,road=road) if case!='no-lead' else None
        other = _spawn(world,road_map,stack,'vehicle.mini.cooper',target,ego_s+direction*2.,road=road) if case in ('occupied', 'departing-lead') else None
        route = [road_map.get_waypoint_xodr(road,lane,ego_s+direction*i).transform.location for i in range(-3,86)]
        navigation = Navigation(route)
        rig = Rig(world,ego,range_only=not native_rgb)
        stack.callback(rig.close)
        detections = []
        if native_rgb:
            from .lamp import RouteLamps
            from .vision import Detector
            detector = Detector()
            lamps = RouteLamps(world, route)
        chase = None
        if record:
            from .robotaxi import BodyKit
            from .showcase import ChaseCapture, encode_and_check
            body = BodyKit(world,ego)
            stack.callback(body.close)
            tmp = Path(stack.enter_context(TemporaryDirectory(dir=folder)))
            raw = tmp/'chase.mjpeg'
            stream = stack.enter_context(raw.open('wb'))
            chase = ChaseCapture(960,540,75,stream)
            chase.attach(world,ego)
            stack.callback(chase.close)
        for _ in range(9):
            rig.read(world.tick())
        ego_start = _footprint(ego,corridor)[0]
        lead_start = _footprint(lead,corridor)[0] if lead is not None else None
        rows = []
        for step in range(420):
            _npc(lead,8. if case == 'departing-lead' and step >= 40 else 1.45)
            # Scenario intervention on the other driver only.
            _npc(other,9. if case == 'occupied' and step >= 28 else 0.)
            frame = world.tick()
            obs = rig.read(frame)
            if chase:
                chase.append(frame)
            speed = vehicle_speed(ego)
            defaults = navigation.read(ego,LOOK)
            points = planner.preview(ego, obs['lidar'], defaults, speed,LOOK)
            light = (0., 60.)
            if native_rgb:
                if step % 2 == 0:
                    detections = detector.detect(obs['front'])
                if getattr(observer, 'uses_conflict_profile', False):
                    obs.update(progress_frame=frame, progress_camera_age_s=(step % 2) * .1,
                               progress_tactical=planner.info)
                reading = lamps.observe(ego, obs)
                if reading is not None:
                    light = (model.sensor_lamp_classifier.red_probability(reading[1]), reading[0])
                elif getattr(observer, 'uses_conflict_profile', False) and lamps.governing_gap is not None:
                    light = (0., lamps.governing_gap)
            x = observer.observe(obs,detections,points[:6],speed,route_preview=points[6:],stopline_gap=60.)
            if getattr(model,'uses_posted_limit',False):
                model.posted_limit_kmh = float(ego.get_speed_limit())
            if planner.measured_hold:
                model.tactical_speed_limit = planner.speed_limit
            result = model.predict(np.asarray([*x,*light],dtype=np.float32))
            steer, throttle, brake = planner.apply(result,speed)
            ego.apply_control(carla.VehicleControl(steer=steer,throttle=throttle,brake=brake))

            # All following ground truth measures outcomes after issuing control.
            rear,nose,left,right = _footprint(ego,corridor)
            row=dict(frame=frame,time_s=round(.1*step,2),ego_travel_m=round(rear-ego_start,3),speed_mps=round(speed,3),
                     ego_offset_m=round((left+right)/2,3),in_legal_lanes=left >= -1.5*corridor.width and right <= .5*corridor.width,
                     steer=round(steer,4),throttle=round(throttle,4),brake=round(brake,4),
                     tactical=planner.info,collisions=len(rig.collisions()))
            if native_rgb:
                row.update(posted_limit_kmh=float(model.posted_limit_kmh), speed_request_mps=float(model.last_speed_request),
                           progress_evidence=dict(getattr(model, 'progress_evidence', {})),
                           rgb_detection_count=len(detections))
            if lead is not None:
                lr,lf,ll,_ = _footprint(lead,corridor)
                row.update(lead_bumper_gap_m=round(lr-nose,3),lead_speed_mps=round(vehicle_speed(lead),3),
                           lead_travel_m=round(lr-lead_start,3),alongside_lead=nose>=lr and rear<=lf,
                           lead_lateral_clearance_m=round(ll-right,3),physically_overtaken=rear>=lf+1.)
            if other is not None:
                ar,_,_,_ = _footprint(other,corridor)
                row['adjacent_bumper_gap_m'] = round(ar-nose,3)
            rows.append(row)
            if row.get('physically_overtaken') or row['collisions'] or rear > corridor.arc[-1]-3:
                break
            if case=='no-lead' and rear-ego_start>=45.:
                break
            if case == 'departing-lead' and rear-ego_start >= 30.:
                break
        # Flush any delayed physics collision callback.
        rig.read(world.tick())
        rows[-1]['collisions'] = len(rig.collisions())
        if chase:
            stream.flush()
            encode_and_check(raw,folder/(case+'.mp4'),(960,540),0,len(rows))
    with gzip.open(folder/(case+'.jsonl.gz'),'wt',encoding='utf8') as stream:
        for row in rows:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    score = _score(rows,case)
    print(json.dumps(score),flush=True)
    return score


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('models/city-cruise/candidate.npz'))
    parser.add_argument('--port',type=int,default=2968)
    parser.add_argument('--attach',action='store_true')
    parser.add_argument('--record',action='store_true',help='Save a frame-aligned native chase video for each trial')
    parser.add_argument('--output',type=Path,default=Path('results/tactical-pass'))
    parser.add_argument('--cases',nargs='+',choices=('clear','occupied','no-lead'),default=('clear','occupied','no-lead'))
    args=parser.parse_args()
    if args.record and not args.attach:
        parser.error('--record requires a graphics-capable attached server')
    args.output.mkdir(exist_ok=True,parents=True)
    folder=Path(mkdtemp(prefix='paired-',dir=args.output))
    print(folder,flush=True)
    with simulator(args.port,args.attach,physics_only=not args.attach,map_name='Town03_Opt') as world:
        manifest=dict(producer=producer_info(world),speed_model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                      tactical_model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                      features='real 360-degree LiDAR, radar, localized lane geometry, speed; no RGB detections',
                      city_model=str(args.model),cases=[])
        for name in ('city_pace.py', 'overtake.py', 'overtake_trials.py'):
            source = Path(__file__).with_name(name)
            manifest.setdefault('runtime_sources_sha256', {})[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        for case in args.cases:
            manifest['cases'].append(evaluate(world,case,args.model,folder,args.record))
            (folder/'summary.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    return all(r['passed'] for r in manifest['cases'])


if __name__=='__main__':
    raise SystemExit(0 if main() else 1)
