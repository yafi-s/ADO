"""Record the city driver's real moving overtakes with six synchronized native views."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace

import carla

from .city_map import export
from .control import vehicle_speed
from .drive import episode
from .experiment import PROJECT, fingerprint, read_json, write_json
from .journey_capture import (JourneyCamera, PerimeterCapture, PERIMETER_VIEWS, _camera_interval,
                              _load_inputs, _observed_settings, _poster)
from .overtake import PassingCorridor
from .overtake_trials import _footprint, _npc, _score, _spawn
from .replay_pack import pack
from .runtime import producer_info, simulator
from .showcase import FRONT_SIZE, encode_and_check, write_replay
from .world import _raised


class PassingStudyScene:
    """NPC staging and independent scoring. The ego uses the normal city driving loop."""
    kind = 'overtake'
    road_id, lane_id = 4, 3
    spawn_lateral_m = spawn_yaw_deg = 0.
    ambient = None
    _junction_ranges = ()

    def __init__(self, world, case):
        self.world, self.case = world, case
        self._map = world.get_map()
        self._urban = SimpleNamespace(lane_change=SimpleNamespace(source_lane=(4,3)))
        self.corridor = PassingCorridor.from_map(self._map)
        self.route = [self._map.get_waypoint_xodr(4,3,float(s)).transform.location for s in range(86,17,-1)]
        self.stack = ExitStack()
        self.stack.callback(world.set_weather,world.get_weather())
        self.weather = 'ClearNoon'
        self.frame_scores = {}
        self.step = 0
        self.length = 68.
        try:
            world.set_weather(carla.WeatherParameters.ClearNoon)
            ego_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz')
            ego_bp.set_attribute('role_name','hero')
            self.ego = world.spawn_actor(ego_bp,_raised(self._map.get_waypoint_xodr(4,3,86.).transform,.35))
            self.stack.callback(self.ego.destroy)
            self.ego_id = self.ego.id
            self.ego.apply_control(carla.VehicleControl(brake=1.))
            self.hazard = _spawn(world,self._map,self.stack,'vehicle.tesla.model3',3,62.)
            self.traffic = _spawn(world,self._map,self.stack,'vehicle.mini.cooper',2,84.) if case=='occupied' else None
            self.ego_start = self.lead_start = None
        except BaseException:
            self.close()
            raise

    def close(self):
        self.stack.close()

    def before_tick(self):
        _npc(self.hazard,1.45)
        _npc(self.traffic,9. if self.step >= 28 else 0.)
        self.step += 1

    def observer_positions(self):
        result = []
        for actor, role in ((self.hazard,'moving_lead'),(self.traffic,'adjacent_traffic')):
            if actor is None:
                continue
            pose, extent = actor.get_transform(), actor.bounding_box.extent
            result.append(dict(actor_id=actor.id,type='car',role=role,blueprint=actor.type_id,
                               x=round(pose.location.x,3), y=round(pose.location.y,3),z=round(pose.location.z,3),
                               yaw=round(pose.rotation.yaw,3),dimensions_m=[2*extent.x,2*extent.z,2*extent.y]))
        return result

    def metrics(self):
        rear,nose,left,right = _footprint(self.ego,self.corridor)
        lr,lf,ll,_ = _footprint(self.hazard,self.corridor)
        if self.ego_start is None:
            # Body poses and bounding boxes are initialized by the physics
            # ticks, not by the asynchronous spawn acknowledgment.
            self.ego_start,self.lead_start = rear,lr
        progress = rear - self.ego_start
        phase = ('Pass complete' if rear>lf else 'Passing moving lead' if nose>lr
                 else 'Approaching moving lead')
        legal = left >= -1.5*self.corridor.width and right <= .5*self.corridor.width
        offset = (left+right)/2.
        physical = dict(ego_travel_m=round(progress,3),ego_offset_m=round(offset,3),in_legal_lanes=legal,
                        lead_bumper_gap_m=round(lr-nose,3), lead_speed_mps=round(vehicle_speed(self.hazard),3),
                        lead_travel_m=round(lr-self.lead_start,3),alongside_lead=nose>=lr and rear<=lf,
                        lead_lateral_clearance_m=round(ll-right,3),physically_overtaken=rear>=lf+1.)
        if self.traffic:
            ar,_,_,_ = _footprint(self.traffic,self.corridor)
            physical['adjacent_bumper_gap_m'] = round(ar-nose,3)
        self.frame_scores[self.world.get_snapshot().frame] = physical
        return dict(progress_m=progress,route_length_m=self.length,phase=phase,lateral_error_m=min(abs(offset),abs(offset+3.5)),
                    route_departure=not legal,hazard_gap_m=max(0.,lr-nose) if right>ll and nose<=lf else 60.,
                    red_light_violations=0,yield_violations=0,event_triggered=True,event_cleared=rear>=lf+1.,
                    destination='Boulevard',route_id='town03-road4-tactical-study',road_ids=[4],junction_ids=[])


def capture(world, case, folder, model_path):
    model, _, observer_factory, policy = _load_inputs(model_path)
    trial = 'overtake-clear' if case=='clear' else 'overtake-yield'
    export(world,folder/'town.json')
    scenes = []
    def make(world, *args, **kwargs):
        scene = PassingStudyScene(world,case)
        scenes.append(scene)
        return scene

    with ExitStack() as stack:
        def stream(name):
            return stack.enter_context((folder/(name+'.mjpeg')).open('wb'))
        chase = JourneyCamera(960,540,75,stream('chase'))
        rows = stack.enter_context((folder/'measured.jsonl').open('w',encoding='utf8'))
        per = PerimeterCapture(chase,stream('front'),rows,{v:stream(v) for v in PERIMETER_VIEWS})
        from .journey_capture import JourneyWorld
        wrapped = JourneyWorld(world,chase)
        try:
            summary,trace,*_ = episode(wrapped,'overtake',211,'ado-city-cruise-v1',model,recording=per,
                                        max_steps=400,scene_factory=make,observation_factory=observer_factory,
                                        district='Boulevard',destination='Boulevard',run_id=trial,
                                        surround=True,tactical=True,weather='ClearNoon')
        finally:
            chase.close()
    first, frames, telemetry = _camera_interval(summary,trace,chase,per)
    scene, = scenes
    for row in telemetry:
        physical = scene.frame_scores[row['frame']]
        row.update(physical)
        row['tactical'] = row['tactical_pass']
        row['time_s'] = row['sim_time']
        row['collisions'] = row['collision_count']
        if row['tactical'].get('maneuver')=='yield':
            row['phase']='Yielding to adjacent traffic'
    score = _score(telemetry,case)
    if not score['passed'] or summary['outcome']!='complete':
        write_json(folder/'failure.json',dict(score=score,summary=summary))
        raise RuntimeError(f'Passing study outcome: {score}, {summary["outcome"]}')
    write_replay(folder/'replay.json',folder/'measured.jsonl',telemetry)
    write_json(folder/'telemetry.json',telemetry)
    alongside = next(i for i,r in enumerate(telemetry) if r['alongside_lead'])
    _poster(folder/'chase.mjpeg',frames[alongside]['offset'],frames[alongside]['size'],folder/'poster.jpg')
    _poster(folder/'front.mjpeg',*per.front_offsets[alongside],folder/'front.jpg')
    write_json(folder/'source.json',dict(trial_id=trial,case=case,policy=policy,summary=summary,score=score,
               observer=chase.rendering,ego_actor_id=scene.ego_id,producer=producer_info(world),
               settings=_observed_settings(world),start=first,count=len(frames),first_frame=frames[0]['camera_frame'],
               last_frame=frames[-1]['camera_frame'],poster_frame=frames[alongside]['camera_frame']))
    print(f'Captured {trial} with {len(frames)} scored synchronous frames',flush=True)


def publish(stage, output_root):
    import shutil
    source=read_json(stage/'source.json')
    trial=source['trial_id']
    output = output_root/trial
    if output.exists():
        raise FileExistsError(output)
    output_root.mkdir(parents=True,exist_ok=True)
    final = Path(mkdtemp(prefix='delivery-',dir=stage))
    info = {}
    for name, size, start in [('chase',(960,540),source['start']),('front',FRONT_SIZE,0),
                              *((view,(320,180),0) for view in PERIMETER_VIEWS)]:
        info[name] = encode_and_check(stage/(name+'.mjpeg'),final/(name+'.mp4'),size,start,source['count'])
    for name in ('town.json','replay.json','telemetry.json','front.jpg','poster.jpg'):
        shutil.copyfile(stage/name,final/name)
    shutil.copyfile(final/'poster.jpg',final/'chase.jpg')
    pack(final/'replay.json')
    label = 'Pass a moving car' if source['case']=='clear' else 'Yield before overtaking'
    note = 'Supplemental Town03 moving-lead study. The city driver selects its pass from onboard range measurements.'
    catalog = dict(label=label,path=trial,trial_id=trial,cameras=['chase','front',*PERIMETER_VIEWS],
                   cohort='tactical-study',benchmark_member=False,selection_note=note,district='Boulevard',
                   map='Town03_Opt',map_url=f'./media/{trial}/town.json',manifest=f'./media/{trial}/manifest.json',
                   replay_pack_url=f'./media/{trial}/replay-pack.json.gz',
                   replay_pack_sha256=fingerprint(final/'replay-pack.json.gz'))
    sha = {f.name:fingerprint(f) for f in sorted(final.iterdir())}
    write_json(final/'manifest.json',dict(**source,purpose='Supplemental tactical study with a moving lead',
               title=label,catalog_entry=catalog,cohort='tactical-study',benchmark_member=False,selection_note=note,
               camera_videos=info,perimeter_views=list(PERIMETER_VIEWS),quality_requested='Low',passed=True,
               frame_period_s=.1,video=info['chase'],front_video=info['front'],replay='replay.json',telemetry='telemetry.json',
               sha256=sha,map='Town03_Opt',map_asset='town.json',map_url=catalog['map_url'],outcome='complete',
               control_timing='Frame-t readback drives the measured pose. The command inferred from frame t applies on t+1.',
               scene_reference='Independent oriented actor footprints supply the score; observer_map.others contains simulator reference IDs and never enters the driving model.',
               navigation_reference='observer_map.path is the source-lane destination reference. The tactical planner supplies its selected lateral trajectory to the city steering model.',
               inference='Five RGB, full-sweep lidar and five radars run on the inference rig. The pass classifier uses mapped lane geometry and LiDAR; independent surround tracks and RGB lamp classifications are recorded for perception.',
               source_mode='one closed-loop simulation with simultaneous camera capture',teacher=False))
    final.rename(output)
    print(f'Installed {output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=2968)
    parser.add_argument('--model',type=Path,default=PROJECT/'models/city-cruise/candidate.npz')
    parser.add_argument('--output',type=Path,default=PROJECT/'web/media')
    parser.add_argument('--stage',type=Path,help='Publish an existing successful capture without CARLA')
    parser.add_argument('--capture-only',action='store_true')
    parser.add_argument('--cases',nargs='+',choices=('clear','occupied'),default=('clear','occupied'))
    args=parser.parse_args()
    if args.stage:
        publish(args.stage,args.output)
        return
    root=PROJECT/'out/tactical-replays'
    root.mkdir(parents=True,exist_ok=True)
    stages=[]
    with simulator(args.port,attach=True,map_name='Town03_Opt') as world:
        for case in args.cases:
            folder=Path(mkdtemp(prefix=case+'-',dir=root))
            stages.append(folder)
            print(folder,flush=True)
            capture(world,case,folder,args.model)
    if not args.capture_only:
        for stage in stages:
            publish(stage,args.output)


if __name__=='__main__':
    main()
