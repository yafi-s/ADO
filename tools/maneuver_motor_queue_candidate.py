"""Causal stationary motor-vehicle association and tactical choice experiment.

Only positively matched, stationary source-lane road vehicles receive a
queue-follow preference. The original model retains its other choices.
"""
from collections import deque
import numpy as np


class MotorQueue:
    # The qualification thresholds are engineering choices, not fitted
    # parameters. The control detector and the original action tree are fitted.
    def __init__(self):
        self.reset()

    def reset(self):
        self.sightings = deque(maxlen=6)
        self.motion = deque(maxlen=24)
        self.anchor = self.last_surface = None
        self.since = None
        self.preference = None
        self.learned_mode = False
        self.preference_inputs = None
        self.probability_follow = None
        self.intent_gain_m = self.intent_leaf = None
        self.name = None
        self.wait = 0
        self.status = "unconfirmed"
        self.released = False
        self.latest = None

    def _associate(self, frame, station, gap, witness):
        if not witness:
            return
        box_frame = witness["box_frame"]
        if self.sightings and self.sightings[-1][0]==box_frame:
            return
        observed = station+gap
        self.sightings.append((box_frame,observed,witness["label"],witness.get("radial_ground_speed_mps")))
        current = [v for v in self.sightings if 0<=frame-v[0]<=9 and abs(v[1]-observed)<=.85
                   and v[3] is not None and abs(v[3])<=.70]
        if len(current)>=2 and current[-1][0]-current[0][0]>=2:
            if self.anchor is not None and abs(observed-self.anchor)>2.:
                self.wait=0
                self.released=False
                self.motion.clear()
            if self.anchor is None or abs(observed-self.anchor)>2.:
                self.since=current[0][0]
            self.anchor=float(np.median([r[1] for r in current]))
            self.name=witness["label"]
            self.last_surface=frame

    def update(self, frame, station, gap, raw, source_v, source_age, ego_v, witness):
        self.latest=witness
        measured=source_age==0 and raw<59.
        if measured:
            self.motion.append((frame,station+raw))
        # Also accept a freshly positive radar/camera surface during at most
        # two missing source-lidar scans; never count the same image twice.
        if witness and witness.get("stationary_doppler") and source_age in (0,1,2) and gap<59. and source_v is not None and abs(source_v)<.65:
            self._associate(frame,station,gap,witness)
        if self.anchor is None:
            self.status="unconfirmed"
            return
        if measured:
            elapsed=(frame-self.last_surface)*.1
            displacement=station+raw-self.anchor
            if elapsed>2. or abs(displacement)>1.25+5.*elapsed:
                self.reset()
                return
            self.anchor=station+raw
            self.last_surface=frame
        elif frame-self.last_surface>10:
            self.reset()
            return
        velocity=source_v if source_v is not None else 0.
        near = gap-2.5<15. and ego_v<.50
        stationary = measured and abs(velocity)<=.55
        # Give a newly stopped source vehicle a real opportunity to proceed.
        # Only close stopped-following observations earn a wait, not time
        # spent approaching a distant stopped car. Brief missed scans earn
        # no time and do not erase the verified dwell.
        if stationary:
            self.wait=self.wait+1 if near else 0
        elif measured and (abs(velocity)>.85 or ego_v>1.):
            self.wait=0
        if self.wait>=80:
            self.released=True
        recent=[a for a in self.motion if frame-a[0]<=21]
        if len(recent)>=8 and recent[-1][0]-recent[0][0]>=13:
            # Split the observed world-station history, making acceleration
            # during restart distinguishable from sustained slow following.
            cut=(recent[0][0]+recent[-1][0])*.5
            halves=[[p for p in recent if p[0]<=cut],[p for p in recent if p[0]>=cut]]
            rates=[(h[-1][1]-h[0][1])/(.1*(h[-1][0]-h[0][0])) for h in halves if len(h)>1]
            slow=(len(rates)==2 and min(rates)>=.8 and max(rates)<=3.5 and abs(rates[1]-rates[0])<=.65
                  and source_v is not None and .8<=source_v<=3.5)
        else:
            slow=False
        if self.released and stationary:self.status="persistent_stationary_vehicle"
        elif slow:self.status="sustained_slow_vehicle"
        elif source_v is not None and source_v>.55:self.status="vehicle_restarting"
        else:self.status="following_stationary_vehicle"

    def preference_features(self,frame,gap,speed,source_v,gain):
        if self.anchor is None or self.since is None:
            return None
        points=list(self.motion)
        recent=[p for p in points if frame-p[0]<=9]
        earlier=[p for p in points if 10<=frame-p[0]<=19]
        def rate(group,default):
            intervals=[(b[1]-a[1])/(.1*(b[0]-a[0])) for j,b in enumerate(group) for a in group[:j]
                       if b[0]-a[0]>=3]
            return float(np.clip(np.median(intervals),0.,30.)) if intervals else default
        now=rate(recent,max(0.,source_v or 0.))
        previous=rate(earlier,now)
        interval=(np.mean([p[0] for p in recent])-np.mean([p[0] for p in earlier]))*.1 if earlier and recent else 1.
        return dict(source_gap_m=gap,ego_speed_mps=speed,source_speed_mps=now,previous_source_speed_mps=previous,
                    source_acceleration_mps2=(now-previous)/max(.3,interval),
                    observed_stopped_wait_s=.1*self.wait,source_observed_s=.1*(frame-self.since),
                    projected_lane_gain_m=gain)

    def learned_preference(self,model,features):
        self.preference_inputs=features
        self.learned_mode=model is not None
        if model is None or features is None:
            self.preference=None
            self.probability_follow=None
            self.intent_gain_m=self.intent_leaf=None
        elif callable(getattr(type(model),"details",None)):
            self.preference,self.probability_follow,self.intent_gain_m,self.intent_leaf=model.details(features)
        else:
            self.preference,self.probability_follow=model.predict(features)
            self.intent_gain_m=self.intent_leaf=None

    def choose(self, fitted, required_merge=False):
        if required_merge or self.anchor is None:
            return fitted
        if self.learned_mode:
            return "follow" if self.preference=="follow" else fitted
        if self.status in ("persistent_stationary_vehicle","sustained_slow_vehicle"):
            return fitted if fitted!="yield" else ("follow" if self.status=="persistent_stationary_vehicle" else fitted)
        return "follow"

def candidate_planner(intent=None):
    from crosswalk.route_maneuvers import RouteManeuvers,compare_progress
    import importlib.util
    from pathlib import Path
    donor=Path(__file__).with_name("maneuver_source_vehicle.py")
    spec=importlib.util.spec_from_file_location("motor_source_association",donor)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_surface_vehicle=module.source_surface_vehicle

    class VehicleChoice:
        def __init__(self, planner, model):
            self.planner = planner
            self.original = model
            self.fitted = None

        def predict(self, values):
            self.fitted, prob = self.original.predict(values)
            p = self.planner
            action = p.queue.choose(self.fitted) if p.evaluating_source else self.fitted
            if p.defer_source and action=="follow":
                p.holding=False
                p.pass_votes=0
            return action, prob

    class QueuePlanner(RouteManeuvers):
        def __init__(self,corridors,model=None):
            super().__init__(corridors,model)
            self.model=VehicleChoice(self,self.model)
            if intent is not None:
                self.stationary_intent_checkpoint=getattr(intent,"checkpoint_path",None)
                self.stationary_intent_sha256=getattr(intent,"sha256",None)

        def _reset(self):
            super()._reset()
            self.queue=MotorQueue()
            self.evaluating_source=False
            self.defer_source=False

        def _measure(self,pose,scan,station,speed,lateral=0.):
            result=super()._measure(pose,scan,station,speed,lateral)
            x,raw=result[:2]
            data=self.camera_inputs
            self.defer_source=False
            early_outbound=(self.start is not None and self.return_start is None and
                            self.phase=="change_left" and lateral>-.35)
            self.evaluating_source=(data is not None and self.corridor.merge_at is None
                                    and (self.start is None or early_outbound))
            if self.evaluating_source:
                witness=source_surface_vehicle(data["boxes"],scan,data["radar"],pose,data["box_pose"],
                    self.corridor,station,x[0],speed,data["frame"],data["box_frame"])
                source=self.surfaces[0]
                self.queue.update(data["frame"],station,x[0],raw[0],source.speed,source.age,speed,witness)
                front_v=self.surfaces[2].speed or 0.
                _,gain,_=compare_progress("pass",x[0],x[1],x[2],x[3],front_v,result[3][2])
                features=self.queue.preference_features(data["frame"],x[0],speed,source.speed,gain)
                self.queue.learned_preference(intent,features)
                self.defer_source=self.queue.anchor is not None and (
                    self.queue.preference=="follow" if intent is not None else
                    self.queue.status in ("following_stationary_vehicle","vehicle_restarting"))
                if self.defer_source and self.start is not None:
                    # Abort only before substantive lateral ingress. No attempt
                    # to reverse a straddled car from a semantic observation.
                    self.start=None
                    self.return_start=None
            return result

        def preview(self,ego,lidar,defaults,speed,lookaheads=(3.,7.,13.,18.,25.,32.,40.),*,camera_inputs=None):
            self.camera_inputs=camera_inputs
            values=super().preview(ego,lidar,defaults,speed,lookaheads)
            if "source_gap_m" in self.info:
                if self.model.fitted is not None:
                    self.info["fitted_choice"]=self.model.fitted
                self.info["source_motor_vehicle"]=dict(
                    source_track_state=("associated_vehicle" if self.queue.anchor is not None else "unconfirmed"),
                    evaluated=self.evaluating_source,queue_follow=self.defer_source,
                    last_supported_camera=self.queue.sightings[-1][0] if self.queue.sightings else None,
                    tracked_bumper_m=self.queue.anchor,observed_dwell_s=round(.1*self.queue.wait,1),
                    matched_vehicle=self.queue.latest,preference=self.queue.preference,
                    probability_follow=self.queue.probability_follow,learned_pass_gain_m=self.queue.intent_gain_m,
                    leaf_index=self.queue.intent_leaf,preference_inputs=self.queue.preference_inputs)
            return values
    return QueuePlanner


def isolated_episode(drive,intent=None):
    """Move the stock detector to the synchronous observation boundary, once."""
    import inspect
    import textwrap
    text=textwrap.dedent(inspect.getsource(drive.episode))
    old='''                if step % 2 == 0:
                    detections = detector.detect(obs["front"])
'''
    if text.count(old)!=1:raise ValueError("Unknown native control-camera cadence")
    text=text.replace(old,"")
    before='''            speed = vehicle_speed(scene.ego)
'''
    if text.count(before)!=1:raise ValueError("Unknown native observation ordering")
    text=text.replace(before,before+'''            if not range_only and step % 2 == 0:
                detections = detector.detect(obs["front"])
                control_box_frame = frame
                control_box_pose = scene.ego.get_transform()
''')
    preview='''                    points = tactical_pass.preview(scene.ego, obs["lidar"], points, speed)'''
    if text.count(preview)!=1:raise ValueError("Unknown tactical interface")
    text=text.replace(preview,'''                    if isinstance(tactical_pass, QueuePlanner) and not range_only:
                        camera_inputs=dict(frame=frame,boxes=detections,box_frame=control_box_frame,
                                           box_pose=control_box_pose,radar=obs["radar"])
                        points = tactical_pass.preview(scene.ego,obs["lidar"],points,speed,camera_inputs=camera_inputs)
                    else:
                        points = tactical_pass.preview(scene.ego,obs["lidar"],points,speed)''')
    # The qualification runner patches native Rig/BodyKit just before invoking
    # the episode, so resolve the original module globals on every call.
    planner_type=candidate_planner(intent)
    def entry(*args,**kwargs):
        scope=dict(vars(drive),QueuePlanner=planner_type)
        exec(compile(text,"<stationary motor vehicle research episode>","exec"),scope)
        return scope["episode"](*args,**kwargs)
    return planner_type,entry

def native(arguments):
    import hashlib
    from importlib.util import module_from_spec,spec_from_file_location
    import json
    from pathlib import Path
    import runpy
    import shutil
    import sys
    from unittest.mock import patch
    from zipfile import ZipFile
    from crosswalk import drive, route_maneuvers

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    donor=Path(__file__).with_name("maneuver_source_vehicle.py")
    if digest(donor)!="2f0001ecd8958ce7b2991c2cdde70e407977050d7fa7d38865530d13639d10f7":
        raise ValueError("Source vehicle association changed")
    if digest(Path(route_maneuvers.__file__))!="18c7543ed01beb2fce9b94abd9aac0db426df72dbf3877702e106741da059ada":
        raise ValueError("Expected frozen base maneuver planner")
    with ZipFile(arguments.registered_source) as archive:
        harness=archive.read("market_loop_qualify.py")
    if hashlib.sha256(harness).hexdigest()!="855c00279e4020b4887bfc8a4c0aff3642ef2b1bee194078c41a8e38495bd7b4":
        raise ValueError("Expected the exact 9145 timed traffic fixture")
    spec=spec_from_file_location("crosswalk.queue_intent_qualification",arguments.intent_source)
    module=module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    intent=module.StationaryIntent(arguments.intent_model)
    arguments.output.mkdir(parents=True,exist_ok=False)
    hpath=arguments.output/"market_loop_qualify.py"
    hpath.write_bytes(harness)
    files=[Path(__file__),donor,Path(arguments.intent_source),Path(arguments.intent_model)]
    for path in files:shutil.copyfile(path,arguments.output/path.name)
    if digest(arguments.output/arguments.intent_model.name)!=intent.sha256:
        raise RuntimeError("Snapshotted intent weights differ from the actual loaded inference bytes")
    planner,episode=isolated_episode(drive,intent)
    root=Path(route_maneuvers.__file__).resolve().parents[1]
    protocol=dict(base_git="914560faaa57f6592ffdd3e881715b06104344cf",phase=arguments.phase,
                  pace_sha256=digest(root/"models/city-maneuvers/candidate.npz"),
                  route_choice_sha256=digest(root/"models/route-maneuvers/choice.json"),
                  stationary_intent_sha256=intent.sha256,
                  stationary_intent_inference_sha256=digest(arguments.intent_source),
                  source_vehicle_association_sha256=digest(donor),
                  case=arguments.case,files={p.name:digest(p) for p in [*files,hpath]},
                  policy_inputs="Same control NanoDet boxes evaluated once on the original 5 Hz exposure cadence before tactical preview, synchronized lidar/radar and ego transform. Original pace controls, actors and route unchanged.")
    (arguments.output/"candidate.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
    with patch.object(route_maneuvers,"RouteManeuvers",planner),patch.object(drive,"episode",episode),patch.object(sys,"argv",[str(hpath),arguments.phase,arguments.case]):
        runpy.run_path(str(hpath),run_name="__main__")


if __name__=="__main__":
    import argparse,sys
    from pathlib import Path
    sys.path.insert(0,str(Path.cwd()))
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--registered-source",type=Path,required=True)
    cli.add_argument("--intent-source",type=Path,required=True)
    cli.add_argument("--intent-model",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--phase",choices=("prefix","tail"),default="prefix")
    cli.add_argument("--case",choices=("continuous","later-gap"),default="continuous")
    native(cli.parse_args())
