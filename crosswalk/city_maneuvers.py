"""Route-relative camera, range and curvature evidence for the measured city driver."""
from collections import deque
import io

import numpy as np

from . import behavior_v2 as v2
from .camera_geometry import CAMERAS
from .city_clearance import BODY
from .city_grade import GradeModel, GradeObserver, FEATURE_NAMES as PRIOR_NAMES, supported_grade
from .city_open_road import long_sight
from .city_policy import INDEX
from .city_progress import _profile
from .policy import _returns, LIDAR_Z

VERSION = "ado-city-maneuvers-v1"
FIELDS = ("verified_forward_m", "verified_cross_sight_m", "supported_shoulder_m",
          "camera_stop_horizon_m", "off_path_camera_boxes", "geometry_scans", "verified_grade_m",
          "long_sight_m", "long_body_gap_m")
FEATURE_NAMES = (*PRIOR_NAMES[:-2], *FIELDS, *PRIOR_NAMES[-2:])
BENDS = tuple(INDEX[n] for n in ("bend_7_13", "bend_13_25", "bend_25_40"))
BEND_LENGTHS = np.array([5., 9., 14.])


def camera_horizon(detections, laser, path):
    """Bound progress by the earliest station of each well-resolved lateral person.

    Both the observed laser surfaces and the plausible near edge of the object's
    ground-contact ray must be separated from the swept vehicle. Missing range,
    central people and two-wheelers retain the prior vulnerable-user response.
    """
    boxes = [d for d in detections if d["label"] in ("person", "bicycle", "motorcycle")
             and d["box"][3] - d["box"][1] > .055]
    if not boxes:
        return 60., 0
    cam = CAMERAS["front"]
    dx = laser[:, 0] - cam.x
    valid = dx > .6
    x, y, z = laser[valid, 0], laser[valid, 1], laser[valid, 2] + LIDAR_Z
    depth = dx[valid]
    u, v = .5 + cam.fx * y / depth, .5 - cam.fy * (z-cam.z) / depth
    plane = v2._ground_plane(x,y,z)
    height = z - (plane[0]*x+plane[1]*y+plane[2])
    stop = 60.
    for d in boxes:
        x1, y1, x2, y2 = d["box"]
        if d["label"] != "person" or x1 < .66 and x2 > .34 or y2 >= .99:
            return 0., len(boxes)
        # Check the lower body and foot. Widen the angular sector for sensor
        # registration, stale 0.1s detections, and small lateral movements.
        hits = ((u >= x1-.012) & (u <= x2+.012) & (v >= y1+.15*(y2-y1)) &
                (v <= y2+.025) & (depth < 55.))
        if hits.sum() < 4 or y2 <= .50:
            return 0., len(boxes)
        sector = (u >= x1-.015) & (u <= x2+.015)
        foot_depth = cam.fy*(cam.z-.40)/(y2-.5+.035)
        lower = min(float(min(depth[hits]))-1., foot_depth)
        # A nearer upright surface below the detector box may hide its feet.
        # Shorten the budget to that surface as well. Require positive ground
        # returns through the near sector; blank lidar is never a clearance.
        occluder = sector & (v >= y1) & (v <= 1.) & (height > .26) & (depth < lower)
        if occluder.any():
            lower = min(lower,float(min(depth[occluder]))-.65)
        pavement = sector & (abs(height) < .27) & (depth < lower+.75) & (depth > 2.)
        if pavement.sum() < 3:
            return 0., len(boxes)
        upper = max(lower+.5, np.quantile(depth[hits], .9)+1.)
        # Sample the entire possible depth interval on both edges of the box.
        # A bend can reach a visually peripheral person further along the route.
        sample = np.arange(max(.1, lower), min(58.,upper)+1., .8)
        if not len(sample):
            return 0., len(boxes)
        rng = np.tile(sample, 3)
        rays = np.repeat([x1-.015, (x1+x2)*.5, x2+.015], len(sample))
        station, lateral = v2._project(cam.x+rng, cam.y+(rays-.5)*rng/cam.fx, path)
        if np.any((station < 12.) & (abs(lateral) < BODY + 2.)):
            return 0., len(boxes)
        # Still plan to stop before its earliest plausible longitudinal station.
        stop = min(stop, max(0., float(np.min(station))-1.))
    return stop, len(boxes)


def shoulder_scan(laser, path):
    """Measure compact low surfaces and visible pavement behind each flank."""
    x, y, z = laser[:,0], laser[:,1], laser[:,2]+LIDAR_Z
    a, b, c = v2._ground_plane(x,y,z)
    keep = (x > .05) & (x < 24.) & (abs(y) < 12.)
    x,y,h = x[keep], y[keep], (z-a*x-b*y-c)[keep]
    s,d = v2._project(x,y,path)
    ground = abs(h) < .26
    solid = (h > .28) & (h < 3.)
    if np.any(solid & (s >= 0.) & (s < 12.) & (abs(d) < BODY+.12)):
        return [], 0.
    cells = []
    sight = 16.
    for sign in (-1., 1.):
        side = sign*d
        visible = ground & (side > 3.6) & (side < 7.5) & (s >= 0.) & (s < 16.)
        counts = np.bincount((s[visible]//2).astype(int), minlength=8)
        missing = np.flatnonzero(counts < 2)
        if len(missing):
            sight = min(sight, float(missing[0]*2))
        hits = np.flatnonzero(solid & (s > -2.) & (s < 16.) & (side >= BODY) & (side <= 7.))
        if not len(hits):
            continue
        ordered = hits[np.argsort(s[hits])]
        for group in np.split(ordered, np.flatnonzero(np.diff(s[ordered]) > .55)+1):
            front, end = min(s[group]), max(s[group])
            if end < -.1:
                continue
            margin = float(min(side[group])-BODY)
            beyond = ground & (side > max(side[group])+.25) & (side < max(side[group])+3.3)
            through = np.count_nonzero(beyond & (s > front-.65) & (s < end+.65)) >= 3
            if (len(group) >= 4 and end-front <= 1.35 and np.ptp(side[group]) <= 2.4
                    and max(h[group]) < 1.28 and margin > .17 and through):
                cells.append((float(np.median(s[group])), float(sign*np.median(side[group])), margin))
            else:
                sight = min(sight, max(0., float(front)))
    return cells,sight


class ManeuverObserver(GradeObserver):
    uses_long_sight = True
    def __init__(self):
        super().__init__()
        self.geometry = deque(maxlen=3)
        self.shoulders = deque(maxlen=4)
        self.grades = deque(maxlen=9)
        self.long_ranges = deque(maxlen=3)

    def measure(self, obs, detections, path, speed, prior, *, valid_image, contiguous):
        if not contiguous:
            self.geometry.clear()
            self.shoulders.clear()
            self.grades.clear()
            self.long_ranges.clear()
        laser = _returns(obs["lidar"])
        if not valid_image or len(laser) < 30 or not len(_returns(obs["radar"])):
            self.geometry.clear()
            self.shoulders.clear()
            self.grades.clear()
            self.long_ranges.clear()
            return [0.,0.,0.,0.,0.,0.,0.,0.,0.]
        camera, count = camera_horizon(detections, laser, path)
        # Lost or unresolved camera evidence withdraws the additional proposal.
        corridor, cross, _, _ = _profile(obs["lidar"], path, {})
        cells, sight = shoulder_scan(laser,path)
        settled = []
        for pos,side,margin in cells:
            # Require both ends of an ego-motion-compensated 0.4s history to
            # agree. An unmatched nearer surface consumes the available sight.
            if len(self.shoulders)==self.shoulders.maxlen and all(
                any(abs((pos+self.travel)-(p+travel)) < .42 and abs(side-d) < .19 for p,d,_ in old)
                for travel,old in (self.shoulders[0],self.shoulders[-1])):
                settled.append((pos,margin))
            else:
                sight = min(sight,max(0.,pos-.7))
        self.shoulders.append((self.travel,cells))
        shoulder = 0.
        if settled and min(s for s,_ in settled) < 12.:
            shoulder = sight*min(1., min(m for _,m in settled)/(.75*BODY))
        self.geometry.append((corridor, cross, shoulder, camera))
        distance = min(prior[INDEX["path_body_lidar"]], prior[INDEX["path_body_radar"]])
        extent = prior[-2]
        if camera > 0. and 10. <= distance < 32. and not extent and prior[-7] == 0.:
            extent = supported_grade(obs["lidar"], obs["radar"], path, speed)
        self.grades.append((extent,distance))
        grade = (min(row[0] for row in list(self.grades)[-3:]) if len(self.grades) == 9 and
                 all(proof > 0. or gap >= 24. for proof,gap in self.grades) else 0.)
        profile = np.min(self.geometry,axis=0) if len(self.geometry)==3 else [0.,0.,0.,0.]
        far = obs.get("long_sight_preview")
        extended = np.vstack((path,far)) if far is not None else path
        tactical = obs.get("progress_tactical",{})
        if (camera==60. and not tactical.get("committed",False) and
                prior[INDEX["junction_gap"]]>22. and distance>=30. and
                min(prior[INDEX["left_edge_lidar"]],prior[INDEX["right_edge_lidar"]])>18.):
            long,body=long_sight(obs.get("long_sight_lidar",obs["lidar"]),obs["radar"],extended,speed)
        else:
            long,body=0.,max(0.,float(distance))
        self.long_ranges.append(long)
        verified=min(self.long_ranges) if len(self.long_ranges)==3 else 0.
        return [*profile,count,len(self.geometry),grade,verified,body]

    def observe(self, obs, detections, local_route, speed, *, route_preview=None, stopline_gap=None):
        frame, previous = obs.get("progress_frame"),self.prior_frame
        x = super().observe(obs,detections,local_route,speed,route_preview=route_preview,stopline_gap=stopline_gap)
        image = obs.get("front")
        pose = np.asarray(obs.get("progress_ego_pose"),dtype=float)
        valid = (isinstance(frame,(int,np.integer)) and frame>=0 and pose.shape==(3,) and
                 np.isfinite(pose).all() and 0. <= obs.get("progress_camera_age_s",np.inf) <= .15)
        if valid and image is not None and image.ndim==3 and image.size:
            patch=image[image.shape[0]//3:3*image.shape[0]//4:12,::12]
            valid=bool(patch.size and np.quantile(patch,.75)>24 and np.std(patch)>7)
        else:
            valid=False
        measured=self.measure(obs,detections,v2._checked_route(local_route,route_preview),speed,x,
                              valid_image=valid,contiguous=previous is not None and frame==previous+1)
        return np.r_[x,measured].astype(np.float32)


class ManeuverModel(GradeModel):
    uses_route_maneuvers = True
    feature_names = FEATURE_NAMES

    def __init__(self, parent, curve, stopping):
        prior = GradeModel.load(io.BytesIO(parent))
        super().__init__(prior.parent_checkpoint,prior.coefficients)
        self.maneuver_parent = parent
        self.high_stopping = np.asarray(stopping,dtype=float)
        if (self.high_stopping.shape != (4,) or not np.isfinite(self.high_stopping).all()
                or np.any(self.high_stopping<=0.) or not 11. < self.high_stopping[3] < 25.):
            raise ValueError("Invalid native high-speed braking envelope")
        self.signed_preview_ceiling_mps = float(self.high_stopping[3])
        self.allow_high_speed = True
        self.curve = np.asarray(curve, dtype=float)
        if self.curve.shape != (2,4) or not np.isfinite(self.curve).all() or np.any(self.curve<0.):
            raise ValueError("Invalid measured curvature calibration")

    def new_observer(self):
        super().new_observer()
        return ManeuverObserver()

    def predict(self,x):
        x=np.asarray(x)
        if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
            raise ValueError("Invalid route maneuver measurements")
        self.extra=x[-len(FIELDS)-2:-2]
        steer,throttle,brake=super().predict(np.r_[x[:-len(FIELDS)-2],x[-2:]])
        speed=x[INDEX["ego_speed_mps"]]
        delay,quadratic,guard,_=self.high_stopping
        # The stopping trials use a fixed 0.45 brake after a 0.5s coast.
        # Meet that actuation when a real obstruction consumes that envelope.
        if speed>11. and self.extra[-1]<delay*speed+quadratic*speed**2+guard+self.standoff:
            return steer,0.,max(.45,brake)
        return steer,throttle,brake

    def _finish_request(self,x,baseline,requested):
        original=super()._finish_request(x,baseline,requested)
        evidence=self.progress_evidence
        forward,cross,shoulder,camera,boxes,scans,grade,long_sight_m,long_body=map(float,self.extra)
        evidence.update(maneuver_prior_request_mps=float(original), maneuver_added_mps=0.,
                        maneuver_binding="prior",camera_stop_horizon_m=camera,
                        off_path_camera_candidates=int(boxes), shoulder_sight_m=shoulder,
                        speed_regime="urban",cruise_ceiling_mps=11.,high_speed_sight_m=long_sight_m,
                        long_body_gap_m=long_body)
        if (scans<3 or camera<=0. or self.profile[5]<60. or x[INDEX["mapped_lamp_gap_m"]]<59.
                or max(self.light_window,default=0.)>.25 or
                min(x[INDEX[n]] for n in ("oncoming_gap","oncoming_gap_recent","oncoming_near_gap_recent"))<34.
                or max(x[INDEX[n]] for n in ("rgb_person_height","rgb_person_height_recent"))>.055):
            return original
        raw=min(x[INDEX[n]] for n in ("path_body_lidar","path_body_radar","path_body_lidar_recent","path_body_radar_recent"))
        if raw<24. and grade<=0.:
            return original
        # An adjacent street needs opposing-side visibility. Roadside low
        # surfaces may supplement that sight only away from an intersection.
        use_shoulder=shoulder if x[INDEX["junction_gap"]]>18. else 0.
        sight=min(forward,max(cross,use_shoulder),max(grade,raw),camera)
        if sight<=self.standoff:
            return original
        bends = np.abs(x[list(BENDS)])/BEND_LENGTHS
        near, upcoming = self.curve @ np.r_[1.,bends]
        lateral = self.coefficients[1]
        limits={"maneuver_sight":self._stoppable_speed(sight-self.standoff),
                "near_curvature":np.sqrt(lateral/max(.001,near)),
                "upcoming_curvature":np.sqrt(lateral/max(.001,upcoming)+2.*1.5*7.),
                "demonstrated_cruise":self.coefficients[0],
                "posted":(self.posted_limit_kmh if np.isfinite(self.posted_limit_kmh) and self.posted_limit_kmh>0. else 30.)/3.6}
        # A higher request needs enough positively observed ground to stop at
        # the tested braking response. A forthcoming junction consumes the
        # same budget as an obstruction or the end of clear visible pavement.
        delay,quadratic,guard,tested=self.high_stopping
        free=min(long_sight_m,float(x[INDEX["junction_gap"]]),long_body)
        budget=max(0.,free-self.standoff-guard)
        stop_speed=2.*budget/(delay+np.sqrt(delay*delay+4.*quadratic*budget))
        if (self.allow_high_speed and limits["posted"]>11. and camera==60. and
                not boxes and cross>=12. and forward>=16. and raw>=30. and
                self.tactical_speed_limit is None and stop_speed>11.):
            ceiling=min(tested,limits["posted"])
            limits["demonstrated_cruise"]=ceiling
            limits["maneuver_sight"]=stop_speed
            evidence.update(speed_regime="verified_open_road",cruise_ceiling_mps=float(ceiling),
                            high_speed_braking_cap_mps=float(stop_speed))
        if self.signed_speed_limit is not None:
            limits["signed_approach"]=self.signed_speed_limit
        if self.tactical_speed_limit is not None:
            limits["tactical"]=self.tactical_speed_limit
        binding=min(limits,key=limits.get)
        target=float(limits[binding])
        # On abstentions the parent owns the frame. Recovery from any genuine
        # hazard never accelerates faster than the native pedal calibration.
        if self.last_speed_request is not None:
            target=min(target,self.last_speed_request+.1*self.coefficients[2])
        added=max(0.,target-original)
        evidence.update(maneuver_binding=binding if added>0. else "prior",maneuver_added_mps=added,
                        verified_sight_m=sight,fitted_curvature_near=near,fitted_curvature_upcoming=upcoming)
        return original+added

    def save(self,path):
        np.savez_compressed(path,version=VERSION,revision=1,feature_names=FEATURE_NAMES,
                            curvature=self.curve,high_stopping=self.high_stopping,base=np.frombuffer(self.maneuver_parent,np.uint8))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if str(d["version"].item())!=VERSION or int(d["revision"])!=1 or tuple(d["feature_names"])!=FEATURE_NAMES:
                raise ValueError("Incompatible city maneuver model")
            return cls(d["base"].tobytes(),d["curvature"],d["high_stopping"])
