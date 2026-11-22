"""Local pavement-height fit for distant grade returns in the additional speed proposal."""
from collections import deque

import numpy as np

from . import behavior_v2 as v2
from .city_clearance import BODY
from .city_policy import INDEX
from .city_progress import (FEATURE_NAMES as PRIOR_NAMES, PARAMETERS, VERSION, ProgressModel, ProgressObserver)
from .policy import BUMPER_X, LIDAR_Z, RADAR_X, RADAR_Z, _returns


GRADE_FIELDS = ("pavement_fit_extent_m", "confirmed_pavement_extent_m")
FEATURE_NAMES = (*PRIOR_NAMES[:-2], *GRADE_FIELDS, *PRIOR_NAMES[-2:])


def _ribbon_nodes(station, side, height, beam, radar_stations):
    # The three strips share longitudinal support. Only the outer strips fit
    # elevation; center returns are independent checks on the result.
    nodes = []
    for channel in np.unique(beam):
        sample = beam == channel
        values = []
        if np.count_nonzero(sample) < 7:
            continue
        for lo, hi in ((-3.4,-1.1), (-BODY,BODY), (1.1,3.4)):
            ix = sample & (side >= lo) & (side <= hi)
            if not ix.any():
                if lo != -BODY:
                    break
                middle = np.median(station[sample])
                center = station[abs(side) <= BODY]
                # A sparse ray ring can miss the center between two positive
                # laser cross-sections. Require current near-station center
                # radar and actual laser immediately on *both* sides in range.
                radar_here = radar_stations[abs(radar_stations-middle) < 1.]
                if (len(radar_here) < 2 or not np.any((center<middle)&(center>middle-3.5)) or
                        not np.any((center>middle)&(center<middle+3.5))):
                    break
                # Only station is consumed for the center witness. The height
                # model is fitted exclusively to measured left/right lidar.
                values.append([np.median(radar_here),0.,0.])
                continue
            values.append([np.median(station[ix]),np.median(side[ix]),np.median(height[ix])])
        if len(values) != 3:
            continue
        node = np.array(values)
        if nodes:
            ds = node[:,0] - nodes[-1][:,0]
            if np.max(ds) > 5. or np.min(ds) < .3:
                if np.max(ds) > 5.:
                    break
                continue
            slope = (node[[0,2],2] - nodes[-1][[0,2],2]) / ds[[0,2]]
            if max(abs(slope)) > .19:
                break
            if len(nodes) >= 2:
                previous = (nodes[-1][[0,2],2] - nodes[-2][[0,2],2]) / (nodes[-1][[0,2],0] - nodes[-2][[0,2],0])
                if max(abs(slope-previous)) > .055 + .012*min(ds):
                    break
        elif np.max(node[:,0]) > 4.5 or max(abs(node[:,2])) > .40:
            return None
        nodes.append(node)
    return np.asarray(nodes) if len(nodes) >= 5 else None


def _floor(nodes, station, side):
    left, right = nodes[:,0,:], nodes[:,2,:]
    zl=np.interp(station, left[:,0], left[:,2])
    zr=np.interp(station, right[:,0], right[:,2])
    yl=np.interp(station, left[:,0], left[:,1])
    yr=np.interp(station, right[:,0], right[:,1])
    bank=(zr-zl)/(yr-yl)
    return zl+(side-yl)*bank, bank


def supported_grade(lidar, radar, path, speed):
    """Distance with positive pavement support, or zero for an unverified surface."""
    laser, doppler = _returns(lidar), _returns(radar)
    if len(laser) < 30 or not len(doppler):
        return 0.
    x, y, z = laser[:,0], laser[:,1], laser[:,2]+LIDAR_Z
    plane = v2._ground_plane(x,y,z)
    roi = (x >= -.5) & (x < 42.) & (abs(y) < 15.)
    x, y, z = x[roi], y[roi], z[roi]
    s, d = v2._project(x, y, path)
    local = (s >= 0.) & (s < 39.) & (abs(d) <= 3.4)
    s, d, z, x, y = (v[local] for v in (s,d,z,x,y))
    relative = z - (plane[0]*x + plane[1]*y + plane[2])
    body = abs(d) <= BODY
    old_laser = body & (relative > .28) & (relative < 2.5)
    laser_gap = v2._gap(s,old_laser,support=2)

    velocity, az, alt, depth = doppler.T
    dist = depth*np.cos(alt)
    rx,ry,rz = dist*np.cos(az)+RADAR_X,dist*np.sin(az),depth*np.sin(alt)+RADAR_Z
    rs,rd = v2._project(rx,ry,path)
    rh = rz - (plane[0]*rx + plane[1]*ry + plane[2])
    rbody = (abs(rd) <= BODY) & (rs >= 0.) & (rx >= -.5) & (rx < 36.+BUMPER_X)
    old_radar = rbody & (rh > .28) & (rh < 2.5)
    radar_gap = v2._gap(rs,old_radar)
    # A genuine close return always stays in the original obstacle path.
    # Both modalities must independently resolve a distant raised surface.
    near, far = min(laser_gap,radar_gap), max(laser_gap,radar_gap)
    if near < 10. or near >= 32. or far >= 36.:
        return 0.
    # Group by laser elevation channel, so a continuous surface is not split
    # when a ring falls on an arbitrary fixed station-bin edge.
    beam = np.rint((np.rad2deg(np.arctan2(z-LIDAR_Z,np.hypot(x,y)))+28.)*31./38.).astype(int)
    nodes = _ribbon_nodes(s,d,z,beam,rs[rbody])
    if nodes is None:
        return 0.
    extent = min(36.,float(min(nodes[-1,:,0])))
    if extent < far+2.:
        return 0.
    confirmed = s <= extent
    ground, bank = _floor(nodes,s,d)
    if np.max(abs(bank[confirmed]), initial=0.) > .095:
        return 0.
    # All three strips must agree locally. Reject any unexplained elevated
    # central return, including single low, wheel-width objects, and any
    # broken cross-section along the approach or beyond the apparent obstacle.
    if np.any(abs(z[confirmed]-ground[confirmed]) > .10):
        return 0.
    radar_check = rbody & (rs <= extent)
    rground, _ = _floor(nodes,rs,rd)
    rv = velocity + speed*np.cos(az)*np.cos(alt)
    if np.any(abs(rz[radar_check]-rground[radar_check]) > .10) or np.any(abs(rv[radar_check]) > .75):
        return 0.
    rising, _ = _floor(nodes,np.array([near-2., near+1.]), np.zeros(2))
    if rising[1]-rising[0] < .06:
        return 0.
    return extent


class GradeObserver(ProgressObserver):
    def __init__(self):
        super().__init__()
        self.grade_history = deque(maxlen=9)

    def observe(self, obs, detections, local_route, speed, *, route_preview=None, stopline_gap=None):
        current = obs.get("progress_frame")
        if current is None or self.prior_frame is None or current != self.prior_frame + 1:
            self.grade_history.clear()
        x = super().observe(obs,detections,local_route,speed,route_preview=route_preview,stopline_gap=stopline_gap)
        distance = min(x[INDEX["path_body_lidar"]],x[INDEX["path_body_radar"]])
        if current is None or len(self.recent) < 1 or distance >= 32.:
            extent = 0.
        else:
            extent = supported_grade(obs["lidar"],obs["radar"],v2._checked_route(local_route,route_preview),speed)
        self.grade_history.append((extent,float(distance)))
        # The original recent body feature covers the current and eight previous
        # scans. A fresh road fit cannot erase an unexplained past return.
        recent = list(self.grade_history)
        confirmed = (min(row[0] for row in recent[-3:]) if len(recent)==9 and
                     all(proof > 0. or gap >= 24. for proof,gap in recent) else 0.)
        return np.r_[x,extent,confirmed].astype(np.float32)


class GradeModel(ProgressModel):
    feature_names = FEATURE_NAMES

    def new_observer(self):
        super().new_observer()
        return GradeObserver()

    def predict(self,x):
        x = np.asarray(x)
        if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
            raise ValueError("Invalid grade-aware city measurements")
        self.grade_extent, self.confirmed_grade = map(float,x[-4:-2])
        action = super().predict(np.r_[x[:-4],x[-2:]])
        self.progress_evidence.update(raw_lidar_gap_m=float(x[INDEX["path_body_lidar"]]),
            raw_radar_gap_m=float(x[INDEX["path_body_radar"]]),
            grade_extent_m=self.grade_extent, confirmed_grade_extent_m=self.confirmed_grade)
        return action

    def _finish_request(self,x,baseline,requested):
        original = super()._finish_request(x,baseline,requested)
        evidence = self.progress_evidence
        raw = min(float(x[INDEX[n]]) for n in ("path_body_lidar","path_body_radar",
                                               "path_body_lidar_recent","path_body_radar_recent"))
        evidence.update(raw_body_memory_gap_m=raw,grade_added_request_mps=0.)
        if evidence["binding"] != "occupied_corridor" or not 0. < self.confirmed_grade <= 36.:
            return original
        # The v3 head has already screened signals, opposing/crossing traffic,
        # vulnerable users and observation age. Only its additional progress
        # proposal may use the locally supported stopping distance.
        limits = {"grade_sight":self._stoppable_speed(self.confirmed_grade-self.standoff),
                  "sight_distance":self._stoppable_speed(evidence["sight_m"]-self.standoff),
                  "turn":np.sqrt(self.coefficients[1]/max(.001,evidence["curvature_per_m"])),
                  "demonstrated_cruise":self.coefficients[0],
                  "posted":(self.posted_limit_kmh if np.isfinite(self.posted_limit_kmh)
                            and self.posted_limit_kmh>0. else 30.)/3.6}
        if self.tactical_speed_limit is not None:
            limits["tactical"] = self.tactical_speed_limit
        if self.signed_speed_limit is not None:
            limits["signed_approach"] = self.signed_speed_limit
        binding = min(limits,key=limits.get)
        target = float(limits[binding])
        evidence.update(binding=binding, proposal_mps=target)
        if self.last_speed_request is not None and target > self.last_speed_request+.1*self.coefficients[2]:
            target = self.last_speed_request+.1*self.coefficients[2]
            evidence["binding"] = "recovery"
        chosen = max(original,target)
        evidence["grade_added_request_mps"] = float(chosen-original)
        return float(chosen)

    def save(self,path):
        np.savez_compressed(path,version=VERSION,revision=4,feature_names=FEATURE_NAMES,parameters=PARAMETERS,
                            coefficients=self.coefficients,base=np.frombuffer(self.parent_checkpoint,dtype=np.uint8))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if (str(d["version"].item())!=VERSION or int(d["revision"].item())!=4 or
                    tuple(d["feature_names"])!=FEATURE_NAMES or tuple(d["parameters"])!=PARAMETERS):
                raise ValueError("Incompatible grade-aware city checkpoint")
            return cls(d["base"].tobytes(),d["coefficients"])
