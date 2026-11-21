"""Predict lateral surface occupancy from ordinary consecutive roof lidar scans and ego odometry."""
from collections import deque
import numpy as np

from . import behavior_v2 as v2
from .city_clearance import BODY
from .lidar_motion import Sweep
from .policy import BUMPER_X, LIDAR_Z


EMPTY = (60., 6., 0., 0.)  # bumper distance, seconds to path entry, inward speed, observation age


class CrossingObserver:
    def __init__(self):
        self.sweeps = deque(maxlen=7)
        self.anchor = None
        self.release_after = 0.
        self.last_seen = 0.
        self.last_gap = 60.
        self.eta = 6.
        self.inward = 0.
        self.clear_frames = 0
        self.pending = None

    @staticmethod
    def _prior(cluster, previous, elapsed):
        if previous is None:
            return None
        candidates = [old for old in previous.clusters
                      if .4 < cluster.scale / old.scale < 2.4 and
                      np.linalg.norm(cluster.center-old.center) < 18.*elapsed+.4]
        return min(candidates, key=lambda old: np.linalg.norm(cluster.center-old.center), default=None)

    @staticmethod
    def _route_axes(path, center):
        starts = path[:-1]
        delta = np.diff(path, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        tangents = delta / np.maximum(lengths[:,None], .01)
        along = np.clip(np.sum((center-starts)*tangents,axis=1),0.,lengths)
        projections = starts + along[:,None]*tangents
        i = int(np.argmin(np.linalg.norm(projections-center,axis=1)))
        tangent = tangents[i]
        normal = np.array([-tangent[1], tangent[0]])
        return tangent, normal, projections[i], float(lengths[:i].sum() + along[i] - BUMPER_X)

    def _incoming(self, current, previous, older, path, ego_speed):
        candidates = []
        dt = current.now - previous.now
        for cluster in current.clusters:
            center = (cluster.center-current.origin) @ current.rotation
            if center[0] < -2. or np.linalg.norm(center) > 34.:
                continue
            old = self._prior(cluster, previous, dt)
            if old is None or self._prior(old, current, dt) is not cluster:
                continue
            velocity = (cluster.center-old.center) @ current.rotation / dt
            tangent, normal, foot, station = self._route_axes(path, center)
            lateral = (center-foot) @ normal
            sign = np.sign(lateral)
            inward = float(-sign * (velocity @ normal))
            if inward < .8 or inward > 18. or inward < .55*abs(velocity @ tangent):
                continue
            # Require either measured free-space change or coherent independent
            # displacement on two successive intervals. Test both directions.
            coherent = False
            if older is not None:
                older_dt = previous.now - older.now
                ancestor = self._prior(old, older, older_dt)
                if ancestor is not None and self._prior(ancestor, previous, older_dt) is old:
                    old_velocity = ((old.center-ancestor.center) @ current.rotation) / older_dt
                    prev_inward = float(-sign * (old_velocity @ normal))
                    coherent = prev_inward >= .7 and np.linalg.norm(old_velocity-velocity) < max(1.5,.5*inward)
            if not coherent or not (previous.was_clear(cluster.world) or current.was_clear(old.world)):
                continue
            points = cluster.surface[:,:2]-foot
            lateral_edges = sign * (points @ normal)
            # Positive measured vehicle/body extent plus a buffer for a partially
            # observed face; avoid interpreting the nearest point as a full body.
            entry = max(0., float(min(lateral_edges))-BODY-.6) / inward
            leave = (max(0., float(max(lateral_edges)))+BODY+1.) / inward
            advance = float(velocity @ tangent) * entry
            first = float(np.min(points @ tangent)) - .8
            gap = station + advance + first
            if not -1. < gap < 34. or entry > 4.:
                continue
            # A vehicle already well past the meeting point before the earliest
            # lateral arrival does not need to stop. Avoid predicting a speedup.
            rear_clear_s = (gap + max(5.,cluster.scale) + 6.) / max(ego_speed, .5)
            if rear_clear_s + .8 < entry:
                continue
            intercept = (foot + tangent*(advance+first)) @ current.rotation.T + current.origin
            candidates.append((max(0.,gap), entry, inward, intercept, leave, cluster.center))
        return min(candidates, key=lambda c:c[0], default=None)

    @staticmethod
    def _open_at(scan, local, path):
        # Confirm the old conflict footprint and both of its shoulders with real
        # ground returns; an empty/missing sweep cannot release a measured yield.
        pts = np.asarray(scan)[:,:3]
        plane = v2._ground_plane(pts[:,0],pts[:,1],pts[:,2]+LIDAR_Z)
        delta = pts[:,:2]-local
        tangent, normal, _, _ = CrossingObserver._route_axes(path,local)
        along, side = delta @ tangent, delta @ normal
        h = pts[:,2]+LIDAR_Z - (pts[:,0]*plane[0]+pts[:,1]*plane[1]+plane[2])
        window = (along >= -1.) & (along <= 7.)
        ground = window & (abs(h)<.32)
        # The anchor is the near edge of the approaching surface, slightly
        # upstream of the conflict; include its length on the far side.
        if np.count_nonzero(window & (abs(side)<BODY+.5) & (h>.3) & (h<2.8)) >= 2:
            return False
        return all(np.count_nonzero(ground & (side>=lo) & (side<hi)) >= 2
                   for lo,hi in ((-3.6,-BODY),(-BODY,BODY),(BODY,3.6)))

    def observe(self, scan, pose, frame, speed, path):
        if frame is None:
            return EMPTY if self.anchor is None else (max(0., self.last_gap-speed*.1),self.eta,self.inward,6.)
        now = frame*.1
        if pose is None or len(scan) < 30:
            self.sweeps.clear()
            self.pending = None
            self.clear_frames = 0
            if self.anchor is None:
                return EMPTY
            self.last_gap = max(0., self.last_gap - speed*.1)
            return self.last_gap, max(0.,self.eta-(now-self.last_seen)), self.inward, max(0.,now-self.last_seen)
        x,y,yaw = pose
        c,s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        rotation = np.array([[c,-s],[s,c]])
        origin = np.array([x,y])
        if self.sweeps and abs(now - self.sweeps[-1].now - .1) > .001:
            self.sweeps.clear()
            self.pending = None
        current = Sweep(scan,origin,rotation,now)
        previous = self.sweeps[-3] if len(self.sweeps)>=3 else None
        older = self.sweeps[-6] if len(self.sweeps)>=6 else None
        conflict = self._incoming(current,previous,older,path,speed) if previous is not None else None
        self.sweeps.append(current)
        if conflict is not None:
            gap, eta, inward, anchor, leave, center = conflict
            count = 1
            if self.pending is not None:
                old_time, old_center, old_anchor, votes = self.pending
                if now-old_time < .21 and np.linalg.norm(center-old_center) < 3. and np.linalg.norm(anchor-old_anchor) < 2.5:
                    count += votes
            self.pending = now, center, anchor, count
            # A later, more distant moving surface cannot displace an unresolved
            # nearer stop. Keep the original world-space reference until clear.
            if count >= 2 and (self.anchor is None or np.linalg.norm(self.anchor-anchor) < 4. or gap <= self.last_gap):
                self.anchor = anchor
                self.eta, self.inward = eta, inward
                self.release_after = max(self.release_after,now+min(5.,leave)+.5)
                self.last_seen = now
                self.clear_frames = 0
        if self.anchor is None:
            return EMPTY
        local = (self.anchor-origin) @ rotation
        _,_,_,gap = self._route_axes(path,local)
        self.last_gap = max(0.,gap)
        if now >= self.release_after and self._open_at(scan,local,path):
            self.clear_frames += 1
        else:
            self.clear_frames = 0
        if self.clear_frames >= 3 or local[0]<-4.:
            self.anchor = None
            self.release_after = 0.
            return EMPTY
        age = max(0., now-self.last_seen)
        return self.last_gap, max(0.,self.eta-age), self.inward, age
