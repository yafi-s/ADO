"""Learned keep/yield/pass decision over two legal native lane alternatives.

Localized LiDAR points supply the same-lane lead gap and speed plus the
adjacent lane's front and rear gaps. No actor state enters these measurements.
The selected trajectory is a smooth geometric maneuver executed by the
unchanged learned steering and longitudinal controllers.
"""
from collections import deque
import json
import math
from pathlib import Path

import numpy as np

from .urban_map import _left_lane

MODEL = Path(__file__).resolve().parents[1] / 'models/tactical-pass/choice.json'
FEATURES = ('source_gap_m', 'source_speed_mps', 'target_front_m', 'target_rear_m')
ACTIONS = ('keep', 'yield', 'pass')


class GapChoice:
    def __init__(self, path=MODEL):
        data = json.loads(path.read_text(encoding='utf8'))
        if tuple(data['features']) != FEATURES or tuple(data['actions']) != ACTIONS:
            raise ValueError('Incompatible tactical pass classifier')
        self.tree = data

    def predict(self, measurements):
        node, t = 0, self.tree
        while t['left'][node] >= 0:
            edge = 'left' if measurements[t['feature'][node]] <= t['threshold'][node] else 'right'
            node = t[edge][node]
        probability = t['probability'][node]
        return ACTIONS[int(np.argmax(probability))], probability


def _points(wp):
    return np.asarray([(p.transform.location.x, p.transform.location.y, p.transform.location.z) for p in wp])


class PassingCorridor:
    """The straight dashed-line section of a two-lane navigation alternative."""
    def __init__(self, source, target):
        self.source, self.target = _points(source), _points(target)
        self.forward = self.source[-1,:2]-self.source[0,:2]
        self.forward /= np.linalg.norm(self.forward)
        self.right = np.asarray((-self.forward[1], self.forward[0]))
        self.arc = (self.source[:,:2]-self.source[0,:2]) @ self.forward
        self.width = float(np.median((self.source[:,:2]-self.target[:,:2]) @ self.right))

    @classmethod
    def from_map(cls, road_map, source_lane=(4, 3)):
        samples = []
        for s in range(1, 160):
            wp = road_map.get_waypoint_xodr(*source_lane, float(s))
            if wp is not None:
                samples.append((wp, _left_lane(wp)))
        if len(samples) < 35:
            raise ValueError('Insufficient legal two-lane corridor')
        if samples[0][0].lane_id > 0:
            samples.reverse()
        return cls(*zip(*samples))

    def coordinates(self, xy):
        delta = xy - self.source[0,:2]
        return delta @ self.forward, delta @ self.right

    def lane_points(self, stations, alpha):
        # The deployed corridor consists of parallel, straight OpenDRIVE lanes.
        # Extend the final tangent for waypoint lookahead at the test endpoint.
        source = self.source[0,:2] + stations[:,None]*self.forward
        return source - (self.width*alpha)[:,None]*self.right

    def planned_centers(self, stations, commit):
        """World-space preview for a recorded or active lateral commitment."""
        alpha = np.zeros_like(stations)
        if commit is not None:
            t = np.clip((stations-commit)/24., 0., 1.)
            alpha = t**3*(10.-15.*t+6.*t**2)
        return self.lane_points(stations, alpha)


def _nearest(distances):
    """Reject isolated single-beam outliers, retain a measured surface distance."""
    values = np.sort(distances[(distances >= 0.) & (distances < 59.)])
    if len(values) < 3:
        return 60.
    support = np.flatnonzero(values[2:]-values[:-2] < 1.)
    return float(values[support[0]+1]) if len(support) else 60.


class TacticalPass:
    def __init__(self, corridor, model=None, *, measured_hold=False):
        self.corridor = corridor
        self.model = model or GapChoice()
        self.history = deque(maxlen=10)
        self.commit = None
        self.pass_votes = 0
        self.scan = 0
        self.coverage = (False, False)
        self.info = {}
        self.yielding = False
        self.measured_hold = measured_hold
        self.source_age = None
        self.source_speed = None
        self.raw_source_gap = 60.
        self.source_clear = False

    @property
    def speed_limit(self):
        return 0. if self.yielding else None

    @classmethod
    def from_scene_map(cls, scene, *, measured_hold=False):
        urban = getattr(scene, '_urban', None)
        if urban is None:
            return None
        corridor = PassingCorridor.from_map(scene._map, urban.lane_change.source_lane)
        scene.tactical_lane_scoring = True
        return cls(corridor, measured_hold=measured_hold)

    def measure(self, pose, lidar, station):
        # CARLA's 360-degree ray-cast sensor is 2.30 m above the vehicle origin.
        matrix = np.asarray(pose.get_matrix())
        xyz = np.asarray(lidar)[:,:3]
        xyz = xyz[(abs(xyz[:,0]) > 3.1) | (abs(xyz[:,1]) > 1.36)]
        mount = matrix[:3,3] + 2.30*matrix[:3,2]
        world = xyz @ matrix[:3,:3].T + mount
        arc, side = self.corridor.coordinates(world[:,:2])
        road_z = np.interp(arc, self.corridor.arc, self.corridor.source[:,2])
        # At the road boundary, rear traffic on the approach can already
        # occupy the merge gap. Include its returns in the lane projection.
        height = world[:,2]-road_z
        valid = (height > .44) & (height < 3.7)
        relative = arc - station
        source_gap = _nearest(relative[valid & (abs(side)<1.30) & (relative > 2.8)])
        in_target = abs(side+self.corridor.width)<1.36
        adjacent = relative[valid & in_target]
        front, rear = _nearest(adjacent), _nearest(-adjacent)
        # A missing sector or a dropped scan provides no evidence of open
        # pavement. A finite obstacle distance is already positive evidence.
        pavement = relative[in_target & (abs(height)<.43)]
        covered = tuple(np.count_nonzero((side*pavement > 4.) & (side*pavement < 24.)) >= 8 for side in (1.,-1.))
        self.coverage = covered
        if front == 60. and not covered[0]:
            front = 0.
        if rear == 60. and not covered[1]:
            rear = 0.
        source_pavement = relative[(abs(side) < 1.) & (abs(height) < .43)] if self.measured_hold else ()
        source_gap, source_speed = self._source_motion(station, source_gap, source_pavement)
        return (source_gap, source_speed, front, rear)

    def _source_motion(self, station, gap, pavement):
        self.scan += 1
        self.raw_source_gap = gap
        self.source_age = None
        self.source_clear = False
        if self.measured_hold and self.history:
            last_scan, last_surface = self.history[-1]
            age = self.scan - last_scan
            elapsed = age / 10.
            expected = last_surface + (self.source_speed or 0.) * min(elapsed, .5) - station
            if gap < 60.:
                # A jump to a different surface cannot supply velocity for a
                # newly appearing lead. Allow for ordinary surface jitter.
                tolerance = 1.5 + elapsed * (30. if self.source_speed is None else 5.)
                if age > 5 or abs(expected - gap) > tolerance:
                    self.history.clear()
                    self.source_speed = None
            else:
                self.source_age = age
                if age <= 5 and self.source_speed is not None:
                    return float(np.clip(expected, 0., 59.)), self.source_speed
                # Empty returns alone cannot release an established wait. A
                # visible patch of source lane beyond the old bumper can.
                self.source_clear = bool(np.count_nonzero((pavement > expected + 1.) & (pavement < expected + 15.)) >= 3)
                if not self.source_clear:
                    return 60., 30.
        if gap < 60.:
            observed = station + gap
            self.history.append((self.scan, observed))
            self.source_age = 0
        else:
            self.history.clear()
        # Fit velocity in world coordinates, robust to returns switching
        # surfaces and to changes in ego speed while braking.
        velocity = [10.*(b[1]-a[1])/(b[0]-a[0]) for a in self.history for b in self.history
                    if b[0]-a[0]>=3]
        self.source_speed = float(np.clip(np.median(velocity),0.,30.)) if velocity else None
        return gap, self.source_speed if self.source_speed is not None else 30.

    def preview(self, ego, lidar, defaults, speed, lookaheads=(3.,7.,13.,18.,25.,32.,40.)):
        pose = ego.get_transform()
        xy = np.array((pose.location.x, pose.location.y))
        station, lateral = self.corridor.coordinates(xy)
        angle = math.radians(pose.rotation.yaw)
        heading = np.array((math.cos(angle), math.sin(angle)))
        was_yielding = self.yielding
        self.info, self.yielding = {}, False
        if (station < 0 or station > self.corridor.arc[-1] or lateral < -self.corridor.width-2.
                or lateral > 1.85 or heading @ self.corridor.forward < .85):
            if self.measured_hold:
                self.history.clear()
                self.source_speed = self.commit = None
                self.pass_votes = 0
            return defaults
        measured = self.measure(pose, lidar, station)
        action, probabilities = self.model.predict(measured)
        self.pass_votes = self.pass_votes+1 if action=='pass' else 0
        # The branch is reversible while most of the car is still centred
        # in its original lane. A newly occupied gap cancels the invitation.
        if self.commit is not None and lateral > -.45 and action == 'yield':
            self.commit = None
        if self.commit is None and self.pass_votes >= 3 and (not self.measured_hold or self.source_age == 0):
            self.commit = float(station)
        self.yielding = self.commit is None and action=='yield'
        if self.measured_hold and was_yielding and self.commit is None:
            # Maintain the stop through confirmation of the adjacent opening.
            # A freshly measured lead that gains enough speed (or distance)
            # to warrant ordinary following also releases the wait.
            following = action == 'keep' and self.source_age == 0 and self.source_speed is not None
            self.yielding = not (following or self.source_clear)
        stations = station+np.asarray(lookaheads)
        # Use the learned trajectory choice as long as this road segment is in range.
        # After completing a pass the original route and selected lane coincide.
        goals = self.corridor.planned_centers(stations, self.commit) - xy
        right = np.array((-heading[1], heading[0]))
        local = np.column_stack((goals @ heading, goals @ right))
        if self.commit is not None:
            beyond = stations > self.corridor.arc[-1]
            local[beyond] = np.asarray(defaults).reshape(-1,2)[beyond]
        self.info = dict(choice=action, maneuver='passing' if self.commit is not None else action,
                         probabilities=[round(float(v), 3) for v in probabilities],
                         **{k:round(v,3) for k,v in zip(FEATURES, measured)},
                         corridor_station_m=round(float(station),3),
                         adjacent_pavement_seen=[bool(v) for v in self.coverage],
                         source_lateral_m=round(float(lateral),3), committed=self.commit is not None)
        if self.measured_hold:
            if self.yielding:
                self.info['maneuver'] = 'yield'
            self.info.update(raw_source_gap_m=round(self.raw_source_gap, 3), source_age_scans=self.source_age,
                             source_pavement_clear=self.source_clear, tactical_speed_limit_mps=self.speed_limit)
        return local.ravel()

    def apply(self, predicted, speed):
        if self.yielding and not self.measured_hold:
            steer, _, brake = predicted
            return steer, 0., max(float(brake), min(.65, .22+.06*speed))
        return predicted
