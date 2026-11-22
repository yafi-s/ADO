"""Discretionary lane choice from measured surfaces on connected native lanes."""
from collections import deque
import math

import carla
import numpy as np

from .maneuver_choice import FEATURES, ManeuverChoice
from .maneuver_map import discover
from .overtake import _nearest

LOOK = (3., 7., 13., 18., 25., 32., 40.)


def blend(stations, start, length):
    t = np.clip((stations-start)/length, 0., 1.)
    return t**3*(10.+t*(-15.+6.*t))


def usable_gap(front, rear, fc, rc, speed):
    # Distances start at the ego origin and end on the adjoining surface.
    # Reserve the longitudinal body envelope, two seconds of headway to
    # traffic ahead, and at least five seconds before a closing encounter.
    front_room, rear_room = front-2.5, rear-2.5
    return (front_room > max(15., 2.*speed, 5.*fc)
            and rear_room > max(12., 2.*max(0., speed+rc), 5.*rc))


class Surface:
    """Short, association-gated history of a lane's nearest supported surface."""
    def __init__(self):
        self.history = deque(maxlen=10)
        self.speed = None
        self.age = None

    def update(self, tick, station, separation, sign):
        if self.history:
            last, prior = self.history[-1]
            self.age = tick-last
            elapsed = self.age*.1
            expected = prior + (self.speed or 0.)*elapsed
            if separation < 60. and (self.age > 6 or
                    abs(station+sign*separation-expected) > 1.25+elapsed*(20. if self.speed is None else 7.)):
                self.history.clear()
                self.speed = None
            if separation >= 60.:
                if self.age <= 8:
                    return float(np.clip(sign*(expected-station), 0., 59.))
                self.speed = None
                return 60.
        if separation < 60.:
            self.age = 0
            self.history.append((tick, station+sign*separation))
            velocities = [(b-a)/(.1*(tb-ta)) for ta,a in self.history for tb,b in self.history if tb-ta >= 3]
            self.speed = float(np.clip(np.median(velocities), -8., 40.)) if velocities else None
        return separation


class RouteManeuvers:
    measured_hold = True

    def __init__(self, corridors, model=None):
        self.corridors = corridors
        self.model = model or ManeuverChoice()
        self.corridor = None
        self.info = {}
        self.last_light = None
        self._reset()

    @classmethod
    def from_route(cls, road_map, route):
        return cls(discover(road_map, route))

    def _reset(self):
        self.surfaces = [Surface() for _ in range(4)]
        self.tick = self.pass_votes = self.return_votes = self.clear_votes = 0
        self.phase = "keep"
        self.start = self.return_start = None
        self.lead_station = self.lead_speed = None
        self.return_signalling = False
        self.travel_length = 23.
        self.speed_limit = None
        self.turn_signal = "off"

    def _measure(self, pose, scan, station, speed):
        lidar = np.asarray(scan).reshape(-1,4)[:,:3]
        lidar = lidar[(abs(lidar[:,0]) > 3.1) | (abs(lidar[:,1]) > 1.36)]
        matrix = np.asarray(pose.get_matrix())
        world = lidar @ matrix[:3,:3].T + matrix[:3,3] + 2.30*matrix[:3,2]
        rel, side, height, width, inside = self.corridor.project_scan(world, station)
        solid = inside & (height > .25) & (height < 3.6)
        pavement = inside & (abs(height) < .20)
        source, target = abs(side)<1.20, abs(side+width)<1.42
        measurements, covered, raw, visibility = [], [], [], []
        for surface, lane, sign in zip(self.surfaces, (source,source,target,target), (1.,-1.,1.,-1.)):
            distance = sign*rel
            # Ignore the ego's body. Side-by-side traffic in the other lane
            # can be centred at zero longitudinal separation.
            mask = solid & lane
            measured = _nearest(distance[mask])
            if np.count_nonzero(mask & (abs(rel)<2.)) >= 3:
                measured = 0.
            road = distance[pavement & lane]
            sight = all(np.count_nonzero((road>lo)&(road<hi)) >= 2
                        for lo,hi in ((4.,11.),(11.,19.),(19.,31.)))
            visibility.append(float(np.max(road,initial=0.)))
            covered.append(bool(sight))
            raw.append(measured)
            measurements.append(surface.update(self.tick, station, measured, sign))
        gap,srear,front,rear = measurements
        lead,_,ahead,behind = self.surfaces
        # An unmeasured adjoining vehicle's speed is uncertain. Assume it can
        # approach until several returns establish longitudinal velocity.
        fc = max(0.,speed-(ahead.speed if ahead.speed is not None else 0.)) if front < 60. else 0.
        rc = max(0.,(behind.speed if behind.speed is not None else 15.)-speed) if rear < 60. else 0.
        source_speed = max(0.,lead.speed) if lead.speed is not None else 20.
        x = (gap,source_speed,front,rear,fc,rc,speed)
        safe = (covered[2] and covered[3] and visibility[2] > max(24., 2.*speed+5.)
                and visibility[3] > 24. and usable_gap(front,rear,fc,rc,speed))
        # A far detected surface still limits sight. Use the surveyed lane
        # tangent and actual pavement ahead; the map alone never opens a gap.
        source_rear_v = self.surfaces[1].speed
        src_rc = max(0.,(source_rear_v if source_rear_v is not None else 15.)-speed) if srear<60. else 0.
        src_fc = max(0.,speed-source_speed) if gap<60. else 0.
        source_return_safe = (covered[0] and covered[1] and visibility[0]>max(24.,2.*speed+5.)
                              and visibility[1]>24. and usable_gap(gap,srear,src_fc,src_rc,speed))
        return x, raw, covered, visibility, safe, source_return_safe, srear

    def preview(self, ego, lidar, defaults, speed, lookaheads=LOOK):
        pose = ego.get_transform()
        xy = np.array((pose.location.x,pose.location.y))
        a = math.radians(pose.rotation.yaw)
        heading = np.array((math.cos(a),math.sin(a)))
        corridor = self.corridor
        located = corridor.locate(xy,heading) if corridor else None
        if located is None:
            self.corridor = None
            self._reset()
            for c in self.corridors:
                if (located := c.locate(xy,heading)) is not None:
                    self.corridor = c
                    break
        self.info = {"planner":"route-maneuvers-v1", "choice":"keep", "maneuver_phase":"keep",
                     "maneuver":"keep","committed":False, "turn_signal":"off"}
        if located is None:
            return defaults
        c, (s, lateral) = self.corridor, located
        self.tick += 1
        x, raw, cover, sight, safe, rejoin_safe, source_rear = self._measure(pose,lidar,s,speed)
        gap,lv,front,rear,_,rear_closing,_ = x
        action,probs = self.model.predict(x)
        source = self.surfaces[0]
        fresh = source.age == 0 and source.speed is not None
        recent = source.age in (0,1,2) and source.speed is not None
        if source.age is None or source.age > 8:
            kind = "none"
        elif not recent:
            kind = "unconfirmed"
        elif lv < .55:
            kind = "static"
        elif lv < 4.3:
            kind = "slow_moving"
        else:
            kind = "departing" if lv > speed+.5 else "moving"
        room = c.end-s
        route_merge = c.merge_at is not None
        required = route_merge and s >= c.merge_at-16.
        enough_room = room > (24. if route_merge else max(57.,gap+33.+lv*4.))
        self.speed_limit = None
        self.turn_signal = "off"
        width = float(np.interp(s,c.arc,c.width))
        if self.start is None:
            waiting = self.phase == "yield"
            # Positive observations over the old source position release a
            # wait after occlusion. A dropped scan cannot start the car.
            clear = cover[0] and raw[0]>=60. and (not source.history or
                    sight[0] > source.history[-1][1]-s+3.)
            self.clear_votes = self.clear_votes+1 if clear else 0
            invitation = (required or (action=="pass" and recent and kind in ("static","slow_moving")))
            self.pass_votes = self.pass_votes+1 if invitation and safe and enough_room else 0
            if self.pass_votes >= 6 and (fresh or required):
                self.start = s
                self.travel_length = min(24., max(12., (gap-3.)*1.05)) if not required else 23.
                self.lead_station = s+gap if fresh else None
                self.lead_speed = lv if fresh else 0.
                self.phase = "change_left"
            elif self.pass_votes:
                self.phase = "signal_left"
                self.turn_signal = "left"
                if waiting or speed < .35:
                    self.speed_limit = 0.
            elif ((required and not safe) or (action in ("pass","yield") and fresh and enough_room and not safe)
                    or (waiting and not fresh and self.clear_votes < 3)):
                self.phase = "yield"
                self.speed_limit = 0.
            else:
                self.phase = "follow" if fresh else "keep"
        if self.start is not None:
            self.turn_signal = "left"
            if self.return_start is None and lateral > -.5 and (not safe or
                    (fresh and kind in ("moving","departing") and not required)):
                self.start = None
                self.pass_votes = 0
                self.phase = "yield" if not safe else "follow"
                self.turn_signal = "off"
                self.speed_limit = 0. if self.phase=="yield" else None
            elif self.return_start is not None:
                self.phase = "change_right"
                self.turn_signal = "right"
                if lateral < -width+.50 and not rejoin_safe:
                    self.return_start = None
                    self.return_votes = 0
                    self.phase = "wait_rejoin"
                    self.turn_signal = "off"
                elif s >= self.return_start+20. and lateral > -.48:
                    self.phase = "rejoined"
                    self.return_signalling = False
                    self.turn_signal = "off"
                    self.start = self.return_start = None
                    self.return_votes = self.pass_votes = 0
            elif lateral <= -.77*width:
                self.turn_signal = "off"
                if route_merge:
                    self.phase = "route_merge"
                else:
                    self.phase = "passing"
                    # Wait until the passed object's measured rearward return
                    # actually lies behind the car; pavement checks include both
                    # sides of the return envelope. No simulated actor lookup.
                    if fresh and 0.<gap<40. and self.lead_station is not None and abs(s+gap-self.lead_station)<7.:
                        self.lead_station = s+gap
                        self.lead_speed = lv
                    elif self.lead_station is not None:
                        self.lead_station += .1*self.lead_speed
                    # When approaching a second source-lane vehicle, make
                    # room to return behind it. Ease speed using its measured
                    # headway and motion; no speed request can open the gap.
                    if (recent and self.lead_station is not None and s+gap > self.lead_station+12.
                            and 20. < gap < 55. and s > self.lead_station+2.):
                        self.speed_limit = max(3., lv+max(0.,gap-20.)/6.)
                    past = self.lead_station is not None and s-self.lead_station>8.
                    seen_behind = self.surfaces[1].age == 0 and 9.<source_rear<50.
                    self.return_votes = self.return_votes+1 if past and seen_behind and rejoin_safe else 0
                    if past:
                        # Maintain the physical indicator during sparse pavement
                        # scans. Fresh measured clearance still gates movement.
                        self.return_signalling |= bool(self.return_votes)
                        if source_rear < 8. or gap < 14.:
                            self.return_signalling = False
                        self.phase = "signal_right" if self.return_signalling else "wait_rejoin"
                        self.turn_signal = "right" if self.return_signalling else "off"
                    if self.return_votes >= 5:
                        self.phase = "change_right"
                        self.return_start = s
                    if room < 22. and self.return_start is None:
                        self.speed_limit = max(0., min(speed, (room-6.)*.32))
            else:
                self.phase = "change_left"
            if self.start is not None and front < max(5., .85*speed):
                self.speed_limit = 0.
        committed = self.start is not None
        self.info = dict(planner="route-maneuvers-v1", choice=action, maneuver_phase=self.phase,
                         maneuver="passing" if committed else self.phase, committed=committed,
                         probabilities=probs, turn_signal=self.turn_signal, lead_kind=kind,
                         source_lane=list(c.source_key), target_lane=list(c.target_key),
                         corridor_station_m=round(s,3), corridor_remaining_m=round(room,3),
                         source_lateral_m=round(lateral,3), source_age_scans=source.age,
                         raw_source_gap_m=round(raw[0],3), adjacent_pavement_seen=cover[2:],
                         pavement_reach_m=[round(a,2) for a in sight],
                         target_gap_accepted=bool(safe), return_gap_accepted=bool(rejoin_safe),
                         source_rear_m=round(source_rear,3), navigation_merge=route_merge,
                         tactical_speed_limit_mps=self.speed_limit,
                         **{k:round(float(v),3) for k,v in zip(FEATURES,x)})
        if not committed and not route_merge:
            return defaults
        stations = s+np.asarray(lookaheads)
        alpha = blend(stations,self.start,self.travel_length) if committed else np.zeros(len(stations))
        if self.return_start is not None:
            alpha *= 1.-blend(stations,self.return_start,20.)
        delta = c.goals(stations,alpha)-xy
        local = np.column_stack((delta@heading, delta@(-heading[1],heading[0])))
        beyond = stations>c.end
        local[beyond] = np.asarray(defaults).reshape(-1,2)[beyond]
        return local.ravel()

    def update_lights(self, ego):
        state = int(ego.get_light_state())
        left = int(carla.VehicleLightState.LeftBlinker)
        right = int(carla.VehicleLightState.RightBlinker)
        selected = {"off":0, "left":left,"right":right}[self.turn_signal]
        command = (state & ~(left|right)) | selected
        if self.last_light != command or state != command:
            ego.set_light_state(carla.VehicleLightState(command))
            self.last_light = command
        self.info["signal_command_bits"] = command
        self.info["previous_signal_bits"] = state & (left|right)

    def legal_footprint(self, ego, road_map):
        """Independent native-lane membership for scoring while a pass is active."""
        if not self.info.get("committed") or self.corridor is None:
            return False
        keys = {self.corridor.source_key, self.corridor.target_key}
        for v in ego.bounding_box.get_world_vertices(ego.get_transform()):
            # Project the horizontal corner onto the road surface first. CARLA's
            # non-projecting query tests 3-D distance to the lane center and
            # can reject a roof corner above an interior dashed line.
            wp = road_map.get_waypoint(v, project_to_road=True, lane_type=carla.LaneType.Driving)
            if not wp or wp.is_junction or (wp.road_id,wp.lane_id) not in keys:
                return False
            # Native nearest-waypoint lookup can select the other side of a
            # dashed line by several centimetres. Test the *union* of the two
            # legal lanes against their actual outer boundaries.
            other = self.corridor.target_key if (wp.road_id,wp.lane_id)==self.corridor.source_key else self.corridor.source_key
            neighbor = road_map.get_waypoint_xodr(*other, wp.s)
            for lane in (wp,neighbor):
                if lane is None:
                    continue
                surface = lane.transform
                dx, dy = v.x-surface.location.x, v.y-surface.location.y
                yaw = math.radians(surface.rotation.yaw)
                if abs(-dx*math.sin(yaw)+dy*math.cos(yaw)) <= lane.lane_width/2.+.01:
                    break
            else:
                return False
        return True

    @staticmethod
    def apply(predicted, speed):
        return predicted
