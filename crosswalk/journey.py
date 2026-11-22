"""Continuous, metre-sampled CARLA city drive with native, independently scored encounters."""
from bisect import bisect_left
import math
import random
from dataclasses import dataclass, field

import carla

from .district import connectors
from .world import Scenario, _angle, _clip, _oncoming_pair, _raised, _vehicle_blueprint

WALKERS = {"pedestrian", "pedestrian-delayed", "occluded"}
LEADS = {"lead", "braking"}


def required_event_kinds(map_name, *, barriers=False, maneuvers=False):
    """Immutable encounter contract for each supported continuous journey course."""
    town = map_name.rsplit("/",1)[-1].lower()
    base = ("pedestrian","pedestrian-delayed","occluded","lead","braking",
            "left","signal","cyclist","roadworks")
    if town in ("town01","town01_opt"):
        return base
    if town in ("town03","town03_opt"):
        fixed = (*base,"bus","barrier") if barriers else (*base,"bus")
        return (*fixed, "in-lane-obstruction", "slow-traffic") if maneuvers else fixed
    raise ValueError(f"No commissioned journey layout for {map_name}")


def journey_points(road_map, seed, destination=None):
    """Follow planned OpenDRIVE connectors. The two destinations fork at the last junction."""
    destination = destination or ("east" if seed % 2 else "south")
    if destination not in ("east", "south"):
        raise ValueError("Journey destinations are east and south")
    streets = (6, 19, 12, 23, 4, 17, 10, 25, 1, 0 if destination == "east" else 16)
    links = connectors(road_map)
    planned = [links[a, b] for a, b in zip(streets, streets[1:])]
    start = planned[0].previous(random.Random(seed).uniform(170, 185))
    if len(start) != 1 or start[0].road_id != streets[0]:
        raise RuntimeError("Town01 journey start does not match planned approach")
    points = [start[0]]
    turn = final_steps = 0
    for _ in range(1600):
        choices = points[-1].next(1.)
        if len(choices) > 1 and turn < len(planned):
            target = planned[turn]
            choices = [p for p in choices if (p.road_id, p.lane_id) == (target.road_id, target.lane_id)]
        if len(choices) != 1:
            raise RuntimeError("Town01 journey disconnected at a planned movement")
        nxt = choices[0]
        points.append(nxt)
        if turn < len(planned) and nxt.road_id == planned[turn].road_id:
            turn += 1
        if nxt.road_id == streets[-1] and turn == len(planned):
            final_steps += 1
            if final_steps == 23:
                return points
    raise RuntimeError("Town01 journey endpoint unreachable")


def _junctions(points):
    ranges = []
    start = None
    for i, p in enumerate(points):
        if p.is_junction and start is None:
            start = i
        elif not p.is_junction and start is not None:
            yaw = _angle(p.transform.rotation.yaw - points[start-1].transform.rotation.yaw)
            ranges.append((start, i, "left" if yaw < -50 else "right" if yaw > 50 else "straight"))
            start = None
    return ranges


def _signal_site(points, junctions, signals):
    """Bind the exact stop waypoint to a lane and to this local approach/connector."""
    stops = [(s, stop, {(w.road_id, w.lane_id) for w in s.get_affected_lane_waypoints() if w})
             for s in signals for stop in s.get_stop_waypoints() if stop]
    # Prefer the isolated intersection following the occluded crossing.
    for j in (6, 7, 8, 4, 3, 2, 1, 5):
        start, end, _ = junctions[j]
        local = {(p.road_id, p.lane_id) for p in points[start-5:end]}
        hits = []
        for signal, stop, affected in stops:
            if not (affected & local):
                continue
            for i in range(start-15, start+6):
                p = points[i]
                if (p.road_id, p.lane_id) == (stop.road_id, stop.lane_id):
                    d = p.transform.location.distance(stop.transform.location)
                    if d <= 1.2:
                        hits.append((d, signal.id, i, signal, stop))
        if hits:
            _, _, idx, signal, stop = min(hits, key=lambda h: h[:3])
            return idx, signal, stop
    raise RuntimeError("No physical traffic light controls a planned Town01 journey approach")


def _offset(point, location, direction):
    delta = location - point
    return delta.x * direction.x + delta.y * direction.y


def _van_blocks_walker(ego, van, walker):
    """Intersect the front-camera-to-walker torso segment with the physical van box."""
    eye = ego.get_transform().transform(carla.Location(1.3, 0, 1.65))
    box = walker.bounding_box
    torso = walker.get_transform().transform(carla.Location(box.location.x, box.location.y,
                                                             box.location.z))
    transform = van.get_transform()
    box = van.bounding_box
    vehicle_inverse = transform.get_inverse_matrix()
    box_inverse = carla.Transform(box.location, box.rotation).get_inverse_matrix()

    def local(point):
        xyz = [point.x, point.y, point.z, 1.]
        for matrix in (vehicle_inverse, box_inverse):
            xyz = [sum(c*v for c, v in zip(row, xyz)) for row in matrix]
        return carla.Location(*xyz[:3])

    origin, end = local(eye), local(torso)
    lo, hi = 0., 1.
    for axis in ("x", "y", "z"):
        center, radius = 0., getattr(box.extent, axis)
        a, delta = getattr(origin, axis), getattr(end, axis)-getattr(origin, axis)
        if abs(delta) < 1.e-8:
            if abs(a-center)>radius:
                return False
        else:
            near, far = sorted(((center-radius-a)/delta, (center+radius-a)/delta))
            lo, hi = max(lo,near), min(hi,far)
            if lo > hi:
                return False
    return 0. < lo < 1.


def _speed(actor):
    v = actor.get_velocity()
    return math.hypot(v.x, v.y)


@dataclass
class Encounter:
    kind: str
    index: int
    pose: object
    s: float
    actor: object = None
    state: str = "waiting"
    observed: dict = field(default_factory=dict)
    min_gap: float = 60.
    stopped: bool = False
    violation: bool = False
    last_line: float | None = None
    release_pos: object = None
    next_action: float | None = None
    target: object = None
    conflict: object = None
    direction: object = None
    exit_path: list = field(default_factory=list)
    previous_speed: float | None = None
    peak_decel: float = 0.
    pass_station: float | None = None
    exit_road: int | None = None
    exit_lanes: set = field(default_factory=set)
    exit_arc: list = field(default_factory=list)
    exit_limits: list = field(default_factory=list)
    lane_change: object = None
    queue: object = None
    walk_setback: float = 0.
    walk_finish_side: float = -6.
    cones: list = field(default_factory=list)
    encountered_cones: set = field(default_factory=set)
    cone_origins: list = field(default_factory=list)
    barriers: list = field(default_factory=list)
    barrier_origins: list = field(default_factory=list)
    encountered_barriers: set = field(default_factory=set)

    def mark(self, name, time):
        self.observed.setdefault(name, round(time, 3))

    def record(self):
        o = self.observed
        evidence = ["triggered_s", "exposed_s", "cleared_s", "passed_s"]
        if self.kind == "occluded":
            evidence.extend(("blocked_view_s", "emerged_s"))
        elif self.kind == "signal":
            evidence.append("crossed_green_s")
            if self.queue is not None:
                evidence.extend(("queue_exposed_s","queue_departed_green_s"))
        elif self.kind == "bus":
            evidence.extend(("source_lane_seen_s","measured_lane_change_s","bus_alongside_s"))
        elif self.kind in LEADS:
            evidence.extend(("restarted_s","left_route_s"))
            if self.kind == "braking":
                evidence.append("lead_stopped_s")
        elif self.kind == "cyclist":
            evidence.extend(("moving_ahead_s", "left_route_s"))
        elif self.kind in ("in-lane-obstruction", "slow-traffic"):
            evidence.extend(("before_target_s", "alternate_alongside_s", "rear_cleared_target_s", "returned_to_route_s"))
            if self.kind == "slow-traffic":
                evidence.extend(("moving_target_s", "moving_alongside_s", "occupied_gap_wait_s", "adjacent_traffic_cleared_s"))
        elif self.kind == "barrier":
            evidence.append("all_barriers_passed_s")
        elif self.kind == "roadworks":
            evidence.append("all_cones_passed_s")
        physically_ordered = True
        if self.kind in ("in-lane-obstruction", "slow-traffic"):
            phases = ["before_target_s"]
            if self.kind == "slow-traffic":
                phases.extend(("occupied_gap_wait_s", "adjacent_traffic_cleared_s"))
                physically_ordered = o.get("occupied_gap_frames", 0) >= 3
            phases.extend(("alternate_alongside_s", "rear_cleared_target_s", "returned_to_route_s"))
            physically_ordered = physically_ordered and all(key in o for key in phases) and all(
                o[a] < o[b] for a, b in zip(phases, phases[1:]))
        return dict(kind=self.kind, station_m=round(self.s, 2), phase=self.state,
                    triggered="triggered_s" in o, cleared="cleared_s" in o,
                    exposed="exposed_s" in o, passed="passed_s" in o,
                    handled=physically_ordered and all(k in o for k in evidence)
                            and not self.violation and "runout_reentered_s" not in o,
                    violation=self.violation, stopped_for_event=self.stopped,
                    minimum_gap_m=round(self.min_gap, 2), **o,
                    **({"peak_lead_deceleration_mps2": round(self.peak_decel, 2)}
                       if self.kind=="braking" else {}))


class JourneyScenario(Scenario):
    """All movement commands target NPCs. The caller drives and samples the single ego."""
    def __init__(self, world, kind, seed, perturb=False, district=None, crossing_delay=0., destination=None, *,
                 barriers=False, population_profile="standard", maneuver_case="continuous"):
        if population_profile not in ("standard", "city-life", "city-maneuvers"):
            raise ValueError(f"Unknown journey population profile: {population_profile}")
        self.population_profile = population_profile
        # The original crowded registration remains selectable for exact-geometry
        # stress comparisons. The normal Market course retains the downstream
        # stopped-sedan encounter at an independently fixed native site.
        original_spacing = maneuver_case in ("original-continuous", "original-later-gap")
        if original_spacing:
            maneuver_case = maneuver_case.removeprefix("original-")
        spaced_market = (population_profile == "city-maneuvers" and destination != "market-loop" and
                         maneuver_case in ("continuous", "later-gap") and not original_spacing)
        self.maneuver_fixture = None
        if kind != "journey":
            raise ValueError("JourneyScenario expects kind='journey'")
        self.kind = self.name = kind
        self.world, self.seed = world, int(seed)
        self.district = "townwide"
        self.destination = destination or ("east" if seed % 2 else "south")
        self.crossing_delay = float(crossing_delay)
        if not math.isfinite(self.crossing_delay):
            raise ValueError("crossing_delay must be finite")
        self._actors = []
        self._closed = False
        self._previous_weather = world.get_weather()
        self._signal_states, self._signals_were_frozen = [], None
        self.ego = self.hazard = self.occluder = self.traffic = self.ambient = self.signal = None
        self.events = []
        self._time, self._last_observation = 0., None
        self._dt = world.get_settings().fixed_delta_seconds
        if self._dt is None or self._dt <= 0:
            raise ValueError("Journey requires a synchronous fixed timestep")
        self._event_start = self.turn_direction = None
        self._cleared = False
        self.red_light_violations = self.yield_violations = 0
        self._line = None
        self.phase = "clear road"
        self._active = None
        self._gap = 60.
        rng = random.Random(self.seed ^ 0x173)
        self.spawn_lateral_m = rng.choice((-1, 1))*rng.uniform(.15, .35) if perturb else 0.
        self.spawn_yaw_deg = rng.choice((-1, 1))*rng.uniform(2., 4.) if perturb else 0.
        self.weather = "ClearNoon"
        self._map = world.get_map()
        try:
            town = self._map.name.rsplit("/",1)[-1].lower()
            self._urban = None
            required = required_event_kinds(town, barriers=barriers)
            if town in ("town03","town03_opt"):
                from .urban_map import urban_route
                self._urban = urban_route(self._map, self.seed, destination)
                self.destination = getattr(self._urban, "destination", None) or destination or "urban"
                wp = self._points = self._urban.points
                if spaced_market:
                    from .urban_map import _station_index
                    # Road 67 ends at its next junction at OpenDRIVE s310.4.
                    # Keep the held Tesla about 35 m upstream of that junction
                    # and retain its original native turn-off and release logic.
                    self._urban.sites["lead"] = _station_index(wp, 67, -2, 275.)
            else:
                wp = self._points = journey_points(self._map, self.seed, self.destination)
            self._set_route([w.transform.location for w in wp])
            self._station = [seg[-1] for seg in self._segments] + [self.length]
            if len(self._station) != len(wp):
                raise RuntimeError("Journey contains duplicated waypoint geometry")
            self.turns = _junctions(wp)
            self._junction_ranges = [(self._station[a], self._station[b]) for a,b,_ in self.turns]
            self._turn_observations = [dict(junction_id=wp[a].junction_id,direction=d,
                                             entry_m=round(self._station[a],2),
                                             exit_m=round(self._station[b],2))
                                       for a,b,d in self.turns]
            self._clear_distance = 0.
            self._last_progress = 0.
            self._was_clear = False
            self.road_id, self.lane_id = wp[0].road_id, wp[0].lane_id
            self.road_ids = tuple(dict.fromkeys(w.road_id for w in wp if not w.is_junction))
            self.junction_ids = tuple(wp[a].junction_id for a, _, _ in self.turns)
            self._route_lanes = {(w.road_id, w.lane_id) for w in wp if not w.is_junction}
            self.route_id = f"{town}-journey:{wp[0].id}:{wp[-1].id}"
            self._catalog = world.get_blueprint_library()
            bp = _vehicle_blueprint(self._catalog, "vehicle.lincoln.mkz", "vehicle.lincoln.mkz_2020")
            bp.set_attribute("role_name", "hero")
            spawn = _raised(wp[0].transform, .35)
            spawn.location = spawn.location + spawn.get_right_vector()*self.spawn_lateral_m
            spawn.rotation = carla.Rotation(pitch=spawn.rotation.pitch, yaw=spawn.rotation.yaw+self.spawn_yaw_deg)
            self.ego = self._required(bp, spawn)
            signals = list(world.get_actors().filter("traffic.traffic_light"))
            idx, self.signal, stop = (self._urban.signal_site(signals) if self._urban
                                      else _signal_site(wp, self.turns, signals))
            self._signal_states = [(s, s.get_state()) for s in signals]
            self._signals_were_frozen = signals[0].is_frozen()
            world.freeze_all_traffic_lights(True)
            if self._urban:
                self._set_urban_lights(signals)
            else:
                for lamp in signals:
                    lamp.set_state(carla.TrafficLightState.Green)
            signal = self._event("signal", idx, stop.transform)
            if self._urban:
                self._spawn_queue(signal)
            # All hazards exist at setup so the learner's progress cannot pop actors into view.
            for k, street, offset in (("pedestrian",6,55), ("pedestrian-delayed",19,50),
                                      ("lead",12,70), ("braking",4,65), ("occluded",10,65)):
                idx = (self._urban.sites[k] if self._urban else
                       next(i for i,w in enumerate(wp) if w.road_id == street) + offset)
                event = self._event(k, idx)
                if k == "lead" and population_profile == "city-maneuvers":
                    event.observed.update(native_road=wp[idx].road_id, native_lane=wp[idx].lane_id,
                                          native_road_s=round(wp[idx].s, 3),
                                          stopped_lead_layout="junction-approach" if spaced_market else "original")
                if k in WALKERS:
                    self._spawn_walker(event, rng)
                else:
                    self._spawn_lead(event)
            if self._urban:
                start, end = self._urban.left_turn
                pair = self._urban.oncoming
            else:
                start, end, turn = self.turns[0]
                approach = wp[start-24:end+12]
                pair = _oncoming_pair(self._map, approach, connectors(self._map)) if turn=="left" else None
            left = self._event("left", start-2)
            if pair is None:
                raise RuntimeError("Journey left-turn conflict has no through movement")
            near, far = pair
            bp = _vehicle_blueprint(self._catalog,"vehicle.mini.cooper","vehicle.audi.a2")
            bp.set_attribute("role_name","oncoming")
            self.traffic = left.actor = self._required(bp, _raised(far.transform,.35))
            left.actor.apply_control(carla.VehicleControl(brake=1.))
            left.target = near.transform.location
            left.direction = far.transform.get_forward_vector()
            left.conflict = min(self.route[start:end], key=lambda p: abs(_offset(far.transform.location,p,
                                                                               far.transform.get_right_vector())))
            # A tapered work zone along one edge of the first straight block.
            cone_index = self._urban.sites["roadworks"] if self._urban else 109
            self._spawn_roadworks(self._event("roadworks", cone_index))
            cycle_index = self._urban.sites["cyclist"] if self._urban else self.turns[-1][0]-33
            cycle = self._event("cyclist", cycle_index)
            self._spawn_cyclist(cycle)
            if self._urban:
                lane_change = self._urban.lane_change
                self._spawn_bus(self._event("bus", lane_change.start), lane_change)
                if barriers:
                    from .barrier_scene import spawn_barriers
                    spawn_barriers(self)
            if population_profile == "city-maneuvers":
                from .maneuver_scene import ManeuverFixture
                if self.destination == "market-loop":
                    from .market_loop_traffic import LoopFixture
                    self.maneuver_fixture = LoopFixture(self, maneuver_case)
                else:
                    self.maneuver_fixture = ManeuverFixture(self, maneuver_case)
                required = (*required, *self.maneuver_fixture.event_names)
            self.events.sort(key=lambda e:e.s)
            if sorted(e.kind for e in self.events) != sorted(required):
                raise RuntimeError("Journey setup does not match the course encounter contract")
            if population_profile == "city-maneuvers":
                from .maneuver_scene import ManeuverPopulation
                if self.destination == "market-loop":
                    from .market_loop_traffic import LoopPopulation
                    self.ambient = LoopPopulation(self)
                else:
                    self.ambient = ManeuverPopulation(self)
            elif population_profile == "city-life":
                from .city_population import CityPopulation
                self.ambient = CityPopulation(self)
            else:
                from .journey_ambient import JourneyAmbient
                self.ambient = JourneyAmbient(self)
            self.world.set_weather(getattr(carla.WeatherParameters, self.weather))
        except BaseException:
            self.close()
            raise

    def _required(self, bp, pose):
        actor = self._spawn(bp, pose)
        if actor is None:
            raise RuntimeError(f"Cannot spawn physical {bp.id} at {pose.location}")
        return actor

    def _event(self, kind, index, pose=None):
        e = Encounter(kind, index, pose or self._points[index].transform, self._station[index])
        if pose is not None:
            e.s = self._coordinates(pose.location)[0]
        self.events.append(e)
        return e

    def _spawn_walker(self, e, rng):
        wp = self._points[e.index]
        bp = rng.choice(sorted(self._catalog.filter("walker.pedestrian.*"), key=lambda p:p.id))
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible","false")
        if e.kind == "occluded":
            van_bp = _vehicle_blueprint(self._catalog, "vehicle.sprinter.mercedes", "vehicle.mercedes.sprinter")
            # Roadside trees, poles and construction differ by block. Move the
            # pair together along the same straight block until both bodies fit.
            for shift in (0, 9, -9, 18, -18, 27, -27, 36, -36):
                index = e.index + shift
                site = self._points[index]
                upstream = self._points[index - 5]
                if site.road_id != wp.road_id or upstream.road_id != wp.road_id:
                    continue
                parking = upstream.get_right_lane()
                offsets = (.65, .85)
                if self._urban and parking and parking.lane_type == carla.LaneType.Parking:
                    offsets = (parking.lane_width/2, parking.lane_width/2-.3, parking.lane_width/2+.3)
                for offset in offsets:
                    van_pose = _raised(upstream.transform, .65)
                    van_pose.location = (van_pose.location + van_pose.get_right_vector()
                                         * (upstream.lane_width / 2 + offset))
                    van = self._spawn(van_bp, van_pose)
                    if van is None:
                        continue
                    pose = self._curb_pose(site, .85)
                    walker = self._spawn(bp, pose)
                    if walker is None:
                        van.destroy()
                        self._actors.remove(van)
                        continue
                    self.occluder = van
                    van.apply_control(carla.VehicleControl(hand_brake=True))
                    e.index, e.s, e.pose, e.actor = index, self._station[index], site.transform, walker
                    break
                if self.occluder is not None:
                    break
            if self.occluder is None:
                raise RuntimeError("No Town01 curb site fits both the parked van and the crossing pedestrian")
        else:
            e.actor = self._required(bp, self._curb_pose(wp, .4))
        wp = self._points[e.index]
        right = e.pose.get_right_vector()
        e.direction = carla.Vector3D(-right.x, -right.y, 0)
        e.walk_setback = _offset(e.pose.location, self._curb_pose(wp, .85 if e.kind=='occluded' else .4).location, right)
        e.observed["curb_origin_lateral_m"] = round(e.walk_setback, 3)
        if self._urban:
            from .ambient import _sidewalk
            opposite = wp.get_left_lane()
            if opposite and abs(_angle(opposite.transform.rotation.yaw-wp.transform.rotation.yaw)) > 150:
                sidewalk = _sidewalk(opposite)
                if sidewalk:
                    e.walk_finish_side = _offset(wp.transform.location, sidewalk.transform.location, right)

    def _curb_pose(self, wp, setback):
        pose = wp.transform
        pos = pose.location + pose.get_right_vector() * (wp.lane_width / 2 + setback)
        if self._urban:
            from .ambient import _sidewalk
            sidewalk = _sidewalk(wp)
            if sidewalk is None:
                raise RuntimeError("Urban crossing has no sidewalk on its approach side")
            curb = sidewalk.transform
            pos = curb.location - curb.get_right_vector() * (sidewalk.lane_width / 2 - .25)
        return carla.Transform(carla.Location(pos.x, pos.y, pos.z + 1.),
                               carla.Rotation(yaw=pose.rotation.yaw - 90))

    def _spawn_lead(self, e):
        bp = self._catalog.find("vehicle.tesla.model3")
        bp.set_attribute("role_name", e.kind)
        e.actor = self._required(bp, _raised(e.pose,.35))
        e.actor.apply_control(carla.VehicleControl(brake=1.))
        self._exit_path(e)

    def _exit_path(self, e, start_index=None):
        # Turn off the planned route on departure, then park well beyond the intersection.
        index = e.index if start_index is None else start_index
        wp = self._points[index]
        forked = False
        for i in range(index+1,len(self._points)):
            choices = wp.next(1.)
            if len(choices) > 1:
                if forked:
                    break
                planned = self._points[i]
                alternatives = [w for w in choices if w.road_id != planned.road_id]
                if alternatives:
                    wp = min(alternatives,key=lambda w:(w.road_id,w.lane_id))
                    forked = True
                    if e.kind=="cyclist" or e.kind in LEADS:
                        planned_exit = next((j for j in range(i+1,len(self._points))
                                             if not self._points[j].is_junction), i+1)
                        e.pass_station = self._station[min(planned_exit+5,len(self._points)-1)]
                else:
                    wp = min(choices,key=lambda w:w.transform.location.distance(planned.transform.location))
            elif len(choices) == 1:
                wp = choices[0]
            else:
                break
            e.exit_path.append(wp.transform.location)
            if forked and not wp.is_junction and len(e.exit_path)>1:
                e.exit_road = wp.road_id
                e.exit_lanes.add((wp.road_id,wp.lane_id))
                # Continue in metre steps so the follower never aims across an
                # unobserved bend or brakes in the junction on a long final chord.
                for _ in range(22):
                    extra = wp.next(1.)
                    if not extra:
                        break
                    wp = min(extra,key=lambda w:abs(_angle(w.transform.rotation.yaw-wp.transform.rotation.yaw)))
                    e.exit_lanes.add((wp.road_id,wp.lane_id))
                    e.exit_path.append(wp.transform.location)
                break
        if not forked or e.exit_road is None:
            raise RuntimeError(f"No diverging native runout for {e.kind} at waypoint {e.index}")
        self._prepare_exit(e)

    @staticmethod
    def _prepare_exit(e):
        """Speed limits from actual lane-center curvature and distance to stopping point."""
        points = e.exit_path
        e.exit_arc = [0.]
        for a,b in zip(points,points[1:]):
            e.exit_arc.append(e.exit_arc[-1]+a.distance(b))
        limits = [10.]*len(points)
        for i in range(1,len(points)-1):
            a,b,c = points[i-1:i+2]
            delta = abs(_angle(math.degrees(math.atan2(c.y-b.y,c.x-b.x)-math.atan2(b.y-a.y,b.x-a.x))))
            curvature = math.radians(delta)/max(.15,(e.exit_arc[i+1]-e.exit_arc[i-1])/2.)
            if curvature > .01:
                limits[i] = min(limits[i],math.sqrt(1.35/curvature))
        limits[-1] = 0.
        for i in range(len(limits)-2,-1,-1):
            travel = e.exit_arc[i+1]-e.exit_arc[i]
            limits[i] = min(limits[i],math.sqrt(limits[i+1]**2+2.*1.7*travel))
        e.exit_limits = limits

    def _spawn_cyclist(self, e):
        candidates = []
        for bp in self._catalog.filter("vehicle.*"):
            name = bp.id.lower()
            bicycle_type = (bp.has_attribute("base_type") and
                            str(bp.get_attribute("base_type")).lower() == "bicycle")
            if bicycle_type or any(brand in name for brand in ("crossbike", "cross.bike", "roadbike",
                                   "road.bike", "leisurebike", "leisure.bike", "omafiets", "century")):
                candidates.append(bp)
        if not candidates:
            raise RuntimeError("Town01 journey requires a native drivable bicycle blueprint")
        bp = min(candidates, key=lambda b:b.id)
        bp.set_attribute("role_name", "journey_cyclist")
        e.actor = self._required(bp, _raised(e.pose, .35))
        e.actor.apply_control(carla.VehicleControl(brake=1.))
        e.observed["blueprint"] = bp.id
        self._exit_path(e)

    def _set_urban_lights(self, signals):
        # Keep the route and its competing through movement open. Cross-street
        # approaches remain red while the scripted native traffic uses the junction.
        planned = {(w.road_id, w.lane_id) for w in self._points}
        opposing_start = self._urban.oncoming[1]
        compatible = planned | {(opposing_start.road_id,opposing_start.lane_id)}
        for lamp in signals:
            lane_sets = [{(w.road_id,w.lane_id) for w in group if w} for group in (
                         lamp.get_stop_waypoints(), lamp.get_affected_lane_waypoints())]
            route_approach = bool(lane_sets[0] & compatible or lane_sets[1] & planned)
            state = carla.TrafficLightState.Green if route_approach else carla.TrafficLightState.Red
            lamp.set_state(carla.TrafficLightState.Red if lamp.id == self.signal.id else state)

    def _spawn_bus(self, e, change):
        bp = self._catalog.find("vehicle.fuso.mitsubishi")
        bp.set_attribute("role_name", "journey_bus")
        e.lane_change = change
        e.actor = self._required(bp, _raised(change.bus_pose, .45))
        e.actor.apply_control(carla.VehicleControl(hand_brake=True))
        e.pass_station = self._station[change.bus] + e.actor.bounding_box.extent.x + 4.
        e.observed.update(blueprint=bp.id, source_lane=list(change.source_lane),
                          target_lane=list(change.target_lane))

    def _spawn_queue(self, e):
        # The queued driver's front bumper is several metres behind the real stop line.
        origin = e.index - 7
        bp = self._catalog.find("vehicle.tesla.model3")
        bp.set_attribute("role_name","journey_signal_queue")
        e.actor = e.queue = self._required(bp, _raised(self._points[origin].transform, .35))
        e.queue.apply_control(carla.VehicleControl(brake=1.))
        self._exit_path(e, origin)

    def _bus_gap(self, e, progress):
        bus = e.actor
        bpose = bus.get_transform()
        ego_pose = self.ego.get_transform()
        change = e.lane_change
        bp, ep = bpose.location, ego_pose.location
        right, forward = bpose.get_right_vector(), bpose.get_forward_vector()
        station_offset = _offset(bp, ep, forward)
        if station_offset < -70.:
            return 60.
        stopped = _speed(bus) < .2
        lane = self._map.get_waypoint(ep, project_to_road=False, lane_type=carla.LaneType.Driving)
        lane_key = (lane.road_id,lane.lane_id) if lane else None
        if lane_key == tuple(change.source_lane) and progress < self._station[change.end]:
            e.mark("source_lane_seen_s", self._time)
            if e.release_pos is None:
                e.release_pos = ep
        if stopped and -48. <= station_offset < 0.:
            e.mark("triggered_s", self._time)
            e.mark("exposed_s", self._time)
            e.state = "approaching stopped bus"
        if lane_key == tuple(change.target_lane) and e.release_pos is not None:
            displacement = abs(_offset(e.release_pos,ep,right))
            legal_window = self._station[change.start]-4 <= progress <= self._station[change.end]+6
            if getattr(self, "tactical_lane_scoring", False):
                # A deferred merge can occur anywhere along the dashed approach
                # before the bus. Lane identity and displacement remain measured.
                legal_window = station_offset < -bus.bounding_box.extent.x
            if (legal_window
                    and displacement >= 2.1):
                e.mark("measured_lane_change_s", self._time)
                e.observed.setdefault("measured_lane_change_m", round(displacement, 3))
        bv = bus.bounding_box.get_world_vertices(bpose)
        ev = self.ego.bounding_box.get_world_vertices(ego_pose)
        front, rear = max(_offset(bp,v,forward) for v in ev), min(_offset(bp,v,forward) for v in ev)
        bus_front, bus_rear = max(_offset(bp,v,forward) for v in bv), min(_offset(bp,v,forward) for v in bv)
        if front >= bus_rear and rear <= bus_front:
            # The actual bus is curbside, the legal adjacent driving lane is on its left.
            clearance = min(_offset(bp,v,right) for v in bv) - max(_offset(bp,v,right) for v in ev)
            e.observed["minimum_bus_clearance_m"] = round(min(e.observed.get("minimum_bus_clearance_m",
                                                                            math.inf),clearance),3)
            if stopped and lane_key == tuple(change.target_lane):
                e.mark("bus_alongside_s",self._time)
                e.state = "passing stopped bus"
            if clearance < .25 or not stopped or lane_key != tuple(change.target_lane):
                e.violation = True
        if rear > bus_front + .5 and "bus_alongside_s" in e.observed:
            e.mark("cleared_s",self._time)
            e.state = "bus cleared"
        return 60.

    def _spawn_roadworks(self, e):
        available = sorted(self._catalog.filter("static.prop.*cone*"),
                           key=lambda b: (not b.id.lower().endswith("trafficcone01"), b.id))
        bp = next((b for b in available if "trafficcone" in b.id.lower() or
                  "constructioncone" in b.id.lower()), None)
        if bp is None:
            raise RuntimeError("Town01 journey requires a native traffic cone")
        for advance, taper in zip((0,4,8,12,16,20), (0.,.20,.42,.42,.20,0.)):
            wp = self._points[e.index+advance]
            pose = _raised(wp.transform, .04)
            # Leave enough clearance for the Lincoln even on narrower urban streets.
            offset = max(1.70, wp.lane_width/2 + .25 - taper)
            pose.location = pose.location + pose.get_right_vector() * offset
            actor = self._required(bp, pose)
            box = actor.bounding_box.extent
            if not (.05 <= min(box.x, box.y) and max(box.x, box.y) <= .85):
                raise RuntimeError(f"{bp.id} has no usable native cone footprint: {box}")
            e.cones.append(actor)
            e.cone_origins.append(pose.location)
        e.actor = e.cones[0]
        e.observed.update(cone_count=len(e.cones), blueprint=bp.id)
        e.pass_station = self._station[e.index+20] + 4.

    def _roadworks_gap(self, e, progress):
        pose = self.ego.get_transform()
        forward, right = e.pose.get_forward_vector(), e.pose.get_right_vector()
        if any(not cone.is_alive for cone in e.cones):
            e.violation = True
            return 60.
        if 0 < e.s-progress < 30.:
            e.mark("triggered_s", self._time)
            e.state = "approaching cone taper"
        if e.s - 26. <= progress <= e.pass_station + 2.:
            ego_vertices = self.ego.bounding_box.get_world_vertices(pose)
            projected = [(_offset(e.pose.location,v,forward), _offset(e.pose.location,v,right))
                         for v in ego_vertices]
            front, rear = max(x for x,_ in projected), min(x for x,_ in projected)
            curbside = max(y for _,y in projected)
            for index, cone in enumerate(e.cones):
                origin = e.cone_origins[index]
                current = cone.get_location()
                drift = math.hypot(current.x-origin.x,current.y-origin.y)
                if drift > .25:
                    e.violation = True
                e.observed["max_cone_displacement_m"] = round(max(drift,
                    e.observed.get("max_cone_displacement_m",0.)),3)
                vertices = cone.bounding_box.get_world_vertices(cone.get_transform())
                cs = [_offset(e.pose.location,v,forward) for v in vertices]
                if 0. < min(cs)-front < 24.:
                    e.mark("exposed_s", self._time)
                if min(cs) <= front and max(cs) >= rear:
                    e.encountered_cones.add(index)
                    clearance = min(_offset(e.pose.location,v,right) for v in vertices) - curbside
                    previous = e.observed.get("minimum_cone_clearance_m", math.inf)
                    e.observed["minimum_cone_clearance_m"] = round(min(previous, clearance),3)
                    lane = self._map.get_waypoint(pose.location, project_to_road=False,
                                                  lane_type=carla.LaneType.Driving)
                    original = self._points[e.index]
                    if (clearance < .10 or not lane
                            or (lane.road_id,lane.lane_id) != (original.road_id,original.lane_id)):
                        e.violation = True
                    e.state = "passing cones"
            e.observed["cones_alongside"] = len(e.encountered_cones)
        if progress >= e.pass_station and len(e.encountered_cones) == len(e.cones):
            e.mark("all_cones_passed_s", self._time)
            e.mark("cleared_s", self._time)
            e.state = "cone taper cleared"
        # The center of the planned driving lane remains open throughout.
        return 60.

    def _cyclist_gap(self, e, distance, progress, lateral, nose):
        moving = _speed(e.actor)
        pos = e.actor.get_location()
        ahead, side = self._coordinates(pos)
        if e.state == "waiting" and 0. < distance < 31.:
            e.state = "cycling"
        if e.state != "waiting" and moving > .7:
            e.mark("triggered_s",self._time)
            e.observed["peak_bicycle_speed_mps"] = round(max(e.observed.get("peak_bicycle_speed_mps",0.),
                                                             moving),3)
        gap = _clip(ahead-progress-nose-e.actor.bounding_box.extent.x,0.,60.)
        on_route = abs(side-lateral) < 1.8 and ahead >= progress-nose
        if e.state == "cycling" and moving > .7 and on_route and gap < 25.:
            e.mark("moving_ahead_s",self._time)
            e.mark("exposed_s",self._time)
        if e.state == "cycling":
            lane = self._map.get_waypoint(pos, project_to_road=False, lane_type=carla.LaneType.Driving)
            if lane and not lane.is_junction and lane.road_id == e.exit_road and abs(side)>4.:
                e.mark("left_route_s",self._time)
                e.mark("cleared_s",self._time)
                e.state = "cyclist departed"
        return gap if on_route and e.state != "cyclist departed" else 60.

    def _line_distance(self, e):
        pose, box = self.ego.get_transform(), self.ego.bounding_box
        nose = pose.transform(carla.Location(box.location.x+box.extent.x,box.location.y,box.location.z))
        return _offset(e.pose.location, nose, e.pose.get_forward_vector())

    def _walker_gap(self, e, distance, lateral, nose):
        gap = 60.
        right = e.pose.get_right_vector()
        side = _offset(e.pose.location,e.actor.get_location(),right)
        if e.kind == "occluded" and 0 < distance <= 45.:
            if _van_blocks_walker(self.ego,self.occluder,e.actor):
                e.mark("blocked_view_s",self._time)
            elif "blocked_view_s" in e.observed and e.state in ("crossing","dwell","leaving"):
                e.mark("emerged_s",self._time)
        trigger = 23. if e.kind=="pedestrian-delayed" else 20.
        if self._urban:
            # A walker also traverses the intervening parking strip and shoulder.
            trigger += max(0., e.walk_setback-2.5) * (5./1.25)
        if e.state == "waiting" and 0 < distance <= trigger:
            e.mark("approach_s",self._time)
            delay = (1.5 if e.kind=="pedestrian-delayed" else 0.) + self.crossing_delay
            e.next_action = e.observed["approach_s"] + max(0., delay)
            e.state = "curb"
        if e.state == "curb" and self._time >= e.next_action:
            e.state = "crossing"
            e.mark("triggered_s",self._time)
        if e.state in ("crossing","dwell","leaving") and abs(side) <= self.ego.bounding_box.extent.y+.6:
            if distance >= -nose:
                e.mark("exposed_s",self._time)
                gap = _clip(distance-nose-.35,0.,60.)
            if e.state == "crossing" and side <= .25:
                e.state = "dwell"
                e.next_action = self._time+2.
        if e.state == "dwell" and self._time >= e.next_action:
            e.state = "leaving"
        if e.state == "leaving" and side < -self._points[e.index].lane_width/2-.55:
            e.mark("cleared_s",self._time)
            if side < e.walk_finish_side:
                e.state = "across street"
        if (distance < nose and distance > -nose and
            abs(side-lateral) < self.ego.bounding_box.extent.y+.45):
            e.violation = True
        return gap

    def _lead_gap(self, e, distance, progress, lateral, nose):
        gap = 60.
        pos = e.actor.get_location()
        lead_s, lead_l = self._coordinates(pos)
        bumper_gap = lead_s-progress-nose-e.actor.bounding_box.extent.x
        if lead_s >= progress-nose and abs(lead_l-lateral)<1.9:
            gap = _clip(bumper_gap,0.,60.)
        if e.kind=="braking" and e.state=="waiting" and 0 < distance < 34.:
            e.state = "cruising"
        moving = _speed(e.actor)
        if (e.kind=="lead" and e.state=="waiting" and moving < .25 and gap < 20.
            or e.kind=="braking" and e.state=="cruising" and moving >= 1.6 and gap < 22.):
            e.mark("triggered_s",self._time)
            if e.kind == "lead" and getattr(self, "population_profile", None) == "city-maneuvers":
                e.observed["target_speed_at_exposure_mps"] = round(moving, 3)
            e.state = "braking" if e.kind=="braking" else "stopped"
            if e.state=="stopped":
                e.next_action = self._time+4.
        if e.kind=="braking" and e.state=="braking":
            if e.previous_speed is not None:
                e.peak_decel = max(e.peak_decel, (e.previous_speed-moving)/self._dt)
            e.observed.setdefault("brake_entry_speed_mps",round(moving,3))
        e.previous_speed = moving
        if e.state=="braking" and moving < .2:
            e.mark("lead_stopped_s",self._time)
            e.state = "stopped"
            e.next_action = self._time+3.
        if e.state in ("braking","stopped") and gap < 22.:
            e.mark("exposed_s",self._time)
        if e.state=="stopped" and self._time >= e.next_action:
            e.state = "departing"
            e.release_pos = pos
            e.mark("release_command_s",self._time)
        if e.state=="departing":
            if moving >= 1. and pos.distance(e.release_pos) >= 4.:
                e.mark("restarted_s",self._time)
            if "restarted_s" in e.observed and "left_route_s" not in e.observed:
                actual_lane = self._map.get_waypoint(pos, project_to_road=False,lane_type=carla.LaneType.Driving)
                if actual_lane and (actual_lane.road_id,actual_lane.lane_id) in e.exit_lanes:
                    footprint = e.actor.bounding_box.get_world_vertices(e.actor.get_transform())
                    edge = min(abs(self._coordinates(vertex)[1]) for vertex in footprint)
                    if edge > self.ego.bounding_box.extent.y + .75:
                        e.observed["clearance_at_exit_m"] = round(edge-self.ego.bounding_box.extent.y,3)
                        e.mark("left_route_s",self._time)
                        e.mark("cleared_s",self._time)
        return gap

    def _signal_gap(self, e):
        gap = 60.
        line = self._line_distance(e)
        red = self.signal.get_state()==carla.TrafficLightState.Red
        if red and -28. <= line < 0:
            e.mark("triggered_s",self._time)
            e.mark("exposed_s",self._time)
            e.state = "red"
            gap = -line
        if e.last_line is not None and e.last_line<0<=line:
            if red:
                e.violation = True
            else:
                e.mark("crossed_green_s",self._time)
        e.last_line = line
        if self.signal.get_state()==carla.TrafficLightState.Green and "triggered_s" in e.observed:
            e.mark("cleared_s",self._time)
            e.state = "green"
        if e.queue is not None:
            pos = e.queue.get_location()
            qs, qlat = self._coordinates(pos)
            ego_s, ego_lat = self._coordinates(self.ego.get_location())
            queue_gap = qs - ego_s - e.queue.bounding_box.extent.x - self.ego.bounding_box.extent.x
            if abs(qlat-ego_lat) < 1.9 and qs >= ego_s:
                gap = min(gap, _clip(queue_gap,0.,60.))
                if red and _speed(e.queue) < .25 and 0 <= queue_gap <= 25.:
                    e.mark("queue_exposed_s",self._time)
            if (not red and "queue_exposed_s" in e.observed and _speed(e.queue) > .7 and
                _offset(e.pose.location,pos,e.pose.get_forward_vector()) > -3.):
                e.mark("queue_departed_green_s",self._time)
        return gap

    def _left_gap(self, e, distance, nose):
        gap = 60.
        line = self._line_distance(e)
        if e.state=="waiting" and -21 <= line < 0:
            e.mark("triggered_s",self._time)
            e.state = "yield to oncoming"
        if e.state=="yield to oncoming":
            pos = e.actor.get_location()
            projected = _offset(e.conflict,pos,e.direction)
            moving = _speed(e.actor)
            if moving > .7 and projected < 0 and distance > -nose:
                e.mark("exposed_s",self._time)
            if projected > e.actor.bounding_box.extent.x + 1.:
                e.mark("cleared_s",self._time)
                e.state = "oncoming cleared"
            else:
                gap = _clip(-line,0.,60.)
                separation = self.ego.get_location().distance(e.conflict)
                # Count a physical conflict if the ego occupies the crossing
                # while the oncoming car is within 2.0s of entering it.
                if (separation < nose+1.5 and moving > .7 and
                    -moving*2.-e.actor.bounding_box.extent.x < projected):
                    e.violation = True
        return gap

    def _vehicle_gap(self, actor, ego_progress, ego_lateral, nose):
        """Occupied corridor from the live vehicle's oriented body, including its turn."""
        pose, box = actor.get_transform(), actor.bounding_box
        center = pose.transform(carla.Location(box.location.x,box.location.y,box.location.z))
        if center.distance(self.ego.get_location()) > 65.:
            return 60.
        station, lateral = self._coordinates(center)
        if station < ego_progress-nose:
            return 60.
        segment = self._segments[max(0,min(len(self._segments)-1,bisect_left(self._station,station)-1))]
        ux, uy = segment[2:4]
        forward,right = pose.get_forward_vector(),pose.get_right_vector()
        along = abs(forward.x*ux+forward.y*uy)*box.extent.x + abs(right.x*ux+right.y*uy)*box.extent.y
        width = abs(forward.y*ux-forward.x*uy)*box.extent.x + abs(right.y*ux-right.x*uy)*box.extent.y
        if abs(lateral-ego_lateral) > self.ego.bounding_box.extent.y+width+.1:
            return 60.
        return _clip(station-ego_progress-nose-along,0.,60.)

    def _observe(self):
        if self._last_observation == self._time:
            return
        self._last_observation = self._time
        progress, lateral = self._coordinates(self.ego.get_location())
        speed = _speed(self.ego)
        nose = self.ego.bounding_box.location.x + self.ego.bounding_box.extent.x
        self._gap = 60.
        self._active = None
        self.hazard = None
        in_turn = False
        for event in self._turn_observations:
            if event["entry_m"] <= progress <= event["exit_m"]:
                event.setdefault("entered_s",round(self._time,3))
                in_turn = True
            elif "entered_s" in event and progress > event["exit_m"]:
                event.setdefault("completed_s",round(self._time,3))
        for e in self.events:
            distance = e.s-progress
            trailing_lead = e.kind in LEADS and e.state=="departing" and "passed_s" not in e.observed
            if e.kind not in ("in-lane-obstruction", "slow-traffic") and not trailing_lead and not (-140 if e.kind in ("bus","cyclist") else -65) < distance < 65:
                continue
            if "passed_s" in e.observed:
                continue
            if e.kind in ("in-lane-obstruction", "slow-traffic"):
                gap = self.maneuver_fixture.observe(e, progress)
            elif e.kind in WALKERS:
                gap = self._walker_gap(e, distance, lateral, nose)
            elif e.kind in LEADS:
                gap = self._lead_gap(e, distance, progress, lateral, nose)
            elif e.kind == "cyclist":
                gap = self._cyclist_gap(e, distance, progress, lateral, nose)
            elif e.kind == "barrier":
                from .barrier_scene import barrier_gap
                gap = barrier_gap(self, e, progress)
            elif e.kind == "roadworks":
                gap = self._roadworks_gap(e, progress)
            elif e.kind == "bus":
                gap = self._bus_gap(e, progress)
            elif e.kind == "signal":
                gap = self._signal_gap(e)
            else:
                gap = self._left_gap(e, distance, nose)
            if gap < 60:
                e.min_gap = min(e.min_gap,gap)
                if speed < .35 and gap < 15. and "exposed_s" in e.observed:
                    e.stopped = True
            self._gap = min(self._gap, gap)
            inside_extended = e.pass_station is not None and e.s <= progress <= e.pass_station
            if -6. <= distance < 45. or gap < 30 or inside_extended:
                self._active = e
                self.hazard = e.actor
            if e.pass_station is not None:
                if progress >= e.pass_station and "cleared_s" in e.observed:
                    e.mark("passed_s", self._time)
            elif distance < -(14. if e.kind=="left" else 6.):
                if e.kind not in LEADS or "cleared_s" in e.observed or distance < -45:
                    e.mark("passed_s",self._time)
        # Vehicles can remain physically ahead long after passing the stationary
        # event's anchor. A frame's reference never discards an occupied corridor
        # on the strength of an event timestamp or scheduled departure alone.
        for e in self.events:
            if e.kind not in LEADS and e.queue is None:
                continue
            gap = self._vehicle_gap(e.actor,progress,lateral,nose)
            if gap < 60. and "left_route_s" in e.observed:
                e.mark("runout_reentered_s",self._time)
            if gap < self._gap:
                self._gap = gap
                if gap < 30.:
                    self._active, self.hazard = e, e.actor
        clear = self._active is None and not in_turn
        if clear and self._was_clear:
            self._clear_distance += max(0.,progress-self._last_progress)
        self._last_progress, self._was_clear = progress, clear
        self.red_light_violations = sum(e.violation for e in self.events if e.kind=="signal")
        self.yield_violations = sum(e.violation for e in self.events if e.kind=="left")
        self.phase = f"{self._active.kind}: {self._active.state}" if self._active else "clear road"
        self._cleared = all(e.record()["handled"] for e in self.events)
        self._event_start = min((e.observed["triggered_s"] for e in self.events if "triggered_s" in e.observed),
                                default=None)

    def _follow_exit(self, e, speed):
        # Metre sampling in OpenDRIVE uses the road reference line. At the
        # tight inside of a turn five samples may span only 2-3 actual metres.
        # Use arc-length lookahead; reserve zero speed for the actual endpoint.
        pose = e.actor.get_transform()
        pos = pose.location
        nearest = min(range(len(e.exit_path)),key=lambda i:pos.distance(e.exit_path[i]))
        if e.kind not in LEADS:
            self._drive_to(e.actor,e.exit_path[min(nearest+5,len(e.exit_path)-1)],speed)
            return
        current = _speed(e.actor)
        requested = min(speed,e.exit_limits[nearest])
        arc = e.exit_arc[nearest]
        lookahead = _clip(2.5+.30*current,2.5,5.)
        goal_index = min(bisect_left(e.exit_arc,arc+lookahead),len(e.exit_path)-1)
        goal = e.exit_path[goal_index]
        dx,dy = goal.x-pos.x,goal.y-pos.y
        forward, right = pose.get_forward_vector(),pose.get_right_vector()
        ahead, side = dx*forward.x+dy*forward.y, dx*right.x+dy*right.y
        if nearest >= len(e.exit_path)-2 and (ahead <= .7 or pos.distance(e.exit_path[-1]) < 1.):
            requested = 0.
        # Pure pursuit with a sedan wheelbase and conservative urban bend speeds.
        steer = math.atan2(2.*2.75*side,max(.25,dx*dx+dy*dy))/math.radians(53.)
        error = requested-current
        e.actor.apply_control(carla.VehicleControl(steer=_clip(steer,-1.,1.),
                throttle=_clip(.4*error,0.,.75),
                brake=1. if requested <= .05 else _clip(-.45*error,0.,1.)))

    def before_tick(self):
        self._observe()
        self.ambient.before_tick()
        if self.maneuver_fixture is not None:
            self.maneuver_fixture.before_tick()
        for e in self.events:
            if e.kind in WALKERS:
                if e.state == "leaving":
                    side = _offset(e.pose.location, e.actor.get_location(), e.pose.get_right_vector())
                    if side < e.walk_finish_side:
                        e.state = "across street"
                e.actor.apply_control(carla.WalkerControl(direction=e.direction,
                                      speed=1.25 if e.state in ("crossing","leaving") else 0.))
            elif e.kind in LEADS or e.kind == "cyclist":
                speed = {"cruising": 2.8, "departing": 8.5, "cycling": 3.1,
                         "cyclist departed": 3.1}.get(e.state, 0.)
                self._follow_exit(e,speed)
            elif e.kind=="left":
                self._drive_to(e.actor,e.target,5.8 if e.state!="waiting" else 0.)
            elif e.kind=="signal":
                if e.queue is not None:
                    self._follow_exit(e, 7.5 if e.state == "green" else 0.)
                distance = e.s-self._coordinates(self.ego.get_location())[0]
                if e.state=="waiting" and -2 < distance < 75:
                    self.signal.set_state(carla.TrafficLightState.Red)
                if e.state=="red" and self._time-e.observed["triggered_s"] >= 7.:
                    self.signal.set_state(carla.TrafficLightState.Green)
        self._time += self._dt

    def expert_gap(self):
        self._observe()
        return self._gap

    def observer_positions(self, radius=100.):
        """Nearby ground-truth actors for the spectator map only."""
        actors = [(self.occluder, "van", "occluder")]
        if self.maneuver_fixture is not None:
            actors.extend(self.maneuver_fixture.spectator_actors())
        for event in self.events:
            if event.kind in ("in-lane-obstruction", "slow-traffic"):
                continue
            if event.kind == "roadworks":
                actors.extend((cone,"cone","roadworks") for cone in event.cones)
            elif event.kind == "barrier":
                actors.extend((board, "barrier", "workzone_barrier") for board in event.barriers)
            elif event.kind in WALKERS:
                actors.append((event.actor,"pedestrian",event.kind))
            elif event.kind == "cyclist":
                actors.append((event.actor,"cyclist",event.kind))
            elif event.kind == "bus":
                actors.append((event.actor,"bus","stopped_bus"))
            else:
                role = {"signal":"signal_queue","left":"oncoming"}.get(event.kind,event.kind)
                actors.append((event.actor,"car",role))
        actors.extend((actor,"pedestrian" if actor.type_id.startswith("walker.") else "car","background")
                      for actor in self.ambient._actors)
        center = self.ego.get_location()
        positions = []
        for actor, kind, role in actors:
            if actor is None or not actor.is_alive:
                continue
            pose = actor.get_transform()
            p = pose.location
            if math.hypot(p.x-center.x,p.y-center.y) <= radius:
                bounds = actor.bounding_box
                box = bounds.extent
                positions.append(dict(actor_id=actor.id, type=kind, role=role, blueprint=actor.type_id,
                                      dimensions_m=[round(2*box.x, 2), round(2*box.z, 2), round(2*box.y, 2)],
                                      x=round(p.x,2),y=round(p.y,2),z=round(p.z,2),
                                      yaw=round(pose.rotation.yaw,2)))
                if getattr(self, "population_profile", None) in ("city-life", "city-maneuvers"):
                    box_center = bounds.location
                    local = bounds.rotation
                    positions[-1].update(bbox_offset_m=[round(v, 3) for v in (box_center.x, box_center.y, box_center.z)],
                                         bbox_rotation_deg=[round(v, 2) for v in (local.pitch, local.yaw, local.roll)],
                                         pitch=round(pose.rotation.pitch, 2), roll=round(pose.rotation.roll, 2))
        return positions

    def metrics(self):
        self._observe()
        state = super().metrics()
        if getattr(self, "tactical_lane_scoring", False):
            lane = self._map.get_waypoint(self.ego.get_location(), project_to_road=False,
                                          lane_type=carla.LaneType.Driving)
            change = self._urban.lane_change
            alternatives = (change.source_lane, change.target_lane)
            if lane and (lane.road_id, lane.lane_id) in alternatives:
                pose = self.ego.get_transform()
                displacement = abs(_offset(lane.transform.location, pose.location, lane.transform.get_right_vector()))
                state["lateral_error_m"] = displacement
                state["route_departure"] = displacement > state["lane_tolerance_m"]
                # In the interior of the boulevard the whole car must stay in
                # the two legal driving lanes, including during a delayed merge.
                if 4. < lane.s < 88.:
                    corners = self.ego.bounding_box.get_world_vertices(pose)
                    lanes = [self._map.get_waypoint(v, project_to_road=False,
                                                    lane_type=carla.LaneType.Driving) for v in corners]
                    state["route_departure"] |= any(not w or (w.road_id,w.lane_id) not in alternatives for w in lanes)
        state.update(events=[e.record() for e in self.events],
                     turn_counts={k:sum(d==k for _,_,d in self.turns) for k in ("left","right","straight")},
                     junction_traversals=[dict(turn) for turn in self._turn_observations],
                     turns_completed={k:sum(t["direction"]==k and "completed_s" in t
                                            for t in self._turn_observations)
                                      for k in ("left","right","straight")},
                     clear_road_m=round(self._clear_distance,2), sidewalk_activity=self.ambient.activity,
                     background_traffic_activity=self.ambient.traffic_activity,
                     active_event=self._active.kind if self._active else None)
        return state
