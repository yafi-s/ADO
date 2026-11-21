"""Seeded Town01 routes and non-ego encounters. Actor state is for scoring."""

import math
import random

import carla

from .district import DESTINATIONS, DISTRICTS, admits, connectors, trip_points

KINDS = ("clear", "pedestrian", "lead", "braking", "turn", "occluded", "signal", "left", "trip")
_WALKING = ("pedestrian", "occluded", "trip")
ROUTE_METERS = 64
_EXTENSION = 18  # let the departing lead keep driving past the ego's finish


def _angle(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def _clip(value, low, high):
    return max(low, min(value, high))


def _raised(transform, height):
    p = transform.location
    return carla.Transform(carla.Location(p.x, p.y, p.z + height), transform.rotation)


def _vehicle_blueprint(library, preferred, legacy):
    """Resolve the released UE5 vehicle names and the UE4 reproductions."""
    for name in (preferred, legacy):
        matches = library.filter(name)
        if matches:
            return library.find(name)
    raise RuntimeError(f"CARLA catalog contains neither {preferred} nor {legacy}")


def _straight_lane(start):
    """Return a metre-spaced drivable route and a runout, or reject the start."""
    yaw = start.transform.rotation.yaw
    points = [start]
    for _ in range(ROUTE_METERS + _EXTENSION):
        choices = points[-1].next(1.0)
        if len(choices) != 1:
            return None
        point = choices[0]
        if point.is_junction or abs(_angle(point.transform.rotation.yaw - yaw)) > 2.0:
            return None
        points.append(point)
    return points


def _opposite_lane(road_map, waypoint):
    transform = waypoint.transform
    right = transform.get_right_vector()
    p = transform.location
    # Town01's two-way streets have an opposing lane on the driver's left.
    offset = waypoint.lane_width
    probe = carla.Location(p.x - right.x * offset, p.y - right.y * offset, p.z)
    neighbor = road_map.get_waypoint(probe, project_to_road=False, lane_type=carla.LaneType.Driving)
    if neighbor and abs(_angle(neighbor.transform.rotation.yaw - transform.rotation.yaw)) > 170:
        return neighbor
    return None


def _routes(road_map, seed):
    candidates = sorted(road_map.generate_waypoints(9.0), key=lambda w: (w.road_id, w.lane_id, w.s))
    random.Random(seed).shuffle(candidates)
    for start in candidates:
        if start.is_junction or not 2.7 <= start.lane_width <= 5.0:
            continue
        points = _straight_lane(start)
        if points is None:
            continue
        near, far = (_opposite_lane(road_map, points[i]) for i in (2, ROUTE_METERS - 4))
        if near and far:
            yield points, near, far


def _turn_routes(road_map, seed, direction):
    """Follow one OpenDRIVE intersection connector, then its outbound lane."""
    candidates = []
    for entry, exit_wp in road_map.get_topology():
        if entry.is_junction:
            change = _angle(exit_wp.transform.rotation.yaw - entry.transform.rotation.yaw)
            # CARLA's positive yaw turns toward the right.
            if ((direction == "straight" and abs(change) <= 6)
                or (direction != "straight" and 60 <= (change if direction == "right" else -change) <= 120)):
                candidates.append(entry)
    candidates.sort(key=lambda w: (w.road_id, w.lane_id, w.s))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    approach = rng.randint(21, 27)
    for entry in candidates:
        previous = entry.previous(float(approach))
        if len(previous) != 1 or previous[0].is_junction:
            continue
        points = [previous[0]]
        for _ in range(ROUTE_METERS):
            choices = points[-1].next(1.0)
            if len(choices) > 1:
                choices = [w for w in choices if (w.road_id, w.lane_id) == (entry.road_id, entry.lane_id)]
            if len(choices) != 1:
                break
            nxt = choices[0]
            if nxt.is_junction and nxt.junction_id != entry.junction_id:
                break
            points.append(nxt)
        if len(points) != ROUTE_METERS + 1 or points[-1].is_junction:
            continue
        if not any(p.road_id == entry.road_id and p.lane_id == entry.lane_id for p in points):
            continue
        # The opposing car starts on the inbound straight, facing away from the junction.
        near = _opposite_lane(road_map, points[2])
        far = _opposite_lane(road_map, points[approach - 6])
        if near and far:
            yield points, near, far


def _oncoming_pair(road_map, points, links):
    """Find the through movement crossing this left turn, from the opposite arm."""
    inbound = points[0].road_id
    junction = next(w.junction_id for w in points if w.is_junction)
    for (other, outbound), entry in links.items():
        if outbound != inbound or entry.junction_id != junction:
            continue
        start = entry.previous(24.0)
        if len(start) != 1 or start[0].road_id != other:
            continue
        yaw = start[0].transform.rotation.yaw
        if abs(_angle(yaw - points[0].transform.rotation.yaw)) < 174:
            continue
        # Driving straight down this axis carries the actor through the conflict
        # and into the opposing lane alongside the ego's original approach.
        end = _opposite_lane(road_map, points[2])
        if end:
            return end, start[0]
    return None


def _scene_routes(road_map, seed, kind, district, destination, direction):
    if kind == "trip":
        points = trip_points(road_map, district, destination)
        near, far = (_opposite_lane(road_map, points[i]) for i in (52, 105))
        if near and far and admits(points, district):
            yield points, near, far
        return
    routes = (_turn_routes(road_map, seed, direction) if kind in ("turn", "left", "signal")
              else _routes(road_map, seed))
    links = connectors(road_map) if kind == "left" else {}
    for points, near, far in routes:
        if kind == "left":
            pair = _oncoming_pair(road_map, points, links)
            if pair is None:
                continue
            near, far = pair
        if admits([*points, near, far], district):
            yield points, near, far


class Scenario:
    """Spawn a scene in a synchronous world. The caller owns all world ticks."""

    def __init__(self, world, kind, seed, perturb=False, district=None, crossing_delay=0.0, destination=None):
        kind = {"crossing": "pedestrian", "stopped_lead": "lead"}.get(kind, kind)
        if kind not in KINDS:
            raise ValueError(f"Unknown encounter {kind!r}, choose from {KINDS}")
        self.world = world
        self.name = self.kind = kind
        self.seed = int(seed)
        if district is not None and district not in DISTRICTS:
            raise ValueError(f"Unknown district {district!r}")
        self.district = district or ("heldout" if kind == "trip" else None)
        if kind == "trip":
            destinations = DESTINATIONS[self.district]
            self.destination = destination or tuple(destinations)[self.seed % len(destinations)]
            if self.destination not in destinations:
                raise ValueError(f"Unknown destination {self.destination!r}, choose from {tuple(destinations)}")
        else:
            self.destination = None
        self.crossing_delay = float(crossing_delay)
        if not math.isfinite(self.crossing_delay):
            raise ValueError("crossing_delay must be finite")
        self._actors = []
        self.occluder = self.signal = None
        self.ego = self.hazard = self.traffic = None
        self._previous_weather = world.get_weather()
        self._closed = False
        self._signal_states = []
        self._signals_were_frozen = None
        self.turn_direction = ("right" if self.seed % 2 else "left") if kind == "turn" else ("left" if kind == "left" else None)
        self.spawn_lateral_m = self.spawn_yaw_deg = 0.0
        if perturb:
            pose_rng = random.Random(self.seed ^ 0x7A23)
            self.spawn_lateral_m = pose_rng.choice((-1, 1)) * pose_rng.uniform(0.15, 0.45)
            self.spawn_yaw_deg = pose_rng.choice((-1, 1)) * pose_rng.uniform(2.0, 5.0)
        dt = world.get_settings().fixed_delta_seconds
        if dt is None or dt <= 0:
            raise ValueError("Set synchronous mode and a fixed timestep before spawning a scenario")
        self._dt = dt
        self._time = 0.0
        self._release_at = None
        self._event_start = None
        self._walk_at = None
        self._cleared = False
        self._line = None
        self._last_line_distance = None
        self.red_light_violations = 0
        self.yield_violations = 0
        self.phase = {
            "clear": "clear road", "pedestrian": "waiting at curb",
            "lead": "lead stopped", "braking": "lead cruising", "turn": f"{self.turn_direction} turn",
            "occluded": "waiting at curb", "trip": "waiting at curb",
            "signal": "red signal", "left": "oncoming waiting",
        }[kind]
        rng = random.Random(self.seed ^ 0x2AF1)
        self._hazard_index = rng.randint(29, 39)
        self._walk_speed = rng.uniform(0.95, 1.35)
        self._trigger_gap = rng.uniform(15.0, 19.0)
        self._lead_speed = rng.uniform(5.0, 6.5)
        self.weather = rng.choice(("ClearNoon", "CloudyNoon", "WetNoon", "ClearSunset"))
        if kind == "braking":
            self._hazard_index = rng.randint(18, 24)
        elif kind == "trip":
            self._hazard_index = 76
        try:
            road_map = self._map = world.get_map()
            direction = "straight" if kind == "signal" else self.turn_direction
            routes = _scene_routes(road_map, self.seed, kind, self.district, self.destination, direction)
            if kind == "signal":
                signals = world.get_actors().filter("traffic.traffic_light")
                stop_lines = []
                for signal in signals:
                    movements = {(p.road_id, p.lane_id) for p in signal.get_affected_lane_waypoints() if p}
                    for w in signal.get_stop_waypoints():
                        stop_lines.append((signal, w, movements))
            for points, near, far in routes:
                if kind == "signal" and not self._match_signal(points, stop_lines):
                    continue
                if self._populate(points, near, far):
                    break
                self._destroy_actors()
            else:
                raise RuntimeError(f"No unobstructed {kind} route available in {road_map.name}")
            if kind in ("turn", "left", "trip", "signal"):
                if kind != "signal":
                    signals = world.get_actors().filter("traffic.traffic_light")
                if signals:
                    self._signal_states = [(signal, signal.get_state()) for signal in signals]
                    self._signals_were_frozen = signals[0].is_frozen()
                    world.freeze_all_traffic_lights(True)
                    for signal in signals:
                        signal.set_state(carla.TrafficLightState.Green)
                    if self.signal is not None:
                        self.signal.set_state(carla.TrafficLightState.Red)
            self.world.set_weather(getattr(carla.WeatherParameters, self.weather))
        except BaseException:
            self.close()
            raise

    def _spawn(self, blueprint, transform):
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            self._actors.append(actor)
        return actor

    def _match_signal(self, points, stop_lines):
        junction_entry = next(i for i, w in enumerate(points) if w.is_junction)
        route_lanes = {(w.road_id, w.lane_id) for w in points}
        # CARLA's stop waypoints are derived from each real light's trigger
        # volume. Some lie just inside the connector rather than on its approach.
        candidates = []
        for signal, stop, controlled_lanes in stop_lines:
            if not controlled_lanes.intersection(route_lanes):
                continue
            pos = stop.transform.location
            for i, w in enumerate(points[7:junction_entry + 6], 7):
                if w.road_id != stop.road_id or w.lane_id != stop.lane_id:
                    continue
                distance = pos.distance(w.transform.location)
                if distance <= 1.2:
                    candidates.append((distance, signal.id, i, signal, stop))
        if not candidates:
            return False
        _, _, _, self.signal, stop = min(candidates, key=lambda c: c[:3])
        self._line = stop.transform
        return True

    def _populate(self, points, near, far):
        library = self.world.get_blueprint_library()
        ego_bp = _vehicle_blueprint(library, "vehicle.lincoln.mkz", "vehicle.lincoln.mkz_2020")
        ego_bp.set_attribute("role_name", "hero")
        if ego_bp.has_attribute("color"):
            ego_bp.set_attribute("color", "225,225,225")
        start = _raised(points[0].transform, 0.35)
        right = start.get_right_vector()
        spawn = carla.Transform(start.location + right * self.spawn_lateral_m, carla.Rotation(
            pitch=start.rotation.pitch, yaw=start.rotation.yaw + self.spawn_yaw_deg, roll=start.rotation.roll))
        self.ego = self._spawn(ego_bp, spawn)
        if self.ego is None:
            return False
        traffic_bp = _vehicle_blueprint(library, "vehicle.mini.cooper", "vehicle.audi.a2")
        traffic_bp.set_attribute("role_name", "oncoming")
        self.traffic = self._spawn(traffic_bp, _raised(far.transform, 0.35))
        if self.traffic is None:
            return False
        self._traffic_target = near.transform.location
        route_points = points if self.kind == "trip" else points[:ROUTE_METERS + 1]
        self._set_route([w.transform.location for w in route_points])
        self.road_id, self.lane_id = points[0].road_id, points[0].lane_id
        self.road_ids = tuple(dict.fromkeys(p.road_id for p in route_points if not p.is_junction))
        self._route_lanes = {(p.road_id, p.lane_id) for p in route_points if not p.is_junction}
        self.junction_ids = tuple(dict.fromkeys(p.junction_id for p in route_points if p.is_junction))
        self.route_id = f"{self.district or 'town'}:{points[0].id}:{route_points[-1].id}"
        self._junction_ranges = []
        entry = None
        for i, p in enumerate(route_points):
            if p.is_junction and entry is None:
                entry = i
            elif not p.is_junction and entry is not None:
                self._junction_ranges.append((self._segments[entry][5], self._segments[i-1][5] + self._segments[i-1][4]))
                if self.kind == "left":
                    self._line = route_points[entry-2].transform
                entry = None
        if self._line is not None:
            self._stop_s = self._coordinates(self._line.location)[0]
            self._last_line_distance = self._bumper_line_distance()
            if self.kind == "left":
                # The two lane centerlines actually intersect inside the turn.
                direction = far.transform.get_forward_vector()
                start = far.transform.location
                def distance_to_oncoming_axis(w):
                    delta = w.transform.location - start
                    return abs(delta.x * direction.y - delta.y * direction.x)
                conflict = min((w for w in route_points if w.is_junction), key=distance_to_oncoming_axis)
                self._conflict = conflict.transform.location
                self._oncoming_direction = direction
        self._lead_target = points[-1].transform.location
        hazard_transform = points[self._hazard_index].transform
        if self.kind in _WALKING:
            people = sorted(library.filter("walker.pedestrian.*"), key=lambda bp: bp.id)
            walker_bp = random.Random(self.seed).choice(people)
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            offset = points[self._hazard_index].lane_width / 2 + (0.8 if self.kind == "occluded" else 0.35)
            right = hazard_transform.get_right_vector()
            self._walk_direction = carla.Vector3D(-right.x, -right.y, 0)
            p = hazard_transform.location + right * offset
            at_curb = carla.Transform(
                carla.Location(p.x, p.y, p.z + 1.0),
                carla.Rotation(yaw=hazard_transform.rotation.yaw - 90),
            )
            self.hazard = self._spawn(walker_bp, at_curb)
            if self.kind == "occluded":
                van_wp = points[self._hazard_index - 5]
                van = _raised(van_wp.transform, 0.65)
                van.location = van.location + van.get_right_vector() * (van_wp.lane_width / 2 + 0.65)
                van_bp = _vehicle_blueprint(library, "vehicle.sprinter.mercedes", "vehicle.mercedes.sprinter")
                self.occluder = self._spawn(van_bp, van)
                if self.occluder is None:
                    return False
                self.occluder.apply_control(carla.VehicleControl(hand_brake=True))
        elif self.kind in ("lead", "braking"):
            lead_bp = library.find("vehicle.tesla.model3")
            lead_bp.set_attribute("role_name", self.kind)
            self.hazard = self._spawn(lead_bp, _raised(hazard_transform, 0.35))
        return self.kind in ("clear", "turn", "left", "signal") or self.hazard is not None

    def _bumper_line_distance(self):
        pose, box = self.ego.get_transform(), self.ego.bounding_box
        front = pose.transform(carla.Location(box.location.x + box.extent.x, box.location.y, box.location.z))
        d = front - self._line.location
        f = self._line.get_forward_vector()
        return d.x * f.x + d.y * f.y

    def _score_line(self):
        if self._line is None:
            return
        distance = self._bumper_line_distance()
        if self._last_line_distance < 0 <= distance:
            if self.kind == "signal" and self.phase == "red signal":
                self.red_light_violations += 1
            elif self.kind == "left" and not self._cleared:
                self.yield_violations += 1
        self._last_line_distance = distance

    def _set_route(self, locations):
        self.route = locations
        self.route_start = locations[0]
        self._segments = []
        self.length = 0.0
        for start, end in zip(locations, locations[1:]):
            dx, dy = end.x - start.x, end.y - start.y
            length = math.hypot(dx, dy)
            if length > 1e-6:
                self._segments.append((start.x, start.y, dx / length, dy / length, length, self.length))
                self.length += length
        if not self._segments:
            raise ValueError("Scenario route has zero length")
        self._fx, self._fy = self._segments[0][2:4]

    def _coordinates(self, location):
        """Nearest point on the route polyline, in metres along and right of it."""
        nearest = math.inf
        px, py = location.x, location.y
        for sx, sy, ux, uy, length, start_s in self._segments:
            x, y = px - sx, py - sy
            tangent = x * ux + y * uy
            along = _clip(tangent, 0.0, length)
            distance_sq = (x - along * ux) ** 2 + (y - along * uy) ** 2
            if distance_sq < nearest:
                nearest = distance_sq
                progress = start_s + along
                side = y * ux - x * uy
                extension = tangent - along
        # Actors can sit behind the start or drive into the runout. Extend the
        # end tangents for their clearance; metrics() caps route completion.
        if progress == 0.0 or progress == self.length:
            return progress + extension, side
        return progress, math.copysign(math.sqrt(nearest), side)

    def _drive_to(self, actor, destination, speed):
        pose = actor.get_transform()
        x, y = destination.x - pose.location.x, destination.y - pose.location.y
        forward = pose.get_forward_vector()
        if x * forward.x + y * forward.y < 2.0:
            speed = 0.0
        velocity = actor.get_velocity()
        error = speed - math.hypot(velocity.x, velocity.y)
        heading_error = _angle(math.degrees(math.atan2(y, x)) - pose.rotation.yaw)
        actor.apply_control(carla.VehicleControl(
            steer=_clip(heading_error / 38.0, -0.6, 0.6),
            throttle=_clip(error * 0.40, 0.0, 0.75),
            brake=1.0 if speed == 0.0 else _clip(-error * 0.35, 0.0, 1.0),
        ))

    def before_tick(self):
        """Schedule the actors' controls for the upcoming fixed physics step."""
        self._time += self._dt
        oncoming_speed = 6.0
        if self.kind in _WALKING:
            hazard_s, hazard_l = self._coordinates(self.hazard.get_location())
            ego_s, _ = self._coordinates(self.ego.get_location())
            # Negative offsets trigger farther upstream at nominal cruise;
            # positive offsets hold the walker at the curb for actual seconds.
            trigger = self._trigger_gap + 2.5 - 6.0 * min(self.crossing_delay, 0.0)
            if self._walk_at is None and hazard_s - ego_s <= trigger:
                self._walk_at = self._time + max(self.crossing_delay, 0.0)
            if self.phase == "waiting at curb" and self._walk_at is not None and self._time >= self._walk_at:
                self.phase = "crossing"
                self._event_start = self._time
            if self.phase == "crossing" and hazard_l <= 0.25:
                self.phase = "paused in lane"
                self._release_at = self._time + 2.5
            if self.phase == "paused in lane" and self._time >= self._release_at:
                self.phase = "leaving lane"
            if self.phase == "leaving lane" and hazard_l <= -6.4:
                self.phase = "across street"
                self._cleared = True
            walking = self.phase in ("crossing", "leaving lane")
            self.hazard.apply_control(carla.WalkerControl(
                direction=self._walk_direction,
                speed=self._walk_speed if walking else 0.0,
            ))
            other_s, _ = self._coordinates(self.traffic.get_location())
            # Keep the opposing car out of the pedestrian's swept path as well.
            if self.phase != "across street" and -2.0 < other_s - hazard_s < 15.0:
                oncoming_speed = 0.0
        elif self.kind == "lead":
            if self._release_at is None and (self.expert_gap() < 12.0 or self._time > 14.0):
                self._event_start = self._time
                self._release_at = self._time + 3.0
            moving = self._release_at is not None and self._time >= self._release_at
            self._cleared = moving
            self.phase = "lead departing" if moving else "lead stopped"
            self._drive_to(self.hazard, self._lead_target, self._lead_speed if moving else 0.0)
        elif self.kind == "braking":
            # A low-speed leader being caught by the ego brakes sharply.
            # The distance trigger makes each controller cause its own encounter.
            # A downstream fallback also exercises the brake for a cautious ego.
            leader_s, _ = self._coordinates(self.hazard.get_location())
            if self.phase == "lead cruising" and self._time > 2.0 and (self.expert_gap() < 13.0 or leader_s >= 47):
                self._event_start = self._time
                self.phase = "lead braking"
            if self.phase == "lead braking":
                velocity = self.hazard.get_velocity()
                if math.hypot(velocity.x, velocity.y) <= 0.15:
                    self.phase = "lead stopped"
                    self._release_at = self._time + 3.0
            if self.phase == "lead stopped" and self._time >= self._release_at:
                self.phase = "lead departing"
                self._cleared = True
            requested_speed = {"lead cruising": 2.7, "lead departing": self._lead_speed}.get(self.phase, 0.0)
            self._drive_to(self.hazard, self._lead_target, requested_speed)
        elif self.kind == "signal":
            if self._event_start is None and self._bumper_line_distance() >= -23.0:
                self._event_start = self._time
            if self._event_start is not None and not self._cleared and self._time >= self._event_start + 6.0:
                self.signal.set_state(carla.TrafficLightState.Green)
                self.phase = "green signal"
                self._cleared = True
        elif self.kind == "left":
            if self._event_start is None and self._bumper_line_distance() >= -20.0:
                self._event_start = self._time
                self.phase = "yield to oncoming"
            if self._event_start is None:
                oncoming_speed = 0.0
            else:
                delta = self.traffic.get_location() - self._conflict
                d = self._oncoming_direction
                if delta.x * d.x + delta.y * d.y > self.traffic.bounding_box.extent.x + 1.0:
                    self.phase = "oncoming cleared"
                    self._cleared = True
        self._drive_to(self.traffic, self._traffic_target, oncoming_speed)

    def expert_gap(self):
        """Ground truth for scenario triggers and scoring. Bumper clearance caps at 60 m."""
        if self.kind in ("signal", "left"):
            return 60.0 if self._cleared else _clip(-self._bumper_line_distance(), 0.0, 60.0)
        if self.hazard is None:
            return 60.0
        hazard_s, hazard_l = self._coordinates(self.hazard.get_location())
        ego_s, ego_l = self._coordinates(self.ego.get_location())
        ego_box, hazard_box = self.ego.bounding_box, self.hazard.bounding_box
        if hazard_s < ego_s - ego_box.extent.x:
            return 60.0
        if self.kind in _WALKING:
            # Include a small lateral buffer for the moving walker.
            if abs(hazard_l - ego_l) > ego_box.extent.y + 0.60:
                return 60.0
            tail = max(hazard_box.extent.x, hazard_box.extent.y)
        else:
            if abs(hazard_l - ego_l) > ego_box.extent.y + hazard_box.extent.y:
                return 60.0
            tail = hazard_box.extent.x - hazard_box.location.x
        nose = ego_box.location.x + ego_box.extent.x
        return _clip(hazard_s - ego_s - nose - tail, 0.0, 60.0)

    def metrics(self):
        progress, lateral = self._coordinates(self.ego.get_location())
        self._score_line()
        gap = self.expert_gap()
        tolerance = 1.85
        if any(entry - 3 <= progress <= end + 3 for entry, end in self._junction_ranges):
            # A legal wide arc can depart from one narrow connector model. A
            # junction or the planned approach/exit driving surface is required.
            pos = self.ego.get_location()
            lane = self._map.get_waypoint(pos, project_to_road=False, lane_type=carla.LaneType.Driving)
            if lane and ((lane.is_junction and lane.junction_id in self.junction_ids)
                         or (lane.road_id, lane.lane_id) in self._route_lanes):
                tolerance = 3.6
        return {
            "progress_m": _clip(progress, 0.0, self.length),
            "route_length_m": self.length,
            "lateral_error_m": abs(lateral),
            "phase": self.phase,
            "hazard_gap_m": gap,
            "hazard_in_path": gap < 60.0,
            "event_start_s": self._event_start,
            "turn_direction": self.turn_direction,
            "spawn_lateral_m": self.spawn_lateral_m,
            "spawn_yaw_deg": self.spawn_yaw_deg,
            "route_id": self.route_id,
            "road_ids": self.road_ids,
            "junction_ids": self.junction_ids,
            "district": self.district,
            "destination": self.destination,
            "crossing_delay_s": self.crossing_delay,
            "event_triggered": self._event_start is not None,
            "event_cleared": self._cleared,
            "red_light_violations": self.red_light_violations,
            "yield_violations": self.yield_violations,
            "route_departure": abs(lateral) > tolerance,
            "lane_tolerance_m": tolerance,
        }

    def _destroy_actors(self):
        for actor in reversed(self._actors):
            if actor.is_alive:
                actor.destroy()
        self._actors.clear()
        self.occluder = None
        self.ego = self.traffic = self.hazard = None

    def close(self):
        if not self._closed:
            try:
                self._destroy_actors()
            finally:
                try:
                    for signal, state in self._signal_states:
                        signal.set_state(state)
                finally:
                    if self._signals_were_frozen is not None:
                        self.world.freeze_all_traffic_lights(self._signals_were_frozen)
                    self.world.set_weather(self._previous_weather)
                    self._closed = True
