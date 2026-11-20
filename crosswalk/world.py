"""Seeded Town01 routes and non-ego encounters. Actor state is for scoring."""

import math
import random

import carla


KINDS = ("clear", "pedestrian", "lead", "braking", "turn")
ROUTE_METERS = 64
_EXTENSION = 18  # let the departing lead keep driving past the ego's finish


def _angle(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def _clip(value, low, high):
    return max(low, min(value, high))


def _raised(transform, height):
    p = transform.location
    return carla.Transform(carla.Location(p.x, p.y, p.z + height), transform.rotation)


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
            if 60 <= (change if direction == "right" else -change) <= 120:
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


class Scenario:
    """Spawn a scene in a synchronous world. The caller owns all world ticks."""

    def __init__(self, world, kind, seed, perturb=False):
        kind = {"crossing": "pedestrian", "stopped_lead": "lead"}.get(kind, kind)
        if kind not in KINDS:
            raise ValueError(f"Unknown encounter {kind!r}, choose from {KINDS}")
        self.world = world
        self.name = self.kind = kind
        self.seed = int(seed)
        self._actors = []
        self.ego = self.hazard = self.traffic = None
        self._previous_weather = world.get_weather()
        self._closed = False
        self._signal_states = []
        self._signals_were_frozen = None
        self.turn_direction = ("right" if self.seed % 2 else "left") if kind == "turn" else None
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
        self.phase = {
            "clear": "clear road", "pedestrian": "waiting at curb",
            "lead": "lead stopped", "braking": "lead cruising", "turn": f"{self.turn_direction} turn",
        }[kind]
        rng = random.Random(self.seed ^ 0x2AF1)
        self._hazard_index = rng.randint(29, 39)
        self._walk_speed = rng.uniform(0.95, 1.35)
        self._trigger_gap = rng.uniform(15.0, 19.0)
        self._lead_speed = rng.uniform(5.0, 6.5)
        self.weather = rng.choice(("ClearNoon", "CloudyNoon", "WetNoon", "ClearSunset"))
        if kind == "braking":
            self._hazard_index = rng.randint(18, 24)
        try:
            road_map = world.get_map()
            routes = (_turn_routes(road_map, self.seed, self.turn_direction) if kind == "turn"
                      else _routes(road_map, self.seed))
            for points, near, far in routes:
                if self._populate(points, near, far):
                    break
                self._destroy_actors()
            else:
                raise RuntimeError(f"No unobstructed {kind} route available in {road_map.name}")
            if kind == "turn":
                # This isolated experiment has one approach and no cross traffic.
                # Include signal actors associated with adjacent approach roads.
                signals = world.get_actors().filter("traffic.traffic_light")
                if signals:
                    self._signal_states = [(signal, signal.get_state()) for signal in signals]
                    self._signals_were_frozen = signals[0].is_frozen()
                    world.freeze_all_traffic_lights(True)
                    for signal in signals:
                        signal.set_state(carla.TrafficLightState.Green)
            self.world.set_weather(getattr(carla.WeatherParameters, self.weather))
        except BaseException:
            self.close()
            raise

    def _spawn(self, blueprint, transform):
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            self._actors.append(actor)
        return actor

    def _populate(self, points, near, far):
        library = self.world.get_blueprint_library()
        ego_bp = library.find("vehicle.lincoln.mkz_2020")
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
        traffic_bp = library.find("vehicle.audi.a2")
        traffic_bp.set_attribute("role_name", "oncoming")
        self.traffic = self._spawn(traffic_bp, _raised(far.transform, 0.35))
        if self.traffic is None:
            return False
        self._traffic_target = near.transform.location
        self._set_route([w.transform.location for w in points[: ROUTE_METERS + 1]])
        self.road_id, self.lane_id = points[0].road_id, points[0].lane_id
        self._lead_target = points[-1].transform.location
        hazard_transform = points[self._hazard_index].transform
        if self.kind == "pedestrian":
            people = sorted(library.filter("walker.pedestrian.*"), key=lambda bp: bp.id)
            walker_bp = random.Random(self.seed).choice(people)
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            offset = points[self._hazard_index].lane_width / 2 + 0.35
            p = hazard_transform.location
            at_curb = carla.Transform(
                carla.Location(p.x - self._fy * offset, p.y + self._fx * offset, p.z + 1.0),
                carla.Rotation(yaw=hazard_transform.rotation.yaw - 90),
            )
            self.hazard = self._spawn(walker_bp, at_curb)
        elif self.kind in ("lead", "braking"):
            lead_bp = library.find("vehicle.tesla.model3")
            lead_bp.set_attribute("role_name", self.kind)
            self.hazard = self._spawn(lead_bp, _raised(hazard_transform, 0.35))
        return self.kind in ("clear", "turn") or self.hazard is not None

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
        if self.kind == "pedestrian":
            hazard_s, hazard_l = self._coordinates(self.hazard.get_location())
            ego_s, _ = self._coordinates(self.ego.get_location())
            if self.phase == "waiting at curb" and hazard_s - ego_s <= self._trigger_gap + 2.5:
                self.phase = "crossing"
                self._event_start = self._time
            if self.phase == "crossing" and hazard_l <= 0.25:
                self.phase = "paused in lane"
                self._release_at = self._time + 2.5
            if self.phase == "paused in lane" and self._time >= self._release_at:
                self.phase = "leaving lane"
            if self.phase == "leaving lane" and hazard_l <= -6.4:
                self.phase = "across street"
            walking = self.phase in ("crossing", "leaving lane")
            self.hazard.apply_control(carla.WalkerControl(
                direction=carla.Vector3D(self._fy, -self._fx, 0),
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
            requested_speed = {"lead cruising": 2.7, "lead departing": self._lead_speed}.get(self.phase, 0.0)
            self._drive_to(self.hazard, self._lead_target, requested_speed)
        self._drive_to(self.traffic, self._traffic_target, oncoming_speed)

    def expert_gap(self):
        """Ground truth for scenario triggers and scoring. Bumper clearance caps at 60 m."""
        if self.hazard is None:
            return 60.0
        hazard_s, hazard_l = self._coordinates(self.hazard.get_location())
        ego_s, ego_l = self._coordinates(self.ego.get_location())
        ego_box, hazard_box = self.ego.bounding_box, self.hazard.bounding_box
        if hazard_s < ego_s - ego_box.extent.x:
            return 60.0
        if self.kind == "pedestrian":
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
        gap = self.expert_gap()
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
        }

    def _destroy_actors(self):
        for actor in reversed(self._actors):
            if actor.is_alive:
                actor.destroy()
        self._actors.clear()
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
