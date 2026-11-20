"""Seeded encounters on uninterrupted two-way streets in CARLA Town01.

Scenario state is available for ground-truth labels and scoring. The learned
driver and the sensor-only rules baseline must not call expert_gap().
"""

import math
import random

import carla


KINDS = ("clear", "pedestrian", "lead", "braking")
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


class Scenario:
    """Spawn a scene in a synchronous world. The caller owns all world ticks."""

    def __init__(self, world, kind, seed):
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
        dt = world.get_settings().fixed_delta_seconds
        if dt is None or dt <= 0:
            raise ValueError("Set synchronous mode and a fixed timestep before spawning a scenario")
        self._dt = dt
        self._time = 0.0
        self._release_at = None
        self._event_start = None
        self.phase = {
            "clear": "clear road", "pedestrian": "waiting at curb",
            "lead": "lead stopped", "braking": "lead cruising",
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
            for points, near, far in _routes(road_map, self.seed):
                if self._populate(points, near, far):
                    break
                self._destroy_actors()
            else:
                raise RuntimeError(f"No unobstructed two-way street long enough for this scenario in {road_map.name}")
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
        self.ego = self._spawn(ego_bp, _raised(points[0].transform, 0.35))
        if self.ego is None:
            return False
        traffic_bp = library.find("vehicle.audi.a2")
        traffic_bp.set_attribute("role_name", "oncoming")
        self.traffic = self._spawn(traffic_bp, _raised(far.transform, 0.35))
        if self.traffic is None:
            return False
        self._traffic_target = near.transform.location
        self.route = [w.transform.location for w in points[: ROUTE_METERS + 1]]
        self.route_start = self.route[0]
        self.road_id, self.lane_id = points[0].road_id, points[0].lane_id
        self._lead_target = points[-1].transform.location
        first, last = self.route[0], self.route[-1]
        self.length = math.hypot(last.x - first.x, last.y - first.y)
        self._fx = (last.x - first.x) / self.length
        self._fy = (last.y - first.y) / self.length
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
        return self.kind == "clear" or self.hazard is not None

    def _coordinates(self, location):
        x, y = location.x - self.route[0].x, location.y - self.route[0].y
        return x * self._fx + y * self._fy, y * self._fx - x * self._fy

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
        """For imitation labels only. Longitudinal bumper clearance, capped at 60 m."""
        if self.hazard is None:
            return 60.0
        hazard_s, hazard_l = self._coordinates(self.hazard.get_location())
        ego_s, ego_l = self._coordinates(self.ego.get_location())
        ego_box, hazard_box = self.ego.bounding_box, self.hazard.bounding_box
        if hazard_s < ego_s - ego_box.extent.x:
            return 60.0
        if self.kind == "pedestrian":
            # Allow the teacher a small lateral buffer for the moving walker.
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
                self.world.set_weather(self._previous_weather)
                self._closed = True
