"""Background actors whose paths stay clear of a staged Town01 trip."""

from dataclasses import dataclass
import math
import random

import carla

from .district import trip_points
from .world import _raised

_WALK_METERS = 5.0
_WALK_CLEARANCE = 3.4
_CAR_CLEARANCE = 12.0
_ATTEMPTS_PER_STREET = 5
_VEHICLE_MODELS = (
    "vehicle.lincoln.mkz",
    "vehicle.mini.cooper",
    "vehicle.tesla.model3",
)


def _distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _clearance(location, route):
    return min(_distance(location, point) for point in route)


def _same_lane(first, last):
    return (not last.is_junction
            and (first.road_id, first.lane_id, first.lane_type)
            == (last.road_id, last.lane_id, last.lane_type))


def _sidewalk(driving):
    while driving:
        driving = driving.get_right_lane()
        if driving and driving.lane_type == carla.LaneType.Sidewalk:
            return driving
    return None


def _walking_segment(start, route, crossing):
    origin = start.transform.location
    if (start.lane_width < 2.0 or _clearance(origin, route) < _WALK_CLEARANCE
            or _distance(origin, crossing) < 14.0):
        return None
    before, after = start.previous(5.0), start.next(_WALK_METERS + 5.0)
    if any(len(ends) != 1 or not _same_lane(start, ends[0]) for ends in (before, after)):
        return None
    locations = []
    for step in range(1, int(_WALK_METERS) + 1):
        choices = start.next(float(step))
        if len(choices) != 1 or not _same_lane(start, choices[0]):
            return None
        locations.append(choices[0].transform.location)
    # Validate the straight path actually commanded to the walker.
    destination = locations[-1]
    delta = destination - origin
    for step, sidewalk_center in enumerate(locations, 1):
        chord = origin + delta * (step / _WALK_METERS)
        if (_distance(chord, sidewalk_center) > 0.2 or _clearance(chord, route) < _WALK_CLEARANCE
                or _distance(chord, crossing) < 14.0):
            return None
    return destination


def _walker_sites(points, route, crossing, rng):
    by_street = {}
    for index, point in enumerate(points):
        if not point.is_junction:
            by_street.setdefault(point.road_id, []).append((index, point))
    for street_number, street in enumerate(by_street.values()):
        if street_number == 0:
            preferred = street[len(street) // 2][0]
        elif street_number == 1:
            preferred = street[int(len(street) * 0.42)][0]
        else:
            preferred = street[int(len(street) * 0.55)][0]
        ranked = sorted(street[::2], key=lambda pair: abs(pair[0] - preferred) + rng.uniform(0, 10))
        candidates = []
        for index, point in ranked:
            sides = [point, point.get_left_lane()]
            rng.shuffle(sides)
            for side in sides:
                sidewalk = _sidewalk(side)
                if sidewalk is None:
                    continue
                target = _walking_segment(sidewalk, route, crossing)
                if target is not None:
                    candidates.append((index, _raised(sidewalk.transform, 0.85), target))
            if len(candidates) >= _ATTEMPTS_PER_STREET:
                break
        yield candidates[:_ATTEMPTS_PER_STREET]


def _junction_centers(points):
    junctions = {}
    for point in points:
        if point.is_junction:
            junctions.setdefault(point.junction_id, []).append(point)
    return [group[len(group) // 2].transform.location for group in junctions.values()]


def _vehicle_sites(road_map, points, route, rng):
    streets = {point.road_id for point in points}
    junctions = _junction_centers(points)
    waypoints = sorted(road_map.generate_waypoints(4.0),
                       key=lambda point: (point.road_id, point.lane_id, point.s))
    off_route = [point for point in waypoints if not point.is_junction
                 and point.road_id not in streets and point.lane_width >= 3.5]
    for junction in junctions:
        candidates = []
        for point in off_route:
            pose = _raised(point.transform, 0.35)
            pose.location = pose.location + pose.get_right_vector() * 0.55
            gap = _distance(pose.location, junction)
            if gap > 25.0 or _clearance(pose.location, route) < _CAR_CLEARANCE:
                continue
            if any(len(ends) != 1 or not _same_lane(point, ends[0])
                   for ends in (point.previous(7.0), point.next(7.0))):
                continue
            candidates.append((gap + rng.uniform(0, 4), point.road_id, pose))
        candidates.sort(key=lambda item: item[0])
        yield [(road, pose) for _, road, pose in candidates[:_ATTEMPTS_PER_STREET]]


@dataclass
class _Walker:
    actor: object
    origin: carla.Location
    direction: carla.Vector3D
    length: float
    speed: float
    start_at: float
    stopped: bool = False

    def update(self, elapsed, dt):
        if self.stopped or elapsed < self.start_at or not self.actor.is_alive:
            return
        offset = self.actor.get_location() - self.origin
        traveled = offset.x * self.direction.x + offset.y * self.direction.y
        lateral = offset.y * self.direction.x - offset.x * self.direction.y
        remaining = self.length - traveled
        # A blocked or deflected actor should stop rather than drift off its sidewalk.
        self.stopped = remaining <= 0.2 or abs(lateral) > 0.45 or elapsed > self.start_at + 8.0
        self.actor.apply_control(carla.WalkerControl(
            direction=self.direction, speed=0.0 if self.stopped else min(self.speed, remaining / dt),
        ))


class Ambient:
    """Own optional background actors. The caller owns every simulation tick."""

    def __init__(self, world, scene):
        self._actors = []
        self._walkers = []
        self._counts = {"walkers": 0, "vehicles": 0}
        self._elapsed = 0.0
        if scene.kind != "trip":
            return
        self._dt = world.get_settings().fixed_delta_seconds
        road_map = world.get_map()
        points = trip_points(road_map, scene.district, scene.destination)
        crossing = scene.route[scene._hazard_index]
        rng = random.Random(scene.seed ^ 0x36D2)
        placement_rng = random.Random(scene.seed ^ 0x3F91)
        library = world.get_blueprint_library()
        walker_models = sorted(library.filter("walker.pedestrian.*"), key=lambda bp: bp.id)
        vehicle_models = [bp for model in _VEHICLE_MODELS for bp in library.filter(model)]
        try:
            if walker_models:
                for candidates in _walker_sites(points, scene.route, crossing, placement_rng):
                    for index, pose, target in candidates:
                        blueprint = rng.choice(walker_models)
                        actor = world.try_spawn_actor(blueprint, pose)
                        if actor is None:
                            continue
                        self._actors.append(actor)
                        delta = target - pose.location
                        length = math.hypot(delta.x, delta.y)
                        direction = carla.Vector3D(delta.x / length, delta.y / length, 0.0)
                        # Estimate an arrival window from cruise speed, without using ego state.
                        start_at = max(0.0, index / 6.0 - 7.0) + rng.uniform(0.5, 2.0)
                        self._walkers.append(_Walker(actor, pose.location, direction, length,
                                                     rng.uniform(0.9, 1.25), start_at))
                        self._counts["walkers"] += 1
                        break
            if vehicle_models:
                used_streets = set()
                for candidates in _vehicle_sites(road_map, points, scene.route, placement_rng):
                    for road, pose in candidates:
                        if road in used_streets:
                            continue
                        blueprint = rng.choice(vehicle_models)
                        blueprint.set_attribute("role_name", "background")
                        if blueprint.has_attribute("color"):
                            colors = sorted(blueprint.get_attribute("color").recommended_values)
                            if colors:
                                blueprint.set_attribute("color", rng.choice(colors))
                        actor = world.try_spawn_actor(blueprint, pose)
                        if actor is None:
                            continue
                        self._actors.append(actor)
                        actor.apply_control(carla.VehicleControl(hand_brake=True))
                        self._counts["vehicles"] += 1
                        used_streets.add(road)
                        break
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                exc.add_note(f"Ambient cleanup failed: {cleanup_error}")
            raise

    @property
    def counts(self):
        return self._counts.copy()

    def observer_positions(self):
        positions = []
        for actor in self._actors:
            if actor.is_alive:
                pose = actor.get_transform()
                positions.append(dict(type="walker" if actor.type_id.startswith("walker.") else "vehicle",
                                      x=round(pose.location.x,2), y=round(pose.location.y,2),
                                      yaw=round(pose.rotation.yaw,2)))
        return positions

    def before_tick(self):
        if self._walkers:
            self._elapsed += self._dt
            for walker in self._walkers:
                walker.update(self._elapsed, self._dt)

    def close(self):
        errors = []
        for actor in reversed(self._actors):
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception as exc:
                errors.append(exc)
        self._actors.clear()
        self._walkers.clear()
        if errors:
            raise errors[0]
