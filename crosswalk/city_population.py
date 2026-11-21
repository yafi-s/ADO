"""Seeded native vehicles and walking pairs for the Town03 Market showcase."""
from dataclasses import dataclass
import math
import random

import carla

from .journey_ambient import _Traffic, JourneyAmbient
from .world import _raised


# Each corridor stays on a single driving lane, ending before the junction.
# Same-lane following pairs have separate stopping points and preserve their order.
# Fields are OpenDRIVE road/lane, start/end s, release station on the ego route, movement.
CARS = (
    (24, 1, 20, 3, 0, "opposing"),
    (30, -1, 18, 66, 9, "side street"), (30, -2, 32, 65, 12, "side street"),
    (76, 1, 49, 8, 76, "opposing"), (76, 1, 61, 20, 78, "opposing"),
    (76, 1, 144, 82, 160, "opposing"), (76, 1, 164, 94, 163, "opposing"),
    (67, 2, 95, 9, 335, "opposing"), (67, 3, 105, 12, 340, "opposing"),
    (67, -1, 77, 174, 399, "parallel"),
    (67, 2, 267, 125, 499, "opposing"), (67, 3, 250, 125, 494, "opposing"),
    (69, 2, 48, 6, 705, "opposing"), (69, 3, 40, 6, 707, "opposing"),
    (4, -1, 8, 85, 807, "opposing"), (4, -2, 17, 84, 810, "opposing"),
    (3, -2, 12, 105, 909, "opposing"),
    (19, -1, 4, 46, 1000, "side street"),
    (20, 1, 85, 12, 1073, "opposing"), (20, 1, 105, 25, 1076, "opposing"),
    (62, -1, 3, 34, 1223, "side street"), (62, 1, 34, 5, 1230, "side street"),
    (23, 1, 49, 7, 1348, "opposing"),
)

# Each pair shares a sidewalk segment. A different segment of the same route can
# appear close in Euclidean space, so new paths must also clear existing walkers.
# The sidewalk lane nearest the road with enough width is selected at each site.
WALKS = (
    (24, -1, 4, 16), (24, 1, 6, 18), (30, 1, 49, 70),
    (76, 1, 14, 38), (76, -1, 87, 109), (76, 1, 120, 146),
    (47, -1, 4, 20), (67, 1, 8, 35), (67, -1, 64, 92),
    (67, 1, 110, 137), (67, -1, 163, 188), (67, 1, 228, 255),
    (67, -1, 270, 295), (69, -1, 15, 37), (4, 1, 48, 72),
    (3, -1, 24, 53), (3, 1, 84, 107), (19, 1, 12, 34),
    (20, 1, 32, 60), (20, -1, 84, 105), (62, 1, 13, 37),
    (23, -1, 8, 33),
)


def lane_segment(road_map, road, lane, start, end, lane_type):
    step = 1 if end > start else -1
    points = [road_map.get_waypoint_xodr(road, lane, s)
              for s in range(start, end + step, step)]
    if any(p is None or p.is_junction or p.lane_type != lane_type
           or p.road_id != road or p.lane_id != lane for p in points):
        return []
    return points


def sidewalk_segment(road_map, road, side, start, end):
    for lane in range(1, 10):
        points = lane_segment(road_map, road, side * lane, start, end, carla.LaneType.Sidewalk)
        if points and min(p.lane_width for p in points) >= 1.85:
            return points
    return []


@dataclass
class Pedestrian:
    actor: object
    path: list
    index: int
    direction: int
    speed: float
    pause: float
    wait: float = 0.
    previous: object = None
    moved_near: float = 0.
    traveled: float = 0.
    anchor: object = None
    anchor_time: float = 0.

    def step(self, time, ego):
        pos = self.actor.get_location()
        if self.previous is not None:
            moved = math.hypot(pos.x - self.previous.x, pos.y - self.previous.y)
            if moved < 1.:
                self.traveled += moved
                if pos.distance(ego) < 35.:
                    self.moved_near += moved
        self.previous = pos
        if time < self.wait:
            return
        # If a native bench or other obstacle blocks the authored sidewalk,
        # turn back along the same segment. Physics remains enabled throughout.
        if self.anchor is None:
            self.anchor, self.anchor_time = pos, time
        elif time - self.anchor_time >= 2.:
            if pos.distance(self.anchor) < .25:
                self.direction *= -1
                self.index = min(range(len(self.path)), key=lambda i: pos.distance(self.path[i]))
            self.anchor, self.anchor_time = pos, time
        target = self.path[self.index]
        dx, dy = target.x - pos.x, target.y - pos.y
        distance = math.hypot(dx, dy)
        if distance < .55:
            nxt = self.index + self.direction
            if not 0 <= nxt < len(self.path):
                self.direction *= -1
                self.wait = time + self.pause
                self.anchor = None
                self.actor.apply_control(carla.WalkerControl(speed=0.))
            self.index += self.direction
            return
        if distance > 3.:
            self.actor.apply_control(carla.WalkerControl(speed=0.))
            return
        self.actor.apply_control(carla.WalkerControl(
            direction=carla.Vector3D(dx / distance, dy / distance, 0.), speed=self.speed))


class CityPopulation(JourneyAmbient):
    """Populate the whole route once. NPC controls never write ego observations."""

    def __init__(self, scene):
        if not scene._urban:
            raise ValueError("The city-life population requires Town03")
        self._scene, self._dt, self._elapsed = scene, scene._dt, 0.
        self._actors, self._walkers, self._cars = [], [], []
        self._contact_sensors, self._contacts = [], {}
        self._counts = {"vehicles": 0, "walkers": 0}
        rng = random.Random(scene.seed ^ 0xC170)
        try:
            self._traffic(scene, rng)
            self._pedestrians(scene, rng)
            if len(self._cars) < 8 or len(self._walkers) < 12:
                raise RuntimeError(f"Too few native city-life actors spawned: {self._counts}")
            bp = scene._catalog.find("sensor.other.collision")
            for car in self._cars:
                sensor = scene.world.spawn_actor(bp, carla.Transform(), attach_to=car.actor)
                self._contact_sensors.append(sensor)
                sensor.listen(self._contact)
        except BaseException:
            self.close()
            raise

    def _contact(self, event):
        # Deduplicate mirrored sensor callbacks and prolonged contact on later ticks.
        if event.other_actor.type_id.startswith(("vehicle.", "walker.")):
            pair = tuple(sorted((event.actor.id, event.other_actor.id)))
            self._contacts.setdefault(pair, dict(frame=int(event.frame), actor_id=event.actor.id,
                                                 other_id=event.other_actor.id, other_type=event.other_actor.type_id))

    def close(self):
        for sensor in self._contact_sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._contact_sensors.clear()
        super().close()

    def _traffic(self, scene, rng):
        models = [b for name in ("vehicle.lincoln.mkz", "vehicle.mini.cooper", "vehicle.tesla.model3",
                                 "vehicle.audi.a2", "vehicle.dodge.charger_2020")
                  for b in scene._catalog.filter(name)]
        # Planned event runouts have priority even well beyond their anchor.
        runouts = [point for event in scene.events for point in event.exit_path[::2]]
        stops = {}
        for light, _ in scene._signal_states:
            for point in light.get_stop_waypoints():
                if point:
                    stops.setdefault((point.road_id, point.lane_id), []).append(point.s)
        for road, lane, start, end, activation, movement in CARS:
            points = lane_segment(scene._map, road, lane, start, end, carla.LaneType.Driving)
            # Keep the front bumper behind each native stop line throughout
            # the runout, including a line located inside an OpenDRIVE road.
            direction = 1 if end > start else -1
            for station in stops.get((road, lane), []):
                if (station - start) * direction >= 0.:
                    points = [p for p in points if (station - p.s) * direction >= 4.]
            if len(points) < 9:
                continue
            path = [p.transform.location for p in points]
            if any(p.distance(q) < 3. for p in path[::2] for q in runouts):
                continue
            bp = rng.choice(models)
            bp.set_attribute("role_name", "city_life_background")
            if bp.has_attribute("color"):
                colors = sorted(bp.get_attribute("color").recommended_values)
                if colors:
                    bp.set_attribute("color", rng.choice(colors))
            actor = scene.world.try_spawn_actor(bp, _raised(points[0].transform, .35))
            if actor is None:
                continue
            actor.apply_control(carla.VehicleControl(brake=1.))
            self._actors.append(actor)
            self._cars.append(_Traffic(actor, path, activation, 4.8 if road in (76, 20) else 5.8,
                                       movement, road, lane))
        self._counts["vehicles"] = len(self._cars)

    def _pedestrians(self, scene, rng):
        models = sorted(scene._catalog.filter("walker.pedestrian.*"), key=lambda bp: bp.id)
        for site, (road, side, start, end) in enumerate(WALKS):
            points = sidewalk_segment(scene._map, road, side, start, end)
            if not points:
                continue
            # Stagger pairs along and across the sidewalk. A two-metre strip
            # allows two 0.6 m wide bodies with clearance from each curb.
            width = min(p.lane_width for p in points)
            offsets = (-.48, .48) if width >= 1.95 and site % 3 == 0 else (0.,)
            speed = rng.uniform(.92, 1.30)
            direction = rng.choice((-1, 1))
            pause = rng.uniform(.9, 2.6)
            for member, offset in enumerate(offsets):
                path = [p.transform.location + p.transform.get_right_vector() * offset for p in points]
                start_index = 3 + member if direction == 1 else len(points) - 4 - member
                bp = rng.choice(models)
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                pose = _raised(points[start_index].transform, .87)
                pose.location += pose.get_right_vector() * offset
                actor = scene.world.try_spawn_actor(bp, pose)
                if actor is None:
                    continue
                self._actors.append(actor)
                self._walkers.append(Pedestrian(actor, path, start_index + direction, direction, speed, pause))
        self._counts["walkers"] = len(self._walkers)

    def before_tick(self):
        self._elapsed += self._dt
        self._move_cars()
        ego = self._scene.ego.get_location()
        for walker in self._walkers:
            if walker.actor.is_alive:
                walker.step(self._elapsed, ego)

    @property
    def activity(self):
        return dict(spawned=len(self._walkers), moved_on_approach=sum(w.moved_near >= 1. for w in self._walkers),
                    motion_on_approach_m=[round(w.moved_near, 2) for w in self._walkers],
                    total_walked_m=[round(w.traveled, 2) for w in self._walkers],
                    actor_ids=[w.actor.id for w in self._walkers],
                    missing_pedestrians=sum(not w.actor.is_alive for w in self._walkers))

    @property
    def traffic_activity(self):
        activity = super().traffic_activity
        for result, car in zip(activity["vehicles"], self._cars, strict=True):
            result.update(actor_id=car.actor.id, activation_station_m=car.activation)
        activity["participant_contacts"] = sorted(self._contacts.copy().values(),
                                                 key=lambda event: (event["frame"], event["actor_id"]))
        return activity
