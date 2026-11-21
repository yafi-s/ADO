"""Sidewalk activity along the Town01 journey, with bounded native walker paths."""
from dataclasses import dataclass
import math
import random

import carla

from .ambient import Ambient, _sidewalk, _Walker, _walking_segment
from .world import _raised


@dataclass
class _Traffic:
    actor: object
    path: list
    activation: float
    speed: float
    movement: str
    road: int
    lane: int
    active: bool = False
    traveled: float = 0.
    closest: float | None = None
    peak_speed: float = 0.
    previous: object = None
    departed_lane: bool = False


class JourneyAmbient(Ambient):
    def __init__(self, scene):
        self._actors, self._walkers, self._sites = [], [], []
        self._visible_motion = []
        self._cars = []
        self._counts = {"walkers": 0, "vehicles": 0}
        self._elapsed = 0.
        self._dt = scene._dt
        self._scene = scene
        library = scene.world.get_blueprint_library()
        models = sorted(library.filter("walker.pedestrian.*"), key=lambda b: b.id)
        rng = random.Random(scene.seed ^ 0x431)
        try:
            used = set()
            for base in self._preferred_sites(scene):
                road = scene._points[base].road_id
                if road in used:
                    continue
                for shift in (0, -4, 4, -8, 8, -12, 12):
                    index = base + shift
                    driving = scene._points[index]
                    if driving.road_id != road:
                        continue
                    sides = [driving, driving.get_left_lane()]
                    rng.shuffle(sides)
                    if any(self._place(scene, index, side, models, rng) for side in sides):
                        used.add(road)
                        break
                if len(used) >= 5:
                    break
            if scene._urban:
                self._place_traffic(scene)
            if len(self._walkers) < 2:
                raise RuntimeError("Town01 journey lacks unobstructed sidewalk activity on two distinct streets")
        except BaseException:
            self.close()
            raise

    def _place_traffic(self, scene):
        # Isolated, verified same-road stretches between the scored encounters.
        # Args are OpenDRIVE road/lane, start/end s, ego activation s, speed.
        locations = ((76, 1, 157.,100.,90.,5.4,"opposing","vehicle.mini.cooper"),
                     (67,-1,  85.,177.,77.,6.8,"parallel","vehicle.tesla.model3"),
                     (20, 1, 104., 42.,48.,5.5,"opposing","vehicle.lincoln.mkz"))
        for road, lane, start, end, activation, speed, movement, model in locations:
            route_idx = min((i for i,p in enumerate(scene._points) if p.road_id==road),
                            key=lambda i: abs(scene._points[i].s-activation))
            for shift in (0.,-3.,3.,-6.,6.):
                native = scene._map.get_waypoint_xodr(road,lane,start+shift)
                if native is None or native.lane_type != carla.LaneType.Driving:
                    continue
                path = [native.transform.location]
                wp = native
                for _ in range(round(abs(end-start))):
                    choices = wp.next(1.)
                    if len(choices) != 1 or (choices[0].road_id,choices[0].lane_id)!=(road,lane):
                        break
                    wp = choices[0]
                    path.append(wp.transform.location)
                if len(path) < abs(end-start)-1 or not self._isolated(scene,path):
                    continue
                bp = scene._catalog.find(model)
                bp.set_attribute("role_name","journey_background")
                actor = scene.world.try_spawn_actor(bp,_raised(native.transform,.35))
                if actor is None:
                    continue
                actor.apply_control(carla.VehicleControl(brake=1.))
                self._actors.append(actor)
                self._cars.append(_Traffic(actor,path,scene._station[route_idx],speed,movement,road,lane))
                self._counts["vehicles"] += 1
                break
        if len(self._cars) < 2:
            raise RuntimeError("Town03 journey lacks two clear, separate background traffic corridors")

    @staticmethod
    def _isolated(scene, path):
        for e in scene.events:
            # Crossing people can use either lane. Keep a buffer around every
            # crossing, stop line, bus pass and work zone for the whole runout.
            buffer = 26. if e.kind=="roadworks" else 11.
            if any(p.distance(e.pose.location) < buffer for p in path):
                return False
            if e.exit_path and any(q.distance(p) < 3.2 for q in e.exit_path[::3] for p in path[::3]):
                return False
        return True

    def _move_cars(self):
        for car in self._cars:
            if not car.active and self._scene._last_progress >= car.activation:
                car.active = True
            if not car.active:
                continue
            pos = car.actor.get_location()
            if car.previous is None and pos.distance(car.path[0]) > 2.:
                # A newly spawned actor may still have a missing first snapshot.
                continue
            if car.previous is not None:
                step = math.hypot(pos.x-car.previous.x,pos.y-car.previous.y)
                if step > max(3.,2.*car.speed*self._dt):
                    car.departed_lane = True
                else:
                    car.traveled += step
            car.previous = pos
            v = car.actor.get_velocity()
            speed = math.hypot(v.x,v.y)
            car.peak_speed = max(car.peak_speed,speed)
            if speed > .7:
                gap = pos.distance(self._scene.ego.get_location())
                car.closest = gap if car.closest is None else min(car.closest,gap)
            nearest = min(range(len(car.path)), key=lambda i:pos.distance(car.path[i]))
            if pos.distance(car.path[nearest]) > 1.7:
                car.departed_lane = True
            remaining = pos.distance(car.path[-1])
            requested = 0. if car.departed_lane or remaining < 2. else min(car.speed,math.sqrt(3.*remaining))
            self._scene._drive_to(car.actor,car.path[min(nearest+5,len(car.path)-1)],requested)

    @property
    def traffic_activity(self):
        records = [dict(blueprint=c.actor.type_id, movement=c.movement,road=c.road,lane=c.lane,
                        traveled_m=round(c.traveled,2),peak_speed_mps=round(c.peak_speed,2),
                        closest_while_moving_m=None if c.closest is None else round(c.closest,2),
                        left_corridor=c.departed_lane) for c in self._cars]
        return dict(spawned=len(records),moving_near_ego=sum(c.traveled >= 3. and c.closest is not None
                    and c.closest < 16. and not c.departed_lane for c in self._cars), vehicles=records)

    @staticmethod
    def _preferred_sites(scene):
        if not scene._urban:
            return [next(i for i, w in enumerate(scene._points) if w.road_id == road) + offset
                    for road, offset in ((19, 76), (12, 138), (4, 155), (10, 112), (1, 50))]
        points = scene._points
        candidates = [i for i in range(25,len(points)-25,5)
                      if not points[i].is_junction and points[i-12].road_id == points[i+12].road_id
                      and all(abs(e.s-scene._station[i]) > 18. for e in scene.events)]
        desired = (.16,.32,.54,.72,.88)
        preferred = [min(candidates,key=lambda i:abs(scene._station[i]-fraction*scene.length))
                     for fraction in desired] if candidates else []
        return list(dict.fromkeys(preferred+candidates))

    def _place(self, scene, i, side, models, rng):
        sidewalk = _sidewalk(side)
        if sidewalk is None:
            return False
        origin = sidewalk.transform.location
        crossing = min((e.pose.location for e in scene.events), key=origin.distance)
        target = _walking_segment(sidewalk, scene.route[i-20:i+21], crossing)
        if target is None:
            return False
        pose = _raised(sidewalk.transform, .85)
        bp = rng.choice(models)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        actor = scene.world.try_spawn_actor(bp, pose)
        if actor is None:
            return False
        delta = target - pose.location
        length = math.hypot(delta.x, delta.y)
        self._actors.append(actor)
        self._sites.append(scene._station[i])
        self._visible_motion.append(0.)
        self._walkers.append(_Walker(actor, pose.location, carla.Vector3D(delta.x/length, delta.y/length, 0.),
                                      length, rng.uniform(.95,1.2), math.inf))
        self._counts["walkers"] += 1
        return True

    def before_tick(self):
        self._elapsed += self._dt
        self._move_cars()
        for i, (station, walker) in enumerate(zip(self._sites, self._walkers)):
            distance = station - self._scene._last_progress
            if math.isinf(walker.start_at) and 0. < distance < 30.:
                walker.start_at = self._elapsed
            walker.update(self._elapsed, self._dt)
            if not math.isinf(walker.start_at) and -4. < distance < 32. and walker.actor.is_alive:
                offset = walker.actor.get_location() - walker.origin
                progress = offset.x * walker.direction.x + offset.y * walker.direction.y
                self._visible_motion[i] = max(self._visible_motion[i], progress)

    @property
    def activity(self):
        return dict(spawned=len(self._walkers), moved_on_approach=sum(m >= 1. for m in self._visible_motion),
                    motion_on_approach_m=[round(m,2) for m in self._visible_motion])
