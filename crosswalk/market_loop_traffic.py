"""Commission independent traffic on the return carriageway of Town03 Market Loop."""
import math

import carla

from .city_population import CARS, lane_segment
from .camera_geometry import CAMERAS
from .city_progress_bus import body_gap, footprint
from .showcase import MOUNT
from .maneuver_scene import ManeuverFixture, ManeuverPopulation, MovingVehicle
from .world import _raised


def boulevard_run(road, start_road, lane, initial_s, final_s):
    """Follow physical driving-lane successors through the two straight continuations."""
    wp = road.get_waypoint_xodr(start_road, lane, initial_s)
    points = [wp]
    for _ in range(450):
        if wp.road_id == 67 and wp.s <= final_s:
            return points
        choices = [v for v in wp.next(1.) if v.lane_id == lane and
                   v.road_id in (69, 1960, 68, 101, 67) and v.lane_type == carla.LaneType.Driving]
        if len(choices) != 1:
            raise ValueError(f"Disconnected return traffic route {(wp.road_id, lane, wp.s)}")
        wp = choices[0]
        points.append(wp)
    raise ValueError("Return traffic runout exceeds its surveyed approach")


class LoopPopulation(ManeuverPopulation):
    # Retain the original Market parallel sedan. The extra outer-lane convoy
    # vehicle travels another 12 m to provide a safe downstream convoy berth.
    traffic_corridors = CARS + tuple(
        (road,lane,start,140 if (road,lane,start)==(67,3,281) else end,release,movement)
        for road,lane,start,end,release,movement in ManeuverPopulation.traffic_corridors
        if (road,lane,start,end,release,movement) not in CARS)

    def _traffic(self, scene, rng):
        super()._traffic(scene,rng)
        for car in self._cars:
            if car.road == 69 and car.lane in (2,3):
                start = 48. if car.lane == 2 else 40.
                stop = 139. if car.lane == 2 else 155.
                points = boulevard_run(scene._map,69,car.lane,start,stop)
                if points[0].transform.location.distance(car.path[0]) > .05:
                    raise ValueError("Extended background traffic differs from original launch")
                car.path = [w.transform.location for w in points]
        if len([v for v in self._cars if v.road==69 and v.lane in (2,3)]) != 2:
            raise ValueError("Both original opposing cars must continue physically ahead of the return trip")

    def monitor_arrival(self, actor):
        sensor = self._scene.world.spawn_actor(self._scene._catalog.find("sensor.other.collision"),
                                                carla.Transform(),attach_to=actor)
        self._contact_sensors.append(sensor)
        sensor.listen(self._contact)


class LoopFixture(ManeuverFixture):
    """Existing outbound cone fixture plus an independent late, reverse-carriageway pass."""

    def _traffic(self, case):
        if case not in ("continuous","later-gap"):
            raise ValueError("Market Loop supports the commissioned continuous and later-gap traffic schedules")
        scene = self.scene
        ix = min((i for i,w in enumerate(scene._points) if (w.road_id,w.lane_id)==(67,3)),
                 key=lambda i:abs(scene._points[i].s-292.))
        self.lead_event = e = scene._event("slow-traffic",ix)
        e.pass_station = e.s
        e.observed.update(blueprint="vehicle.lincoln.mkz",source_lane=[67,3],passing_lane=[67,2])
        self.event_names.append(e.kind)
        self.clock_index = next(i for i,w in enumerate(scene._points) if (w.road_id,w.lane_id)==(68,3))
        self.clock_start = self.spawn_frame = None
        self.last_occlusion_station = -100.
        self.locations = {
            "slow-traffic": (67,3,292,166,0.,1.90 if case=="later-gap" else 1.65,"vehicle.lincoln.mkz"),
            "adjacent-front": (67,2,288,153,4.,5.8,"vehicle.mini.cooper"),
            "adjacent-rear": (68,2,11,168,5.2 if case=="later-gap" else 4.5,11.5,"vehicle.tesla.model3"),
        }

    def _occlusion(self):
        scene = self.scene
        ego = scene.ego.get_transform()
        eyes = {name:ego.transform(carla.Location(cam.x,cam.y,cam.z)) for name,cam in CAMERAS.items()}
        eyes["chase"] = ego.transform(carla.Location(MOUNT["x"],MOUNT["y"],MOUNT["z"]))
        result = {}
        for role,(road,lane,start,*_) in self.locations.items():
            site = scene._map.get_waypoint_xodr(road,lane,float(start)).transform
            checks = 0
            for eye in eyes.values():
                for x,y in ((-2.5,-1.15),(-2.5,1.15),(2.5,-1.15),(2.5,1.15),(0.,0.)):
                    top = site.transform(carla.Location(x,y,1.95))
                    hits = scene.world.cast_ray(eye,top)
                    if not any(h.label==carla.CityObjectLabel.Buildings and h.location.distance(eye)>5. and
                               h.location.distance(top)>5. for h in hits):
                        return None
                    checks += 1
            result[role] = checks
        return result

    def _arrive(self):
        scene = self.scene
        if not 990. <= scene._last_progress <= 1360.:
            return
        locations = {role:scene._map.get_waypoint_xodr(*values[:3]).transform.location
                     for role,values in self.locations.items()}
        ego = scene.ego.get_location()
        distances = {role:pos.distance(ego) for role,pos in locations.items()}
        if min(distances.values()) < 180.:
            return
        # Both original vehicles physically pass the future arrivals while the
        # ego is elsewhere. Never replace their bodies or introduce a blocker
        # into their right of way.
        convoy = [v for v in scene.ambient._cars if v.road==69]
        actual = [scene._map.get_waypoint(v.actor.get_location()) for v in convoy]
        if len(actual)!=2 or not all(w.road_id==67 and w.s<260. for w in actual):
            return
        nearby = [v.actor.get_location() for v in scene.ambient._cars]
        if any(pos.distance(p)<12. for pos in locations.values() for p in nearby):
            return
        if scene._last_progress-self.last_occlusion_station < 3.:
            return
        self.last_occlusion_station = scene._last_progress
        blocked_rays = self._occlusion()
        if blocked_rays is None:
            return
        for role,(road_id,lane,start,end,release,speed,name) in self.locations.items():
            points = (boulevard_run(scene._map,road_id,lane,float(start),float(end)) if road_id==68 else
                      lane_segment(scene._map,road_id,lane,start,end,carla.LaneType.Driving))
            if not points:raise ValueError(f"Missing native Market Loop path for {role}")
            bp = scene._catalog.find(name)
            bp.set_attribute("role_name",role)
            actor = scene._required(bp,_raised(points[0].transform,.35))
            actor.apply_control(carla.VehicleControl(brake=1.))
            path = [p.transform.location for p in points]
            self.moving[role] = MovingVehicle(role,actor,path,release,speed,lane)
            scene.ambient.monitor_arrival(actor)
        self.lead_event.actor = self.moving["slow-traffic"].actor
        self.spawn_frame = scene.world.get_snapshot().frame
        self.lead_event.observed.update(arrival_world_frame=self.spawn_frame,
            arrival_ego_station_m=round(scene._last_progress,3), building_occluded_rays=blocked_rays,
            actual_convoy_before_arrival=[dict(road=w.road_id,lane=w.lane_id,native_s=round(w.s,2)) for w in actual],
            arrival_distances_to_ego_m={k:round(v,3) for k,v in distances.items()},
            corridor_clock_station_m=round(scene._station[self.clock_index],3))
        if hasattr(self,"_participants"):
            del self._participants

    def before_tick(self):
        scene = self.scene
        if not self.moving:
            self._arrive()
            if not self.moving:
                if scene._last_progress > 1360.:
                    raise RuntimeError("Return traffic could not enter outside the sensor horizon")
                return
        # First published positions independently corroborate the distant spawn.
        if scene.world.get_snapshot().frame > self.spawn_frame and "first_native_arrival_check" not in self.lead_event.observed:
            err={role:round(math.hypot(v.actor.get_location().x-v.path[0].x,v.actor.get_location().y-v.path[0].y),4)
                 for role,v in self.moving.items()}
            if max(err.values())>.30:raise RuntimeError("Native return-traffic placement differs from the committed launch")
            self.lead_event.observed["first_native_arrival_check"] = err
            bodies = [v.actor for v in self.moving.values()]
            nearby = [*scene.ambient._actors, scene.ego, *bodies]
            gaps = [body_gap(footprint(actor),footprint(other)) for actor in bodies for other in nearby
                    if actor.id != other.id and other.is_alive and actor.get_location().distance(other.get_location())<14.]
            self.lead_event.observed["minimum_arrival_body_gap_m"] = round(min(gaps,default=14.),3)
            if min(gaps,default=14.) < 1.:
                raise RuntimeError("New traffic did not physically clear existing participants")
        if self.clock_start is None and scene._last_progress >= scene._station[self.clock_index]:
            self.clock_start = scene._time
            self.lead_event.mark("corridor_clock_started_s", scene._time)
        elapsed = -1. if self.clock_start is None else scene._time-self.clock_start
        for role,v in self.moving.items():
            leaders = [scene.ego]
            if role=="adjacent-rear":
                leaders.append(self.moving["adjacent-front"].actor)
            v.drive(scene,elapsed,leaders)
