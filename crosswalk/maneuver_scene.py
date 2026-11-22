"""Town03 Market maneuver participants and independent, native-body event scoring."""
from dataclasses import dataclass
import math

import carla
import numpy as np
import cv2

from .city_population import CARS, WALKS, CityPopulation, lane_segment
from .city_progress_bus import body_gap, footprint
from .world import _raised


class ManeuverPopulation(CityPopulation):
    # Preserve the Market population except for its short parallel path, now occupied
    # by the independently driven front and rear gap vehicles. Each added runout ends
    # before its own stop line and behind any preceding vehicle's final position.
    walking_corridors = (*WALKS, (67, 1, 53, 76), (76, -1, 131, 157), (3, -1, 70, 90))
    traffic_corridors = tuple(c for c in CARS if c[5] != "parallel") + (
        (24, 1, 38, 15, 1, "opposing"),
        (76, 1, 78, 35, 80, "opposing"),
        (67, 2, 117, 29, 343, "opposing"),
        (67, 3, 281, 152, 498, "opposing"),
        (3, -1, 20, 97, 916, "opposing"),
        (20, 1, 131, 48, 1081, "opposing"),
        (23, 1, 65, 24, 1348, "opposing"),
    )


    def __init__(self, scene):
        super().__init__(scene)
        sensor = scene._catalog.find("sensor.other.collision")
        try:
            for participant in scene.maneuver_fixture.moving.values():
                monitor = scene.world.spawn_actor(sensor, carla.Transform(), attach_to=participant.actor)
                self._contact_sensors.append(monitor)
                monitor.listen(self._contact)
        except BaseException:
            self.close()
            raise

    @property
    def traffic_activity(self):
        report = super().traffic_activity
        report["maneuver_vehicles"] = [
            dict(role=v.role, actor_id=v.actor.id, blueprint=v.actor.type_id, traveled_m=round(v.distance, 3),
                 maximum_step_m=round(v.maximum_step, 3), maximum_lane_center_error_m=round(v.corridor_error, 3))
            for v in self._scene.maneuver_fixture.moving.values()]
        return report


@dataclass
class MovingVehicle:
    role: str
    actor: object
    path: list
    activate_at: float
    speed: float
    lane: int
    last: object = None
    distance: float = 0.
    maximum_step: float = 0.
    corridor_error: float = 0.

    def drive(self, scene, progress, leaders=(), late_clearance=False):
        pose = self.actor.get_location()
        if self.last is not None:
            step = math.hypot(pose.x-self.last.x, pose.y-self.last.y)
            self.maximum_step = max(self.maximum_step, step)
            self.distance += step
        self.last = pose
        index = min(range(len(self.path)), key=lambda i: pose.distance(self.path[i]))
        nearest = self.path[index]
        self.corridor_error = max(self.corridor_error, math.hypot(pose.x-nearest.x, pose.y-nearest.y))
        remaining = len(self.path) - 1 - index
        target = min(self.speed, math.sqrt(2.5 * remaining)) if progress >= self.activate_at and remaining > 2 else 0.
        transform = self.actor.get_transform()
        for leader in leaders:
            delta = leader.get_location() - pose
            ahead = delta.dot(transform.get_forward_vector())
            if not 0. < ahead < 50.:
                continue
            leader_speed = leader.get_velocity().length()
            room = ahead - self.actor.bounding_box.extent.x - leader.bounding_box.extent.x
            if late_clearance:
                # Reproduce the former follower for the uncooperative rear-car control.
                across = abs(delta.dot(transform.get_right_vector()))
                if ahead < 32. and across < self.actor.bounding_box.extent.y + leader.bounding_box.extent.y + .20:
                    target = min(target, max(0., leader_speed + (room-7.)*.45))
                    if room < 3.5:
                        target = 0.
                continue
            native = scene._map.get_waypoint(leader.get_location(), project_to_road=True,
                                             lane_type=carla.LaneType.Driving)
            if native is None or native.road_id != 67:
                continue
            # The leader can straddle a dashed line on a curve. Test its full
            # footprint in the follower lane's frame at the leader's station.
            point = scene._map.get_waypoint_xodr(67, self.lane, native.s)
            axis, origin = point.transform.get_right_vector(), point.transform.location
            sides = (_polygon(leader) - [origin.x, origin.y]) @ [axis.x, axis.y]
            half = self.actor.bounding_box.extent.y + 1.10
            if min(sides) > half or max(sides) < -half:
                continue
            current = self.actor.get_velocity().length()
            desired_gap = max(13., 4.5 + 1.5 * current)
            target = min(target, max(0., leader_speed + (room-desired_gap)*.40))
            if room < 4.:
                target = 0.
        scene._drive_to(self.actor, self.path[min(index + 6, len(self.path) - 1)], target)


def _polygon(actor):
    if not actor.type_id.startswith("static.prop."):
        return footprint(actor)
    # CARLA 1.0 reports the prop box rotated into the world axes already; applying
    # the actor's rotation a second time would rotate the measured footprint twice.
    p = actor.get_location()
    vertices = actor.bounding_box.get_world_vertices(carla.Transform(p))
    return cv2.convexHull(np.asarray([(v.x, v.y) for v in vertices], np.float32)).reshape(-1, 2)


class ManeuverFixture:
    def __init__(self, scene, case):
        self.scene, self.case = scene, case
        self.moving, self.obstacles = {}, []
        self.accelerated_after_pass = False
        self.event_names = []
        self.stations = {round(wp.s): i for i, wp in enumerate(scene._points)
                         if (wp.road_id, wp.lane_id) == (67, -2)}
        if case not in ("continuous", "cones", "board", "slow", "blocked", "no-lead", "departing",
                        "clear-bend", "inside-curb", "outside-curb", "closing-rejoin", "uncooperative-rear", "later-gap"):
            raise ValueError(f"Unsupported maneuver fixture {case}")
        if case in ("continuous", "cones", "board", "uncooperative-rear", "later-gap"):
            self._workzone("board" if case == "board" else "cones")
        if case in ("continuous", "slow", "blocked", "no-lead", "departing", "closing-rejoin", "uncooperative-rear", "later-gap"):
            self._traffic(case)

    def _event(self, kind, station):
        event = self.scene._event(kind, self.stations[station])
        self.event_names.append(kind)
        # Disable the generic stationary-event completion condition. Only an observed
        # rejoin with the entire rear past the target can complete a maneuver.
        event.pass_station = event.s
        return event

    def _workzone(self, kind):
        e = self._event("in-lane-obstruction", 70)
        bp = self.scene._catalog.find("static.prop.streetbarrier" if kind == "board" else "static.prop.trafficcone01")
        sites = [(70, 0.)] if kind == "board" else [(70, -.50), (71, .53), (73, -.10)]
        for station, lateral in sites:
            wp = self.scene._points[self.stations[station]]
            pose = _raised(wp.transform, .045)
            pose.location += pose.get_right_vector() * lateral
            if kind == "board":
                pose.rotation = carla.Rotation(yaw=pose.rotation.yaw + 90.)
            actor = self.scene._required(bp, pose)
            self.obstacles.append((actor, pose.location))
        e.actor = self.obstacles[0][0]
        e.observed.update(blueprint=bp.id, target_count=len(sites), source_lane=[67,-2], passing_lane=[67,-1])

    def _car(self, role, blueprint, lane, start, end, release, speed):
        scene = self.scene
        waypoints = lane_segment(scene._map, 67, lane, start, end, carla.LaneType.Driving)
        if not waypoints:
            raise RuntimeError(f"Disconnected {role} runout")
        bp = scene._catalog.find(blueprint)
        bp.set_attribute("role_name", role)
        actor = scene._required(bp, _raised(waypoints[0].transform,.35))
        actor.apply_control(carla.VehicleControl(brake=1.))
        self.moving[role] = MovingVehicle(role, actor, [w.transform.location for w in waypoints],
                                          scene._station[self.stations[release]], speed, lane)
        return actor

    def _traffic(self, case):
        if case != "no-lead":
            e = self._event("slow-traffic", 149)
            e.actor = self._car("slow-traffic", "vehicle.lincoln.mkz", -2, 149,
                                229, 90 if case=="departing" else 105,
                                8.0 if case=="departing" else 1.90 if case=="later-gap" else 1.65)
            e.observed.update(blueprint=e.actor.type_id, source_lane=[67,-2], passing_lane=[67,-1])
        self._car("adjacent-front", "vehicle.mini.cooper", -1, 166, 293, 111, 7.)
        if case == "blocked":
            self._car("adjacent-rear", "vehicle.tesla.model3", -1, 143, 262, 105, 1.65)
        else:
            # Short control runs start the hero at s94; preserve the closing
            # rear approach with a matching local run-up in both model arms.
            start, activation = (17, 81) if case in ("continuous", "uncooperative-rear", "later-gap") else (72, 94)
            if case == "later-gap":
                activation += 6
            self._car("adjacent-rear", "vehicle.tesla.model3", -1, start, 256, activation, 12.0)

    def before_tick(self):
        if self.case == "closing-rejoin" and not self.accelerated_after_pass:
            lead = self.moving["slow-traffic"]
            pose = lead.actor.get_transform()
            delta = self.scene.ego.get_location() - pose.location
            # A lead notices the passing car level with its front door and
            # accelerates along its own lane. This NPC receives no policy state.
            if -1. <= delta.dot(pose.get_forward_vector()) <= 5. and -4.6 < delta.dot(pose.get_right_vector()) < -2.3:
                self.accelerated_after_pass = True
                lead.speed = 12.
        for role, car in self.moving.items():
            leaders = [self.scene.ego]
            if role == "slow-traffic":
                leaders.extend(e.actor for e in self.scene.events if e.kind == "lead")
            if role == "adjacent-rear":
                leaders.append(self.moving["adjacent-front"].actor)
            car.drive(self.scene, self.scene._last_progress, leaders,
                      late_clearance=self.case == "uncooperative-rear" and role == "adjacent-rear")

    def legal_roadside_pass(self, event):
        """A continued overtake past the roadside boards must still use legal pavement."""
        scene = self.scene
        traffic = next((e for e in scene.events if e.kind == "slow-traffic"), None)
        if traffic is None or "alternate_alongside_s" not in traffic.observed or "returned_to_route_s" in traffic.observed:
            return False
        vertices = scene.ego.bounding_box.get_world_vertices(scene.ego.get_transform())
        lanes = set()
        for vertex in vertices:
            wp = scene._map.get_waypoint(vertex, project_to_road=True, lane_type=carla.LaneType.Driving)
            if not wp or wp.is_junction or (wp.road_id, wp.lane_id) not in ((67, -2), (67, -1)):
                return False
            pose = wp.transform
            if abs((vertex-pose.location).dot(pose.get_right_vector())) > wp.lane_width / 2 + .015:
                return False
            lanes.add(wp.lane_id)
        floor_gap = min(body_gap(footprint(scene.ego), _polygon(actor)) for actor in event.barriers)
        event.observed["overtake_board_minimum_body_gap_m"] = round(min(
            floor_gap, event.observed.get("overtake_board_minimum_body_gap_m", 60.)), 3)
        return -1 in lanes and floor_gap >= .25

    def spectator_actors(self):
        return ([(a, "barrier" if a.type_id.endswith("streetbarrier") else "cone", "in-lane-obstruction")
                 for a, _ in self.obstacles] + [(v.actor, "car", role) for role, v in self.moving.items()])

    def _score_moving_neighbors(self, event, ego_polygon):
        scene = self.scene
        if not hasattr(self, "_participants"):
            people = [(v.actor, v.role) for v in self.moving.values()]
            people += [(a, "background") for a in scene.ambient._actors]
            people += [(e.actor, e.kind) for e in scene.events if e.actor is not None and
                       e.actor.type_id.startswith(("walker.", "vehicle."))]
            self._participants = {a.id: (a, role) for a, role in reversed(people)}.values()
        position = scene.ego.get_location()
        event.observed["moving_neighbor_search_radius_m"] = 65.
        for actor, role in self._participants:
            if not actor.is_alive or actor.get_location().distance(position) > 65.:
                continue
            speed = actor.get_velocity().length()
            if speed <= .5:
                continue
            gap = body_gap(ego_polygon, _polygon(actor))
            observed = event.observed
            if gap < observed.get("minimum_moving_participant_gap_m", 60.):
                observed["minimum_moving_participant_gap_m"] = round(gap, 4)
                observed["nearest_moving_participant"] = dict(frame=scene.world.get_snapshot().frame,
                    actor_id=actor.id, role=role, blueprint=actor.type_id, speed_mps=round(speed, 3),
                    ego_speed_mps=round(scene.ego.get_velocity().length(), 3))
            lane = scene._map.get_waypoint(actor.get_location(), project_to_road=True,
                                           lane_type=carla.LaneType.Driving) if actor.type_id.startswith("vehicle.") else None
            if lane and (lane.road_id, lane.lane_id) == (67, -1):
                key = "minimum_moving_passing_lane_car_gap_m"
                observed[key] = round(min(observed.get(key, 60.), gap), 4)
                if gap < 1.0:
                    event.violation = True

    def observe(self, event, progress):
        scene, o = self.scene, event.observed
        if progress < event.s - 65. or progress > event.s + 125.:
            return 60.
        ego = scene.ego
        wp = scene._map.get_waypoint(ego.get_location(), project_to_road=False, lane_type=carla.LaneType.Driving)
        if wp is None or wp.road_id != 67:
            return 60.
        original = scene._map.get_waypoint_xodr(67,-2,wp.s)
        transform = original.transform
        origin, forward = transform.location, transform.get_forward_vector()
        ego_poly = _polygon(ego)
        right = transform.get_right_vector()
        lateral = (ego.get_location()-origin).dot(right)
        ego_s = (ego_poly - [origin.x, origin.y]) @ [forward.x, forward.y]
        footprint_lateral = (ego_poly - [origin.x, origin.y]) @ [right.x, right.y]
        centered_in_source = (wp.lane_id == -2 and abs(lateral) < .60 and
                              max(abs(footprint_lateral)) < original.lane_width/2)
        moving = event.kind == "slow-traffic"
        targets = [event.actor] if moving else [a for a, _ in self.obstacles]
        polygons = [_polygon(actor) for actor in targets]
        stations = np.concatenate([(poly - [origin.x, origin.y]) @ [forward.x, forward.y] for poly in polygons])
        rear, front = min(stations), max(stations)
        headway = float(rear-max(ego_s))
        if headway > 50.:
            return 60.
        event.mark("triggered_s", scene._time)
        if headway > 0.:
            event.state = "approaching moving traffic" if moving else "approaching lane obstruction"
        if headway > 0.:
            event.mark("before_target_s",scene._time)
            if headway < 30.:
                event.mark("exposed_s", scene._time)
        self._score_moving_neighbors(event, ego_poly)
        distances = [body_gap(ego_poly, poly) for poly in polygons]
        o["minimum_body_clearance_m"] = round(min(o.get("minimum_body_clearance_m", 60.), *distances),3)
        if min(distances) < .25:
            event.violation = True
        if moving:
            target = self.moving["slow-traffic"]
            v = target.actor.get_velocity()
            speed = math.hypot(v.x,v.y)
            if headway > 0. and speed > .65:
                event.mark("moving_target_s",scene._time)
            o["target_traveled_m"] = round(target.distance,3)
            o["target_maximum_step_m"] = round(target.maximum_step,3)
            others = [self.moving[role] for role in ("adjacent-front","adjacent-rear")]
            deltas = [(car.actor.get_location()-ego.get_location()).dot(forward) for car in others]
            speeds = [math.hypot(car.actor.get_velocity().x,car.actor.get_velocity().y) for car in others]
            for other in others:
                gap = body_gap(ego_poly, footprint(other.actor))
                key = other.role.replace("-", "_") + "_minimum_gap_m"
                o[key] = round(min(o.get(key, 100.), gap), 3)
                if gap < .25 or other.maximum_step > 3.:
                    event.violation = True
            if target.maximum_step > 3.:
                event.violation = True
            # The driver can begin following early. The immediate beside/behind
            # envelope of the passing lane determines occupancy; record the measured
            # forward headway independently, within the 60 m control lidar range.
            if 0 < headway <= 35. and speed > .65 and centered_in_source and -18 <= deltas[1] <= 10 and speeds[1] > 1. and speeds[0] > 1.:
                event.state = "beside moving adjacent traffic"
                event.mark("occupied_gap_wait_s",scene._time)
                o["occupied_gap_frames"] = o.get("occupied_gap_frames",0)+1
                ego_speed = ego.get_velocity().length()
                o["minimum_ego_while_occupied_mps"] = round(min(ego_speed, o.get("minimum_ego_while_occupied_mps", 100.)), 3)
                o["stopped_while_occupied_s"] = round(o.get("stopped_while_occupied_s", 0.)
                                                    + (scene._dt if ego_speed <= .30 else 0.), 2)
                o.setdefault("adjacent_gap_witness", dict(rear_relative_m=round(deltas[1],2),
                             front_relative_m=round(deltas[0],2), rear_mps=round(speeds[1],2),
                             front_mps=round(speeds[0],2), source_bumper_gap_m=round(headway,2),
                             moving_lead_mps=round(speed,2), ego_mps=round(ego.get_velocity().length(),2)))
            if "occupied_gap_wait_s" in o and deltas[1] > 9.:
                event.mark("adjacent_traffic_cleared_s",scene._time)
        else:
            drift = max(math.hypot(a.get_location().x - loc.x, a.get_location().y - loc.y) for a,loc in self.obstacles)
            o["maximum_target_displacement_m"] = round(max(o.get("maximum_target_displacement_m",0.),drift),4)
            event.violation |= drift > .12
        if min(ego_s) <= front and max(ego_s) >= rear and wp.lane_id == -1:
            event.state = "alongside in the passing lane"
            event.mark("alternate_alongside_s",scene._time)
            o["alongside_ego_lane"] = [wp.road_id,wp.lane_id]
            o["alongside_minimum_body_gap_m"] = round(min(o.get("alongside_minimum_body_gap_m",60.),*distances),3)
            if moving and speed > .65:
                event.mark("moving_alongside_s",scene._time)
                o.setdefault("alongside_target_mps",round(speed,3))
        if "alternate_alongside_s" in o and min(ego_s) - front > 1.:
            event.state = "rear clear of target"
            event.mark("rear_cleared_target_s",scene._time)
            if centered_in_source:
                event.mark("returned_to_route_s",scene._time)
                event.mark("cleared_s",scene._time)
                o["rejoin_station_m"] = round(progress,3)
                o["rejoin_rear_clear_m"] = round(float(min(ego_s) - front),3)
                event.state = "back in route lane"
        return max(0., min(60., headway)) if headway > 0 and abs(lateral)<1.3 else 60.


def score_sequence(events, trace):
    """Match physical evidence to applied control decisions from the same trial."""
    by_kind = {e["kind"]: e for e in events}
    obstruction = by_kind["in-lane-obstruction"]
    slow = by_kind["slow-traffic"]
    before = obstruction.get("returned_to_route_s")
    wait = slow.get("occupied_gap_wait_s")
    waited_frames = []
    previous = 0
    for row in trace:
        target = next(e for e in row["events"] if e["kind"] == "slow-traffic")
        count = target.get("occupied_gap_frames", 0)
        tactical = row.get("tactical_pass", {})
        if count > previous and tactical.get("maneuver_phase") == "yield" and not tactical.get("committed"):
            waited_frames.append(row["frame"])
        previous = count
    distinct = before is not None and wait is not None and before < wait
    return dict(passed=bool(obstruction["handled"] and slow["handled"] and distinct and waited_frames),
                obstruction_rejoin_precedes_occupied_wait=distinct,
                actual_wait_policy_frames=len(waited_frames),
                first_actual_wait_frame=waited_frames[0] if waited_frames else None,
                last_actual_wait_frame=waited_frames[-1] if waited_frames else None)
