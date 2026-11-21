"""Independent stopped-bus crossing fixture. No evaluation state is passed to the driver."""
import math

import carla
import cv2
import numpy as np

from .journey import JourneyScenario, _offset, _van_blocks_walker


def footprint(actor):
    vertices = actor.bounding_box.get_world_vertices(actor.get_transform())
    xy = np.asarray([(p.x, p.y) for p in vertices], dtype=np.float32)
    return cv2.convexHull(xy).reshape(-1, 2)


def body_gap(a, b):
    if cv2.intersectConvexConvex(a, b)[0] > 0:
        return 0.
    return min(abs(cv2.pointPolygonTest(contour, tuple(map(float, xy)), True))
               for contour, points in ((a, b), (b, a)) for xy in points)


class BusCrossing(JourneyScenario):
    """A walker steps from the far end of the stopped bus into the passing lane."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.bus = next(e.actor for e in self.events if e.kind == "bus")
            base = self._urban.lane_change.bus_pose
            self.origin = base.location
            self.forward, self.right = base.get_forward_vector(), base.get_right_vector()
            self.passing_lane_width = self._map.get_waypoint_xodr(4, 2, 23.).lane_width
            nose = max(_offset(self.origin, p, self.forward)
                       for p in self.bus.bounding_box.get_world_vertices(base))
            self.cross_station = nose + 1.6
            if not 2. < self.cross_station < 13.:
                raise ValueError("Crossing station must be immediately beyond the native bus nose")
            loc = self.origin + self.forward * self.cross_station
            loc.z = base.location.z + 1.
            bp = sorted(self._catalog.filter("walker.pedestrian.*"), key=lambda entry: entry.id)[0]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            self.crossing_person = self._required(bp, carla.Transform(loc, carla.Rotation(yaw=base.rotation.yaw - 90.)))
            self.across = carla.Vector3D(-self.right.x, -self.right.y, 0.)
            self.crossing_started = False
            self.walk_rows = {}
        except BaseException:
            self.close()
            raise

    def before_tick(self):
        super().before_tick()
        corners = self.ego.bounding_box.get_world_vertices(self.ego.get_transform())
        distance = self.cross_station - max(_offset(self.origin, v, self.forward) for v in corners)
        self.crossing_started |= distance <= 8.
        side = _offset(self.origin, self.crossing_person.get_location(), self.right)
        speed = 1.25 if self.crossing_started and side > -7. else 0.
        self.crossing_person.apply_control(carla.WalkerControl(direction=self.across, speed=speed))

    def metrics(self):
        state = super().metrics()
        frame = self.world.get_snapshot().frame
        ego = self.ego.bounding_box.get_world_vertices(self.ego.get_transform())
        person = self.crossing_person
        walker = person.bounding_box.get_world_vertices(person.get_transform())
        ego_front = max(_offset(self.origin, p, self.forward) for p in ego)
        near_crossing = abs(self.cross_station - ego_front) < 40.
        wl = [_offset(self.origin, p, self.right) for p in walker]
        # Fixed, native passing-lane center and body envelope. Include the person's full box.
        lane_width = self.passing_lane_width
        lane_left = -lane_width - self.ego.bounding_box.extent.y
        lane_right = -lane_width + self.ego.bounding_box.extent.y
        row = dict(near_crossing=near_crossing, walker_started=self.crossing_started,
                   walker_in_passing_path=min(wl) <= lane_right and max(wl) >= lane_left,
                   walker_cleared_passing_path=max(wl) < lane_left - .25,
                   bus_hides_walker=_van_blocks_walker(self.ego, self.bus, person) if near_crossing else None,
                   front_to_crossing_m=round(self.cross_station - ego_front, 3),
                   walker_lateral_m=round(_offset(self.origin, person.get_location(), self.right), 3),
                   person_speed_mps=round(math.hypot(person.get_velocity().x, person.get_velocity().y), 3))
        if near_crossing:
            row["person_body_gap_m"] = round(body_gap(footprint(self.ego), footprint(person)), 3)
        self.walk_rows[frame] = row
        return state

    def observer_positions(self, radius=100.):
        rows = super().observer_positions(radius)
        person = self.crossing_person
        pose = person.get_transform()
        box = person.bounding_box
        if self.ego.get_location().distance(pose.location) <= radius:
            rows.append(dict(actor_id=person.id, type="pedestrian", role="bus_crossing",
                blueprint=person.type_id, x=pose.location.x, y=pose.location.y, z=pose.location.z,
                yaw=pose.rotation.yaw, dimensions_m=[2 * box.extent.x, 2 * box.extent.z, 2 * box.extent.y]))
        return rows


def crossing_score(trace):
    moving = next((r for r in trace if r["person_speed_mps"] > .5), None)
    hidden = next((r for r in trace if moving and r["frame"] <= moving["frame"] and r["bus_hides_walker"]), None)
    visible = next((r for r in trace if hidden and r["frame"] > hidden["frame"] and
                    r["walker_started"] and r["bus_hides_walker"] is False), None)
    occupied = [r for r in trace if r["walker_started"] and r["walker_in_passing_path"]]
    clear = next((r for r in trace if r["walker_cleared_passing_path"]), None)
    passed = next((r for r in trace if r["front_to_crossing_m"] <= 0.), None)
    gap = min((r["person_body_gap_m"] for r in trace if "person_body_gap_m" in r), default=None)
    return dict(hidden_frame=hidden and hidden["frame"], moving_frame=moving and moving["frame"],
                emerged_frame=visible and visible["frame"],
                ego_speed_at_emergence_kmh=visible and round(3.6 * visible["speed_mps"], 2),
                front_distance_at_emergence_m=visible and visible["front_to_crossing_m"],
                path_occupied_frames=len(occupied), clear_frame=clear and clear["frame"],
                front_at_crossing_frame=passed and passed["frame"],
                minimum_body_gap_m=gap,
                minimum_speed_while_occupied_kmh=min((3.6 * r["speed_mps"] for r in occupied), default=None),
                passed=bool(hidden and moving and visible and occupied and clear and passed and gap and gap > 0. and
                            clear["frame"] < passed["frame"] and not trace[-1]["collision_count"]))
