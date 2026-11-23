"""A continuous Town03 city route sampled from the native OpenDRIVE driving graph."""
from dataclasses import dataclass
import math
import random

import carla


# Every fork names the junction connector reached by Waypoint.next, including
# the straight-through movements. The unlisted short roads have one successor.
_FORKS = {24: 139, 80: 1059, 47: 1363, 67: 100, 69: 271, 4: 874, 3: 831, 20: 1829}
_MARKET_LOOP_LANES = ((23, -1), (332, -1), (69, 3), (1960, 3), (68, 3), (101, 3), (67, 3))
_MARKET_LOOP_END_S = 175.
# Road, lane, OpenDRIVE s. All walkers start beside a parking lane, and all
# three moving vehicles can turn off the route at the next physical junction.
_SITES = {
    "pedestrian": (24, -1, 31.0),
    "occluded": (76, -1, 67.0),
    "roadworks": (67, -2, 23.0),
    "lead": (67, -2, 238.0),
    "braking": (3, 2, 68.0),
    "cyclist": (20, -1, 122.0),
    "pedestrian-delayed": (21, -1, 24.0),
}


def _lane(point):
    return point.road_id, point.lane_id


def _angle(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


def _station_index(points, road, lane, station):
    candidates = (i for i, wp in enumerate(points) if _lane(wp) == (road, lane))
    try:
        return min(candidates, key=lambda i: abs(points[i].s - station))
    except ValueError as exc:
        raise RuntimeError(f"Town03 route did not reach road {road}, lane {lane}") from exc


@dataclass(frozen=True)
class LaneChange:
    start: int
    end: int
    bus: int
    source_lane: tuple
    target_lane: tuple
    bus_pose: object


class _TransitionSample:
    """Native lane metadata with the tangent and location on the driving path."""
    def __init__(self, native, pose):
        self._native = native
        self.transform = pose

    def __getattr__(self, name):
        return getattr(self._native, name)


def _left_lane(source):
    target = source.get_left_lane()
    if (source.is_junction or target is None or target.is_junction
            or target.lane_type != carla.LaneType.Driving
            or source.left_lane_marking.type != carla.LaneMarkingType.Broken
            or source.lane_change not in (carla.LaneChange.Left, carla.LaneChange.Both)
            or source.road_id != target.road_id or source.lane_id * target.lane_id <= 0
            or target.get_right_lane() is None or _lane(target.get_right_lane()) != _lane(source)
            or min(source.lane_width, target.lane_width) < 3.4
            or abs(_angle(target.transform.rotation.yaw - source.transform.rotation.yaw)) > 2.0):
        raise RuntimeError("Town03 bus pass requires two adjacent same-direction driving lanes and a broken line")
    return target


def _transition(source, target, travel, length):
    # Quintic blend with zero first and second derivatives at both ends.
    # Road 4's parallel lane centers are straight and separated by 3.5 m.
    t = min(1.0, max(0.0, travel / length))
    alpha = t ** 3 * (10.0 - 15.0 * t + 6.0 * t ** 2)
    derivative = 30.0 * t ** 2 * (1.0 - t) ** 2 / length
    original = source.transform
    delta = target.transform.location - original.location
    pos = original.location + delta * alpha
    forward = original.get_forward_vector()
    yaw = math.degrees(math.atan2(forward.y + derivative * delta.y,
                                  forward.x + derivative * delta.x))
    pose = carla.Transform(pos, carla.Rotation(pitch=original.rotation.pitch, yaw=yaw,
                                               roll=original.rotation.roll))
    return _TransitionSample(source if alpha < 0.5 else target, pose)


@dataclass
class UrbanRoute:
    points: list
    sites: dict
    lane_change: LaneChange
    left_turn: tuple
    oncoming: tuple
    destination: str = "market"
    market_prefix_count: int | None = None

    def signal_site(self, signals):
        """Bind an actor's actual stop waypoint AND affected lane to the driven approach."""
        stops = []
        for signal in signals:
            affected = {_lane(p) for p in signal.get_affected_lane_waypoints() if p}
            stops.extend((signal, stop, affected) for stop in signal.get_stop_waypoints() if stop)
        candidates = []
        # A queue at the end of the long lead's block would obstruct its exit.
        # Prefer the signalled junction where the residential street joins the boulevard.
        preference = {1352: 0, 1038: 1, 238: 2, 103: 3}
        occupied = [*self.sites.values(), self.lane_change.start, self.lane_change.bus, self.left_turn[0]]
        point_count = self.market_prefix_count or len(self.points)
        for start in range(20, point_count - 2):
            if self.points[start - 1].is_junction or not self.points[start].is_junction:
                continue
            jid = self.points[start].junction_id
            if jid not in preference:
                continue
            end = start
            while end < point_count and self.points[end].junction_id == jid:
                end += 1
            lanes = {_lane(p) for p in self.points[start - 4:end]}
            for signal, stop, affected in stops:
                if not affected.intersection(lanes):
                    continue
                for i in range(start - 18, min(start + 7, end)):
                    sample = self.points[i]
                    if (_lane(sample) != _lane(stop)
                            or min(abs(i - event) for event in occupied) < 28):
                        continue
                    distance = sample.transform.location.distance(stop.transform.location)
                    if distance <= 1.2:
                        candidates.append((preference[jid], distance, signal.id, i, signal, stop))
        if not candidates:
            raise RuntimeError("No native signal stop waypoint and affected lane match the planned Town03 approaches")
        _, _, _, index, signal, stop = min(candidates, key=lambda c: c[:4])
        return index, signal, stop


def _extend_market_loop(points):
    native = points[-1]
    if _lane(native) != _MARKET_LOOP_LANES[0]:
        raise RuntimeError("Market Loop requires the original Market endpoint on road 23")
    segment = 0
    for _ in range(400):
        permitted = _MARKET_LOOP_LANES[segment:segment + 2]
        choices = [point for point in native.next(1.) if _lane(point) in permitted]
        if len(choices) != 1:
            raise RuntimeError(f"Market Loop disconnected at road {native.road_id}, lane {native.lane_id}")
        successor = choices[0]
        if _lane(successor) != _lane(native):
            segment += 1
        points.append(successor)
        native = successor
        if segment == len(_MARKET_LOOP_LANES) - 1 and native.s <= _MARKET_LOOP_END_S:
            return
    raise RuntimeError("Market Loop endpoint unreachable")


def urban_route(road_map, seed=0, destination=None):
    """Follow a reproducible native city itinerary with one legal left-lane bus pass."""
    if destination not in (None, "market", "east", "market-loop"):
        raise ValueError("Town03 city journey destination must be market or market-loop")
    if road_map.name.split("/")[-1] not in ("Town03", "Town03_Opt"):
        raise ValueError("Urban journey requires Town03_Opt")
    origin = random.Random(int(seed)).uniform(1.0, 4.0)
    native = road_map.get_waypoint_xodr(24, -1, origin)
    if native is None:
        raise RuntimeError("Missing native Town03 residential departure lane")
    points = [native]
    start = end = start_s = None
    for _ in range(1800):
        options = native.next(1.0)
        if len(options) > 1:
            options = [wp for wp in options if wp.road_id == _FORKS.get(native.road_id)]
        if len(options) != 1:
            raise RuntimeError(f"Town03 journey disconnected at road {native.road_id}, lane {native.lane_id}")
        native = point = options[0]
        if _lane(native) == (4, 3) and native.s <= 80.0:
            target = _left_lane(native)
            if start is None:
                start, start_s = len(points), native.s
            if start_s - native.s >= 34.0 - 1e-5:
                native = point = target
                end = len(points)
            else:
                point = _transition(native, target, start_s - native.s, 34.0)
        points.append(point)
        if native.road_id == 23 and native.s >= 54.0:
            break
    else:
        raise RuntimeError("Town03 journey endpoint unreachable")
    if start is None or end is None:
        raise RuntimeError("Town03 journey did not complete the boulevard lane change")

    bus = _station_index(points, 4, 2, 23.0)
    # Keep the physical bus in the original curb lane, aligned at exactly the
    # same longitudinal station as the sample in the passing lane.
    bus_wp = road_map.get_waypoint_xodr(4, 3, points[bus].s)
    if _lane(_left_lane(bus_wp)) != _lane(points[bus]):
        raise RuntimeError("Bus and completed lane change do not share the planned boulevard")
    sites = {name: _station_index(points, *site) for name, site in _SITES.items()}
    left_start = next(i for i, p in enumerate(points) if p.road_id == 831)
    left_end = next(i for i in range(left_start, len(points)) if points[i].road_id == 20)
    # Road 2's through lane crosses connector 831 on connector 743, then
    # enters road 3 opposite the ego. Both endpoints are on that same movement.
    far = road_map.get_waypoint_xodr(2, -1, 90.0)
    near = road_map.get_waypoint_xodr(3, -1, 22.0)
    if near is None or far is None:
        raise RuntimeError("Missing native opposing through movement")
    route = UrbanRoute(points, sites, LaneChange(start, end, bus, (4, 3), (4, 2), bus_wp.transform),
                       (left_start, left_end), (near, far))
    if destination == "market-loop":
        route.market_prefix_count = len(points)
        route.destination = destination
        _extend_market_loop(points)
    return route
