"""Native Town03 roadside barricades and their independent physical clearance score."""
import math

import carla

from .world import _raised


def spawn_barriers(scene):
    """Place two striped boards beyond the right edge of the open driving corridor."""
    candidates = [i for i, w in enumerate(scene._points) if (w.road_id, w.lane_id) == (67, -2)]
    start = min(candidates, key=lambda i: abs(scene._points[i].s - 198.))
    event = scene._event("barrier", start)
    bp = scene._catalog.find("static.prop.streetbarrier")
    for offset, angle in ((0, 90.), (7, 73.)):
        point = scene._points[start + offset]
        pose = _raised(point.transform, .045)
        # The wide striped face is across the line of sight of approaching traffic.
        pose.location = pose.location + pose.get_right_vector() * (point.lane_width / 2 + .85)
        pose.rotation = carla.Rotation(yaw=pose.rotation.yaw + angle)
        actor = scene._required(bp, pose)
        event.barriers.append(actor)
        event.barrier_origins.append(pose.location)
    event.actor = event.barriers[0]
    event.pass_station = scene._station[start + 7] + 5.
    event.observed.update(barrier_count=len(event.barriers), blueprint=bp.id)


def barrier_gap(scene, event, progress):
    if not event.s - 32. <= progress <= event.pass_station + 4.:
        return 60.
    event.mark("triggered_s", scene._time)
    event.state = "approaching striped barricades"
    origin = event.pose.location
    forward, right = event.pose.get_forward_vector(), event.pose.get_right_vector()

    def project(p, axis):
        return (p.x - origin.x) * axis.x + (p.y - origin.y) * axis.y

    ego_pose = scene.ego.get_transform()
    vertices = scene.ego.bounding_box.get_world_vertices(ego_pose)
    front = max(project(v, forward) for v in vertices)
    rear = min(project(v, forward) for v in vertices)
    curbside = max(project(v, right) for v in vertices)
    for index, actor in enumerate(event.barriers):
        if not actor.is_alive:
            event.violation = True
            continue
        pos = actor.get_location()
        old = event.barrier_origins[index]
        drift = math.hypot(pos.x - old.x, pos.y - old.y)
        event.observed["max_barrier_displacement_m"] = round(max(
            drift, event.observed.get("max_barrier_displacement_m", 0.)), 3)
        if drift > .25:
            event.violation = True
        # Native static props in CARLA 1.0 expose world-aligned horizontal extents.
        # Project their enclosing rectangle conservatively onto the lane axes.
        extent = actor.bounding_box.extent
        sf = abs(forward.x) * extent.x + abs(forward.y) * extent.y
        sr = abs(right.x) * extent.x + abs(right.y) * extent.y
        along = project(pos, forward)
        if 0. < along - sf - front <= 24.:
            event.mark("exposed_s", scene._time)
        if rear <= along + sf and front >= along - sf:
            gap = project(pos, right) - sr - curbside
            event.encountered_barriers.add(index)
            event.observed["minimum_barrier_clearance_m"] = round(min(
                gap, event.observed.get("minimum_barrier_clearance_m", math.inf)), 3)
            lane = scene._map.get_waypoint(ego_pose.location, project_to_road=False, lane_type=carla.LaneType.Driving)
            legal = lane and (lane.road_id, lane.lane_id) == (67, -2)
            fixture = getattr(scene, "maneuver_fixture", None)
            if not legal and fixture and fixture.legal_roadside_pass(event):
                legal = True
                event.observed["roadside_pass_during_legal_overtake"] = True
            if gap < .25 or not legal:
                event.violation = True
            event.state = "passing striped barricades"
    event.observed["barriers_alongside"] = len(event.encountered_barriers)
    if progress >= event.pass_station and len(event.encountered_barriers) == len(event.barriers):
        event.mark("all_barriers_passed_s", scene._time)
        event.mark("cleared_s", scene._time)
        event.state = "striped barricades cleared"
    # The scenario grader never asks the driver to brake beside an open corridor.
    return 60.
