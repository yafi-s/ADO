"""Native Town03 fixture commissioning. The ego remains stationary throughout."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import queue
from types import SimpleNamespace

import carla

from .journey import JourneyScenario, WALKERS, _offset, _speed, _van_blocks_walker
from .runtime import producer_info, simulator


def _camera(world, actor, pending, cleanup):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    for key, value in (("image_size_x", "480"), ("image_size_y", "320"), ("fov", "52")):
        bp.set_attribute(key, value)
    pose = carla.Transform(carla.Location(-3.5,3.5,1.8),carla.Rotation(pitch=-12,yaw=-45))
    cam = world.spawn_actor(bp,pose,attach_to=actor)
    cleanup.callback(cam.destroy)
    def receive(image):
        try:
            pending.get_nowait()
        except queue.Empty:
            pass
        pending.put_nowait(image)
    cam.listen(receive)
    cleanup.callback(cam.stop)
    return cam


def inspect(world, report_dir, seed=211, photo=False):
    report_dir.mkdir(parents=True, exist_ok=True)
    result_path = report_dir / "native-actor-commission.json"
    if result_path.exists():
        raise FileExistsError(f"Choose a fresh commissioning directory: {report_dir}")
    with ExitStack() as cleanup:
        scene = JourneyScenario(world,"journey",seed,destination="market")
        cleanup.callback(scene.close)
        by = {e.kind:e for e in scene.events}
        bike, bus, sig, left, brake = (by[k] for k in ("cyclist","bus","signal","left","braking"))
        walkers = [e for e in scene.events if e.kind in WALKERS]
        if photo:
            images = queue.Queue(maxsize=1)
            photo_cleanup = cleanup.enter_context(ExitStack())
            _camera(world,bike.actor,images,photo_cleanup)
        # New CARLA actors have no reliable position until the first published snapshot.
        for _ in range(12):
            world.tick()
        scene.metrics()
        if photo:
            image = images.get(timeout=6.)
            image.save_to_disk(str(report_dir / "native-bicycle.png"))
            photo_cleanup.close()
        physical = dict(subject="Native Town03 fixture movements with a stationary ego; no driving-performance score",
                        producer=producer_info(world),seed=seed,route_m=scene.length,actors={},
                        ambient=scene.ambient.counts)
        for e in scene.events:
            physical["actors"][e.kind] = dict(route_m=round(e.s,2), actor=e.actor.type_id if e.actor else None)
        physical["actors"]["signal"].update(native_light=scene.signal.id,
                    native_stop_lanes=[[w.road_id,w.lane_id] for w in scene.signal.get_stop_waypoints() if w],
                    affected_lanes=[[w.road_id,w.lane_id] for w in scene.signal.get_affected_lane_waypoints() if w],
                    queue_vehicle=sig.queue.type_id)
        occluded = by["occluded"]
        eye = SimpleNamespace(get_transform=lambda:scene._points[occluded.index-17].transform)
        geometry = {e.kind:dict(start_side_m=_offset(e.pose.location,e.actor.get_location(),
                                                     e.pose.get_right_vector())) for e in walkers}
        blocked_initial = _van_blocks_walker(eye,scene.occluder,occluded.actor)
        ever_blocked, emerged = blocked_initial,False
        bike_start = bike.actor.get_location()
        peak_bike = peak_brake = measured_decel = peak_bus = 0.
        previous_brake = 0.
        lead = by["lead"]
        lead_exited = False
        native_bike_exit = False
        oncoming_crossed = False
        queue_moved_on_green = False
        held_red = green_seen = False
        queue_start = sig.queue.get_location()
        ego_origin = scene.ego.get_location()
        scene.signal.set_state(carla.TrafficLightState.Red)
        for walker in scene.ambient._walkers:
            walker.start_at = 0.
        for car in scene.ambient._cars:
            car.active = True
        dt = world.get_settings().fixed_delta_seconds
        for step in range(350):
            scene.ambient._move_cars()
            for walker in scene.ambient._walkers:
                walker.update(step*dt,dt)
            if step==35:
                scene.signal.set_state(carla.TrafficLightState.Green)
            green = scene.signal.get_state()==carla.TrafficLightState.Green
            for e in walkers:
                side = _offset(e.pose.location,e.actor.get_location(),e.pose.get_right_vector())
                e.actor.apply_control(carla.WalkerControl(direction=e.direction,
                                                speed=1.25 if side>e.walk_finish_side else 0.))
            scene._follow_exit(lead,8.5)
            scene._follow_exit(bike,3.1)
            scene._follow_exit(brake,3. if step<35 else 0.)
            scene._follow_exit(sig,7.5 if green else 0.)
            scene._drive_to(left.actor,left.target,5.8)
            world.tick()
            lead_lane = scene._map.get_waypoint(lead.actor.get_location(),project_to_road=False,
                                                 lane_type=carla.LaneType.Driving)
            lead_footprint = lead.actor.bounding_box.get_world_vertices(lead.actor.get_transform())
            if lead_lane and (lead_lane.road_id,lead_lane.lane_id) in lead.exit_lanes:
                lead_exited |= min(abs(scene._coordinates(v)[1]) for v in lead_footprint) > scene.ego.bounding_box.extent.y+.75
            actual_bike = _speed(bike.actor)
            actual_brake = _speed(brake.actor)
            peak_bike, peak_brake = max(peak_bike,actual_bike), max(peak_brake,actual_brake)
            if step>=35:
                measured_decel = max(measured_decel,(previous_brake-actual_brake)/dt)
            previous_brake = actual_brake
            peak_bus = max(peak_bus,_speed(bus.actor))
            bike_lane = scene._map.get_waypoint(bike.actor.get_location(),project_to_road=False,
                                                lane_type=carla.LaneType.Driving)
            native_bike_exit |= bool(bike_lane and not bike_lane.is_junction and bike_lane.road_id==bike.exit_road)
            oncoming_crossed |= (_offset(left.conflict,left.actor.get_location(),left.direction)
                                 > left.actor.bounding_box.extent.x+1.)
            queue_moved_on_green |= green and _speed(sig.queue)>.7 and sig.queue.get_location().distance(queue_start)>4.
            held_red |= scene.signal.get_state()==carla.TrafficLightState.Red and _speed(sig.queue)<.25
            green_seen |= green
            blocked = _van_blocks_walker(eye,scene.occluder,occluded.actor)
            ever_blocked |= blocked
            emerged |= ever_blocked and not blocked and geometry["occluded"]["start_side_m"] - _offset(
                occluded.pose.location,occluded.actor.get_location(),occluded.pose.get_right_vector())>.5
            for e in walkers:
                side = _offset(e.pose.location,e.actor.get_location(),e.pose.get_right_vector())
                geometry[e.kind]["minimum_side_m"] = min(geometry[e.kind].get("minimum_side_m",math.inf),side)
        sidewalk_motion = []
        for walker in scene.ambient._walkers:
            pos = walker.actor.get_location()
            distance = math.hypot(pos.x-walker.origin.x,pos.y-walker.origin.y)
            lane = scene._map.get_waypoint(pos,project_to_road=False,lane_type=carla.LaneType.Sidewalk)
            sidewalk_motion.append(dict(distance_m=round(distance,2),on_sidewalk=bool(lane)))
        cones = by["roadworks"]
        cone_clearances = []
        for i, cone in enumerate(cones.cones):
            lane = scene._points[cones.index+i*4]
            front = cone.bounding_box.get_world_vertices(cone.get_transform())
            inside = min(_offset(lane.transform.location,v,lane.transform.get_right_vector()) for v in front)
            cone_clearances.append(round(inside-scene.ego.bounding_box.extent.y,3))
        bus_lane = scene._map.get_waypoint(bus.actor.get_location(),project_to_road=False,lane_type=carla.LaneType.Driving)
        metrics = dict(walker_paths=geometry, van_blocks_fixed_approach=ever_blocked,
                       moving_walker_emerged=emerged, bicycle_peak_mps=peak_bike,
                       stopped_lead_exited=lead_exited,
                       stopped_lead_terminal_error_m=lead.actor.get_location().distance(lead.exit_path[-1]),
                       bicycle_travel_m=bike.actor.get_location().distance(bike_start),
                       bicycle_exit_seen=native_bike_exit, brake_entry_mps=peak_brake,
                       lead_peak_deceleration_mps2=measured_decel, lead_final_mps=previous_brake,
                       bus_peak_mps=peak_bus, bus_lane=[bus_lane.road_id,bus_lane.lane_id] if bus_lane else None,
                       sidewalk_walkers=sidewalk_motion, background_cars=scene.ambient.traffic_activity, cone_count=len(cones.cones), cone_planned_centerline_gaps_m=cone_clearances,
                       oncoming_crossed=oncoming_crossed, queue_held_red=held_red,
                       queue_moved_on_green=queue_moved_on_green, real_green_seen=green_seen,
                       ego_distance_m=scene.ego.get_location().distance(ego_origin))
        checks = dict(walkers=all(v["minimum_side_m"] < -2.5 for v in geometry.values()),
                      occlusion=ever_blocked and emerged, cycle=native_bike_exit and peak_bike>1.,
                      bus=peak_bus<.25 and metrics["bus_lane"]==list(bus.lane_change.source_lane),
                      stopped_lead_runout=lead_exited and lead.actor.get_location().distance(lead.exit_path[-1]) < 4.,
                      lead=peak_brake>1.6 and measured_decel>1. and previous_brake<.25,
                      cone=len(cones.cones)==6 and min(cone_clearances)>.10, oncoming=oncoming_crossed,
                      queue=held_red and green_seen and queue_moved_on_green,
                      sidewalk=sum(w["on_sidewalk"] and w["distance_m"]>=1. for w in sidewalk_motion)>=2,
                      ordinary_traffic=sum(c.traveled>15. and c.peak_speed>1. and not c.departed_lane
                                           for c in scene.ambient._cars)>=2)
        physical.update(measurements=metrics, checks=checks, all_fixture_checks_passed=all(checks.values()))
        result_path.write_text(json.dumps(physical,indent=2)+"\n",encoding="utf8")
        print(json.dumps(dict(checks=checks,metrics=metrics),indent=2),flush=True)
        return all(checks.values())


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=2868)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=211)
    parser.add_argument("--photo",action="store_true")
    args=parser.parse_args()
    with simulator(args.port,attach=True,map_name="Town03_Opt") as active_world:
        ok = inspect(active_world,args.output,args.seed,args.photo)
    raise SystemExit(0 if ok else 1)
