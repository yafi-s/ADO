"""Matched CARLA work-zone trials: an occupied lane reopens, or the same barrier stays curbside."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path

import carla
import cv2

from .city_drive import load_driver
from .drive import episode
from .experiment import PROJECT, fingerprint, write_json, write_trace
from .runtime import producer_info, simulator
from .showcase import encode_and_check
from .world import Scenario, _raised


class WorkZone:
    """Own the work-zone fixture and physics scores. Its clearance never enters the policy."""
    road_id, lane_id = 67, -2
    spawn_lateral_m = spawn_yaw_deg = 0.
    ambient = traffic = None
    _junction_ranges = ()
    _set_route = Scenario._set_route
    _coordinates = Scenario._coordinates

    def __init__(self, world, occupied, weather, start_station=45):
        self.world, self.occupied = world, occupied
        self.stack = ExitStack()
        self.step = 0
        self.rows = {}
        self.weather = weather
        road = world.get_map()
        points = [road.get_waypoint_xodr(self.road_id, self.lane_id, float(station))
                  for station in range(start_station, start_station+91)]
        if any(p is None or p.is_junction for p in points):
            raise ValueError("Work zone requires a connected nonjunction lane")
        self._set_route([p.transform.location for p in points])
        self.board_waypoint = points[55]
        self.width = self.board_waypoint.lane_width
        self.board_station = self._coordinates(self.board_waypoint.transform.location)[0]
        # Once the board has moved to this location, its nearest edge is clear of the entire lane.
        self.shoulder = self.width/2 + 1.15
        self.last_offset = None
        try:
            self.stack.callback(world.set_weather, world.get_weather())
            world.set_weather(getattr(carla.WeatherParameters, weather))
            library = world.get_blueprint_library()
            bp = library.find("vehicle.lincoln.mkz")
            bp.set_attribute("role_name", "hero")
            self.ego = world.spawn_actor(bp, _raised(points[0].transform, .35))
            self.stack.callback(self.ego.destroy)
            self.ego.apply_control(carla.VehicleControl(brake=1.))
            self.hazard = world.spawn_actor(library.find("static.prop.streetbarrier"),
                                            self.board_pose(0. if occupied else self.shoulder))
            self.stack.callback(self.hazard.destroy)
            self.hazard.set_simulate_physics(False)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.stack.close()

    def board_pose(self, lateral):
        pose = _raised(self.board_waypoint.transform, .045)
        pose.location += pose.get_right_vector()*lateral
        pose.rotation = carla.Rotation(yaw=pose.rotation.yaw+90.)
        return pose

    def before_tick(self):
        # Scheduled environment movement is identical on every occupied trial, independent of the ego.
        t = self.step*.1
        u = min(1., max(0., (t-17.)/3.))
        offset = self.shoulder*(u*u*(3.-2.*u)) if self.occupied else self.shoulder
        if self.last_offset is not None and offset != self.last_offset:
            self.hazard.set_transform(self.board_pose(offset))
        self.last_offset = offset
        self.step += 1

    def observer_positions(self):
        pose = self.hazard.get_transform()
        extent = self.hazard.bounding_box.extent
        return [dict(actor_id=self.hazard.id, type="barrier", role="workzone_barrier",
                     blueprint=self.hazard.type_id, x=pose.location.x, y=pose.location.y,
                     z=pose.location.z, yaw=pose.rotation.yaw, dimensions_m=[2*extent.x,2*extent.z,2*extent.y])]

    def metrics(self):
        ego_pose = self.ego.get_transform()
        progress, lateral = self._coordinates(ego_pose.location)
        verts = self.ego.bounding_box.get_world_vertices(ego_pose)
        footprint = [self._coordinates(v) for v in verts]
        rear, front = min(p[0] for p in footprint), max(p[0] for p in footprint)
        left, right = min(p[1] for p in footprint), max(p[1] for p in footprint)
        pos = self.hazard.get_location()
        station, offset = self._coordinates(pos)
        axes = self.board_waypoint.transform
        along, side = axes.get_forward_vector(), axes.get_right_vector()
        ext = self.hazard.bounding_box.extent
        half_length = abs(along.x)*ext.x + abs(along.y)*ext.y
        half_width = abs(side.x)*ext.x + abs(side.y)*ext.y
        blocked = offset-half_width <= right and offset+half_width >= left
        gap = station-half_length-front
        crossing = rear <= station+half_length and front >= station-half_length
        phase = "blocked lane" if blocked else "lane reopening" if self.occupied and self.last_offset < self.shoulder else "open lane"
        row = dict(board_m=station, board_lateral_m=offset, board_in_ego_path=blocked,
                   bumper_gap_m=gap, lateral_clearance_m=offset-half_width-right,
                   alongside=crossing, footprint_intersection=bool(blocked and crossing))
        self.rows[self.world.get_snapshot().frame] = row
        return dict(progress_m=progress, route_length_m=self.length, phase=phase,
                    lateral_error_m=lateral, route_departure=abs(lateral)>self.width/2,
                    hazard_gap_m=max(0.,gap) if blocked and rear<station+half_length else 60.,
                    red_light_violations=0, yield_violations=0, event_triggered=front>station-35.,
                    event_cleared=rear>station+half_length, destination="Work zone",
                    route_id="Town03-road67-workzone",road_ids=[self.road_id], junction_ids=[])


class Witness:
    primary_rig_only = True

    def __init__(self, stream, scene):
        self.stream = stream
        self.scene = scene
        self.observations = {}

    def append(self, row, images):
        ok, jpeg = cv2.imencode(".jpg", images["front"][:,:,::-1], [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise RuntimeError(f"Missing camera exposure on frame {row['frame']}")
        self.stream.write(jpeg.tobytes())
        pose = self.scene.ego.get_transform()
        board = self.scene.hazard.get_location()
        dx, dy = board.x-pose.location.x, board.y-pose.location.y
        yaw = math.radians(pose.rotation.yaw)
        bx, by = dx*math.cos(yaw)+dy*math.sin(yaw), -dx*math.sin(yaw)+dy*math.cos(yaw)
        tracks = [t for t in row["tracks"] if t["label"]=="barrier" and not t["coasting"]]
        errors = [math.hypot(t["x"]-bx,t["y"]-by) for t in tracks]
        boxes = row["camera_detections"].get("front",[])
        self.observations[row["frame"]] = dict(barrier_boxes=sum(d["label"]=="barrier" for d in boxes),
            barrier_tracks=len(tracks), nearest_barrier_track_error_m=min(errors,default=None))


def longest_stop(rows):
    longest = current = 0
    for row in rows:
        if row["board_in_ego_path"] and row["bumper_gap_m"]<20 and row["progress_m"]>5 and row["speed_mps"]<=.3:
            current += 1
            longest = max(longest,current)
        else:
            current = 0
    return round(.1*longest,1)


def run(world, model_path, occupied, folder, *, weather="ClearNoon", grip=1.):
    folder.mkdir(parents=True,exist_ok=False)
    model, _, schema = load_driver(model_path)
    holder = []
    def factory(*args, **kwargs):
        scene = WorkZone(args[0], occupied, weather)
        holder.append(scene)
        witness.scene=scene
        return scene
    with (folder/"front.mjpeg").open("wb") as stream:
        witness = Witness(stream,None)
        summary, trace, _, _ = episode(world, "workzone", 241, schema, model, max_steps=450,
            scene_factory=factory, district="Town03",destination="Work zone",surround=True,tactical=False,
            recording=witness, run_id=folder.name, grip=grip)
    scene, = holder
    for row in trace:
        row.update(scene.rows[row["frame"]], **witness.observations[row["frame"]])
    safe = summary["outcome"]=="complete" and summary["collision_events"]==0 and not any(r["footprint_intersection"] for r in trace)
    stopped = longest_stop(trace)
    after = [r["speed_mps"] for r in trace if r["sim_time"]>=20 and not r["board_in_ego_path"]]
    open_approach = [r["speed_mps"] for r in trace if 12<r["progress_m"]<45]
    passing = [r for r in trace if r["alongside"]]
    score = dict(case="closed-then-open" if occupied else "curbside", model_sha256=fingerprint(model_path),
        producer=producer_info(world), sensor_configuration=summary["sensor_configuration"], summary=summary,
        longest_stop_before_blocked_board_s=stopped, max_speed_after_reopening_mps=max(after,default=0.) if occupied else None,
        minimum_stopped_gap_m=min((r["bumper_gap_m"] for r in trace if r["board_in_ego_path"] and
                                  r["progress_m"]>5 and r["speed_mps"]<=.3),default=None),
        minimum_lateral_clearance_at_pass_m=min((r["lateral_clearance_m"] for r in passing),default=None),
        approach_peak_kmh=round(3.6*max(open_approach,default=0.),2),
        unnecessary_stop_during_curbside_approach=not occupied and any(v<.3 for v in open_approach),
        matched_barrier_track_frames=sum(r["nearest_barrier_track_error_m"] is not None and r["nearest_barrier_track_error_m"] < 1. for r in trace),
        passed=bool(safe and (stopped>=1. and max(after,default=0.)>3. if occupied else
                              open_approach and min(open_approach)>.3 and max(open_approach)>5.)))
    write_trace(folder/"trace.jsonl.gz",trace)
    write_json(folder/"result.json",score)
    print(score["case"],summary["outcome"],"stopped",stopped,"matched",score["matched_barrier_track_frames"],"pass",score["passed"],flush=True)
    return score


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port",type=int,default=2968)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--weather",choices=("ClearNoon","WetNoon"),default="ClearNoon")
    p.add_argument("--grip",type=float,choices=(.5,.75,1.),default=1.)
    args=p.parse_args()
    with simulator(args.port,attach=True,map_name="Town03_Opt") as world:
        for case in ("closed-then-open","curbside"):
            run(world,args.model,case=="closed-then-open",args.output/case,weather=args.weather,grip=args.grip)
    for folder in args.output.iterdir():
        if (folder/"result.json").exists():
            count=json.loads((folder/"result.json").read_bytes())["summary"]["frames"]
            encode_and_check(folder/"front.mjpeg",folder/"front.mp4",(640,360),0,count)
