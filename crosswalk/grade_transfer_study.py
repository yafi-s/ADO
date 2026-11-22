"""Independent uphill and stopped-lane obstacle transfer, using native physics and sensors."""
import carla
import numpy as np

from .experiment import fingerprint, write_json
from .junction_progress_study import Junction, Site, run, station_speeds, xy
from .runtime import simulator
from .urban_map import urban_route


HILL = Site(1441, (1468, 1), (75, 1), 0., 0.)
WINDOWS = ((0., 10.), (10., 25.), (25., 40.), (0., 40.))


def hill_geometry(road_map):
    market = urban_route(road_map, 881, "market").points
    market_roads = {p.road_id for p in market}
    market_junctions = sorted({p.junction_id for p in market if p.is_junction})
    start = road_map.get_waypoint_xodr(6, -2, 30.)
    points = [start]
    for _ in range(135):
        last = points[-1]
        choices = last.next(1.)
        if len(choices) > 1:
            choices = [p for p in choices if p.road_id == {6: 1181, 75: 1468}.get(last.road_id)]
        point, = choices
        points.append(point)
        if point.road_id == 74:
            break
    roads = list(dict.fromkeys(p.road_id for p in points))
    if roads != [6, 1181, 75, 1468, 74] or set(roads) & market_roads:
        raise ValueError(f"Expected continuous independent grade, found roads {roads}")
    foot = road_map.get_waypoint_xodr(75, 1, 46.)
    park = road_map.get_waypoint_xodr(75, 1, 21.)
    if not 10. < park.transform.rotation.pitch % 360. < 12.:
        raise ValueError("Obstacle must lie on the native uphill ramp")
    return points, foot, park, market_junctions


class HillScene(Junction):
    def initialize(self):
        points, foot, park, self.market = hill_geometry(self._map)
        self.set_guide(points)
        self.foot_station = self._coordinates(foot.transform.location)[0]
        self.center = xy(foot.transform.location)
        forward = xy(foot.transform.get_forward_vector())
        forward /= np.linalg.norm(forward)
        self.axes = [forward, np.array([-forward[1], forward[0]])]
        self.spawn_cars(points[0], park)
        self.set_lights(points)
        self.hold_steps = 0
        if self.traffic:
            self.traffic.apply_control(carla.VehicleControl(brake=1., hand_brake=True))

    def before_tick(self):
        # Parked car keeps full native collision and suspension; no actor is repositioned.
        self.step += 1

    def world_row(self):
        row = super().world_row()
        row["ego_station_m"] = self._coordinates(self.ego.get_location())[0] - self.foot_station
        if self.traffic:
            pose = self.traffic.get_transform()
            row["parked_grade_station_m"] = self._coordinates(pose.location)[0] - self.foot_station
            row["parked_origin_xyz"] = [pose.location.x, pose.location.y, pose.location.z]
            self.hold_steps = self.hold_steps + 1 if (
                self.ego.get_velocity().length() < .20 and 1. <= row["gap_m"] <= 10.) else 0
        return row

    def metrics(self):
        state = super().metrics()
        state.update(phase="climb with parked car" if self.traffic else "clear climb",
                     route_id="independent-climb", junction_ids=[1168, 1441],
                     event_cleared=False, event_triggered=bool(self.traffic))
        return state


def windows(rows):
    result = []
    for start, end in WINDOWS:
        sample = [r for r in rows if start <= r["ego_station_m"] < end]
        if not sample:
            continue
        actual = np.array([r["speed_mps"] for r in sample]) * 3.6
        targets = np.array([r["speed_request_mps"] for r in sample]) * 3.6
        largest_idx = int(np.argmin(np.diff(targets))) + 1 if len(targets) > 1 else 0
        result.append(dict(start_m=start, end_m=end, frames=len(sample), seconds=round(len(sample) * .1, 1),
                           median_actual_kmh=round(float(np.median(actual)), 3),
                           median_request_kmh=round(float(np.median(targets)), 3),
                           largest_request_drop_kmh=round(float(max(0., targets[largest_idx-1]-targets[largest_idx])),3),
                           drop_frame=sample[largest_idx]["frame"],
                           peak_half_second_decel_mps2=round(float(max(0.,-min(
                                   r.get("longitudinal_accel_mps2", 0.) for r in sample))), 3),
                           below_10_kmh_seconds=round(.1 * int((actual < 10).sum()), 1)))
    return result


def score(rows, summary, blocked):
    result = dict(complete=summary["outcome"] == "complete", contacts=summary["collision_events"],
                  windows=windows(rows), same_station_speeds=station_speeds(rows, (-10., 0., 10., 15., 20., 25., 30., 40.)))
    if not blocked:
        result["passed"] = result["complete"] and not result["contacts"]
        return result
    positions = np.asarray([r["parked_origin_xyz"][:2] for r in rows])
    held = rows[-30:]
    clean = all(r["speed_mps"] < .20 and 1. <= r["gap_m"] <= 10. for r in held)
    min_gap = min(r["gap_m"] for r in rows)
    drift = float(np.linalg.norm(positions - positions[0], axis=1).max())
    result.update(parked_horizontal_drift_m=round(drift,4), min_body_gap_m=round(float(min_gap),4),
                  last_body_gap_m=round(float(rows[-1]["gap_m"]),4), last_speed_mps=rows[-1]["speed_mps"],
                  held_3s=len(held)==30 and clean,
                  final_ego_station_m=rows[-1]["ego_station_m"], observed_obstacle_station_m=rows[-1]["parked_grade_station_m"])
    result["passed"] = bool(result["held_3s"] and not result["contacts"] and drift < .4 and min_gap >= 1.
                            and summary["outcome"] == "restarted")
    return result


def hill_runs(args, checkpoint, snapshot):
    write_json(args.output / "protocol.json", dict(map="Town03_Opt", cases=["clear","parked"],
               weather="ClearNoon",hill_road=75,lane=1,foot_xodr_s=46.,park_xodr_s=21.,
               policy_checkpoint_sha256=fingerprint(checkpoint),execution_source_sha256=fingerprint(snapshot),
               windows_m=WINDOWS,stop_requirements="Complete 30 consecutive frames below 0.20 m/s, body clearance 1–10 m, no contact or lane departure; parked car drift <0.4 m.",
               marks="Confirm clear-speed improvement and a smaller target-speed drop under matched conditions. A clear-speed gain cannot compensate for failing the parked car."))
    with simulator(args.port, args.attach, quality="High", map_name="Town03_Opt") as world:
        for blocked in (False, True):
            folder = args.output / ("parked" if blocked else "clear")
            rows, summary = run(world, checkpoint, folder, 0 if blocked else None, HILL,
                                raw=args.raw, scene_type=HillScene, finish=(lambda s: s.hold_steps >= 30) if blocked else None,
                                max_steps=400)
            grade = score(rows, summary, blocked)
            write_json(folder / "grade.json", grade)
            print(folder, grade, flush=True)
