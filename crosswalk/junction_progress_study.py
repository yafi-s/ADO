"""Paired native crossing-traffic transfer on a Town03 junction outside the Market itinerary."""
import argparse
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import carla
import cv2
import numpy as np

from .city_drive import LoggedDriver, load_driver
from .city_progress_bus import body_gap, footprint
from .drive import DT, episode
from .experiment import PROJECT, fingerprint, save_arrays, write_json, write_trace
from .runtime import producer_info, simulator
from .urban_map import urban_route
from .world import Scenario, _raised, _vehicle_blueprint


JUNCTION = 356
EGO_CONNECTOR = (384, -1)
CROSS_CONNECTOR = (434, 1)
CROSS_SPEED = 3.0
SEED = 881
STATIONS = (-20., -12., -8., -4., 0., 4., 10.)
NEAR_MISS_GAP = 1.0
NEAR_MISS_TIME = 1.0


def xy(p):
    return np.array([p.x, p.y], dtype=float)


def geometry(road_map):
    used = sorted({p.junction_id for p in urban_route(road_map, SEED, "market").points if p.is_junction})
    if JUNCTION in used:
        raise ValueError("Test junction is on Market itinerary")
    lanes = {(a.road_id, a.lane_id): (a, b) for a, b in road_map.get_topology()
             if a.is_junction and a.junction_id == JUNCTION}
    entry, end = lanes[EGO_CONNECTOR]
    cross, exit_wp = lanes[CROSS_CONNECTOR]
    start, = entry.previous(40.)
    points = [start]
    for _ in range(80):
        choices = points[-1].next(1.)
        if len(choices) > 1:
            choices = [p for p in choices if (p.road_id, p.lane_id) == EGO_CONNECTOR]
        point, = choices
        points.append(point)
    if any(p.junction_id != JUNCTION for p in points if p.is_junction):
        raise ValueError("Unexpected additional junction")
    axes = []
    for a, b in [(entry, end), (cross, exit_wp)]:
        delta = xy(b.transform.location) - xy(a.transform.location)
        axes.append(delta / np.linalg.norm(delta))
    along, across = axes
    if abs(along @ across) > .06:
        raise ValueError("Movements must cross at nearly a right angle")
    u, v = np.linalg.solve(np.column_stack((along, -across)), xy(cross.transform.location) - xy(entry.transform.location))
    center = xy(entry.transform.location) + along * u
    if not (3. < u < 22. and 3. < v < 22.):
        raise ValueError("Crossing point outside both connectors")
    spawn, = cross.previous(2.5)
    target, = exit_wp.next(12.)
    return points, center, axes, spawn, target, used


class Junction(Scenario):
    """Actor placement and reference geometry are isolated from the production observer."""
    def __init__(self, world, *args, departure=None, **kwargs):
        self.world = world
        self._map = world.get_map()
        self._actors = []
        self._closed = False
        self._signal_states = []
        self._signals_were_frozen = None
        self._previous_weather = world.get_weather()
        self.ambient = self.hazard = self.traffic = None
        self.weather = "ClearNoon"
        self.spawn_lateral_m = self.spawn_yaw_deg = 0.
        self.rows = {}
        self.step = 0
        self.departure = departure
        try:
            self.initialize()
        except BaseException:
            self.close()
            raise

    def initialize(self):
        points, self.center, self.axes, cross_spawn, target, self.market = geometry(self._map)
        self._set_route([w.transform.location for w in points])
        self.road_id, self.lane_id = points[0].road_id, points[0].lane_id
        self.roads = list(dict.fromkeys(p.road_id for p in points))
        junction_indices = [i for i, p in enumerate(points) if p.is_junction]
        self._junction_ranges = [tuple(self._coordinates(points[i].transform.location)[0]
                                       for i in (junction_indices[0], junction_indices[-1] + 1))]
        self.target = target.transform.location
        library = self.world.get_blueprint_library()

        def spawn(preferred, legacy, role, pose):
            bp = _vehicle_blueprint(library, preferred, legacy)
            bp.set_attribute("role_name", role)
            if bp.has_attribute("color"):
                bp.set_attribute("color", "225,225,225" if role == "hero" else "200,45,30")
            actor = self._spawn(bp, _raised(pose, .35))
            if actor is None:
                raise RuntimeError(f"Native spawn failed for {role}")
            actor.apply_control(carla.VehicleControl(brake=1.))
            return actor

        self.ego = spawn("vehicle.lincoln.mkz", "vehicle.lincoln.mkz_2020", "hero", points[0].transform)
        if self.departure is not None:
            self.traffic = spawn("vehicle.mini.cooper", "vehicle.audi.a2", "cross_traffic", cross_spawn.transform)
        lamps = list(self.world.get_actors().filter("traffic.traffic_light"))
        route_lanes = {(p.road_id, p.lane_id) for p in points}
        self.ego_heads = []
        if lamps:
            self._signal_states = [(lamp, lamp.get_state()) for lamp in lamps]
            self._signals_were_frozen = lamps[0].is_frozen()
            self.world.freeze_all_traffic_lights(True)
            for lamp in lamps:
                affected = {(w.road_id, w.lane_id) for w in lamp.get_affected_lane_waypoints() if w}
                governs = bool(affected & route_lanes)
                lamp.set_state(carla.TrafficLightState.Green if governs else carla.TrafficLightState.Red)
                if governs:
                    self.ego_heads.append(lamp.get_opendrive_id())

    def before_tick(self):
        if self.traffic:
            self._drive_to(self.traffic, self.target, CROSS_SPEED if self.step >= self.departure else 0.)
        self.step += 1

    def world_row(self):
        ego = footprint(self.ego)
        row = dict(ego_xy=ego.round(4).tolist(), ego_station_m=float((xy(self.ego.get_location()) - self.center) @ self.axes[0]))
        if self.traffic:
            other = footprint(self.traffic)
            row.update(traffic_xy=other.round(4).tolist(), traffic_station_m=float(
                (xy(self.traffic.get_location()) - self.center) @ self.axes[1]), gap_m=body_gap(ego, other))
        return row

    def metrics(self):
        row = self.rows[self.world.get_snapshot().frame] = self.world_row()
        progress, lateral = self._coordinates(self.ego.get_location())
        return dict(phase="straight through", progress_m=progress, route_length_m=self.length,
                    lateral_error_m=abs(lateral), route_departure=abs(lateral) > 1.8,
                    hazard_gap_m=row.get("gap_m", 100.), red_light_violations=0, yield_violations=0,
                    event_triggered=self.departure is not None and self.step > self.departure,
                    event_cleared=row.get("traffic_station_m", -100.) > 4., route_id="356-westbound",
                    road_ids=self.roads, junction_ids=[JUNCTION], destination="straight")


class Samples:
    primary_rig_only = True

    def __init__(self, folder):
        self.folder = folder

    def append(self, row, obs):
        step = round(row["sim_time"] / DT)
        if step % 5:
            return
        # Both images are untouched RGB exposures apart from resize for the contact strip.
        panels = [cv2.resize(obs[name], (320, 180)) for name in ("left", "front")]
        path = self.folder / f"{step:04d}-left-front.jpg"
        if not cv2.imwrite(str(path), np.hstack(panels)[:, :, ::-1]):
            raise RuntimeError(f"Could not write {path}")


def calibrate(world, folder):
    """Measure time to the center under native throttle before either controller sees this scene."""
    scene = Junction(world, departure=0)
    try:
        for _ in range(8):
            world.tick()
        rows = []
        for i in range(160):
            scene.before_tick()
            world.tick()
            row = scene.world_row()
            rows.append(dict(step=i, **row))
            if row["traffic_station_m"] > 8.:
                break
        if rows[-1]["traffic_station_m"] < 8.:
            raise RuntimeError("Cross traffic did not traverse the reference conflict point")
        write_trace(folder / "unobstructed-npc.jsonl.gz", rows)
        return min(rows, key=lambda r: abs(r["traffic_station_m"]))["step"]
    finally:
        scene.close()


def run(world, checkpoint, folder, departure):
    folder.mkdir(parents=True, exist_ok=False)
    images = folder / "cameras"
    images.mkdir()
    driver, names, schema = load_driver(checkpoint)
    model = LoggedDriver(driver)
    scenes = []
    def make(*args, **kwargs):
        scene = Junction(*args, **kwargs, departure=departure)
        scenes.append(scene)
        return scene
    summary, trace, _, _ = episode(world, "journey", SEED, schema, model, weather="ClearNoon",
        scene_factory=make, max_steps=320, recording=Samples(images), tactical=True)
    scene = scenes[0]
    for row in trace:
        row.update(scene.rows[row["frame"]])
    write_trace(folder / "trace.jsonl.gz", trace)
    save_arrays(folder / "observations.npz", frame=[r["frame"] for r in trace], features=model.observations,
                feature_names=names, requested_mps=model.requested_speed, posted_kmh=model.posted_limits)
    samples = {}
    for s in STATIONS:
        if trace[0]["ego_station_m"] <= s <= trace[-1]["ego_station_m"]:
            row = min(trace, key=lambda r: abs(r["ego_station_m"] - s))
            samples[str(s)] = {k: row[k] for k in ("ego_station_m", "frame", "sim_time", "speed_mps", "speed_request_mps")}
    summary.update(stations=samples, model_sha256=fingerprint(checkpoint), producer=producer_info(world),
                   center_xy=scene.center.tolist(), axes=[a.tolist() for a in scene.axes], green_heads=scene.ego_heads,
                   market_junctions=scene.market, departure_step=departure, cross_speed_requested_mps=CROSS_SPEED)
    write_json(folder / "summary.json", summary)
    print(folder, summary["outcome"], summary["duration_s"], flush=True)
    return trace, summary


def rect(row, field):
    return np.asarray(row[field], dtype=np.float32)


def grade(clear, traffic, summary):
    # Replay the clear ego against the measured threat at the same elapsed tick. Any frame
    # with overlapping world footprints is direct evidence that the clear trajectory is unsafe.
    pairs = [(a,b) for a,b in zip(clear, traffic) if a["sim_time"] == b["sim_time"]]
    if len(pairs) != min(len(clear), len(traffic)):
        raise ValueError("Paired timestamps do not match")
    counter = [body_gap(rect(a, "ego_xy"), rect(b, "traffic_xy")) for a,b in pairs]
    center = np.asarray(summary["center_xy"])
    forward, across = np.asarray(summary["axes"])
    # Use the physical cross-lane body widths from the actual actors, projected perpendicular
    # to their respective travel axes. Gates are fixed strips through the lane-center crossing.
    def perpendicular(axis):
        return np.array([-axis[1], axis[0]])
    def width(key, axis):
        return np.ptp(rect(traffic[0], key) @ perpendicular(axis)) / 2.
    ego_half, cross_half = width("ego_xy", forward), width("traffic_xy", across)
    def occupies(row, key, normal, half):
        projected = (rect(row, key) - center) @ normal
        return min(projected) <= half and max(projected) >= -half
    cross_times = [r["sim_time"] for r in traffic if occupies(r, "traffic_xy", perpendicular(forward), ego_half)]
    ego_times = [r["sim_time"] for r in traffic if occupies(r, "ego_xy", perpendicular(across), cross_half)]
    has_exited = traffic[-1]["traffic_station_m"] > 4.
    pet = None
    order = "incomplete"
    if ego_times and cross_times and has_exited:
        # Each occupancy edge can fall between samples. Subtract one full tick at
        # each edge to give a conservative lower bound on the temporal clearance.
        if cross_times[-1] < ego_times[0]:
            order = "traffic_first"
            pet = round(max(0., ego_times[0] - cross_times[-1] - 2 * DT), 2)
        elif ego_times[-1] < cross_times[0]:
            order = "ego_first"
            pet = round(max(0., cross_times[0] - ego_times[-1] - 2 * DT), 2)
        else:
            order, pet = "overlap", 0.
    overlaps = [pairs[i][0]["sim_time"] for i, gap in enumerate(counter) if gap < .001]
    near = min(r["gap_m"] for r in traffic)
    clean = summary["outcome"] == "complete" and not summary["collision_events"]
    return dict(counterfactual_min_body_gap_m=min(counter), counterfactual_overlap_s=overlaps,
                physical_min_body_gap_m=near, collision_events=summary["collision_events"],
                crossing_body_in_ego_path_s=cross_times and [cross_times[0], cross_times[-1]],
                ego_body_in_cross_path_s=ego_times and [ego_times[0], ego_times[-1]],
                crossing_order=order, post_encroachment_s=pet,
                near_miss=bool(near < NEAR_MISS_GAP or (pet is not None and pet < NEAR_MISS_TIME)),
                passed=bool(overlaps and clean and near >= NEAR_MISS_GAP and order == "traffic_first"
                            and pet >= NEAR_MISS_TIME))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=PROJECT / "models/city-clearance/candidate.npz")
    parser.add_argument("--candidate", type=Path, default=PROJECT / "models/city-progress/candidate.npz")
    parser.add_argument("--port", type=int, default=3528)
    parser.add_argument("--attach", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoints = {}
    for name, source in (("baseline", args.baseline), ("candidate", args.candidate)):
        checkpoints[name] = args.output / f"{name}.npz"
        shutil.copyfile(source, checkpoints[name])
    snapshot = args.output / "source.zip"
    with ZipFile(snapshot, "x", ZIP_DEFLATED) as archive:
        for path in sorted(Path(__file__).parent.glob("*.py")):
            archive.write(path, f"crosswalk/{path.name}")
    write_json(args.output / "protocol.json", dict(seed=SEED, junction=JUNCTION, ego_connector=EGO_CONNECTOR,
               traffic_connector=CROSS_CONNECTOR, near_miss_gap_m=NEAR_MISS_GAP, near_miss_time_s=NEAR_MISS_TIME,
               policy_checkpoints={k:fingerprint(v) for k,v in checkpoints.items()},
               fixture_sha256=fingerprint(Path(__file__)), execution_source_sha256=fingerprint(snapshot),
               release="Shared NPC departure aligned to clear baseline center arrival",
               camera_samples="left and front every 0.5 simulated seconds, same row time as trace"))
    with simulator(args.port, args.attach, quality="High", map_name="Town03_Opt") as world:
        npc_arrival_step = calibrate(world, args.output)
        clear = {arm: run(world, path, args.output / f"{arm}-clear", None)[0] for arm,path in checkpoints.items()}
        if clear["baseline"][-1]["ego_station_m"] < 0.:
            raise RuntimeError("Clear baseline did not reach conflict center; cannot anchor cross traffic")
        free_arrival = min(enumerate(clear["baseline"]), key=lambda i: abs(i[1]["ego_station_m"]))[0]
        departure = max(0, free_arrival - npc_arrival_step)
        write_json(args.output / "timing.json", dict(clear_baseline_center_step=free_arrival,
                   npc_center_step=npc_arrival_step, departure_step=departure))
        for arm, path in checkpoints.items():
            folder = args.output / f"{arm}-crossing"
            rows, summary = run(world, path, folder, departure)
            score = grade(clear[arm], rows, summary)
            write_json(folder / "grade.json", score)
            print(arm, score, flush=True)


if __name__ == "__main__":
    main()
