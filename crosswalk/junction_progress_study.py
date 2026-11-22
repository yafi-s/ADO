"""Paired native crossing-traffic transfer on a Town03 junction outside the Market itinerary."""
import argparse
import gzip
import importlib.abc
import importlib.util
import io
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import carla
import cv2
import numpy as np


class SealedSource(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Execute the saved driver under its original file paths for relative model/asset lookup."""
    def __init__(self, source):
        with ZipFile(source) as archive:
            self.files = {name: archive.read(name) for name in archive.namelist()}

    def find_spec(self, fullname, path, target=None):
        key = fullname.replace(".", "/") + ".py"
        if key in self.files and key not in {"crosswalk/junction_progress_study.py", "crosswalk/grade_transfer_study.py"}:
            return importlib.util.spec_from_loader(fullname, self)

    def exec_module(self, module):
        key = module.__name__.replace(".", "/") + ".py"
        module.__file__ = str(Path(__file__).resolve().parents[1] / key)
        exec(compile(self.files[key], module.__file__, "exec"), module.__dict__)


sealed = None
if __name__ == "__main__" and "--production-source" in sys.argv:
    initial = argparse.ArgumentParser(add_help=False)
    initial.add_argument("--production-source", type=Path, required=True)
    opts, _ = initial.parse_known_args()
    sealed = SealedSource(opts.production_source)
    sys.meta_path.insert(0, sealed)
elif __name__ == "__main__" and "--reproduce" in sys.argv:
    initial = argparse.ArgumentParser(add_help=False)
    initial.add_argument("--reuse-schedule", type=Path, required=True)
    opts, _ = initial.parse_known_args()
    sealed = SealedSource(opts.reuse_schedule / "source.zip")
    sys.meta_path.insert(0, sealed)

from .city_drive import LoggedDriver, load_driver
from .city_progress_bus import body_gap, footprint
from .drive import DT, episode
from .experiment import PROJECT, fingerprint, save_arrays, write_json, write_trace
from .runtime import producer_info, simulator
from .urban_map import urban_route
from .world import Scenario, _raised, _vehicle_blueprint


@dataclass(frozen=True)
class Site:
    junction: int
    ego: tuple
    traffic: tuple
    npc_speed: float
    approach: float


SITES = {"original": Site(356, (384, -1), (434, 1), 3., 2.5),
         "transfer": Site(1221, (1229, 1), (1334, -2), 3.6, 8.)}
SEED = 881
STATIONS = (-20., -12., -8., -4., 0., 4., 10.)
NEAR_MISS_GAP = 1.0
NEAR_MISS_TIME = 1.0


def xy(p):
    return np.array([p.x, p.y], dtype=float)


def geometry(road_map, site=SITES["original"]):
    used = sorted({p.junction_id for p in urban_route(road_map, SEED, "market").points if p.is_junction})
    if site.junction in used:
        raise ValueError("Test junction is on Market itinerary")
    lanes = {(a.road_id, a.lane_id): (a, b) for a, b in road_map.get_topology()
             if a.is_junction and a.junction_id == site.junction}
    entry, end = lanes[site.ego]
    cross, exit_wp = lanes[site.traffic]
    start, = entry.previous(40.)
    points = [start]
    for _ in range(80):
        choices = points[-1].next(1.)
        if len(choices) > 1:
            choices = [p for p in choices if (p.road_id, p.lane_id) == site.ego]
        point, = choices
        points.append(point)
    if any(p.junction_id != site.junction for p in points if p.is_junction):
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
    if not (3. < u < end.transform.location.distance(entry.transform.location) - 3. and
            3. < v < exit_wp.transform.location.distance(cross.transform.location) - 3.):
        raise ValueError("Crossing point outside both connectors")
    spawn, = cross.previous(site.approach)
    target, = exit_wp.next(12.)
    return points, center, axes, spawn, target, cross, used


class Junction(Scenario):
    """Actor placement and reference geometry are isolated from the production observer."""
    def __init__(self, world, *args, departure=None, site=SITES["original"], stops_short=False, **kwargs):
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
        self.site = site
        self.stops_short = stops_short
        try:
            self.initialize()
        except BaseException:
            self.close()
            raise

    def initialize(self):
        points, self.center, self.axes, cross_spawn, target, entry, self.market = geometry(self._map, self.site)
        self.set_guide(points)
        self.target = (entry if self.stops_short else target).transform.location
        self.spawn_cars(points[0], cross_spawn)
        self.set_lights(points)

    def set_guide(self, points):
        self._set_route([w.transform.location for w in points])
        self.road_id, self.lane_id = points[0].road_id, points[0].lane_id
        self.roads = list(dict.fromkeys(p.road_id for p in points))
        junction_indices = [i for i, p in enumerate(points) if p.is_junction]
        groups = np.split(junction_indices, np.flatnonzero(np.diff(junction_indices) != 1) + 1)
        self._junction_ranges = [tuple(self._coordinates(points[i].transform.location)[0]
                                 for i in (group[0], min(group[-1] + 1, len(points) - 1)))
                                 for group in groups if len(group)]

    def spawn_cars(self, start, other):
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

        self.ego = spawn("vehicle.lincoln.mkz", "vehicle.lincoln.mkz_2020", "hero", start.transform)
        if self.departure is not None:
            self.traffic = spawn("vehicle.mini.cooper", "vehicle.audi.a2", "cross_traffic", other.transform)

    def set_lights(self, points):
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
            self._drive_to(self.traffic, self.target, self.site.npc_speed if self.step >= self.departure else 0.)
        self.step += 1

    def world_row(self):
        ego = footprint(self.ego)
        pose = self.ego.get_transform()
        row = dict(ego_xy=ego.round(4).tolist(), ego_matrix=pose.get_matrix(),
                   ego_station_m=float((xy(pose.location) - self.center) @ self.axes[0]))
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
                    event_cleared=row.get("traffic_station_m", -100.) > 4., route_id=f"{self.site.junction}-through",
                    road_ids=self.roads, junction_ids=[self.site.junction], destination="straight")


class Samples:
    primary_rig_only = True

    def __init__(self, folder, raw, side):
        self.folder = folder
        self.raw = raw
        self.side = side

    def append(self, row, obs):
        step = round(row["sim_time"] / DT)
        if self.raw and 30 <= step <= 200:
            stem = f"{step:04d}-{row['frame']}"
            for name in ("lidar", "radar"):
                buffer = io.BytesIO()
                np.save(buffer, obs[name], allow_pickle=False)
                self.raw.writestr(f"{stem}/{name}.npy", buffer.getvalue())
            for name in ("front", "left", "right"):
                ok, data = cv2.imencode(".jpg", obs[name][:,:,::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise RuntimeError(f"Missing {name} camera exposure")
                self.raw.writestr(f"{stem}/{name}.jpg", data.tobytes())
        if step % 5:
            return
        # Both images are untouched RGB exposures apart from resize for the contact strip.
        panels = [cv2.resize(obs[name], (320, 180)) for name in (self.side, "front")]
        path = self.folder / f"{step:04d}-{self.side}-front.jpg"
        if not cv2.imwrite(str(path), np.hstack(panels)[:, :, ::-1]):
            raise RuntimeError(f"Could not write {path}")


def calibrate(world, folder, site):
    """Measure time to the center under native throttle before either controller sees this scene."""
    scene = Junction(world, departure=0, site=site)
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


def station_speeds(trace, stations=STATIONS):
    """Compare actual speed at identical coordinates, keeping the last applied discrete target."""
    result = {}
    for station in stations:
        for previous, current in zip(trace, trace[1:]):
            start, end = previous["ego_station_m"], current["ego_station_m"]
            if not start < station <= end:
                continue
            weight = (station - start) / (end - start)
            result[str(station)] = dict(
                first_frame=previous["frame"], second_frame=current["frame"],
                time_s=round(previous["sim_time"] + weight * DT, 4),
                actual_kmh=round(3.6 * (previous["speed_mps"] + weight *
                                       (current["speed_mps"] - previous["speed_mps"])), 3),
                applied_target_kmh=round(3.6 * previous["speed_request_mps"], 3),
                next_target_kmh=round(3.6 * current["speed_request_mps"], 3))
            break
    return result


def run(world, checkpoint, folder, departure, site=SITES["original"], stops_short=False, raw=False,
        scene_type=Junction, finish=None, max_steps=320):
    folder.mkdir(parents=True, exist_ok=False)
    images = folder / "cameras"
    images.mkdir()
    driver, names, schema = load_driver(checkpoint)
    model = LoggedDriver(driver)
    scenes = []
    def make(*args, **kwargs):
        scene = scene_type(*args, **kwargs, departure=departure, site=site, stops_short=stops_short)
        scenes.append(scene)
        return scene
    archive = ZipFile(folder / "sensor-sweeps.zip", "x", ZIP_DEFLATED, compresslevel=1) if raw else None
    try:
        summary, trace, _, _ = episode(world, "journey", SEED, schema, model, weather="ClearNoon",
            scene_factory=make, max_steps=max_steps, recording=Samples(images, archive, "left" if site == SITES["original"]
                                                                  else "right"), tactical=True,
            change_request=(lambda: finish(scenes[0])) if finish else None)
    finally:
        if archive is not None:
            archive.close()
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
    summary.update(stations=samples, exact_station_speeds=station_speeds(trace),
                   model_sha256=fingerprint(checkpoint), producer=producer_info(world),
                   center_xy=scene.center.tolist(), axes=[a.tolist() for a in scene.axes], green_heads=scene.ego_heads,
                   market_junctions=scene.market, departure_step=departure, cross_speed_requested_mps=site.npc_speed)
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
    parser.add_argument("--site", choices=(*SITES, "hill"), default="original")
    parser.add_argument("--production-source", type=Path, help="Use archived production modules with current scoring")
    parser.add_argument("--reuse-schedule", type=Path, help="Same-site first attempt providing frozen departure")
    parser.add_argument("--reproduce", action="store_true", help="Reuse the matched candidate clear trace and checkpoint")
    parser.add_argument("--raw", action="store_true", help="Archive 10 Hz RGB, lidar and radar for 3-20 s")
    parser.add_argument("--candidate-only", action="store_true", help="Skip baseline on a follow-up candidate")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoints = {}
    for name, source in (("baseline", args.baseline), ("candidate", args.candidate)):
        checkpoints[name] = args.output / f"{name}.npz"
        shutil.copyfile(source, checkpoints[name])
    snapshot = args.output / "source.zip"
    with ZipFile(snapshot, "x", ZIP_DEFLATED) as archive:
        for path in sorted(Path(__file__).parent.glob("*.py")):
            name = f"crosswalk/{path.name}"
            if sealed and path.name not in (Path(__file__).name, "grade_transfer_study.py") and name in sealed.files:
                archive.writestr(name, sealed.files[name])
            else:
                archive.write(path, name)
    if args.site == "hill":
        from .grade_transfer_study import hill_runs
        return hill_runs(args, checkpoints["candidate"], snapshot)
    site = SITES[args.site]
    anchor = "baseline" if args.site == "original" else "candidate"
    if args.candidate_only:
        checkpoints.pop("baseline")
    write_json(args.output / "protocol.json", dict(seed=SEED, junction=site.junction, ego_connector=site.ego,
               traffic_connector=site.traffic, npc_speed_mps=site.npc_speed, near_miss_gap_m=NEAR_MISS_GAP,
               near_miss_time_s=NEAR_MISS_TIME, policy_checkpoints={k:fingerprint(v) for k,v in checkpoints.items()},
               fixture_sha256=fingerprint(Path(__file__)), execution_source_sha256=fingerprint(snapshot),
               release=f"Shared departure aligned to clear {anchor} center arrival",
               schedule_reference=str(args.reuse_schedule) if args.reuse_schedule else None,
               short_stop_control="Same traffic car, spawn and release, brakes at native connector entry",
               camera_samples="incident side and front every 0.5 simulated seconds, same row time as trace"))
    if args.reproduce and not (args.reuse_schedule and args.candidate_only):
        parser.error("--reproduce requires --reuse-schedule and --candidate-only")
    with simulator(args.port, args.attach, quality="High", map_name="Town03_Opt") as world:
        if args.reuse_schedule:
            previous = json.loads((args.reuse_schedule / "protocol.json").read_text())
            if previous["junction"] != site.junction:
                raise ValueError("Reused schedule is for another geometry")
            timing = json.loads((args.reuse_schedule / "timing.json").read_text())
        else:
            npc_arrival = calibrate(world, args.output, site)
        if args.reproduce:
            if previous["policy_checkpoints"]["candidate"] != fingerprint(checkpoints["candidate"]):
                raise ValueError("Reproduction must use the first attempt checkpoint")
            with gzip.open(args.reuse_schedule / "candidate-clear" / "trace.jsonl.gz", "rt") as stream:
                clear = {"candidate": [json.loads(line) for line in stream]}
        else:
            clear = {arm: run(world, path, args.output / f"{arm}-clear", None, site, raw=args.raw)[0]
                     for arm, path in checkpoints.items()}
        if not args.reuse_schedule:
            if clear[anchor][-1]["ego_station_m"] < 0.:
                raise RuntimeError("Clear arm did not reach conflict center")
            free_arrival = min(enumerate(clear[anchor]), key=lambda i: abs(i[1]["ego_station_m"]))[0]
            timing = dict(anchor=anchor, clear_anchor_center_step=free_arrival, npc_center_step=npc_arrival,
                          departure_step=max(0, free_arrival - npc_arrival))
        write_json(args.output / "timing.json", timing)
        for arm, path in checkpoints.items():
            folder = args.output / f"{arm}-crossing"
            rows, summary = run(world, path, folder, timing["departure_step"], site, raw=args.raw)
            score = grade(clear[arm], rows, summary)
            write_json(folder / "grade.json", score)
            print(arm, score, flush=True)
            if args.site == "transfer" and arm == "candidate":
                folder = args.output / "candidate-stops-short"
                rows, summary = run(world, path, folder, timing["departure_step"], site, stops_short=True, raw=args.raw)
                evidence = grade(clear[arm], rows, summary)
                near_center = [r for r in rows if -12. <= r["ego_station_m"] <= 8.]
                stop_score = dict(min_body_gap_m=evidence["physical_min_body_gap_m"],
                                  counterfactual_min_body_gap_m=evidence["counterfactual_min_body_gap_m"],
                                  traffic_displacement_m=rows[-1]["traffic_station_m"] - rows[0]["traffic_station_m"],
                                  traffic_occupancy=evidence["crossing_body_in_ego_path_s"],
                                  near_center_min_speed_kmh=min((r["speed_mps"] * 3.6 for r in near_center), default=None),
                                  passed=bool(summary["outcome"] == "complete" and not summary["collision_events"]
                                              and not evidence["crossing_body_in_ego_path_s"] and near_center
                                              and all(r["speed_mps"] >= .35 for r in near_center)
                                              and evidence["counterfactual_min_body_gap_m"] >= NEAR_MISS_GAP))
                write_json(folder / "grade.json", stop_score)
                print("stops-short", stop_score, flush=True)


if __name__ == "__main__":
    main()
