"""Run matched native Town03 maneuver trials and retain synchronized scoring evidence."""

import argparse
from pathlib import Path
import shutil
import traceback
from zipfile import ZIP_DEFLATED, ZipFile

import carla
import cv2
import numpy as np

from .city_drive import LoggedDriver, load_driver
from .city_progress_bus import body_gap
from .drive import episode
from .experiment import PROJECT, fingerprint, save_arrays, write_json, write_trace
from .journey import JourneyScenario
from .maneuver_scene import _polygon
from .maneuver_trial_grade import ADJACENT_CASES, evaluation_active, grade_case, native_body
from .overtake import MODEL as TACTICAL_MODEL
from .runtime import producer_info, simulator
from .showcase import encode_and_check
from .workzone_study import WorkZone, longest_stop
from .world import _raised


SEED = 212
WEATHER = "ClearNoon"
BEND = (67, -2, 48., 145.)
TRAFFIC = (67, -2, 94., 231.)
OPEN_TRAFFIC = (67, -2, 94., 185.)  # Finish before the unrelated original stopped lead enters pass range.
PERSON = (76, -1, 43., 112.)
REOPENED = (76, -1, 47., 133.)
REAR_LAUNCH_ROUTE_M = 278.
WINDOWS = {
    "cones": BEND, "board": BEND, "slow": TRAFFIC, "blocked": TRAFFIC,
    "no-lead": OPEN_TRAFFIC, "departing": OPEN_TRAFFIC, "clear-bend": BEND,
    "inside-curb": BEND, "outside-curb": BEND, "person": PERSON,
    "reopened": REOPENED, "closing-rejoin": TRAFFIC, "uncooperative-rear": (67, -2, 6., 166.),
}
CASES = tuple(WINDOWS)
CURB_CASES = ("inside-curb", "outside-curb")


def lateral(origin, point, axis):
    return (point.x - origin.x) * axis.x + (point.y - origin.y) * axis.y


def route_pose_at(scene, station):
    """Interpolate the exact departure along the same route polyline the driver uses."""
    index = int(np.searchsorted(scene._station, station, side="right"))
    left, right = (scene._points[i].transform for i in (index - 1, index))
    fraction = (station - scene._station[index - 1]) / (scene._station[index] - scene._station[index - 1])
    location = left.location + (right.location - left.location) * fraction
    yaw_delta = (right.rotation.yaw - left.rotation.yaw + 180.) % 360. - 180.
    rotation = carla.Rotation(
        yaw=left.rotation.yaw + fraction * yaw_delta,
        pitch=left.rotation.pitch + fraction * (right.rotation.pitch - left.rotation.pitch),
        roll=left.rotation.roll + fraction * (right.rotation.roll - left.rotation.roll))
    return carla.Transform(location, rotation)


def add_curb_board(scene, case):
    """Bracket the native sidewalk boundary using one physical prop and heading."""
    source = scene._map.get_waypoint_xodr(67, -2, 84.)
    curb = source.get_right_lane()
    while curb is not None and curb.lane_type == carla.LaneType.Shoulder:
        curb = curb.get_right_lane()
    if curb is None or curb.lane_type != carla.LaneType.Sidewalk:
        raise ValueError("Road 67 fixture requires a sidewalk beside the native shoulder")
    pose = _raised(source.transform, .045)
    # Retain independent coordinates before relocating the prospective spawn pose.
    origin = carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z)
    right = source.transform.get_right_vector()
    curb_pose = curb.transform
    curb_edge = curb_pose.location - curb_pose.get_right_vector() * (curb.lane_width / 2)
    edge = lateral(origin, curb_edge, right)
    if edge < source.lane_width / 2:
        raise ValueError("Road 67 sidewalk boundary overlaps the driving lane")
    pose.location = origin + right * edge
    pose.rotation = carla.Rotation(yaw=pose.rotation.yaw + 90.)
    board = scene._required(scene._catalog.find("static.prop.streetbarrier"), pose)
    board.set_simulate_physics(False)
    # The spawned prop's transform cache can be empty until the first world tick.
    # Use its native box at a zero translation to get only world-oriented offsets,
    # then position those offsets at the surveyed curb. Never infer this starting
    # offset from the unsynchronized actor location.
    relative = board.bounding_box.get_world_vertices(carla.Transform())
    borders = [point.x * right.x + point.y * right.y for point in relative]
    center = edge - max(borders) - .15 if case == "inside-curb" else edge - min(borders) + .15
    pose.location = origin + right * center
    board.set_transform(pose)
    return board, dict(board_native_s=84., driving_edge_m=source.lane_width / 2,
                       sidewalk_edge_m=edge, board_lateral_extent_m=[min(borders) + center, max(borders) + center],
                       requested_center_xy=[pose.location.x, pose.location.y], board_blueprint=board.type_id)


def target_body_gap(ego, target):
    if len(target) >= 3:
        return body_gap(ego, target)
    # Some thin native prop hulls reduce to a line in XY. Test the convex
    # separating axes before measuring vertex-to-segment distances.
    edges = np.vstack((np.roll(ego, -1, axis=0) - ego, target[1] - target[0]))
    axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    a, b = ego @ axes.T, target @ axes.T
    if not np.any((a.max(axis=0) < b.min(axis=0)) | (b.max(axis=0) < a.min(axis=0))):
        return 0.
    return min(abs(cv2.pointPolygonTest(contour, tuple(map(float, point)), True))
               for contour, points in ((ego, target), (target, ego)) for point in points)


def first_curb_placement(road, actor, case, geometry, frame):
    """Attest the real settled prop at the first scored camera exposure."""
    source = road.get_waypoint_xodr(67, -2, geometry["board_native_s"])
    pose = source.transform
    right = pose.get_right_vector()
    polygon = _polygon(actor)
    measured = (polygon - [pose.location.x, pose.location.y]) @ [right.x, right.y]
    lo, hi = float(min(measured)), float(max(measured))
    position = actor.get_location()
    error = float(np.linalg.norm(np.asarray([position.x, position.y]) - geometry["requested_center_xy"]))
    expected = geometry["board_lateral_extent_m"]
    edge = geometry["sidewalk_edge_m"]
    bracket = (lo > geometry["driving_edge_m"] and hi < edge - .05) if case == "inside-curb" else lo > edge + .05
    return dict(frame=int(frame), actor_id=int(actor.id), blueprint=actor.type_id,
                actual_center_xy=[position.x, position.y], center_error_m=round(error, 5),
                physical_lateral_extent_m=[lo, hi], polygon_xy=polygon.round(4).tolist(),
                verified=bool(actor.is_alive and actor.type_id == geometry["board_blueprint"] and
                              error < .05 and max(abs(lo - expected[0]), abs(hi - expected[1])) < .05 and bracket))


def native_lane(road, position):
    wp = road.get_waypoint(position, project_to_road=False, lane_type=carla.LaneType.Any)
    if wp is None:
        return None
    return dict(road_id=wp.road_id, lane_id=wp.lane_id, s=round(wp.s, 3), lane_type=str(wp.lane_type))


def workzone_reference(scene, ego_poly, board):
    axes = scene.board_waypoint.transform
    forward, right = axes.get_forward_vector(), axes.get_right_vector()
    origin = [axes.location.x, axes.location.y]
    ego_local = (ego_poly - origin) @ np.array([[forward.x, right.x], [forward.y, right.y]])
    board_local = (_polygon(board) - origin) @ np.array([[forward.x, right.x], [forward.y, right.y]])
    rear, left = ego_local.min(axis=0)
    front, curb = ego_local.max(axis=0)
    board_rear, board_left = board_local.min(axis=0)
    board_front, board_curb = board_local.max(axis=0)
    in_path = board_left <= curb and board_curb >= left
    alongside = rear <= board_front and front >= board_rear
    return dict(board_in_ego_path=bool(in_path), alongside=bool(alongside),
                bumper_gap_m=float(board_rear - front), lateral_clearance_m=float(board_left - curb),
                footprint_intersection=bool(in_path and alongside and body_gap(ego_poly, _polygon(board)) == 0.))


class ReopenedWorkZone(WorkZone):
    """Retain the existing 17–20 s reopening schedule on a single legal lane."""

    road_id, lane_id = REOPENED[:2]

    def __init__(self, world):
        road = world.get_map()
        for s in range(int(REOPENED[2]), int(REOPENED[3]) + 1):
            source = road.get_waypoint_xodr(self.road_id, self.lane_id, float(s))
            if source is None or source.is_junction or source.lane_change != carla.LaneChange.NONE:
                raise ValueError(f"Expected a nonjunction lane with no legal passing at road 76 s={s}")
            for adjacent in (source.get_left_lane(), source.get_right_lane()):
                if adjacent and adjacent.lane_type == carla.LaneType.Driving and adjacent.lane_id * source.lane_id > 0:
                    raise ValueError(f"Unexpected adjacent same-direction lane at road 76 s={s}")
        super().__init__(world, True, WEATHER, start_station=int(REOPENED[2]))

    def metrics(self):
        state = super().metrics()
        state["route_id"] = "Town03-road76-workzone"
        return state


class Witness:
    primary_rig_only = True

    def __init__(self, stream, case):
        self.stream = stream
        self.case = case
        self.scene = None
        self.road = None
        self.route_range = None
        self.station_map = None
        self.extra_actor = None
        self.geometry = {}
        self.curb_placement = None
        self.reference = []
        self.partial_trace = []

    def append(self, row, obs):
        scene = self.scene
        ego = scene.ego
        ego_poly = _polygon(ego)
        station = float(np.interp(row["progress_m"], *self.station_map))
        source = self.road.get_waypoint_xodr(*WINDOWS[self.case][:2], station)
        axes = source.transform
        sides = [lateral(axes.location, p, axes.get_right_vector())
                 for p in ego.bounding_box.get_world_vertices(ego.get_transform())]
        frame = row["frame"]
        reference = dict(frame=frame, sim_time=row["sim_time"], native_s=round(station, 3),
                         ego=row["observer_map"]["ego"], ego_z=row["observer_map"]["ego_z"],
                         ego_native_lane=native_lane(self.road, ego.get_location()),
                         others=row["observer_map"]["others"], ego_polygon_xy=ego_poly.round(4).tolist(),
                         ego_lateral_extents_m=[round(min(sides), 3), round(max(sides), 3)],
                         left_driving_edge_m=-source.lane_width / 2, right_driving_edge_m=source.lane_width / 2)
        if self.case in CURB_CASES and self.curb_placement is None:
            self.curb_placement = first_curb_placement(self.road, self.extra_actor, self.case, self.geometry, frame)
            reference["curb_placement"] = self.curb_placement
            if not self.curb_placement["verified"]:
                raise RuntimeError(f"Native curb placement differs from the registered fixture: {self.curb_placement}")
        evaluate = evaluation_active(self.geometry, dict(row, native_s=reference["native_s"]))
        if self.case == "uncooperative-rear":
            reference.update(route_progress_m=row["progress_m"], corridor_evaluation_active=evaluate,
                             road67_station_available=bool(row["progress_m"] >= self.station_map[0][0]))
        if evaluate:
            reference["native_geometry"] = native_body(
                self.road, reference, *WINDOWS[self.case][:2], self.case in ADJACENT_CASES)
        targets = []
        if self.extra_actor is not None:
            targets.append((self.case, self.extra_actor))
        if fixture := getattr(scene, "maneuver_fixture", None):
            targets.extend((role, actor) for actor, _, role in fixture.spectator_actors())
        for event in getattr(scene, "events", ()):
            if event.kind in ("roadworks", "barrier", "occluded"):
                actors = [*event.cones, *event.barriers] or [event.actor]
                targets.extend((event.kind, actor) for actor in actors if actor is not None)
                if self.case == "person" and event.kind == "occluded" and scene.occluder is not None:
                    targets.append(("person-occluder", scene.occluder))
        target_polygons = []
        for role, actor in targets:
            if actor.is_alive and ego.get_location().distance(actor.get_location()) <= 100.:
                polygon = _polygon(actor)
                if len(polygon) >= 3 or role in CURB_CASES and len(polygon) == 2:
                    target_polygons.append(dict(actor_id=actor.id, role=role, speed_mps=round(actor.get_velocity().length(), 3),
                                                native_lane=native_lane(self.road, actor.get_location()),
                                                polygon_xy=polygon.round(4).tolist(),
                                                body_gap_m=round(target_body_gap(ego_poly, polygon), 3)))
        reference["target_polygons"] = target_polygons
        trace = dict(row)
        trace.pop("observer_map")
        trace["native_s"] = reference["native_s"]
        trace["ego_native_lane"] = reference["ego_native_lane"]
        if self.case == "uncooperative-rear":
            trace["corridor_evaluation_active"] = evaluate
            trace["road67_station_available"] = reference["road67_station_available"]
        trace["nearest_target_body_gap_m"] = min((r["body_gap_m"] for r in target_polygons), default=None)
        if self.case == "reopened":
            trace.update(scene.rows[frame])
            trace.update(workzone_reference(scene, ego_poly, self.extra_actor))
        ok, jpeg = cv2.imencode(".jpg", obs["front"][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError(f"Missing front exposure at world frame {frame}")
        self.stream.write(jpeg.tobytes())
        self.reference.append(reference)
        self.partial_trace.append(trace)

    def create_scene(self, world, kind, seed, **kwargs):
        road_id, lane_id, start, finish = WINDOWS[self.case]
        self.road = world.get_map()
        if self.case == "reopened":
            scene = ReopenedWorkZone(world)
        else:
            maneuver_case = "clear-bend" if self.case in (*CURB_CASES, "person") else self.case
            scene = JourneyScenario(world, kind, seed, **kwargs, barriers=True,
                                    population_profile="city-maneuvers", maneuver_case=maneuver_case)
        try:
            if self.case == "reopened":
                self.extra_actor = scene.hazard
                samples = [(scene._coordinates(p)[0], start + i) for i, p in enumerate(scene.route)]
                self.geometry["legal_adjacent_lane"] = False
            else:
                samples = [(scene._station[i], wp.s) for i, wp in enumerate(scene._points)
                           if (wp.road_id, wp.lane_id) == (road_id, lane_id)]
                if self.case == "uncooperative-rear":
                    spawn = route_pose_at(scene, REAR_LAUNCH_ROUTE_M)
                else:
                    spawn = self.road.get_waypoint_xodr(road_id, lane_id, start).transform
                scene.ego.set_transform(_raised(spawn, .35))
                scene.spawn_lateral_m = scene.spawn_yaw_deg = 0.
                if self.case in CURB_CASES:
                    self.extra_actor, self.geometry = add_curb_board(scene, self.case)
            route_m, native_s = np.asarray(samples).T
            if np.any(np.diff(native_s) <= 0.) or not (native_s[0] <= start < finish <= native_s[-1]):
                raise ValueError(f"Trial {self.case} is missing or reversing the prescribed native lane")
            self.route_range = np.interp([start, finish], native_s, route_m)
            self.station_map = (route_m, native_s)
            self.geometry.update(road_id=road_id, lane_id=lane_id, start_s=start, end_s=finish,
                                 route_start_m=float(self.route_range[0]), route_end_m=float(self.route_range[1]))
            if self.case == "uncooperative-rear":
                self.geometry.update(launch_route_m=REAR_LAUNCH_ROUTE_M, launch_native_lane=native_lane(self.road, spawn.location),
                                     evaluation_start_s=start, evaluation_route_start_m=float(self.route_range[0]),
                                     evaluation_clock="Full episode from launch. Road 67 grade starts at both evaluation gates.",
                                     native_station_clock="Road 67 route projection, clamped before the first road sample. "
                                                          "See road67_station_available and ego_native_lane for pre-roll.")
            self.scene = scene
            return scene
        except BaseException:
            scene.close()
            raise

    def finished(self):
        progress = self.scene._coordinates(self.scene.ego.get_location())[0]
        return progress >= self.route_range[1] - .005


def record_case(world, checkpoint, folder, case, protocol):
    folder.mkdir()
    raw = folder / "front.mjpeg"
    driver, names, schema = load_driver(checkpoint)
    model = LoggedDriver(driver)
    failure = None
    summary = {}
    with raw.open("xb") as stream:
        witness = Witness(stream, case)
        try:
            summary, trace, _, _ = episode(
                world, "workzone" if case == "reopened" else "journey", SEED, schema, model,
                scene_factory=witness.create_scene, destination="market", weather=WEATHER,
                crossing_delay=.5, recording=witness, tactical=True, change_request=witness.finished,
                max_steps=850 if case == "blocked" else 800 if case == "uncooperative-rear" else
                          560 if case in ("person", "closing-rejoin", "slow") else 450,
                run_id=folder.name)
            if len(trace) != len(witness.partial_trace) or len(trace) != len(model.observations):
                raise RuntimeError("Native frame, trace and policy tape counts differ")
            for row, captured in zip(trace, witness.partial_trace, strict=True):
                captured.update(row)
        except Exception:
            failure = traceback.format_exc()
    frames = witness.partial_trace
    count = len(frames)
    write_trace(folder / "reference.jsonl.gz", witness.reference)
    write_trace(folder / "trace.jsonl.gz", frames)
    save_arrays(folder / "observations.npz", frame=[row["frame"] for row in frames], feature_names=names,
                features=np.asarray(model.observations[:count], dtype=np.float32).reshape(-1, len(names)),
                speed_request_mps=model.requested_speed[:count], posted_limit_kmh=model.posted_limits[:count])
    summary.update(case=case, seed=SEED, producer=producer_info(world), policy_schema=schema, frames=count,
                   model_sha256=protocol["model_sha256"], execution_source_sha256=protocol["execution_source_sha256"],
                   geometry=witness.geometry, first_frame=frames[0]["frame"] if frames else None,
                   last_frame=frames[-1]["frame"] if frames else None,
                   reached_end=bool(count and frames[-1]["native_s"] >= WINDOWS[case][-1] - .005),
                   native_station_m=frames[-1]["native_s"] if frames else None, capture_complete=failure is None,
                   stop_seconds=round(.1 * sum(r["speed_mps"] <= .3 for r in frames), 1),
                   peak_kmh=round(3.6 * max((r["speed_mps"] for r in frames), default=0.), 2))
    if witness.curb_placement is not None:
        summary["curb_placement"] = witness.curb_placement
    if case == "reopened" and frames:
        summary.update(longest_blocked_stop_s=longest_stop(frames),
                       board_contact=any(r["footprint_intersection"] for r in frames))
    if count:
        try:
            movie = folder / "front.mp4"
            encode_and_check(raw, movie, (640, 360), 0, count)
            summary["front_sha256"] = fingerprint(movie)
            if summary["capture_complete"]:
                summary["case_score"] = grade_case(summary, frames, witness.reference)
        except Exception:
            failure = "\n".join(filter(None, (failure, traceback.format_exc())))
            summary["capture_complete"] = False
    if failure is not None:
        write_json(folder / "error.json", dict(case=case, traceback=failure))
        summary.setdefault("outcome", "capture_error")
    for filename in ("front.mjpeg", "trace.jsonl.gz", "reference.jsonl.gz", "observations.npz"):
        summary.setdefault("artifact_sha256", {})[filename] = fingerprint(folder / filename)
    write_json(folder / "summary.json", summary)
    print(case, summary["outcome"], count, "frames", summary["native_station_m"], "m",
          "end reached" if summary["reached_end"] else "end not reached", flush=True)
    return failure is None


def freeze_inputs(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    checkpoint = root / "checkpoint.npz"
    shutil.copyfile(args.model, checkpoint)
    digest = fingerprint(checkpoint)
    if digest != fingerprint(args.model):
        raise RuntimeError("Model changed while freezing the trial")
    snapshot = root / "study-source.zip"
    with ZipFile(snapshot, "x", ZIP_DEFLATED) as archive:
        for path in sorted(Path(__file__).parent.glob("*.py")):
            archive.write(path, path.relative_to(PROJECT))
        renderer = PROJECT / "tools/render_maneuver_trials.py"
        if renderer.is_file():
            archive.write(renderer, renderer.relative_to(PROJECT))
    driver, _, schema = load_driver(checkpoint)
    tactical_path = TACTICAL_MODEL
    if getattr(driver, "uses_route_maneuvers", False):
        from .maneuver_choice import MODEL as tactical_path
    shutil.copyfile(tactical_path, root / "tactical-model.json")
    protocol = dict(policy_schema=schema, cases=args.cases, windows={case: WINDOWS[case] for case in args.cases},
                    seed=SEED, model_sha256=digest, execution_source_sha256=fingerprint(snapshot),
                    tactical_model_sha256=fingerprint(root / "tactical-model.json"), step_s=.1, weather=WEATHER,
                    port=args.port, attach=args.attach, requested_quality="High", map="Town03_Opt")
    if "uncooperative-rear" in args.cases:
        protocol["launch_route_stations_m"] = {"uncooperative-rear": REAR_LAUNCH_ROUTE_M}
    write_json(root / "protocol.json", protocol)
    return checkpoint, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--port", type=int, default=3718)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases):
        parser.error("Each requested case must appear once")
    checkpoint, protocol = freeze_inputs(args)
    outcomes = {}
    with simulator(args.port, args.attach, "High", map_name="Town03_Opt") as world:
        for case in args.cases:
            # Make the preceding case's destruction visible to native actor snapshots.
            world.tick()
            outcomes[case] = record_case(world, checkpoint, args.output / case, case, protocol)
    write_json(args.output / "capture-status.json", outcomes)
    return 0 if all(outcomes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
