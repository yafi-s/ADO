"""Grade maneuver clips from synchronized native body and applied-control evidence."""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np


BODY_MARGIN = .015
MIN_BODY_GAP = .25
MOTION_MPS = .65
ADJACENT_CASES = {"cones", "board", "slow", "blocked", "no-lead", "departing", "closing-rejoin"}
STAY_CASES = {"no-lead", "departing", "clear-bend", "inside-curb", "outside-curb", "person", "reopened"}


def native_body(road, reference, road_id, lane_id, permit_adjacent):
    """Measure horizontal corners against the native lane lines at each corner's own station."""
    import carla

    source = road.get_waypoint_xodr(road_id, lane_id, reference["native_s"])
    pose = source.transform
    f, r = pose.get_forward_vector(), pose.get_right_vector()
    adjacent = source.get_left_lane()
    legal = bool(permit_adjacent and adjacent and not adjacent.is_junction and
                 (adjacent.road_id, adjacent.lane_id) == (road_id, lane_id + 1) and
                 adjacent.lane_type == carla.LaneType.Driving and
                 source.left_lane_marking.type == carla.LaneMarkingType.Broken and
                 source.lane_change in (carla.LaneChange.Left, carla.LaneChange.Both))
    keys = [(road_id, lane_id)] + ([(road_id, lane_id + 1)] if legal else [])
    memberships = []
    for x, y in reference["ego_polygon_xy"]:
        location = carla.Location(x=float(x), y=float(y), z=float(reference["ego_z"]))
        nearest = road.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        inside = []
        if nearest is not None and nearest.road_id == road_id and not nearest.is_junction:
            for key in keys:
                sample = road.get_waypoint_xodr(*key, nearest.s)
                if sample is None or sample.is_junction:
                    continue
                axis = sample.transform.get_right_vector()
                delta = location - sample.transform.location
                if abs(delta.x * axis.x + delta.y * axis.y) <= sample.lane_width / 2 + BODY_MARGIN:
                    inside.append(key[1])
        memberships.append(inside)
    return dict(origin_xy=[pose.location.x, pose.location.y], forward_xy=[f.x, f.y],
                right_xy=[r.x, r.y], source_half_width_m=source.lane_width / 2,
                source_lane=lane_id, adjacent_legal=legal, corner_lanes=memberships)


def bounds(polygon, origin, direction):
    values = (np.asarray(polygon) - origin) @ direction
    return float(values.min()), float(values.max())


def measures(row, reference, native):
    origin, forward, right = (np.asarray(native[key]) for key in ("origin_xy", "forward_xy", "right_xy"))
    ego = bounds(reference["ego_polygon_xy"], origin, forward)
    targets = {}
    for target in reference["target_polygons"]:
        poly = target["polygon_xy"]
        measured = dict(target, longitudinal=bounds(poly, origin, forward), lateral=bounds(poly, origin, right))
        targets.setdefault(target["role"], []).append(measured)
    corners = native["corner_lanes"]
    lane = native["source_lane"]
    return dict(row=row, targets=targets, ego=ego, native=native,
                source_only=all(lane in ids for ids in corners),
                legal=len(corners) >= 4 and all(corners), adjacent_body=any(lane + 1 in ids and lane not in ids for ids in corners),
                entirely_adjacent=native["adjacent_legal"] and all(lane + 1 in ids for ids in corners))


def rows_with_role(frames, role):
    return [(f, targets) for f in frames if (targets := f["targets"].get(role))]


def event_for(summary, kind):
    return next((event for event in summary.get("events", ()) if event["kind"] == kind), {})


def pass_evidence(frames, summary, role, expected_count):
    exposed = rows_with_role(frames, role)
    targets = {t["actor_id"] for _, seen in exposed for t in seen}
    alongside = {}
    for f, seen in exposed:
        rear, nose = f["ego"]
        for target in seen:
            lo, hi = target["longitudinal"]
            if f["entirely_adjacent"] and rear <= hi and nose >= lo and (
                    role != "slow-traffic" or target["speed_mps"] > MOTION_MPS):
                alongside.setdefault(target["actor_id"], f["row"]["frame"])
    final_alongside = max(alongside.values(), default=math.inf)
    returned = next((f["row"]["frame"] for f, seen in exposed
                     if f["row"]["frame"] > final_alongside and f["source_only"] and
                     {t["actor_id"] for t in seen} == targets and
                     all(f["ego"][0] > t["longitudinal"][1] + 1. for t in seen)), None)
    floor = min((t["body_gap_m"] for _, ts in exposed for t in ts), default=None)
    event = event_for(summary, role)
    evidence = dict(target_count=len(targets), measured_alongside_frames=sorted(set(alongside.values())),
                    full_body_return_frame=returned, minimum_target_body_gap_m=floor,
                    event_handled=bool(event.get("handled")), event=event)
    passed = len(targets) == expected_count and set(alongside) == targets and returned is not None
    passed &= bool(event.get("handled")) and floor is not None and floor >= MIN_BODY_GAP
    return bool(passed), evidence


def traffic_evidence(frames):
    witnesses = []
    for f in frames:
        targets = f["targets"]
        if not all(role in targets for role in ("slow-traffic", "adjacent-front", "adjacent-rear")):
            continue
        lead, front, rear = (targets[role][0] for role in ("slow-traffic", "adjacent-front", "adjacent-rear"))
        gap = lead["longitudinal"][0] - f["ego"][1]
        if not (0. < gap <= 35.) or lead["speed_mps"] <= MOTION_MPS:
            continue
        row = f["row"]
        ego_center = sum(f["ego"]) / 2
        rear_relative = sum(rear["longitudinal"]) / 2 - ego_center
        front_relative = sum(front["longitudinal"]) / 2 - ego_center
        rear_closing = rear["speed_mps"] - row["speed_mps"]
        rear_ttc = -rear_relative / rear_closing if rear_relative < 0. and rear_closing > 0. else math.inf
        moving = all(t["speed_mps"] > 1. and (t.get("native_lane") or {}).get("lane_id") == -1
                     for t in (front, rear))
        occupied = -18. <= rear_relative <= 10. or -18. <= front_relative <= 10. or rear_ttc <= 2.5
        if moving and occupied:
            witnesses.append(f)
    centered = [f for f in witnesses if f["source_only"] and abs(f["row"]["lat_error_m"]) <= .6]
    waited = [f for f in centered if
              f["row"].get("tactical_pass", {}).get("maneuver_phase") == "yield" and
              not f["row"]["tactical_pass"].get("committed")]
    unsafe = [f for f in witnesses if f["adjacent_body"] or
              f["row"].get("tactical_pass", {}).get("committed")]
    summary = dict(physical_occupied_frames=len(witnesses), actual_yield_frames=len(waited),
                   centered_occupied_seconds=len(centered) / 10,
                   centered_occupied_minimum_ego_mps=min((f["row"]["speed_mps"] for f in centered), default=None),
                   centered_occupied_stopped_seconds=sum(f["row"]["speed_mps"] <= .3 for f in centered) / 10,
                   centered_occupied_rolling_seconds=sum(f["row"]["speed_mps"] > .3 for f in centered) / 10,
                   actual_yield_seconds=len(waited) / 10,
                   actual_yield_minimum_ego_mps=min((f["row"]["speed_mps"] for f in waited), default=None),
                   actual_yield_stopped_seconds=sum(f["row"]["speed_mps"] <= .3 for f in waited) / 10,
                   actual_yield_rolling_seconds=sum(f["row"]["speed_mps"] > .3 for f in waited) / 10,
                   unsafe_crossing_or_commit_frames=len(unsafe),
                   first_occupied_frame=witnesses[0]["row"]["frame"] if witnesses else None,
                   first_yield_frame=waited[0]["row"]["frame"] if waited else None,
                   last_occupied_frame=witnesses[-1]["row"]["frame"] if witnesses else None,
                   last_yield_frame=waited[-1]["row"]["frame"] if waited else None)
    return summary


def stop_frames(rows):
    """Longest contiguous measured stopping interval."""
    longest = current = 0
    for stopped in rows:
        current = current + 1 if stopped else 0
        longest = max(longest, current)
    return longest


def closing_rejoin_evidence(frames, geometry):
    """Follow one physical lead through alongside, closing from behind, and release."""
    observations = rows_with_role(frames, "slow-traffic")
    alongside = None
    last_rear = None
    first_closing = None
    unsafe = []
    incursions = []
    for f, targets in observations:
        row = f["row"]
        rear, nose = f["ego"]
        for target in targets:
            lane = target.get("native_lane") or {}
            if (lane.get("road_id"), lane.get("lane_id")) != (geometry["road_id"], geometry["lane_id"]):
                continue
            lo, hi = target["longitudinal"]
            if alongside is None:
                if f["entirely_adjacent"] and f["legal"] and target["speed_mps"] > MOTION_MPS and rear <= hi and nose >= lo:
                    alongside = (row["frame"], target["actor_id"])
                continue
            if target["actor_id"] != alongside[1] or row["frame"] <= alongside[0]:
                continue
            delta_s = rear - hi
            if delta_s > 0.:
                prior = last_rear
                last_rear = (row["sim_time"], delta_s)
                if prior and .05 <= row["sim_time"] - prior[0] <= .15:
                    closing = (prior[1] - delta_s) / (row["sim_time"] - prior[0])
                    unsafe_gap = delta_s <= max(12., 2. * target["speed_mps"], 5. * closing)
                    if first_closing is None and unsafe_gap and closing >= .5 and target["speed_mps"] > row["speed_mps"]:
                        first_closing = dict(frame=row["frame"], rear_bumper_gap_m=round(delta_s, 3),
                                             measured_closing_mps=round(closing, 3),
                                             ego_mps=row["speed_mps"], source_vehicle_mps=target["speed_mps"])
                else:
                    unsafe_gap = delta_s <= max(12., 2. * target["speed_mps"])
            else:
                last_rear = None
                unsafe_gap = lo - nose <= max(15., 2. * row["speed_mps"])
            if first_closing is not None and unsafe_gap:
                unsafe.append(row["frame"])
                if any(f["native"]["source_lane"] in ids for ids in f["native"]["corner_lanes"]):
                    incursions.append(row["frame"])
    conclusive = first_closing is not None and len(unsafe) >= 3
    return dict(alongside_frame=alongside[0] if alongside else None,
                source_vehicle_id=alongside[1] if alongside else None,
                first_closing_rejoin=first_closing, threatened_rejoin_frames=len(unsafe),
                first_unsafe_return_frame=unsafe[0] if unsafe else None,
                last_unsafe_return_frame=unsafe[-1] if unsafe else None,
                unsafe_return_crossing_frames=incursions, threatened_rejoin_confirmed=conclusive,
                ended_during_measured_wait=bool(conclusive and not incursions and
                                               unsafe[-1] == frames[-1]["row"]["frame"] and
                                               frames[-1]["entirely_adjacent"]))


def reopened_evidence(frames, geometry):
    blocked_stop = [f["row"]["board_in_ego_path"] and 0. < f["row"]["bumper_gap_m"] < 20. and
                    f["row"]["speed_mps"] <= .3 and bool(f["targets"].get("reopened")) for f in frames]
    longest_stop = stop_frames(blocked_stop)
    last_stop = max((i for i, stopped in enumerate(blocked_stop) if stopped), default=len(frames))
    reopened = next((i for i in range(last_stop + 1, len(frames))
                     if not frames[i]["row"]["board_in_ego_path"] and
                     frames[i]["row"]["lateral_clearance_m"] >= MIN_BODY_GAP), None)
    passing = [f for f in frames if f["row"]["alongside"]]
    floor = min((t["body_gap_m"] for f in passing for t in f["targets"].get("reopened", ())), default=None)
    restarted = next((f["row"]["frame"] for f in frames[reopened:] if f["row"]["speed_mps"] >= 3.),
                    None) if reopened is not None else None
    contact = any(f["row"]["footprint_intersection"] for f in frames)
    evidence = dict(blocked_stop_seconds=longest_stop / 10, clear_frame=frames[reopened]["row"]["frame"]
                    if reopened is not None else None, restart_frame=restarted,
                    minimum_passing_body_gap_m=floor, board_contact=contact)
    return bool(geometry.get("legal_adjacent_lane") is False and longest_stop >= 10 and restarted is not None
                and not contact and floor is not None and floor >= MIN_BODY_GAP), evidence


def person_evidence(frames, summary):
    event = event_for(summary, "occluded")
    exposed = rows_with_role(frames, "occluded")
    occupied = []
    for f, (person, *_) in exposed:
        left, right = person["lateral"]
        width = f["native"]["source_half_width_m"]
        if left <= width and right >= -width:
            occupied.append((f, person))
    moving_in_path = any(p["speed_mps"] > .5 for _, p in occupied)
    occupied_ahead = bool(occupied) and all(f["ego"][1] < p["longitudinal"][0] for f, p in occupied)
    last_occupied = occupied[-1][0]["row"]["frame"] if occupied else math.inf
    passed = next((f["row"]["frame"] for f, (person, *_) in exposed if f["row"]["frame"] > last_occupied
                   and f["ego"][0] > person["longitudinal"][1]), None)
    floor = min((p["body_gap_m"] for _, ts in exposed for p in ts), default=None)
    evidence = dict(walker_moving_in_path=moving_in_path, path_occupied_frames=len(occupied),
                    ego_stayed_behind_while_occupied=occupied_ahead, rear_passed_crossing_frame=passed,
                    minimum_person_body_gap_m=floor, occlusion_witnessed=bool(rows_with_role(frames, "person-occluder")),
                    event=event)
    confirmed = moving_in_path and occupied_ahead and passed is not None and floor is not None and floor >= MIN_BODY_GAP
    confirmed &= bool(event.get("handled") and evidence["occlusion_witnessed"])
    return bool(confirmed), evidence


def control_evidence(case, frames, geometry):
    lead = rows_with_role(frames, "slow-traffic")
    if case == "no-lead":
        return not lead, dict(slow_lead_frames=len(lead))
    if case == "departing":
        departure = [f["row"]["frame"] for f, (t, *_) in lead if t["speed_mps"] >= 5.
                     and t["longitudinal"][0] > f["ego"][1]]
        peak = max((t["speed_mps"] for _, targets in lead for t in targets), default=0.)
        return bool(departure), dict(lead_departure_frame=departure[0] if departure else None,
                                     peak_lead_mps=peak)
    if case == "clear-bend":
        unplanned = any(f["targets"].get("in-lane-obstruction") for f in frames)
        return not unplanned, dict(unplanned_obstruction=unplanned)
    seen = rows_with_role(frames, case)
    board = geometry.get("board_native_s")
    pavement = geometry.get("driving_edge_m")
    curb = geometry.get("sidewalk_edge_m")
    graph_brackets = (board is not None and pavement is not None and curb is not None and pavement < curb)
    alongside = [t for f, (t, *_) in seen if graph_brackets and abs(f["row"]["native_s"] - board) <= 1.
                 and t["longitudinal"][0] <= f["ego"][1] and t["longitudinal"][1] >= f["ego"][0]]
    valid = [t for t in alongside if t["lateral"][0] > pavement + .02 and
             (t["lateral"][1] < curb + .02 if case == "inside-curb" else t["lateral"][0] > curb - .02)]
    floor = min((t["body_gap_m"] for t in alongside), default=None)
    return bool(valid and floor is not None and floor >= MIN_BODY_GAP), dict(
        board_alongside_frames=len(alongside), board_on_expected_side_of_curb_frames=len(valid),
        minimum_board_body_gap_m=floor)


def grade_case(summary, trace, reference, road=None):
    """Report positive evidence, prerequisites and failure reasons for one completed capture."""
    case = summary["case"]
    geometry = summary["geometry"]
    if not trace or len(trace) != len(reference) or any(
            a["frame"] != b["frame"] or a["sim_time"] != b["sim_time"] or
            a["frame"] != trace[0]["frame"] + i for i, (a, b) in enumerate(zip(trace, reference))):
        return dict(passed=False, failures=["missing or unsynchronized evidence"])
    frames = []
    for row, ref in zip(trace, reference):
        native = ref.get("native_geometry")
        if native is None:
            if road is None:
                return dict(passed=False, failures=["native corner geometry missing; reopen with --opendrive"])
            native = native_body(road, ref, geometry["road_id"], geometry["lane_id"], case in ADJACENT_CASES)
        frames.append(measures(row, ref, native))
    failures = []
    if (not summary.get("capture_complete") or not summary.get("front_sha256") or
            summary["outcome"] in ("collision", "red_light", "lane_departure", "capture_error") or
            any(summary.get(key) for key in ("collision_events", "yield_violations", "red_light_violations")) or
            any(r.get("collision_count") for r in trace)):
        failures.append("capture incomplete or unsafe outcome")
    off_road = [f["row"]["frame"] for f in frames if not f["legal"]]
    if off_road:
        failures.append("ego body outside legal driving pavement")
    left_source = [f["row"]["frame"] for f in frames if not f["source_only"]]
    if case in STAY_CASES and left_source:
        failures.append("ego body left the source lane")
    evidence = dict(illegal_body_frames=len(off_road), first_illegal_body_frame=off_road[0] if off_road else None,
                    source_boundary_crossing_frames=len(left_source), reached_end=summary["reached_end"])
    if case == "closing-rejoin":
        evidence.update(closing_rejoin_evidence(frames, geometry))
        evidence["inconclusive"] = not evidence["threatened_rejoin_confirmed"]
        if evidence["inconclusive"]:
            failures.append("no affirmative unsafe rear closure after a measured physical overtake")
        if evidence["unsafe_return_crossing_frames"]:
            failures.append("ego crossed into the source lane during the threatened return")
    if case in ("cones", "board", "slow"):
        movement = "slow-traffic" if case == "slow" else "in-lane-obstruction"
        passed, proof = pass_evidence(frames, summary, movement, 3 if case == "cones" else 1)
        evidence.update(proof)
        if not passed:
            failures.append("missing independent handled event, physical alongside clearance or full-body return")
    if case in ("slow", "blocked"):
        traffic = traffic_evidence(frames)
        evidence.update(traffic)
        if not traffic["physical_occupied_frames"]:
            failures.append("no measured occupied passing gap")
        if traffic["unsafe_crossing_or_commit_frames"]:
            failures.append("crossed or committed while the passing gap was occupied")
        if case == "slow" and (not traffic["actual_yield_frames"] or
                traffic["first_yield_frame"] >= min(evidence["measured_alongside_frames"], default=math.inf)):
            failures.append("no applied yield during the measured approach occupancy")
        if case == "blocked":
            if left_source:
                passed, proof = pass_evidence(frames, summary, "slow-traffic", 1)
                evidence.update(proof)
                if not passed:
                    failures.append("departure from the source lane did not become a verified safe pass and return")
            tail_wait = (not left_source and traffic["last_occupied_frame"] is not None and
                         trace[-1]["frame"] - traffic["last_occupied_frame"] <= 20 and
                         trace[-1]["speed_mps"] <= 3. and
                         abs(trace[-1]["native_s"] - trace[max(0, len(trace) - 20)]["native_s"]) <= 6.)
            evidence["ended_in_measured_safe_wait"] = tail_wait
    if case == "person":
        passed, proof = person_evidence(frames, summary)
        evidence.update(proof)
        if not passed:
            failures.append("missing positive crossing, clearance or yield geometry")
    if case == "reopened":
        passed, proof = reopened_evidence(frames, geometry)
        evidence.update(proof)
        if not passed:
            failures.append("missing proof of blocked stop, board clearance or restart")
    if case in ("no-lead", "departing", "clear-bend", "inside-curb", "outside-curb"):
        passed, proof = control_evidence(case, frames, geometry)
        evidence.update(proof)
        if not passed:
            failures.append("control fixture behavior or measured clearance missing")
    if not summary["reached_end"] and not (
            evidence.get("ended_in_measured_safe_wait") or evidence.get("ended_during_measured_wait")):
        failures.append("prescribed route window did not finish")
    return dict(passed=not failures, failures=failures, **evidence)


def read_frames(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="One output root or one case directory")
    parser.add_argument("--opendrive", type=Path, help="Native Town03_Opt map for captures made before corner logging")
    args = parser.parse_args()
    road = None
    if args.opendrive:
        import carla
        road = carla.Map("Town03_Opt", args.opendrive.read_text(encoding="utf-8-sig"))
    folders = [args.root] if (args.root / "summary.json").is_file() else sorted(
        p for p in args.root.iterdir() if (p / "summary.json").is_file())
    if not folders:
        parser.error(f"No capture summaries found in {args.root}")
    results = {}
    for folder in folders:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        results[summary["case"]] = grade_case(summary, read_frames(folder / "trace.jsonl.gz"),
                                             read_frames(folder / "reference.jsonl.gz"), road)
    print(json.dumps(results, indent=2, allow_nan=False))
    return 0 if all(result["passed"] for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
