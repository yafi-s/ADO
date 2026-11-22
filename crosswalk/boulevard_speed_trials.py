"""Paired clear, native signed-transition and stationary-queue speed trials."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import carla
import cv2
import numpy as np

from .city_boulevard import ClearBoulevard
from .city_drive import LoggedDriver, load_driver
from .city_progress_bus import body_gap, footprint
from .drive import DT, episode
from .experiment import fingerprint, save_arrays, write_json, write_trace
from .runtime import producer_info, simulator
from .traction import _speed
from .world import _raised


class TestBoulevard(ClearBoulevard):
    def __init__(self, world, *args, queue, extended, **kwargs):
        super().__init__(world, *args, **kwargs)
        try:
            self._map = world.get_map()
            self.extended = extended
            if extended:
                # The native 90 sign on road 69 leads into the longer straight
                # on the opposite carriageway of road 67, then two lower signs.
                origin = self._map.get_waypoint_xodr(69, 3, 48.)
                points = [origin]
                for _ in range(438):
                    options = points[-1].next(1.)
                    if len(options) != 1:
                        raise RuntimeError("Unexpected branch on the extended boulevard")
                    points.append(options[0])
                self._set_route([p.transform.location for p in points])
                self.width = min(p.lane_width for p in points)
                junction = np.array([p.is_junction for p in points])
                spans = np.flatnonzero(np.diff(np.r_[False,junction,False])).reshape(-1,2)
                self._junction_ranges = [(self._coordinates(points[a].transform.location)[0],
                                          self._coordinates(points[b-1].transform.location)[0]) for a,b in spans]
                self.route_roads = list(dict.fromkeys(p.road_id for p in points))
                self.junctions = list(dict.fromkeys(p.junction_id for p in points if p.is_junction))
                self.ego.set_transform(_raised(origin.transform, .35))
                self.road_id, self.lane_id = 69, 3
                lights = world.get_actors().filter("traffic.traffic_light")
                if lights:
                    self.stack.callback(world.freeze_all_traffic_lights, lights[0].is_frozen())
                    world.freeze_all_traffic_lights(True)
                    for light in lights:
                        self.stack.callback(light.set_state, light.get_state())
                        light.set_state(carla.TrafficLightState.Green)
            self.lead = self.neighbor = None
            self.settled = 0
            self.truth = []
            if queue:
                # Matching queues on the two lanes make waiting appropriate.
                source, adjacent, position = (3, 2, 178.) if extended else (-2, -1, 180.)
                for lane, model in [(source, "vehicle.lincoln.mkz"), (adjacent, "vehicle.tesla.model3")]:
                    wp = self._map.get_waypoint_xodr(67, lane, position)
                    if not wp or wp.is_junction or wp.lane_type != carla.LaneType.Driving:
                        raise RuntimeError("Missing queue lane")
                    bp = world.get_blueprint_library().find(model)
                    actor = world.spawn_actor(bp, _raised(wp.transform, .35))
                    self.stack.callback(actor.destroy)
                    actor.apply_control(carla.VehicleControl(brake=1., hand_brake=True))
                    if lane == source:
                        self.lead = actor
                    else:
                        self.neighbor = actor
        except BaseException:
            self.close()
            raise

    def metrics(self):
        state = super().metrics()
        if self.extended:
            state.update(route_id="Town03 road 69 lane +3 via native connectors to road 67 lane +3",
                         road_ids=self.route_roads, junction_ids=self.junctions)
        if self.lead:
            ego = footprint(self.ego)
            gap = body_gap(ego, footprint(self.lead))
            neighbor_gap = body_gap(ego, footprint(self.neighbor))
            axis = self.lead.get_transform().get_forward_vector()
            direction = np.array([axis.x,axis.y])
            lead_rear = float(min(footprint(self.lead)@direction))
            ego_front = float(max(ego@direction))
            neighbor_rear = float(min(footprint(self.neighbor)@direction))
            state.update(hazard_gap_m=gap, phase="stationary two-lane queue")
            self.truth.append(dict(lead_body_gap_m=round(gap, 4), other_body_gap_m=round(neighbor_gap, 4),
                                   lead_bumper_longitudinal_gap_m=round(lead_rear-ego_front,4),
                                   other_bumper_longitudinal_gap_m=round(neighbor_rear-ego_front,4),
                                   lead_speed_mps=round(_speed(self.lead), 4),
                                   ego_ahead_of_lead=self._coordinates(self.ego.get_location())[0] >
                                   self._coordinates(self.lead.get_location())[0]))
            self.settled = self.settled+1 if gap < 15. and _speed(self.ego) <= .05 else 0
        return state


def rate(rows, summary, queue, extended=False, window=None):
    speed = np.array([r["speed_mps"] for r in rows])*3.6
    station = np.array([r["progress_m"] for r in rows])
    posted = np.array([r["posted_limit_kmh"] for r in rows])
    # Compare a fixed geographic window entirely inside the native higher zone.
    low, high = window or ((148., 268.) if extended else (45., 165.))
    fixed = (station >= low) & (station < high) & (posted == 90.)
    events = []
    for index in np.flatnonzero(posted[1:] < posted[:-1]) + 1:
        earlier = [i for i in range(index) if rows[i].get("upcoming_posted_kmh") == posted[index]
                   and 0 < station[index]-station[i] <= 100.]
        tracking = [i for i in earlier if abs(rows[i]["signed_approach_cap_mps"]-
                                                rows[i]["speed_request_mps"]) <= .02]
        braking = [i for i in tracking if rows[i]["brake"]>=.05]
        events.append(dict(first_tracking_sign_cap_distance_m=station[index]-station[tracking[0]] if tracking else None,
                           first_braking_while_tracking_distance_m=station[index]-station[braking[0]] if braking else None,
                           speed_at_first_braking_while_tracking_kmh=speed[braking[0]] if braking else None,
                           from_kmh=posted[index-1], to_kmh=posted[index], entry_m=station[index],
                           actual_kmh=speed[index], request_kmh=3.6*rows[index]["speed_request_mps"],
                           first_preview_m=station[earlier[0]] if earlier else None,
                           preview_distance_before_entry_m=station[index]-station[earlier[0]] if earlier else None))
    result = dict(frames=len(rows), outcome=summary["outcome"], collisions=summary["collision_events"],
                  max_abs_lateral_m=max(abs(r["lat_error_m"]) for r in rows),
                  over_posted_tolerance_s=float(np.count_nonzero(speed>posted+1.)*DT),
                  maximum_posted_excess_kmh=float(max(0.,max(speed-posted))),
                  peak_kmh=float(max(speed)), fixed_section=dict(
                      start_m=low, end_m=high, frames=int(fixed.sum()), median_kmh=float(np.median(speed[fixed])) if fixed.any() else None,
                      peak_kmh=float(max(speed[fixed])) if fixed.any() else None), lower_sign_entries=events)
    time = np.array([r["sim_time"] for r in rows])
    requests = np.array([r["speed_request_mps"] for r in rows])*3.6
    def comfort(mask):
        continuous = np.convolve(mask.astype(int), np.ones(6,dtype=int), mode="valid")==6
        accel = (speed[5:]-speed[:-5])/(.5*3.6)
        drops = np.flatnonzero(mask[1:] & mask[:-1] & (np.diff(requests)<=-5.)) + 1
        return dict(actual_median_kmh=float(np.median(speed[mask])) if mask.any() else None,
                    actual_p10_kmh=float(np.quantile(speed[mask],.1)) if mask.any() else None,
                    actual_p90_kmh=float(np.quantile(speed[mask],.9)) if mask.any() else None,
                    max_half_second_braking_mps2=max(0.,float(-min(accel[continuous]))) if continuous.any() else None,
                    target_drops_5_kmh_one_tick=[dict(at_m=station[i], time_s=time[i],
                              from_kmh=requests[i-1], to_kmh=requests[i]) for i in drops])
    result["fixed_section"].update(comfort(fixed))
    result["native_90_zone_comfort"] = comfort(posted==90.)
    result["matched_station_bins"] = [
        dict(start_m=float(s),end_m=float(s+10.),sample_count=int(m.sum()),**comfort(m))
        for s in np.arange(low,high,10.) if (m := (station>=s)&(station<s+10.)&(posted==90.)).any()]
    for threshold in (40,45,50,55,60):
        mask = (posted==90.) & (speed>threshold)
        spans = np.flatnonzero(np.diff(np.r_[False,mask,False])).reshape(-1,2)
        segment = max(spans, key=lambda ab: ab[1]-ab[0], default=None)
        longest = int(segment[1]-segment[0]) if segment is not None else 0
        stats = {}
        if segment is not None:
            a,b = segment
            stats = dict(entry_m=station[a],exit_m=station[b-1], entry_s=time[a],last_sample_s=time[b-1],
                         median_kmh=float(np.median(speed[a:b])),p10_kmh=float(np.quantile(speed[a:b],.1)))
        result[f"above_{threshold}_kmh_in_90_zone"] = dict(seconds=round(mask.sum()*DT,2),
                                                            longest_s=round(float(longest)*DT,2), **stats)
    if queue:
        final = rows[-20:]
        full_stop = len(final)==20 and all(r["speed_mps"] <= .051 and 0 < r["lead_body_gap_m"] < 15. and
                                           not r["ego_ahead_of_lead"] for r in final)
        result["queue"] = dict(whole_body_min_gap_m=min(r["lead_body_gap_m"] for r in rows),
                               adjacent_min_gap_m=min(r["other_body_gap_m"] for r in rows),
                               final_gap_m=rows[-1]["lead_body_gap_m"], full_stop_held_2s=full_stop,
                               peak_reference_lead_speed_mps=max(r["lead_speed_mps"] for r in rows),
                               ever_committed_lane_change=any(r.get("tactical_pass",{}).get("committed",False)
                                                             for r in rows),
                               final_lateral_from_source_m=rows[-1]["lat_error_m"],
                               final_longitudinal_source_bumper_gap_m=rows[-1].get("lead_bumper_longitudinal_gap_m"),
                               final_longitudinal_adjacent_bumper_gap_m=rows[-1].get("other_bumper_longitudinal_gap_m"),
                               speed_at_first_under_50m_kmh=next((speed[i] for i,r in enumerate(rows)
                                            if r["lead_body_gap_m"] < 50.),None),
                               speed_at_first_under_35m_kmh=next((speed[i] for i,r in enumerate(rows)
                                            if r["lead_body_gap_m"] < 35.),None))
    return result


def run(world, model_path, output, case, extended=False, sweeps=None,
        fixture_class=TestBoulevard, window=None):
    output.mkdir(exist_ok=False, parents=True)
    digest = fingerprint(model_path)
    driver, names, schema = load_driver(model_path)
    source_dir = Path(__file__).parent
    sources = {name: fingerprint(source_dir/name) for name in (
        "boulevard_speed_trials.py", "city_maneuvers.py", "city_open_road.py", "drive.py",
        "route_maneuvers.py", "speed_signs.py")}
    choice_sha = fingerprint(source_dir.parent/"models/route-maneuvers/choice.json")
    baseline = case == "matched-40"
    if baseline:
        if not hasattr(driver, "allow_high_speed"):
            raise RuntimeError("Cannot stage requested same-checkpoint ablation")
        driver.allow_high_speed = False
    queue = case == "stationary-queue"
    scenes = []
    def fixture(*args, **kwargs):
        scene = fixture_class(*args, queue=queue, extended=extended, **kwargs)
        scenes.append(scene)
        return scene
    logged = LoggedDriver(driver)
    if sweeps is not None:
        sweeps.mkdir(parents=True, exist_ok=False)
        with ZipFile(sweeps/"controller.zip","w",compression=ZIP_DEFLATED) as bundle:
            for source in source_dir.glob("*.py"):
                bundle.write(source,arcname="crosswalk/"+source.name)
            bundle.write(model_path,arcname="models/city-maneuvers/candidate.npz")
            bundle.write(source_dir.parent/"models/route-maneuvers/choice.json",
                         arcname="models/route-maneuvers/choice.json")
        original_factory = logged.new_observer
        def sampled_observer():
            observer = original_factory()
            original_observe = observer.observe
            def observe(obs, detections, route, speed, **kwargs):
                frame = obs["progress_frame"]
                if frame % 10 == 0:
                    if not cv2.imwrite(str(sweeps/f"{frame}.jpg"),obs["front"][:,:,::-1]):
                        raise RuntimeError("Could not retain sampled front image")
                    np.savez(sweeps/f"{frame}.npz", lidar=obs.get("long_sight_lidar",obs["lidar"]),
                             radar=obs["radar"], speed_mps=speed, local_route=route,
                             preview=kwargs["route_preview"], extended=obs.get("long_sight_preview",[]))
                return original_observe(obs,detections,route,speed,**kwargs)
            observer.observe = observe
            return observer
        logged.new_observer = sampled_observer
    summary, rows, *_ = episode(world, "boulevard", 316, schema, logged, max_steps=700,
                                scene_factory=fixture, destination="Boulevard end", tactical=True,
                                change_request=lambda: scenes[0].settled >= 20)
    if queue:
        assert len(rows) == len(scenes[0].truth)
        for row, truth in zip(rows,scenes[0].truth):
            row.update(truth)
    assert len(rows) == len(logged.observations)
    write_trace(output/"trace.jsonl.gz", rows)
    save_arrays(output/"observations.npz", features=logged.observations, feature_names=names,
                frame=[r["frame"] for r in rows], speed_request_mps=logged.requested_speed,
                posted_limit_kmh=logged.posted_limits)
    grade = rate(rows, summary, queue, extended, window)
    write_json(output/"summary.json", dict(**summary, case=case, model_sha256=digest, implementation=sources,
                maneuver_choice_sha256=choice_sha,
                source_sha256=fingerprint(Path(__file__)), longer_corridor=extended, grade=grade, producer=producer_info(world)))
    print(case, grade, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=3718)
    p.add_argument("--attach", action="store_true")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--longer-corridor", action="store_true")
    p.add_argument("--sampled-sweeps", type=Path, help="Optional one-Hz raw diagnostic tape")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", choices=("matched-40","published-40","higher-speed","stationary-queue"), required=True)
    args = p.parse_args()
    with simulator(args.port, args.attach, "High", map_name="Town03_Opt") as world:
        world.tick()
        run(world, args.model, args.output, args.case, args.longer_corridor, args.sampled_sweeps)


if __name__ == "__main__":
    main()
