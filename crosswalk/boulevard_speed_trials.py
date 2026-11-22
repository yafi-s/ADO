"""Paired clear, native signed-transition and stationary-queue speed trials."""
import argparse
from pathlib import Path

import carla
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
                for lane, model in [(source, "vehicle.lincoln.mkz"), (adjacent, "vehicle.audi.a2")]:
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
            state.update(hazard_gap_m=gap, phase="stationary two-lane queue")
            self.truth.append(dict(lead_body_gap_m=round(gap, 4), other_body_gap_m=round(neighbor_gap, 4),
                                   lead_speed_mps=round(_speed(self.lead), 4),
                                   ego_ahead_of_lead=self._coordinates(self.ego.get_location())[0] >
                                   self._coordinates(self.lead.get_location())[0]))
            self.settled = self.settled+1 if gap < 15. and _speed(self.ego) <= .05 else 0
        return state


def rate(rows, summary, queue, extended=False):
    speed = np.array([r["speed_mps"] for r in rows])*3.6
    station = np.array([r["progress_m"] for r in rows])
    posted = np.array([r["posted_limit_kmh"] for r in rows])
    # Compare a fixed geographic window entirely inside the native higher zone.
    low, high = (148., 268.) if extended else (45., 165.)
    fixed = (station >= low) & (station < high) & (posted == 90.)
    events = []
    for index in np.flatnonzero(posted[1:] < posted[:-1]) + 1:
        earlier = [i for i in range(index) if rows[i].get("upcoming_posted_kmh") == posted[index]
                   and 0 < station[index]-station[i] <= 100.]
        events.append(dict(from_kmh=posted[index-1], to_kmh=posted[index], entry_m=station[index],
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
    for threshold in (40,45,50,55,60):
        mask = (posted==90.) & (speed>threshold)
        spans = np.flatnonzero(np.diff(np.r_[False,mask,False])).reshape(-1,2)
        longest = max((b-a for a,b in spans), default=0)
        result[f"above_{threshold}_kmh_in_90_zone"] = dict(seconds=round(mask.sum()*DT,2),
                                                            longest_s=round(float(longest)*DT,2))
    if queue:
        final = rows[-20:]
        full_stop = len(final)==20 and all(r["speed_mps"] <= .051 and 0 < r["lead_body_gap_m"] < 15. and
                                           not r["ego_ahead_of_lead"] for r in final)
        result["queue"] = dict(whole_body_min_gap_m=min(r["lead_body_gap_m"] for r in rows),
                               adjacent_min_gap_m=min(r["other_body_gap_m"] for r in rows),
                               final_gap_m=rows[-1]["lead_body_gap_m"], full_stop_held_2s=full_stop,
                               peak_reference_lead_speed_mps=max(r["lead_speed_mps"] for r in rows),
                               speed_at_first_under_50m_kmh=next((speed[i] for i,r in enumerate(rows)
                                            if r["lead_body_gap_m"] < 50.),None),
                               speed_at_first_under_35m_kmh=next((speed[i] for i,r in enumerate(rows)
                                            if r["lead_body_gap_m"] < 35.),None))
    return result


def run(world, model_path, output, case, extended=False):
    output.mkdir(exist_ok=False, parents=True)
    driver, names, schema = load_driver(model_path)
    baseline = case == "matched-40"
    if baseline:
        if not hasattr(driver, "allow_high_speed"):
            raise RuntimeError("Cannot stage requested same-checkpoint ablation")
        driver.allow_high_speed = False
    queue = case == "stationary-queue"
    scenes = []
    def fixture(*args, **kwargs):
        scene = TestBoulevard(*args, queue=queue, extended=extended, **kwargs)
        scenes.append(scene)
        return scene
    logged = LoggedDriver(driver)
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
    grade = rate(rows, summary, queue, extended)
    write_json(output/"summary.json", dict(**summary, case=case, model_sha256=fingerprint(model_path),
                source_sha256=fingerprint(Path(__file__)), longer_corridor=extended, grade=grade, producer=producer_info(world)))
    print(case, grade, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=3718)
    p.add_argument("--attach", action="store_true")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--longer-corridor", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", choices=("matched-40","published-40","higher-speed","stationary-queue"), required=True)
    args = p.parse_args()
    with simulator(args.port, args.attach, "High", map_name="Town03_Opt") as world:
        world.tick()
        run(world, args.model, args.output, args.case, args.longer_corridor)


if __name__ == "__main__":
    main()
