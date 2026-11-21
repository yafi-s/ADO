"""Compare the two city drivers on the same empty native Town03 boulevard."""
import argparse
from contextlib import ExitStack
from pathlib import Path

import carla
import numpy as np

from .city_drive import LoggedDriver, load_driver
from .drive import episode
from .experiment import PROJECT, fingerprint, save_arrays, write_json, write_trace
from .runtime import producer_info, simulator
from .world import Scenario, _raised


class ClearBoulevard:
    road_id, lane_id = 67, -2
    weather = "ClearNoon"
    spawn_lateral_m = spawn_yaw_deg = 0.
    ambient = traffic = hazard = None
    _junction_ranges = ()
    _set_route = Scenario._set_route
    _coordinates = Scenario._coordinates

    def __init__(self, world, *_, **kwargs):
        self.stack = ExitStack()
        try:
            existing = world.get_actors()
            if existing.filter("vehicle.*") or existing.filter("walker.*"):
                raise RuntimeError("The empty boulevard fixture requires a world without vehicles or walkers")
            self.stack.callback(world.set_weather, world.get_weather())
            world.set_weather(carla.WeatherParameters.ClearNoon)
            road = world.get_map()
            points = [road.get_waypoint_xodr(67, -2, float(s)) for s in range(1, 301)]
            if any(p is None or p.is_junction or p.lane_type != carla.LaneType.Driving for p in points):
                raise ValueError("Expected continuous Town03 road 67 driving lane -2")
            self._set_route([p.transform.location for p in points])
            self.width = min(p.lane_width for p in points)
            blueprint = world.get_blueprint_library().find("vehicle.lincoln.mkz")
            blueprint.set_attribute("role_name", "hero")
            self.ego = world.spawn_actor(blueprint, _raised(points[0].transform, .35))
            self.stack.callback(self.ego.destroy)
            self.ego.apply_control(carla.VehicleControl(brake=1.))
        except BaseException:
            self.close()
            raise

    def before_tick(self):
        pass

    def close(self):
        self.stack.close()

    def metrics(self):
        progress, lateral = self._coordinates(self.ego.get_location())
        return dict(progress_m=progress, lateral_error_m=lateral, route_length_m=self.length,
                    phase="unobstructed boulevard", route_departure=abs(lateral) > self.width / 2.,
                    hazard_gap_m=60., red_light_violations=0, yield_violations=0,
                    event_triggered=False, event_cleared=False, destination="Boulevard end",
                    route_id="Town03 road 67 lane -2, OpenDRIVE s=1..300", road_ids=[67], junction_ids=[])


def measure(rows, summary):
    speed = np.array([r["speed_mps"]*3.6 for r in rows])
    posted = np.array([r["posted_limit_kmh"] for r in rows])
    station = np.array([r["progress_m"] for r in rows])
    time = np.array([r["sim_time"] for r in rows])
    dt = .1
    np.testing.assert_allclose(np.diff(time), dt)
    # Predeclared station window excludes initial acceleration and the finishing
    # approach. Native posted limits can change within this frame-aligned window.
    interior = (station >= 60.) & (station <= summary["route_length_m"] - 12.) & (posted >= 40.)
    compatible = (posted >= 40.) & (speed <= np.minimum(posted, 40.))
    result = dict(outcome=summary["outcome"], contacts=summary["collision_events"],
                  seed=summary["seed"], frames=len(rows), route_length_m=summary["route_length_m"],
                  max_lateral_error_m=summary["max_lateral_error_m"],
                  native_posted_kmh={str(v):int(np.count_nonzero(posted==v)) for v in np.unique(posted)},
                  over_40_seconds=round(float((speed>40.).sum()*dt), 1),
                  over_posted_plus_1_seconds=round(float((speed>posted+1.).sum()*dt), 1))
    result["lower_zone_entries"] = []
    for i in np.flatnonzero(posted[1:] < posted[:-1]) + 1:
        approach = [(j,r) for j,r in enumerate(rows[:i]) if r.get("upcoming_posted_kmh")==posted[i]
                    and 0. <= station[i]-station[j] <= 60.]
        result["lower_zone_entries"].append(dict(at_m=round(float(station[i]),2),
            prior_limit_kmh=float(posted[i-1]), next_limit_kmh=float(posted[i]),
            entry_speed_kmh=round(float(speed[i]),3), entry_request_kmh=round(rows[i]["speed_request_mps"]*3.6,3),
            preview_distance_before_entry_m=round(float(station[i]-station[approach[0][0]]),2) if approach else None))
    for threshold in (35,38):
        mask = compatible & (speed >= threshold)
        spans = np.flatnonzero(np.diff(np.r_[False,mask,False])).reshape(-1,2)
        lengths = [b-a for a,b in spans]
        settled = next(((a,b) for a,b in spans if b-a >= 20), None)
        result[f"total_seconds_{threshold}_to_40"] = round(float(np.count_nonzero(mask)*dt),1)
        result[f"longest_seconds_{threshold}_to_40"] = round(float(max(lengths, default=0)*dt),1)
        result[f"first_sustained_{threshold}_to_40_at_s"] = float(time[settled[0]]) if settled else None
    if interior.any():
        within = speed[interior]
        braking = np.array([r["brake"] for r in rows])
        change = np.diff(speed)/dt/3.6
        valid = interior[1:] & interior[:-1]
        result["interior"] = dict(from_m=60., to_m=round(summary["route_length_m"]-12.,2),
            frames=int(interior.sum()), median_kmh=round(float(np.median(within)),3),
            p10_kmh=round(float(np.quantile(within,.1)),3), p90_kmh=round(float(np.quantile(within,.9)),3),
            sd_kmh=round(float(np.std(within)),3), variance_kmh2=round(float(np.var(within)),3),
            peak_kmh=round(float(within.max()),3), mean_abs_speed_change_kmh_per_s=round(float(np.mean(abs(change[valid]))*3.6),3),
            seconds_with_brake_above_005=round(float(np.count_nonzero(interior & (braking>.05))*dt),1),
            seconds_decelerating_above_1_mps2=round(float(np.count_nonzero(valid & (change < -1.))*dt),1))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=2968)
    p.add_argument("--attach", action="store_true")
    p.add_argument("--paired-preview", action="store_true",
                   help="Compare the same pace checkpoint with signed preview disabled and enabled")
    args = p.parse_args()
    with simulator(args.port, args.attach, "Low", map_name="Town03_Opt") as world:
        variants = (("without-preview","city-pace",False), ("with-preview","city-pace",True)) if args.paired_preview else (
            ("city-cruise","city-cruise",False), ("city-pace","city-pace",True))
        for name, model_name, preview in variants:
            # CARLA can retain a just-destroyed actor in its snapshot until
            # the next synchronous tick. Confirm a clean world at spawn.
            world.tick()
            directory = args.output/name
            directory.mkdir(parents=True, exist_ok=False)
            checkpoint = PROJECT/"models"/model_name/"candidate.npz"
            driver,names,schema = load_driver(checkpoint)
            if model_name == "city-pace":
                driver.uses_signed_speed_preview = preview
            logged = LoggedDriver(driver)
            summary,rows,*_ = episode(world, "boulevard", 316, schema, logged, max_steps=1100,
                                      destination="Boulevard end", scene_factory=ClearBoulevard)
            summary.update(model_sha256=fingerprint(checkpoint), producer=producer_info(world),
                           signed_speed_preview=preview)
            write_trace(directory/"trace.jsonl.gz", rows)
            write_json(directory/"summary.json", summary)
            save_arrays(directory/"observations.npz", features=logged.observations, feature_names=names,
                        frame=[r["frame"] for r in rows], speed_request_mps=logged.requested_speed,
                        posted_limit_kmh=logged.posted_limits)
            result = measure(rows, summary)
            write_json(directory/"speeds.json", result)
            print(name, result, flush=True)


if __name__ == "__main__":
    main()
