"""Matched closed-loop Town03 trials on an open cone taper, a blocked lane and an occluded crossing."""
import argparse
from pathlib import Path
import math

import cv2
import numpy as np

from .city_drive import LoggedDriver, load_driver
from .drive import episode
from .experiment import fingerprint, save_arrays, write_json, write_trace
from .journey import JourneyScenario
from .runtime import producer_info, simulator
from .workzone_study import WorkZone, longest_stop
from .world import _raised


class LeftMarkerLane(WorkZone):
    """Independent site with five irregularly spaced left-side markers and an open corridor."""
    def __init__(self, world):
        super().__init__(world, False, "ClearNoon", start_station=25)
        self.markers = []
        try:
            # Move the original fixture board away from both the lane and the markers.
            self.hazard.set_transform(self.board_pose(self.width * 3.))
            library = world.get_blueprint_library()
            for station, offset in ((49, 1.9), (53, 1.8), (58, 1.83), (64, 1.92), (69, 1.9)):
                waypoint = world.get_map().get_waypoint_xodr(self.road_id, self.lane_id, float(station))
                pose = _raised(waypoint.transform, .04)
                pose.location -= pose.get_right_vector() * offset
                marker = world.spawn_actor(library.find("static.prop.trafficcone01"), pose)
                self.stack.callback(marker.destroy)
                self.markers.append((marker, pose.location))
        except BaseException:
            self.close()
            raise

    def before_tick(self):
        self.step += 1

    def metrics(self):
        state = super().metrics()
        frame = self.world.get_snapshot().frame
        ego = [self._coordinates(v) for v in self.ego.bounding_box.get_world_vertices(self.ego.get_transform())]
        stations = []
        gaps = []
        rear, front = min(v[0] for v in ego), max(v[0] for v in ego)
        left = min(v[1] for v in ego)
        for actor, original in self.markers:
            stations.append(self._coordinates(original)[0])
            footprint = [self._coordinates(v) for v in actor.bounding_box.get_world_vertices(actor.get_transform())]
            if rear <= max(v[0] for v in footprint) and front >= min(v[0] for v in footprint):
                gaps.append(left - max(v[1] for v in footprint))
        self.rows[frame] = dict(marker_start_m=min(stations), marker_end_m=max(stations),
                                marker_count=len(self.markers), lateral_gap_m=min(gaps) if gaps else None,
                                marker_displacement_m=max(math.hypot(actor.get_location().x - loc.x,
                                                                  actor.get_location().y - loc.y)
                                                          for actor, loc in self.markers))
        return state


class FrontWitness:
    primary_rig_only = True

    def __init__(self, stream, raw):
        self.stream = stream
        self.raw = raw
        self.scene = None
        self.count = 0

    def append(self, row, obs):
        car = self.scene.ego.get_transform()
        ok, jpg = cv2.imencode(".jpg", obs["front"][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93])
        if not ok:
            raise RuntimeError("Front camera frame unavailable")
        self.stream.write(jpg.tobytes())
        if self.raw and self.count % 5 == 0:
            folder = self.raw / f"frame-{row['frame']}"
            folder.mkdir(parents=True)
            (folder / "front.jpg").write_bytes(jpg.tobytes())
            if "right" in obs:
                ok = cv2.imwrite(str(folder / "right.jpg"), obs["right"][:, :, ::-1])
                if not ok:
                    raise RuntimeError("Side camera frame unavailable")
            np.savez_compressed(folder / "range.npz", lidar=obs["lidar"], radar=obs["radar"])
            write_json(folder / "frame.json", dict(frame=row["frame"], time_s=row["sim_time"],
                       progress_m=row["progress_m"], ego=[car.location.x, car.location.y, car.location.z,
                       car.rotation.pitch, car.rotation.yaw, car.rotation.roll],
                       others=row["observer_map"]["others"]))
        self.count += 1


def run(world, checkpoint, folder, case, seed, raw=False):
    driver, names, schema = load_driver(checkpoint)
    model = LoggedDriver(driver)
    folder.mkdir(parents=True, exist_ok=False)
    start, end = {"cone-taper": (346, 429), "occluded": (99, 161), "blocked": (0, 90),
                  "left-markers": (0, 90), "full-route": (0, math.inf)}[case]
    scenes = []

    def factory(world, *args, **kwargs):
        if case == "left-markers":
            scene = LeftMarkerLane(world)
        elif case == "blocked":
            scene = WorkZone(world, True, "ClearNoon")
        else:
            scene = JourneyScenario(world, *args, barriers=True, **kwargs)
            if start:
                index = int(np.argmin(abs(np.asarray(scene._station) - start)))
                scene.ego.set_transform(_raised(scene._points[index].transform, .35))
        scenes.append(scene)
        if witness:
            witness.scene = scene
        return scene

    def done():
        scene = scenes[0]
        return scene._coordinates(scene.ego.get_location())[0] >= end

    witness = None
    kwargs = dict(scene_factory=factory, destination="market", crossing_delay=.5, tactical=case not in ("blocked", "left-markers"),
                  max_steps=8000 if case == "full-route" else 650, change_request=done)
    if case in ("cone-taper", "occluded"):
        with (folder / "front.mjpeg").open("wb") as stream:
            witness = FrontWitness(stream, folder / "raw" if raw else None)
            summary, trace, _, _ = episode(world, "journey", seed, schema, model, recording=witness, **kwargs)
    else:
        summary, trace, _, _ = episode(world, "workzone" if case in ("blocked", "left-markers") else "journey",
                                       seed, schema, model, **kwargs)
    if case in ("blocked", "left-markers"):
        for row in trace:
            row.update(scenes[0].rows[row["frame"]])
    if case == "left-markers":
        alongside = [r for r in trace if r["marker_start_m"] <= r["progress_m"] < r["marker_end_m"]]
        summary["left_markers"] = dict(seconds=.1*len(alongside), median_kmh=(float(np.median([r["speed_mps"]*3.6 for r in alongside])) if alongside else None),
                                  minimum_gap_m=min((r["lateral_gap_m"] for r in trace if r["lateral_gap_m"] is not None),default=None),
                                  max_displacement_m=max(r["marker_displacement_m"] for r in trace))
    if case == "blocked":
        summary["longest_stop_s"] = longest_stop(trace)
        summary["minimum_stopped_gap_m"] = min((r["bumper_gap_m"] for r in trace
                   if r["board_in_ego_path"] and r["progress_m"] > 5 and r["speed_mps"] <= .3), default=None)
        summary["intersects_board"] = any(r["footprint_intersection"] for r in trace)
    summary.update(case=case, seed=seed, start_station_m=start, end_station_m=end if math.isfinite(end) else None,
                   model_sha256=fingerprint(checkpoint), producer=producer_info(world), policy_schema=schema)
    write_json(folder / "summary.json", summary)
    write_trace(folder / "trace.jsonl.gz", trace)
    save_arrays(folder / "observations.npz", frame=[r["frame"] for r in trace], features=model.observations,
                feature_names=names, speed_request_mps=model.requested_speed, posted_limit_kmh=model.posted_limits)
    print(case, summary["outcome"], summary["frames"], summary["progress_m"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=("cone-taper", "occluded", "blocked", "left-markers", "full-route"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=212)
    p.add_argument("--port", type=int, default=3178)
    p.add_argument("--raw", action="store_true")
    p.add_argument("--attach", action="store_true")
    a = p.parse_args()
    with simulator(a.port, a.attach, "High", map_name="Town03_Opt") as world:
        run(world, a.model, a.output, a.case, a.seed, a.raw)


if __name__ == "__main__":
    main()
