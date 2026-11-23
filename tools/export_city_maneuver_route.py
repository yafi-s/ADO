"""Recreate the Market navigation path and its native lane alternatives offline."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import carla

from crosswalk.urban_map import urban_route


def waypoint(point):
    pose, road, lane = point.transform, point.road_id, point.lane_id
    return dict(
        xyz=[pose.location.x, pose.location.y, pose.location.z],
        heading_deg=pose.rotation.yaw,
        road=road, section=point.section_id, lane=lane, s=point.s,
        kind=str(point.lane_type), junction=point.junction_id if point.is_junction else None,
        width_m=point.lane_width,
    )


def export(source, seed, destination="market"):
    raw = source.read_bytes()
    road_map = carla.Map("Town03_Opt", raw.decode("utf-8"))
    route = urban_route(road_map, seed, destination)
    station, samples = 0., []
    for point in route.points:
        sample = waypoint(point)
        if samples:
            station += math.dist(sample["xyz"][:2], samples[-1]["xyz"][:2])
        sample.update(
            station_m=station, change=str(point.lane_change),
            left_marking=str(point.left_lane_marking.type),
            right_marking=str(point.right_lane_marking.type),
        )
        for side, other in (("left", point.get_left_lane()), ("right", point.get_right_lane())):
            sample[side] = waypoint(other) if other else None
        samples.append(sample)
    survey = dict(
        map=road_map.name, seed=seed, opendrive_sha256=hashlib.sha256(raw).hexdigest(),
        route_id=f"town03_opt-journey:{route.points[0].id}:{route.points[-1].id}",
        route_length_m=station,
        route_scheduled_bus_bypass_m=[
            samples[route.lane_change.start]["station_m"], samples[route.lane_change.end]["station_m"],
        ],
        samples=samples,
    )
    if destination == "market-loop":
        survey.update(
            destination=destination,
            shared_market_prefix_count=route.market_prefix_count,
            shared_market_prefix_to_m=samples[route.market_prefix_count - 1]["station_m"],
        )
    return survey


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--seed", type=int, default=212)
    cli.add_argument("--destination", choices=("market", "market-loop"), default="market")
    args = cli.parse_args()
    args.output.write_text(json.dumps(export(args.source, args.seed, args.destination), separators=(",", ":")) + "\n",
                           encoding="utf-8")
