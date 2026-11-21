"""Export the actual town's lane centerlines and building bounds for the 3D observer."""
from collections import defaultdict
from pathlib import Path
import argparse
import json

import carla

from .runtime import simulator, producer_info


def point(v):
    return [round(v.x, 2), round(v.y, 2), round(v.z, 2)]


def geometry(world):
    lanes = defaultdict(list)
    for wp in world.get_map().generate_waypoints(2.):
        if wp.lane_type == carla.LaneType.Driving:
            lanes[wp.road_id, wp.section_id, wp.lane_id].append((wp.s, point(wp.transform.location)))
    roads = [[p for _,p in sorted(line)] for _,line in sorted(lanes.items()) if len(line) >= 2]
    buildings = []
    for obj in world.get_environment_objects(carla.CityObjectLabel.Buildings):
        box = obj.bounding_box
        if min(box.extent.x,box.extent.y) < 1. or box.extent.z < 1.5 or max(box.extent.x,box.extent.y) > 110.:
            continue
        buildings.append(dict(center=point(box.location),extent=point(box.extent),yaw=round(box.rotation.yaw,2)))
    buildings.sort(key=lambda b:b["center"])
    return dict(producer=producer_info(world), roads=roads,buildings=buildings,
                  units="metres, CARLA world coordinates",building_representation="oriented bounding boxes")


def export(world, path):
    result = geometry(world)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,separators=(",",":"))+"\n",encoding="utf-8")
    print(f"{len(result['roads'])} lane paths and {len(result['buildings'])} building bounds",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("web/media/town.json"))
    parser.add_argument("--port",type=int,default=2468)
    args = parser.parse_args()
    with simulator(args.port,args.attach) as world:
        export(world,args.output)
