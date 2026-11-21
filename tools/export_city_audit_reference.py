"""Read parked road users from an existing CARLA map for offline scoring."""

import argparse
import json
from pathlib import Path

import carla


def read_static(client):
    world = client.get_world()
    labels = ("Car", "Truck", "Bus", "Bicycle", "Motorcycle")
    actors = []
    for label in labels:
        for obj in world.get_environment_objects(getattr(carla.CityObjectLabel, label)):
            box = obj.bounding_box
            actors.append(dict(actor_id=f"level-{obj.id}", name=obj.name, role="parked-map-vehicle",
                               type=label.lower(), x=box.location.x, y=box.location.y,
                               z=box.location.z, yaw=box.rotation.yaw,
                               dimensions_m=[2*box.extent.x, 2*box.extent.z, 2*box.extent.y]))
    return dict(map=world.get_map().name.split("/")[-1], server_version=client.get_server_version(),
                client_version=client.get_client_version(), source="World.get_environment_objects",
                queried_labels=list(labels), actors=sorted(actors, key=lambda a: a["actor_id"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("destination", type=Path)
    cli.add_argument("--port", type=int, required=True, help="RPC port of a running CARLA world")
    args = cli.parse_args()
    carla_client = carla.Client("127.0.0.1", args.port)
    carla_client.set_timeout(4)
    info = read_static(carla_client)
    args.destination.write_text(json.dumps(info, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(f"{info['map']} returned {len(info['actors'])} static road users")
