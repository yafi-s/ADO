"""Measure signal colors at fixed CARLA poses with paired physical light states."""
import argparse
from contextlib import ExitStack
from pathlib import Path
from queue import Queue

import carla
import cv2
import numpy as np

from .lamp import head_vertices, project_box
from .runtime import simulator


def collect(world, dest, distances=(3., 12., 25.)):
    dest.mkdir(parents=True, exist_ok=False)
    lights = sorted(world.get_actors().filter("traffic.traffic_light"), key=lambda a: a.get_opendrive_id())
    candidates = [s for s in lights if s.get_stop_waypoints() and head_vertices(s)]
    selected = candidates[::max(1, len(candidates)//9)]
    by_id = {s.get_opendrive_id(): s for s in selected}
    by_id.update({s.get_opendrive_id(): s for s in candidates if s.get_opendrive_id() == "2024"})
    images, ids, labels, sites, sizes = [], [], [], [], []
    with ExitStack() as cleanup:
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        for k,v in dict(image_size_x=640, image_size_y=360, fov=90, sensor_tick=0,
                        enable_dlss="false", use_ray_tracing="false").items():
            if bp.has_attribute(k): bp.set_attribute(k,str(v))
        camera = world.spawn_actor(bp, carla.Transform(carla.Location(z=8)))
        cleanup.callback(camera.destroy)
        q = Queue()
        camera.listen(q.put)
        cleanup.callback(camera.stop)
        for signal in by_id.values():
            old, frozen = signal.get_state(), signal.is_frozen()
            signal.freeze(True)
            try:
                stop = signal.get_stop_waypoints()[0]
                heads = head_vertices(signal)
                for distance in distances:
                    approach = stop.previous(distance)
                    if not approach: continue
                    pose = approach[0].transform
                    pose.location.z += 1.65
                    camera.set_transform(pose)
                    for state in ("Off", "Red", "Yellow", "Green"):
                        signal.set_state(getattr(carla.TrafficLightState,state))
                        for _ in range(5):
                            frame = world.tick()
                            img = q.get(timeout=8)
                            while img.frame < frame: img = q.get(timeout=8)
                            if img.frame != frame: raise RuntimeError("Calibration camera lost synchronization")
                        rgb = np.frombuffer(img.raw_data,np.uint8).reshape(360,640,4)[:,:,2::-1].copy()
                        inv = np.asarray(img.transform.get_inverse_matrix())
                        for j, h in enumerate(heads):
                            box = project_box(h, inv, rgb.shape)
                            if box is None: continue
                            x0,y0,x1,y1 = box
                            crop = rgb[y0:y1,x0:x1]
                            if crop.shape[0] < 5 or crop.shape[1] < 3: continue
                            sizes.append(crop.shape[:2])
                            images.append(cv2.resize(crop,(12,24)))
                            ids.append(signal.get_opendrive_id())
                            sites.append(f"{signal.get_opendrive_id()}/{distance}/{j}")
                            labels.append(state.lower())
            finally:
                signal.set_state(old)
                signal.freeze(frozen)
    if not images: raise RuntimeError("No projected native signal crops collected")
    np.savez_compressed(dest/'paired-colors.npz', crops=images, pole=ids, site=sites, states=labels, sizes=sizes)
    print(f"Captured {len(images)} native signal crops at {len(set(sites))} head/pose pairs.",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=2968)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    with simulator(args.port,attach=True,map_name="Town03_Opt") as world:
        collect(world,args.output)
