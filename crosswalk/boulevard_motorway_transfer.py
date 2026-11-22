"""Transfer a frozen Town03 driver to a signed 626 m divided Town04 motorway."""
import argparse
import hashlib
from contextlib import ExitStack
from pathlib import Path

import carla
import numpy as np

from .boulevard_speed_trials import TestBoulevard, run
from .experiment import fingerprint, write_json
from .runtime import simulator
from .world import _raised


def motorway_points(road_map):
    points = [road_map.get_waypoint_xodr(6,-2,16.)]
    for _ in range(605):
        options = points[-1].next(1.)
        if len(options)!=1:
            raise RuntimeError("Motorway continuation no longer unique")
        points.append(options[0])
    if any(p.is_junction or p.lane_id!=-2 or p.road_id not in (6,45) for p in points):
        raise RuntimeError("Motorway route invalid")
    return points


class Motorway(TestBoulevard):
    road_id, lane_id = 6,-2

    def __init__(self,world,*_,queue,**kwargs):
        self.stack = ExitStack()
        try:
            self._map = world.get_map()
            existing = world.get_actors()
            if existing.filter("vehicle.*") or existing.filter("walker.*"):
                raise RuntimeError("Motorway fixture requires an empty world")
            self.stack.callback(world.set_weather,world.get_weather())
            world.set_weather(carla.WeatherParameters.ClearNoon)
            points = motorway_points(self._map)
            origin = points[0]
            self._set_route([p.transform.location for p in points])
            self.width = min(p.lane_width for p in points)
            self.extended = False
            self.settled = 0
            self.truth = []
            self.lead = self.neighbor = None
            catalog = world.get_blueprint_library()
            def car(p,bp_name,role):
                bp = catalog.find(bp_name)
                bp.set_attribute("role_name",role)
                actor = world.spawn_actor(bp,_raised(p.transform,.35))
                self.stack.callback(actor.destroy)
                actor.apply_control(carla.VehicleControl(brake=1., hand_brake=role!="hero"))
                return actor
            self.ego = car(origin,"vehicle.lincoln.mkz","hero")
            if queue:
                for lane in (-1,-2,-3,-4):
                    p = self._map.get_waypoint_xodr(45,lane,495.)
                    if p is None or p.lane_type != carla.LaneType.Driving:
                        raise RuntimeError("Missing motorway queue lane")
                    actor = car(p,"vehicle.lincoln.mkz" if lane==-2 else "vehicle.tesla.model3","queue")
                    if lane==-2:
                        self.lead = actor
                    if lane==-1:
                        self.neighbor = actor
        except BaseException:
            self.close()
            raise

    def metrics(self):
        state = super().metrics()
        state.update(route_id="Town04 road 6 to road 45, lane -2",road_ids=[6,45],junction_ids=[],
                     phase="stationary four-lane queue" if self.lead else "empty divided motorway")
        return state


def switch(client, short):
    names = client.get_available_maps()
    name = next((n for n in names if n.rsplit("/",1)[-1]==short),None)
    if not name:
        raise ValueError(f"Native map absent: {short}")
    return client.load_world(name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port",type=int,default=3718)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--scratch",type=Path,required=True)
    args=p.parse_args()
    client=carla.Client("127.0.0.1",args.port,worker_threads=2)
    client.set_timeout(120.)
    original=client.get_world()
    old=original.get_map().name.rsplit("/",1)[-1]
    if old!="Town03_Opt":
        raise RuntimeError(f"Expected exclusive Town03 host, got {old}")
    actors=original.get_actors()
    if any(actors.filter(k) for k in ("vehicle.*","walker.*","sensor.*","controller.*")):
        raise RuntimeError("Host is in use")
    settings,weather=original.get_settings(),original.get_weather()
    args.output.mkdir(parents=True,exist_ok=False)
    fields=dict(frozen_checkpoint_sha256=fingerprint(args.model),fixed_window_m=[350,455],
                queue_xodr_s=495.,queue_road_lanes=[45,[-1,-2,-3,-4]],
                predeclared_route="Town04 road 6 s=16 → road 45 lane -2, continuous native 90 sign",
                fixture_sha256=fingerprint(Path(__file__)))
    write_json(args.output/"protocol.json",fields)
    try:
        switch(client,"Town04_Opt")
        with simulator(args.port,True,"High",map_name="Town04_Opt") as world:
            fields["native_opendrive_sha256"]=hashlib.sha256(world.get_map().to_opendrive().encode()).hexdigest()
            write_json(args.output/"protocol.json",fields)
            for case in ("matched-40","higher-speed","stationary-queue"):
                world.tick()
                run(world,args.model,args.output/case,case,sweeps=args.scratch/case,
                    fixture_class=Motorway,window=(350.,455.))
                # Do not score a nominal high-posted route lacking the native signs.
                with np.load(args.output/case/"observations.npz",allow_pickle=False) as tape:
                    if np.count_nonzero(tape["posted_limit_kmh"]==90.)<60:
                        raise RuntimeError("Native 90 km/h carriageway not observed")
    finally:
        restored=switch(client,old)
        restored.apply_settings(settings)
        restored.set_weather(weather)
        restored.tick()
        confirmed=restored.get_map().name.rsplit("/",1)[-1]
        if confirmed != old:
            raise RuntimeError(f"Host failed to restore {old}")
        write_json(args.output/"host-restored.json",dict(map=confirmed,synchronous=restored.get_settings().synchronous_mode,
                   fixed_delta_seconds=restored.get_settings().fixed_delta_seconds))
        print("Restored",confirmed,flush=True)


if __name__=="__main__":
    main()
