"""Matched seed-212 segments with the original city population, roadside props and stopped lead."""
import argparse
import gzip
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

from .city_drive import LoggedDriver, load_driver
from .city_progress_trials import Witness, interval, route_trial
from .drive import episode
from .experiment import fingerprint, save_arrays, write_json, write_trace
from .journey import JourneyScenario
from .runtime import producer_info, simulator
from .showcase import encode_and_check
from .world import _raised

# Both checkpoints get exactly the original City Life arrangement, with the
# same approach from rest. Relative times and traffic trajectories can diverge.
CASES={
    "boulevard": (350.,626.,((365.,401.),(375.,395.),(420.,500.),(525.,550.),(550.,565.),(568.,583.),(600.,620.))),
    "bend": (667.,747.,((675.,700.),(704.7,735.4),(716.,731.))),
    "hidden": (97.,172.,((124.,148.),(150.,168.))),
    "left": (995.,1085.,((1025.,1045.),(1046.,1077.))),
    "red": (278.,363.,((299.,323.),(324.,352.))),
}


def trial(world,checkpoint,folder,raw,case):
    start,end,windows=CASES[case]
    driver,names,schema=load_driver(checkpoint)
    model=LoggedDriver(driver)
    folder.mkdir(parents=True,exist_ok=False)
    raw.mkdir(parents=True,exist_ok=False)
    with (raw/"front.mjpeg").open("xb") as stream, gzip.open(folder/"reference.jsonl.gz","xt",encoding="utf8") as reference:
        target={"boulevard":"lead","hidden":"occluded","left":"left","red":"signal"}.get(case)
        witness=Witness(stream,reference,target)
        def factory(*args,**kwargs):
            scene=JourneyScenario(*args,**kwargs,population_profile="city-life",barriers=True)
            i=int(np.argmin(abs(np.asarray(scene._station)-start)))
            scene.ego.set_transform(_raised(scene._points[i].transform,.35))
            witness.scene=scene
            return scene
        def finished():
            return witness.scene._coordinates(witness.scene.ego.get_location())[0]>=end
        summary,trace,_,_=episode(world,"journey",212,schema,model,scene_factory=factory,
            destination="market",weather="ClearNoon",crossing_delay=.5,recording=witness,tactical=True,
            surround=case=="boulevard",change_request=finished,max_steps=1100)
    write_trace(folder/"trace.jsonl.gz",trace)
    save_arrays(folder/"observations.npz",frame=[r["frame"] for r in trace],features=model.observations,
                feature_names=names,requested_mps=model.requested_speed,posted_kmh=model.posted_limits)
    events={e["kind"]:e for e in summary["events"]}
    required={"boulevard":("roadworks","barrier","lead"),"bend":(),"hidden":("occluded",),"left":("left",),"red":("signal",)}[case]
    passed=(summary["progress_m"]>=end-.7 and not summary["collision_events"] and
            not summary["yield_violations"] and not summary["red_light_violations"] and
            all(events[k]["exposed"] and not events[k]["violation"] for k in required))
    if case in ("left","red","hidden"):
        passed=passed and events[target]["handled"]
    if case=="boulevard":
        passed=passed and events["roadworks"]["handled"] and events["barrier"]["handled"] and events["lead"]["stopped_for_event"]
    summary.update(case=case,original_city_life_geometry=True,surround=case=="boulevard",model_sha256=fingerprint(checkpoint),
        producer=producer_info(world),start_m=start,end_m=end,passed=bool(passed),nearest_target_footprint=witness.closest,
        windows=[interval(trace,*window) for window in windows],first_frame=trace[0]["frame"],last_frame=trace[-1]["frame"])
    write_json(folder/"summary.json",summary)
    print(case,schema,"passed",bool(passed),summary["windows"],flush=True)
    encode_and_check(raw/"front.mjpeg",folder/"front.mp4",(640,360),0,len(trace))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port",type=int,required=True)
    p.add_argument("--attach",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--scratch",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--cases",nargs="+",choices=CASES,default=list(CASES))
    a=p.parse_args()
    cv2.setNumThreads(1)
    a.output.mkdir(parents=True,exist_ok=True)
    with ZipFile(a.output/"source.zip","x",ZIP_DEFLATED) as archive:
        for source in sorted(Path(__file__).parent.glob("*.py")):
            archive.write(source,arcname=f"crosswalk/{source.name}")
    checkpoint=a.output/"checkpoint.npz"
    with checkpoint.open("xb") as f:
        f.write(a.model.read_bytes())
    write_json(a.output/"protocol.json",dict(seed=212,geometry="Town03 original City Life",cases=a.cases,
               windows=CASES,negative_controls="Original city_progress_trials seeds 623/624/625; matched v06 at results/city-progress/trials/matched-candidate-v4.",source_sha256=fingerprint(a.output/"source.zip"),checkpoint_sha256=fingerprint(checkpoint)))
    with simulator(a.port,a.attach,quality="High",map_name="Town03_Opt") as world:
        for case in a.cases:
            frozen={"left":"left","hidden":"occluded","red":"signal"}.get(case)
            if frozen:
                # Reuse the original matched v0.6 control geometry and seeds.
                # The native v0.6 arms and their exact driver are already retained.
                route_trial(world,checkpoint,a.output/case,a.scratch/case,frozen)
            else:
                trial(world,checkpoint,a.output/case,a.scratch/case,case)

if __name__=="__main__":
    main()
