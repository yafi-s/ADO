"""Recover the measured tactical run's inside-corridor preview from its committed lane choice."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import carla
import numpy as np

from crosswalk.overtake import PassingCorridor
from crosswalk.replay_pack import pack


def sha(path):
    with path.open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()


def annotate(directory, road_map):
    manifest_file=directory/"manifest.json"
    manifest=json.loads(manifest_file.read_bytes())
    if manifest.get("cohort")!="tactical-study": raise ValueError("Expected a tactical recording")
    if manifest.get("plan_provenance"): return
    source=directory/"replay.json"
    if sha(source) != manifest["sha256"]["replay.json"]: raise ValueError("Replay digest differs")
    frames=json.loads(source.read_bytes())
    corridor=PassingCorridor.from_map(road_map)
    commit=None
    for row in frames:
        pose=row["ego_pose"]
        xy=np.asarray([pose["x"],pose["y"]])
        info=row["tactical_pass"]
        if not info: continue
        station,_=corridor.coordinates(xy)
        if abs(station-info["corridor_station_m"])>.002: raise ValueError("Recorded pose and lane station differ")
        if info["committed"] and commit is None: commit=station
        if not info["committed"]: commit=None
        stations=station+np.array([3.,7.,13.,18.,25.,32.,40.])
        goals=corridor.planned_centers(stations[stations<=corridor.arc[-1]],commit)
        row["plan_xy"]=np.vstack((xy,goals)).round(3).tolist()
    with TemporaryDirectory(dir=directory.parent) as temp:
        root=Path(temp)
        full=root/"replay.json"
        full.write_text(json.dumps(frames,separators=(",",":"),allow_nan=False)+"\n",encoding="utf8",newline="\n")
        compact=pack(full)
        manifest["plan_provenance"]={"source":"Recorded ego pose, logged commitment, native road 4 OpenDRIVE lanes and the fitted planner's exact geometric preview inside this corridor.",
                                     "original_replay_sha256":sha(source)}
        for item in (full,compact):
            manifest["sha256"][item.name]=sha(item)
            item.replace(directory/item.name)
        manifest["catalog_entry"]["replay_pack_sha256"]=manifest["sha256"]["replay-pack.json.gz"]
        manifest_file.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8",newline="\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xodr",type=Path)
    parser.add_argument("directories",type=Path,nargs="+")
    args=parser.parse_args()
    road_map=carla.Map("Town03_Opt",args.xodr.read_text(encoding="utf8"))
    for directory in args.directories: annotate(directory,road_map)
