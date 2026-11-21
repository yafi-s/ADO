"""Reconstruct the single-scale, 60 m observer on all native city capture sweeps."""
import argparse
from contextlib import nullcontext
from itertools import islice
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosswalk.radar_geometry import CORNER_RADARS
from crosswalk.barrier_posts import ChainHistory, add_observation, board_grade, range_only_decision
from crosswalk.surround import _ImageAssociations
from crosswalk.tracking import Tracker


def ablate(source, sweeps, output, limit=None, verify=False, native_range=False, keep_horizon=False,
           verification_only=False):
    cv2.setNumThreads(1)
    output.mkdir(exist_ok=False, parents=True)
    if not verification_only:
        shutil.copyfile(source / "town.json", output / "town.json")
    observer = Tracker(surround=True)
    images = _ImageAssociations()
    chains = ChainHistory()
    count=0
    verify=verify or verification_only
    native_range = native_range or verify
    keep_horizon = keep_horizon or verify
    source_hash=hashlib.sha256()
    destination=nullcontext() if verification_only else (output / "perception.jsonl").open("x",encoding="utf8",newline="\n")
    with (source / "perception.jsonl").open() as rows, destination as target:
        for index, line in enumerate(islice(rows, limit)):
            count+=1
            source_hash.update(line.encode("utf8"))
            f = json.loads(line)
            now = index*.1
            if abs(now-f["sim_time"]) > 1e-6:
                raise ValueError("Unaligned acquisition time")
            fresh = {}
            for view, observations in f["camera_detections"].items():
                full = [{k: v for k, v in b.items() if k not in
                         ("image_id", "range_supported", "track_id", "range_source")}
                        for b in observations if keep_horizon or b.get("inference_layout") != "front-horizon-2x"]
                fresh[view] = images.update(view, full, now)
            with np.load(sweeps / f"{f['frame']}.npz", allow_pickle=False) as measurement:
                scan = measurement["lidar"]
                xyz = scan[:, :3]
                # Preserve return indices referenced by native striped-board/chain associations.
                if not native_range:
                    scan[np.einsum("ij,ij->i", xyz, xyz) > 60**2] = np.nan
                pose=measurement["ego_xy_yaw"]
                tracks=observer.update(scan, measurement["radar"], fresh, pose, now,
                                       corner_radars={k: measurement[k] for k in CORNER_RADARS})
            if verify and tracks != f["tracks"]:
                raise ValueError(f"Full-sweep reconstruction diverged at {f['frame']}")
            if verification_only:
                continue
            f["tracks"] = tracks
            grades={str(t["id"]):range_only_decision(t["surface_xyz"]) for t in tracks
                    if t.get("barrier_subtype")=="striped_board" and not t["coasting"] and not t["cameras"]}
            by_id={t["id"]:t for t in tracks}
            for view, observations in fresh.items():
                supported = {tuple(track["rgb_boxes"][view]): track["id"]
                             for track in f["tracks"] if view in track["rgb_boxes"]}
                for track in f["tracks"]:
                    cycle = track.get("cycle_association")
                    if cycle and cycle["camera"] == view and cycle["age_s"] == 0:
                        for key in ("rider_box", "bicycle_box"):
                            supported[tuple(cycle[key])] = track["id"]
                for box in observations:
                    track_id = supported.get(tuple(box["box"]))
                    box.update(track_id=track_id, range_supported=track_id is not None)
                    track=by_id.get(track_id,{})
                    if box.get("barrier_subtype")=="striped_board":
                        add_observation(grades,view,box,board_grade(track.get("surface_xyz",[])))
                    elif box.get("barrier_subtype")=="chain" and "post_appearance" in box:
                        grade=chains.qualify(box,box["post_appearance"],pose,now,track.get("chain_support"))
                        add_observation(grades,view,box,grade)
            f["barrier_review"]=grades
            f["camera_detections"] = fresh
            if "front" in fresh:
                f["detections"] = fresh["front"]
            target.write(json.dumps(f, separators=(",", ":")) + "\n")
    with (output/"ablation.json").open("w",encoding="utf8",newline="\n") as log:
        json.dump(dict(frames=count,source_sha256_lf=source_hash.hexdigest(),full_native_range=native_range,
                       horizon_included=keep_horizon,verified_against_native_tracks=verify),log,indent=2)
        log.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "sweeps", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verify-native", action="store_true")
    parser.add_argument("--verification-only", action="store_true",help="Verify the native tracks, saving only a summary")
    parser.add_argument("--retain-native-range", action="store_true")
    parser.add_argument("--retain-horizon", action="store_true")
    args = parser.parse_args()
    ablate(args.source, args.sweeps, args.output,args.limit,args.verify_native,args.retain_native_range,args.retain_horizon,
           args.verification_only)
