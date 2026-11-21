"""Reconstruct the single-scale, 60 m observer on all native city capture sweeps."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosswalk.radar_geometry import CORNER_RADARS
from crosswalk.surround import _ImageAssociations
from crosswalk.tracking import Tracker


def ablate(source, sweeps, output):
    output.mkdir(exist_ok=False, parents=True)
    shutil.copyfile(source / "town.json", output / "town.json")
    observer = Tracker(surround=True)
    images = _ImageAssociations()
    with (source / "perception.jsonl").open() as rows, (output / "perception.jsonl").open("x") as target:
        for line in rows:
            f = json.loads(line)
            fresh = {}
            for view, observations in f["camera_detections"].items():
                full = [{k: v for k, v in b.items() if k not in
                         ("image_id", "range_supported", "track_id", "range_source")}
                        for b in observations if b.get("inference_layout") != "front-horizon-2x"]
                fresh[view] = images.update(view, full, f["sim_time"])
            with np.load(sweeps / f"{f['frame']}.npz", allow_pickle=False) as measurement:
                scan = measurement["lidar"]
                xyz = scan[:, :3]
                # Preserve return indices referenced by native striped-board/chain associations.
                near = scan.copy()
                near[np.einsum("ij,ij->i", xyz, xyz) > 60**2] = np.nan
                f["tracks"] = observer.update(near, measurement["radar"], fresh, measurement["ego_xy_yaw"],
                                              f["sim_time"], corner_radars={k: measurement[k] for k in CORNER_RADARS})
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
            f["camera_detections"] = fresh
            if "front" in fresh:
                f["detections"] = fresh["front"]
            target.write(json.dumps(f, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "sweeps", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    ablate(args.source, args.sweeps, args.output)
