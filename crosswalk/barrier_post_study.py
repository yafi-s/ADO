"""Refit the bollard edge templates and compare provisional and confirmed labels on the native fixtures."""
import argparse
import io
import json
from zipfile import ZipFile

import cv2
import numpy as np

from .barrier import detect
from .barrier_posts import MODEL, THRESHOLD, ChainHistory, board_grade, post_tiles, review
from .barrier_study import ROOT, read_rgb, digest
from .camera_geometry import CAMERAS
from .surround_vision import SurroundDetector
from .tracking import Tracker

DATA = ROOT / "data/barrier-recognition"
RESULTS = ROOT / "results/barrier-recognition"
SOURCES = ("fit_0_chain_5", "fit_1_chain_5")


def sample(archive, name):
    image = read_rgb(archive.read(name + ".jpg"))
    with np.load(io.BytesIO(archive.read(name + ".npz"))) as packet:
        cloud = packet["lidar"].astype(np.float32)
    return image, [d for d in detect(image, cloud, "front", verify_posts=False) if d.get("barrier_subtype") == "chain"]


def fit():
    tiles = []
    with ZipFile(DATA / "town03-chains.zip") as archive:
        for name in SOURCES:
            image, found = sample(archive, name)
            candidate, = found
            tiles.extend(post_tiles(image, CAMERAS["front"], candidate))
    return np.stack(tiles)


def compare(archive_name, raw_name):
    raw = json.loads((RESULTS/raw_name).read_text(encoding="utf-8"))
    frames, confirmed_count, causal_count = [], 0, 0
    histories = {}
    distance_counts = {}
    with ZipFile(DATA/archive_name) as archive:
        manifest={r["name"]:r for r in json.loads(archive.read("manifest.json"))}
        for row in raw["frames"]:
            raw_chain = [m for m in row["measurements"] if m["subtype"]=="chain"]
            if not raw_chain:
                continue
            image, boxes = sample(archive, row["name"])
            grades = [review(image,CAMERAS["front"],box) for box in boxes]
            if len(raw_chain) != len(grades):
                raise ValueError("Unresolved raw box-to-track echo in fixture " + row["name"])
            confirmed_count += sum(item["confirmed"] for item in grades)
            # The three exposures at each fixed pose are one short sequence. Reset history
            # between independent fixtures so a previous, teleported barrier cannot count.
            key=row["name"].rsplit("_s",1)[0]
            sequence=histories.setdefault(key,ChainHistory())
            meta=manifest[row["name"]]
            causal=[]
            for d, appearance, spatial in zip(boxes,grades,raw_chain,strict=True):
                candidate=dict(d,range_supported=True,track_id=1)
                decision=sequence.qualify(candidate,appearance,(0.,0.,0.),meta["world_frame"]*.1,spatial["chain_support"])
                causal.append(decision)
                causal_count+=int(decision["confirmed"])
                distance=meta["distance_m"]
                distance_counts[distance]=distance_counts.get(distance,0)+int(decision["confirmed"])
            frames.append(dict(name=row["name"],raw_chain_tracks=len(raw_chain),post_confirmation=grades,
                               presented_chain=causal))
    return dict(exposures=raw["exposures"], actual_chain_exposures=raw["by_condition"]["chain"]["exposures"],
                raw_chain_tracks=sum(r["raw_chain_tracks"] for r in frames), appearance_confirmed_chain_tracks=confirmed_count,
                causal_confirmed_chain_tracks=causal_count,causal_confirmed_by_distance_m=distance_counts,
                raw_unmatched_predictions=raw["unmatched_predictions"],frames=frames,
                archive_sha256=raw["archive_sha256"])


def board_comparison():
    detector = SurroundDetector()
    output = {}
    for archive_name in ("town03-native.zip", "town03-chains.zip", "town03-chain-validation.zip", "town03-chain-high.zip"):
        exposed = []
        with ZipFile(DATA / archive_name) as archive:
            for meta in json.loads(archive.read("manifest.json")):
                if meta["condition"] != "barricade":
                    continue
                name = meta["name"]
                rgb = read_rgb(archive.read(name+".jpg"))
                with np.load(io.BytesIO(archive.read(name+".npz"))) as packet:
                    cloud = packet["lidar"].astype(np.float32)
                boxes = detector.detect(rgb) + detect(rgb, cloud, "front")
                tracks = Tracker(surround=True).update(cloud,np.empty((0,4)),{"front":boxes},(0.,0.,0.),0.)
                for track in tracks:
                    if track.get("barrier_subtype")=="striped_board" and track["cameras"]:
                        exposed.append(dict(name=name, **board_grade(track["surface_xyz"])))
        output[archive_name] = dict(raw_ranged_boards=len(exposed), confirmed=sum(t["confirmed"] for t in exposed),
                                    minimum_observed_diameter_m=min(t["measured_diameter_m"] for t in exposed),
                                    tracks=exposed)
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit",action="store_true")
    args=p.parse_args()
    cv2.setNumThreads(2)
    fitted=fit()
    if args.fit:
        np.savez_compressed(MODEL,posts=fitted)
    with np.load(MODEL) as packet:
        if not np.array_equal(fitted,packet["posts"]):
            raise ValueError("Published post templates differ from the training crop fit")
    rows=dict(development=compare("town03-chains.zip","chain-development.json"),
              low=compare("town03-chain-validation.zip","chain-validation.json"),
              high=compare("town03-chain-high.zip","chain-high.json"))
    report=dict(schema="ado-post-appearance-fit-v1", threshold=THRESHOLD, model_sha256=digest(MODEL.read_bytes()),
                training_exposures=SOURCES, training_posts=4, fixtures=rows,
                striped_board_breadth=board_comparison())
    (RESULTS/"post-appearance-fit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print({k:(v["raw_chain_tracks"],v["appearance_confirmed_chain_tracks"],v["causal_confirmed_chain_tracks"]) for k,v in rows.items()})
