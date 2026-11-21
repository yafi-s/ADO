"""Audit fresh native barrier labels against physical scene boards and cooked-map references."""
from collections import Counter
import argparse
import hashlib
from zipfile import ZIP_DEFLATED, ZipFile
import gzip
import json
import math
from pathlib import Path

import numpy as np

from .barrier_journey_audit import evaluate, journal_stream
from .barrier_study import ROOT

REGISTRY=ROOT/"results/barrier-recognition/town03-barrier-scoring-registry.json.gz"


def extract_barrier_rows(native, filename):
    """Keep the frame/pose, original barrier observations and independent barrier actor references for every tick."""
    native_sha = hashlib.sha256()
    rows_sha = hashlib.sha256()
    with open(native, "rb") as source, ZipFile(filename, "x", ZIP_DEFLATED) as archive:
        with archive.open("rows.jsonl", "w") as output:
            for line in source:
                native_sha.update(line)
                r = json.loads(line)
                observer = r["observer_map"]
                subset = dict(frame=r["frame"], sim_time=r["sim_time"],observer_map=dict(ego=observer["ego"],
                    others=[a for a in observer["others"] if a["type"]=="barrier"]),
                    tracks=[t for t in r["tracks"] if t["label"]=="barrier"],
                    camera_detections={v:[d for d in boxes if d.get("barrier_subtype")]
                                       for v,boxes in r["camera_detections"].items()})
                raw=(json.dumps(subset,separators=(",",":"))+"\n").encode()
                rows_sha.update(raw)
                output.write(raw)
        archive.writestr("manifest.json",json.dumps(dict(parent_native_journal_sha256=native_sha.hexdigest(),
                        rows_sha256=rows_sha.hexdigest(), selection="all world frames, measured barriers and reference barrier actors")))


def score(journal, sidecar):
    summary=evaluate(journal)
    decisions=json.loads(Path(sidecar).read_bytes())
    native_sha=summary["uncompressed_journal_sha256"]
    if Path(journal).suffix==".zip":
        with ZipFile(journal) as archive:
            meta=json.loads(archive.read("manifest.json"))
        if meta["rows_sha256"]!=native_sha:
            raise ValueError("The compact audit rows differ from their source manifest")
        native_sha=meta["parent_native_journal_sha256"]
    if decisions["partial"] or decisions["native_journal_sha256"]!=native_sha:
        raise ValueError("Barrier review and independently scored full journal differ")
    registry=json.loads(gzip.decompress(REGISTRY.read_bytes()))["objects"]
    coordinates=np.array([[r[2],r[3]] for r in registry])
    native_boards=[r for r in registry if "streetbarrier" in r[0].lower()]
    counts=Counter()
    matched_map_assets=Counter()
    board_misses=[]
    chains=[]
    with journal_stream(journal) as source:
        for line in source:
            row=json.loads(line)
            frame=row["frame"]
            ex,ey,yaw=row["observer_map"]["ego"]
            cs,sn=math.cos(math.radians(yaw)),math.sin(math.radians(yaw))
            def to_world(x,y):
                return ex+cs*x-sn*y,ey+sn*x+cs*y
            scene=[(a["x"],a["y"],a["actor_id"]) for a in row["observer_map"]["others"] if a["type"]=="barrier"]
            for t in row["tracks"]:
                if t["label"]!="barrier" or not t["cameras"] or not {"rgb","lidar"}<=set(t["sources"]):
                    continue
                x,y=to_world(t["x"],t["y"])
                subtype=t.get("barrier_subtype","unspecified")
                counts["raw_"+subtype]+=1
                if subtype=="striped_board":
                    grade=decisions["frames"].get(str(frame),{}).get(str(t["id"]))
                    if not grade:
                        raise ValueError("Missing ranged striped-board breadth review")
                    counts["breadth_confirmed_striped_boards"]+=int(grade["confirmed"])
                    nearest_script=min((math.hypot(x-a,y-b) for a,b,_ in scene),default=math.inf)
                    level=min(native_boards,key=lambda obj:math.hypot(x-obj[2],y-obj[3]))
                    error=math.hypot(x-level[2],y-level[3])
                    if nearest_script<=.65:
                        counts["striped_scene_matches"]+=1
                        counts["confirmed_striped_scene_matches"]+=int(grade["confirmed"])
                    elif error<=.65:
                        counts["striped_map_asset_matches"]+=1
                        counts["confirmed_striped_map_asset_matches"]+=int(grade["confirmed"])
                        if grade["confirmed"]:
                            matched_map_assets[level[0]]+=1
                    else:
                        board_misses.append(dict(frame=frame,id=t["id"],center_xy=[round(x,3),round(y,3)],
                                                nearest_named_board=level[0],center_distance_m=round(error,3),
                                                breadth_confirmed=grade["confirmed"],
                                                views=grade["observations"]))
                elif subtype=="chain":
                    grade=decisions["frames"].get(str(frame),{}).get(str(t["id"]))
                    if not grade:
                        raise ValueError("Missing review of a fresh chain track")
                    counts["post_confirmed_chain_tracks"]+=int(grade["confirmed"])
                    views=[]
                    for view in t["cameras"]:
                        match=[d for d in row["camera_detections"][view] if d.get("track_id")==t["id"]
                               and d.get("barrier_subtype")=="chain" and d["range_supported"]]
                        for d in match:
                            ends=[]
                            for post in d["chain_ends_xy"]:
                                xy=to_world(*post)
                                dist=np.hypot(coordinates[:,0]-xy[0],coordinates[:,1]-xy[1])
                                idx=int(dist.argmin())
                                near=registry[idx]
                                ends.append(dict(xy_m=[round(c,3) for c in xy],nearest_name=near[0],
                                    center_distance_m=round(float(dist[idx]),3),semantic_tag=near[1]))
                            views.append(dict(view=view,image_id=d["image_id"],scorer_nearest_posts=ends))
                    chains.append(dict(frame=frame,sim_time_s=row["sim_time"],track_id=t["id"],
                        raw_chain_support=t["chain_support"], raw_confidence=t["subtype_confidence"],
                        confirmed=grade["confirmed"],post_scores=grade["post_scores"],views=views))
    return dict(schema="ado-full-city-barrier-score-v1",capture_frames=summary["journal_frames"],
        native_journal_sha256=native_sha,raw_counts=dict(counts),
        named_map_barriers=len(native_boards),named_chain_map_objects=sum("chain" in r[0].lower() for r in registry),
        confirmed_named_map_boards=dict(matched_map_assets),
        planted_boards=summary["actors"],unmatched_striped_measurements=board_misses,
        chain_candidates=chains)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("journal",type=Path)
    p.add_argument("sidecar",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--extract",type=Path,help="archive review-relevant original fields from the full native journal")
    args=p.parse_args()
    if args.extract:
        extract_barrier_rows(args.journal,args.extract)
    result=score(args.journal,args.sidecar)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result["raw_counts"]))
