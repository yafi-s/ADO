"""Review recorded chain proposals from synchronized source JPEGs without consulting scene truth.

A compact witness ZIP contains every proposed chain's input geometry and exact
native JPEG. The same review can be reproduced from that ZIP without CARLA.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import cv2
import numpy as np

from .barrier_posts import (MODEL, THRESHOLD, MIN_BOARD_BREADTH_M, ChainHistory, add_observation,
                            board_grade, range_only_decision, review)
from .camera_geometry import CAMERAS


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, separators=(",", ":"), ensure_ascii=False)+"\n").encode("utf-8")


def jpeg_images(path, stop, streams):
    """Read concatenated native JPEGs in order. Hash the complete stream for finished captures."""
    sha = hashlib.sha256()
    pending = b""
    count = 0
    with path.open("rb") as file:
        while stop is None or count<stop:
            end = pending.find(b"\xff\xd9")
            if end >= 0:
                blob, pending = pending[:end+2], pending[end+2:]
                if not blob.startswith(b"\xff\xd8"):
                    raise ValueError(f"Invalid JPEG boundary in {path.name} at image {count}")
                yield count, blob
                count += 1
                continue
            chunk = file.read(4*1024*1024)
            if not chunk:
                if pending and stop is None:
                    raise ValueError(f"Truncated JPEG in {path.name}")
                break
            sha.update(chunk)
            pending += chunk
    if stop is None:
        streams[path.stem] = dict(images=count, sha256=sha.hexdigest())


def collect_witness(raw, output, *, steps=None):
    proposals = defaultdict(dict)
    range_only = []
    hashed = hashlib.sha256()
    count, first, last = 0, None, None
    with (raw/"perception.jsonl").open("rb") as source:
        for index, line in enumerate(source):
            if steps is not None and index>=steps: break
            if not line.endswith(b"\n"):
                if steps is None:
                    raise ValueError("The native journal is still writing")
                break
            hashed.update(line)
            row = json.loads(line)
            frame = row["frame"]
            first = frame if first is None else first
            if last is not None and frame != last + 1:
                raise ValueError("Native world frames are not consecutive")
            last = frame
            count += 1
            tracks={t["id"]:t for t in row["tracks"]}
            for track in tracks.values():
                if track.get("barrier_subtype")=="striped_board" and not track["coasting"] and not track["cameras"]:
                    range_only.append(dict(frame=frame,track_id=track["id"],surface_xyz=track["surface_xyz"]))
            for view, boxes in row["camera_detections"].items():
                hits = [d for d in boxes if d.get("barrier_subtype") in ("chain","striped_board")]
                if not hits:
                    continue
                keys=("box","barrier_subtype","chain_ends_xy","chain_support","image_id","track_id",
                      "range_supported","subtype_confidence")
                filtered = [{k:d[k] for k in keys if k in d} for d in hits]
                for d in filtered:
                    track=tracks.get(d["track_id"],{})
                    if d["barrier_subtype"]=="chain":
                        d["exclusive_support"]=track.get("chain_support")
                    else:
                        d["surface_xyz"]=track.get("surface_xyz",[])
                proposals[view][index] = dict(frame=frame, sim_time_s=row["sim_time"], view=view,
                                              ego_xy_yaw=row["observer_map"]["ego"],proposals=filtered)
    packets, streams = [], {}
    with ZipFile(output, "x") as archive:
        for view in CAMERAS:
            for index, jpeg in jpeg_images(raw/(view+".mjpeg"), steps, streams):
                if index not in proposals[view]:
                    continue
                packet=proposals[view][index]
                name=f"frames/{packet['frame']}-{view}.jpg"
                archive.writestr(ZipInfo(name), jpeg, compress_type=ZIP_STORED)
                packets.append(dict(packet, source_jpeg=name, source_jpeg_sha256=digest(jpeg)))
            if steps is None and streams[view]["images"] != count:
                raise ValueError(f"{view} exposure count differs from the native journal")
        packets.sort(key=lambda x:(x["frame"],x["view"]))
        if len(packets)!=sum(len(items) for items in proposals.values()):
            raise ValueError("At least one source camera exposure is unavailable")
        info=dict(schema="ado-barrier-post-witness-v1", first_frame=first, last_frame=last,
                  frames=count,partial=steps is not None, raw_journal_sha256=hashed.hexdigest(),
                  native_mjpeg=streams, packets=packets,range_only=range_only)
        archive.writestr(ZipInfo("inputs.json"),json_bytes(info),compress_type=ZIP_DEFLATED)


def summarize(witness):
    decisions = {}
    checked = positives = appearances = 0
    subtypes=defaultdict(Counter)
    history=ChainHistory()
    with ZipFile(witness) as archive:
        inputs=json.loads(archive.read("inputs.json"))
        for item in inputs["range_only"]:
            event=decisions.setdefault(str(item["frame"]),{})
            event[str(item["track_id"])]=range_only_decision(item["surface_xyz"])
        for packet in inputs["packets"]:
            jpeg=archive.read(packet["source_jpeg"])
            if digest(jpeg)!=packet["source_jpeg_sha256"]:
                raise ValueError("Native exposure digest differs from witness manifest")
            rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            view=packet["view"]
            frame=decisions.setdefault(str(packet["frame"]),{})
            for raw in packet["proposals"]:
                subtype=raw["barrier_subtype"]
                if subtype=="chain":
                    appearance=review(rgb,CAMERAS[view],raw)
                    grade=history.qualify(raw,appearance,packet["ego_xy_yaw"],packet["sim_time_s"],raw["exclusive_support"])
                    appearances+=int(appearance["confirmed"])
                    subtypes[subtype]["appearance"]+=int(appearance["confirmed"])
                else:
                    grade=board_grade(raw["surface_xyz"])
                subtypes[subtype]["raw"]+=1
                subtypes[subtype]["confirmed"]+=int(grade["confirmed"])
                checked+=1
                positives+=int(grade["confirmed"])
                add_observation(frame,view,raw,grade,source_jpeg_sha256=packet["source_jpeg_sha256"])
    code=Path(__file__).with_name("barrier_posts.py").read_bytes().replace(b"\r\n", b"\n")
    return dict(schema="ado-barrier-post-review-v1",first_frame=inputs["first_frame"],last_frame=inputs["last_frame"],
                capture_frames=inputs["frames"], partial=inputs["partial"], threshold=THRESHOLD,
                reviewed_subtypes=["chain","striped_board"], minimum_board_breadth_m=MIN_BOARD_BREADTH_M,
                counts_by_subtype={k:dict(v) for k,v in subtypes.items()},
                native_journal_sha256=inputs["raw_journal_sha256"],native_mjpeg=inputs["native_mjpeg"],
                witness_sha256=digest(Path(witness).read_bytes()), model_sha256=digest(MODEL.read_bytes()),
                post_classifier_lf_sha256=digest(code),
                review_code_lf_sha256=digest(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
                source_camera_observations=checked,
                appearance_matched_camera_observations=appearances,post_confirmed_camera_observations=positives, frames=decisions)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,help="collect exact proposal inputs from a completed native raw directory")
    parser.add_argument("--witness",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,help="read only this prefix as a partial local inspection")
    args=parser.parse_args()
    cv2.setNumThreads(2)
    if args.raw:
        collect_witness(args.raw,args.witness,steps=args.steps)
    result=summarize(args.witness)
    args.output.write_bytes(json_bytes(result))
    print({k:result[k] for k in ("capture_frames","source_camera_observations","post_confirmed_camera_observations",
                                  "partial","native_journal_sha256","witness_sha256")})
