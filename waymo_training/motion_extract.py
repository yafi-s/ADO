"""Extract a bounded WOMD v1.3.1 longitudinal pilot in an authorized Colab runtime.

The wire projection uses field numbers from the Apache-2.0 official Waymo
scenario.proto and map.proto at 99a4cb3ff07e2fe06c2ce73da001f850f628e45a.
Unknown fields (including image tokens) are never model inputs.
"""
import argparse
import hashlib
import re
import struct
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from filelock import FileLock
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from .extract import atomic_json, masked_crc, sha256_file
from .interaction import ActorTrack, FEATURE_NAMES, HORIZONS, make_features

BUCKET = "waymo_open_dataset_motion_v_1_3_1"
SPLITS = ("training", "validation")
PREFIX = {s: f"uncompressed/scenario/{s}/" for s in SPLITS}
MIB = 1024 ** 2
LIMIT = 2047 * MIB
VERSION = "womd-speed-v1"


def scenario_schema():
    """A read-only protobuf wire projection for the fields consumed below."""
    fd = descriptor_pb2.FileDescriptorProto(name="motion_projection.proto", syntax="proto2")
    def add(name, fields):
        msg = fd.message_type.add(name=name)
        for number, key, typ, repeated in fields:
            field = msg.field.add(name=key, number=number, label=3 if repeated else 1)
            if isinstance(typ, str):
                field.type, field.type_name = 11, "." + typ
            else:
                field.type = typ
    add("Point", [(1,"x",1,False), (2,"y",1,False)])
    add("Lane", [(2,"type",5,False), (8,"polyline","Point",True)])
    add("MapFeature", [(3,"lane","Lane",False)])
    add("State", [(2,"center_x",1,False), (3,"center_y",1,False), (8,"heading",2,False),
                  (9,"velocity_x",2,False), (10,"velocity_y",2,False), (11,"valid",8,False)])
    add("Track", [(2,"object_type",5,False), (3,"states","State",True)])
    add("Scenario", [(1,"timestamps_seconds",1,True), (2,"tracks","Track",True),
                     (5,"scenario_id",9,False), (6,"sdc_track_index",5,False),
                     (8,"map_features","MapFeature",True), (10,"current_time_index",5,False)])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("Scenario"))


SCENARIO = scenario_schema()
KINDS = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}


def lane_geometry(scenario, ego):
    """Pick a heading-compatible lane at the current pose, without future route labels."""
    best, bend = float("inf"), None
    for feature in scenario.map_features:
        if not feature.HasField("lane") or feature.lane.type not in (1, 2):
            continue
        pts = feature.lane.polyline
        if len(pts) < 4:
            continue
        # Waymo lane samples are dense. A coarse search avoids decoding distant lanes.
        coarse = range(0, len(pts), 8)
        i = min(coarse, key=lambda j: (pts[j].x-ego.center_x)**2 + (pts[j].y-ego.center_y)**2)
        if math.hypot(pts[i].x-ego.center_x, pts[i].y-ego.center_y) > 12:
            continue
        i = min(range(max(i-8, 0), min(i+9, len(pts)-1)),
                key=lambda j: (pts[j].x-ego.center_x)**2 + (pts[j].y-ego.center_y)**2)
        dist = math.hypot(pts[i].x-ego.center_x, pts[i].y-ego.center_y)
        def heading(j):
            a, b = pts[max(j-2, 0)], pts[min(j+2, len(pts)-1)]
            return math.atan2(b.y-a.y, b.x-a.x)
        initial = heading(i)
        delta = math.atan2(math.sin(initial-ego.heading), math.cos(initial-ego.heading))
        score = dist + 2 * abs(delta)
        if dist > 4 or abs(delta) > .65 or score >= best:
            continue
        j, length = i, 0.
        while j+1 < len(pts) and length < 15:
            length += math.hypot(pts[j+1].x-pts[j].x, pts[j+1].y-pts[j].y)
            j += 1
        best = score
        bend = (math.atan2(math.sin(heading(j)-initial), math.cos(heading(j)-initial))
                if length >= 12 else None)
    return (0., False) if bend is None else (bend, True)


def valid(state, fields):
    return state.valid and all(state.HasField(f) and math.isfinite(getattr(state, f)) for f in fields)


def decode(payload):
    item = SCENARIO.FromString(payload)
    t = item.current_time_index
    stamps = item.timestamps_seconds
    if (not item.scenario_id or not item.HasField("sdc_track_index") or
            not item.HasField("current_time_index") or not 0 <= item.sdc_track_index < len(item.tracks)):
        raise ValueError("invalid scenario identity or SDC index")
    ego = item.tracks[item.sdc_track_index].states
    offsets = (-10, -5, 0, 10, 20, 30)
    if t < 10 or t+30 >= min(len(stamps), len(ego)):
        return None
    if any(not math.isclose(stamps[t+d] - stamps[t], d / 10, abs_tol=.02) for d in offsets):
        return None
    velocity = ("velocity_x","velocity_y")
    position = ("center_x","center_y")
    if not all(valid(ego[t+d], velocity) for d in offsets) or not valid(ego[t], (*position, "heading")):
        return None
    def speed(state):
        return math.hypot(state.velocity_x, state.velocity_y)
    now = ego[t]
    c, s = math.cos(now.heading), math.sin(now.heading)
    def local(x, y):
        return c*x+s*y, -s*x+c*y
    tracks = []
    for idx, track in enumerate(item.tracks):
        if idx == item.sdc_track_index or track.object_type not in KINDS or len(track.states) <= t:
            continue
        current, past = track.states[t], track.states[t-5]
        if not valid(current, position):
            continue
        x, y = local(current.center_x-now.center_x, current.center_y-now.center_y)
        if x < 0 or x > 80 or abs(y) > 15:
            continue
        if valid(past, position):
            dt = stamps[t] - stamps[t-5]
            vx, vy = local((current.center_x-past.center_x)/dt, (current.center_y-past.center_y)/dt)
        elif valid(current, velocity):
            vx, vy = local(current.velocity_x, current.velocity_y)
        else:
            continue
        tracks.append(ActorTrack(KINDS[track.object_type], x, y, vx, vy))
    bend, mask = lane_geometry(item, now)
    features = make_features([speed(ego[t+d]) for d in offsets[:3]], tracks, bend, mask)
    targets = np.array([speed(ego[t+d]) for d in offsets[3:]], dtype=np.float32)
    return item.scenario_id, features, targets


def staged_objects(raw):
    manifest = json.loads((raw/"manifest.json").read_text())
    if manifest.get("bucket", BUCKET) != BUCKET:
        raise ValueError("unexpected staged dataset bucket")
    objects = []
    seen = set()
    for entry in manifest["objects"]:
        name = entry["name"]
        match = re.fullmatch(r"uncompressed/scenario/(training|validation)/(training|validation)\.tfrecord-\d{5}-of-\d{5}", name)
        if not match or match[1] != match[2] or name in seen:
            raise ValueError("invalid or duplicate official WOMD scenario object")
        seen.add(name)
        if (not str(entry["generation"]).isdigit() or not re.fullmatch(r"[0-9a-f]{64}",entry["sha256"])
                or int(entry["size"])<=0):
            raise ValueError("invalid object identity, hash or size")
        objects.append(dict(split=match[1],name=name,generation=str(entry["generation"]),
                            size=int(entry["size"]),sha256=entry["sha256"]))
    if set(o["split"] for o in objects)!=set(SPLITS) or sum(o["size"] for o in objects)>LIMIT:
        raise ValueError("staged object inventory must include both splits and fit the byte budget")
    return objects


def read_staged(path, job):
    if path.stat().st_size != job["size"]:
        raise ValueError("incorrect staged object size")
    digest = hashlib.sha256()
    read = 0
    with path.open("rb") as fh:
        while read < job["size"]:
            header = fh.read(12)
            if len(header)!=12:
                raise ValueError("truncated TFRecord header")
            length,ucrc = struct.unpack("<QI",header)
            if (masked_crc(header[:8])!=ucrc or not 0<length<=128*MIB
                    or read+length+16>job["size"]):
                raise ValueError("bad TFRecord length or checksum")
            payload=fh.read(length)
            footer=fh.read(4)
            if len(payload)!=length or len(footer)!=4 or masked_crc(payload)!=struct.unpack("<I",footer)[0]:
                raise ValueError("bad TFRecord checksum or truncated payload")
            for part in (header,payload,footer):
                digest.update(part)
            read+=length+16
            yield payload
    if digest.hexdigest()!=job["sha256"]:
        raise ValueError("staged file differs from remote transfer receipt")


def extract_shard(raw, root, job, index):
    path=root/f"part-{index:03}.npz"
    receipt_path=path.with_suffix(".json")
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt["job"]!=job or sha256_file(path)!=receipt["features_sha256"]:
            raise ValueError("motion checkpoint mismatch")
        return receipt
    source=raw/job["split"]/Path(job["name"]).name
    rows,skipped=[],0
    for payload in read_staged(source,job):
        row=decode(payload)
        if row is None:
            skipped+=1
        else:
            rows.append(row)
    if not rows:
        raise ValueError("no eligible scenarios in shard")
    ids,features,targets=zip(*rows)
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh,scenario_id=ids,x=np.stack(features),y=np.stack(targets))
    tmp.replace(path)
    receipt=dict(job=job,bytes=job["size"],prefix_sha256=job["sha256"],file=path.name,
                 features_sha256=sha256_file(path),rows=len(rows),skipped=skipped)
    atomic_json(receipt_path,receipt)
    print(f"{job['name']}: {len(rows)} eligible scenarios; {skipped} ineligible",flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--raw",type=Path,required=True,help="private, content-hashed official Colab shard directory")
    args=parser.parse_args()
    raw,root=args.raw.resolve(),args.out.resolve()
    if any(Path("/content") not in p.parents for p in (raw,root)) or not Path("/content").is_dir():
        parser.error("extract within authorized private Colab /content storage")
    root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/"extract.lock"),timeout=0):
        frozen=staged_objects(raw)
        settings=dict(version=VERSION,bucket=BUCKET,jobs=frozen,
                      source_hashes={n:sha256_file(Path(__file__).with_name(n))
                                     for n in ("motion_extract.py","interaction.py","extract.py")})
        plan=root/"plan.json"
        if plan.exists() and json.loads(plan.read_text())!=settings:
            raise ValueError("existing plan differs in content, object generation, or code")
        atomic_json(plan,settings)
        receipts=[extract_shard(raw,root,job,i) for i,job in enumerate(frozen)]
        seen,rows=set(),Counter()
        for receipt in receipts:
            with np.load(root/receipt["file"],allow_pickle=False) as batch:
                for scenario_id in batch["scenario_id"]:
                    if scenario_id in seen:
                        raise ValueError("repeated scenario ID within or across splits")
                    seen.add(scenario_id)
            rows[receipt["job"]["split"]]+=receipt["rows"]
        summary=dict(bucket=BUCKET,version=VERSION,features=FEATURE_NAMES,horizons_s=HORIZONS,
                     rows=rows,duplicate_scenario_ids=0,plan_sha256=sha256_file(plan),
                     reserved_bytes=sum(o["size"] for o in frozen),shards=receipts)
        atomic_json(root/"manifest.json",summary)
        print(f"Verified {dict(rows)} unique scenarios",flush=True)


if __name__=="__main__":
    main()
