"""Extract a byte-bounded, segment-disjoint pilot from the official Waymo E2E dataset."""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path

import cv2
import google_crc32c
import numpy as np
import requests
import urllib3
from filelock import FileLock
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

BUCKET = "waymo_open_dataset_end_to_end_camera_v_1_0_0"
NAMES = {
    "train": ("training_202504031202_202504151040.tfrecord", 263),
    "val": ("val_202504211843.tfrecord", 93),
}
MIB = 1024 ** 2
LIMIT = 5 * 1024 * MIB
SCHEMA = Path(__file__).with_name("schema") / "e2ed.pb"
PAST = ("pos_x", "pos_y", "vel_x", "vel_y", "accel_x", "accel_y")
VIEWS = (1, 2, 3)


def load_schema():
    pool = descriptor_pool.DescriptorPool()
    for descriptor in descriptor_pb2.FileDescriptorSet.FromString(SCHEMA.read_bytes()).file:
        pool.Add(descriptor)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("waymo.open_dataset.E2EDFrame"))


FRAME = load_schema()


def masked_crc(data):
    value = google_crc32c.value(data)
    return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def complete_records(data):
    """A byte prefix can end in the middle of the last record."""
    offset = 0
    while offset + 12 <= len(data):
        length, checksum = struct.unpack_from("<QI", data, offset)
        if masked_crc(data[offset:offset + 8]) != checksum:
            raise ValueError("TFRecord length checksum failed")
        if not 0 < length < 128 * MIB:
            raise ValueError("invalid TFRecord length")
        end = offset + 12 + length
        if end + 4 > len(data):
            break
        payload = data[offset + 12:end]
        if masked_crc(payload) != struct.unpack_from("<I", data, end)[0]:
            raise ValueError("TFRecord payload checksum failed")
        yield payload
        offset = end + 4


def partition(source, group):
    if source == "train":
        return "train"
    value = hashlib.sha256(("waymo-e2e-pilot-v1/" + group).encode()).digest()
    return "dev" if int.from_bytes(value[:8], "big") % 3 == 0 else "test"


def states(trajectory, fields, length):
    columns = [getattr(trajectory, field) for field in fields]
    if any(len(column) != length for column in columns):
        raise ValueError("incomplete trajectory")
    result = np.asarray(columns, dtype=np.float32).T.copy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite trajectory")
    return result


def decode(payload, source):
    item = FRAME.FromString(payload)
    frame = item.frame
    match = re.fullmatch(r"([0-9a-f]{32})-([0-9]{3})", frame.context.name)
    if not match or not frame.HasField("timestamp_micros"):
        raise ValueError("unexpected frame identifier")
    group = match.group(1)
    cameras = {camera.name: camera.image for camera in frame.images}
    images = []
    for view in VIEWS:
        jpeg = cameras.get(view, b"")
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("invalid or missing camera JPEG")
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
        if bgr is None:
            raise ValueError("camera decode failed")
        thumb = cv2.resize(bgr, (96, 64), interpolation=cv2.INTER_AREA)
        images.append(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
    scores = np.full(3, -1, np.float32)
    rated = np.full((3, 20, 2), np.nan, np.float32)
    for index, trajectory in enumerate(item.preference_trajectories):
        if index >= len(scores):
            raise ValueError("unexpected number of preference trajectories")
        if trajectory.HasField("preference_score") and 0 <= trajectory.preference_score <= 10:
            scores[index] = trajectory.preference_score
            rated[index] = states(trajectory, ("pos_x", "pos_y"), 20)
    if not item.HasField("intent"):
        raise ValueError("missing driving intent")
    return dict(images=np.stack(images), past=states(item.past_states, PAST, 16),
                intent=np.int64(item.intent), future=states(item.future_states, ("pos_x", "pos_y"), 20),
                group=group, frame_index=np.int32(match.group(2)), timestamps=np.int64(frame.timestamp_micros), split=partition(source, group),
                rater_scores=scores, rater_trajectories=rated,
                rater_candidates=np.uint8(len(item.preference_trajectories)))


class Access:
    def __init__(self, root, initial_bytes, auth):
        self.ledger = root / "transfer-budget.json"
        if not self.ledger.exists():
            atomic_json(self.ledger, {"reserved_bytes": initial_bytes})
        self.lock = threading.Lock()
        self.token = ""
        self.expires = 0
        self.credentials = None
        self.refresh_request = None
        self.use_adc = auth == "adc"

    def headers(self):
        with self.lock:
            if self.use_adc:
                if self.credentials is None:
                    import google.auth
                    from google.auth.transport.requests import Request
                    self.credentials, _ = google.auth.default(
                        scopes=["https://www.googleapis.com/auth/devstorage.read_only"])
                    self.refresh_request = Request()
                if not self.credentials.valid:
                    self.credentials.refresh(self.refresh_request)
                token = self.credentials.token
            else:
                if time.monotonic() >= self.expires:
                    exe = shutil.which("gcloud.cmd") or shutil.which("gcloud")
                    if not exe:
                        raise RuntimeError("Install and sign in to the gcloud CLI")
                    self.token = subprocess.check_output(
                        [exe, "auth", "print-access-token"], text=True).strip()
                    self.expires = time.monotonic() + 2400
                token = self.token
            return {"Authorization": "Bearer " + token, "Accept-Encoding": "identity"}

    def reserve(self, size):
        with self.lock:
            used = json.loads(self.ledger.read_text())["reserved_bytes"]
            if used + size > LIMIT:
                raise RuntimeError("5 GiB cumulative transfer reservation limit reached")
            atomic_json(self.ledger, {"reserved_bytes": used + size})


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def extract_one(job, root, prefix, access):
    source, index = job
    stem = f"{source}-{index:05}"
    out = root / "features" / (stem + ".npz")
    receipt = root / "receipts" / (stem + ".json")
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if (record["source"] != source or record["shard"] != index or
                record["prefix_bytes"] != prefix or record["npz_sha256"] != sha256_file(out)):
            raise ValueError(f"checkpoint mismatch for {stem}")
        return record
    base, count = NAMES[source]
    name = f"{base}-{index:05}-of-{count:05}"
    url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{name}"
    with requests.Session() as session:
        headers = access.headers()
        response = session.get(url, params={"fields": "generation,size"}, headers=headers, timeout=30)
        response.raise_for_status()
        meta = response.json()
        total = int(meta["size"])
        if prefix > total:
            raise ValueError("requested range exceeds object size")
        access.reserve(prefix)
        with session.get(url, params={"alt": "media", "generation": meta["generation"]},
                         headers={**headers, "Range": f"bytes=0-{prefix - 1}"},
                         timeout=(15, 90), stream=True) as response:
            response.raise_for_status()
            if (response.status_code != 206 or
                    response.headers.get("Content-Range") != f"bytes 0-{prefix - 1}/{total}" or
                    response.headers.get("x-goog-generation") != meta["generation"]):
                raise ValueError("GCS did not return the pinned object range")
            data = response.raw.read(prefix + 1)
            if len(data) != prefix:
                raise ValueError("incomplete range")
    rows = []
    for payload in complete_records(data):
        rows.append(decode(payload, source))
    if not rows:
        raise ValueError("no complete frame in prefix")
    values = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez(stream, **values)
    tmp.replace(out)
    record = dict(source=source, shard=index, generation=meta["generation"], prefix_bytes=prefix,
                  prefix_sha256=hashlib.sha256(data).hexdigest(), rows=len(rows),
                  npz_sha256=sha256_file(out))
    atomic_json(receipt, record)
    print(f"{stem}: {len(rows)} validated frames", flush=True)
    return record


def make_jobs(train_shards, val_shards):
    jobs = []
    for source, n in (("train", train_shards), ("val", val_shards)):
        total = NAMES[source][1]
        if not 1 <= n <= total:
            raise ValueError("shard count outside split bounds")
        for rank, index in enumerate(np.linspace(0, total - 1, n, dtype=int)):
            jobs.append((rank / n, source, int(index)))
    return [(source, index) for _, source, index in sorted(jobs)]


def summarize(root, receipts):
    groups = collections.defaultdict(set)
    counts = collections.Counter()
    identities = set()
    ratings = collections.Counter()
    classes = collections.defaultdict(collections.Counter)
    zero_timestamps = 0
    for receipt in receipts:
        source, index = receipt["source"], receipt["shard"]
        path = root / "features" / f"{source}-{index:05}.npz"
        with np.load(path, allow_pickle=False) as data:
            zero_timestamps += int(np.count_nonzero(data["timestamps"] == 0))
            for split, group, index_in_segment, intent, scores, candidates in zip(
                    data["split"], data["group"], data["frame_index"], data["intent"],
                    data["rater_scores"], data["rater_candidates"], strict=True):
                ident = (str(group), int(index_in_segment))
                if ident in identities:
                    raise ValueError("duplicate segment/frame index in corpus")
                identities.add(ident)
                groups[str(split)].add(str(group))
                counts[str(split)] += 1
                classes[str(split)][int(intent)] += 1
                if source == "val":
                    ratings["frames_with_any_candidates"] += int(candidates > 0)
                    ratings["frames_with_valid_ratings"] += int(np.any(scores >= 0))
                    ratings["valid_candidate_scores"] += int(np.sum(scores >= 0))
    if sum(map(len, groups.values())) != len(set().union(*groups.values())):
        raise ValueError("cross-split segment overlap")
    charged = json.loads((root / "transfer-budget.json").read_text())["reserved_bytes"]
    summary = dict(bucket=BUCKET, schema_sha256=sha256_file(SCHEMA), rows=dict(counts),
                   groups={s: len(g) for s, g in groups.items()}, intent_counts=dict(classes),
                   official_val_ratings=dict(ratings), zero_timestamp_rows=zero_timestamps,
                   duplicate_segment_frame_indices=0,
                   cross_split_segment_overlap=0, reserved_transfer_bytes=charged,
                   validated_prefix_bytes=sum(r["prefix_bytes"] for r in receipts),
                   shards=receipts)
    atomic_json(root / "manifest.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "shards"}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train-shards", type=int, default=112)
    parser.add_argument("--val-shards", type=int, default=40)
    parser.add_argument("--prefix-mib", type=int, default=32)
    parser.add_argument("--auth", choices=("adc", "gcloud"), default="adc",
                        help="Use Colab application default credentials, or a signed-in gcloud CLI")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--prior-probe-bytes", type=int, default=0,
                        help="Previously transferred exploratory data charged to this pilot")
    args = parser.parse_args()
    root = args.out.resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents:
        parser.error("use a private data directory outside the repository")
    if not 1 <= args.prefix_mib <= 128 or args.prior_probe_bytes < 0:
        parser.error("invalid data budget")
    cv2.setNumThreads(1)
    for folder in ("features", "receipts"):
        (root / folder).mkdir(exist_ok=True, parents=True)
    jobs = make_jobs(args.train_shards, args.val_shards)
    if (len(jobs) * args.prefix_mib * MIB + args.prior_probe_bytes) > LIMIT:
        parser.error("requested cohort exceeds 5 GiB")
    # Keep a single active writer per output directory, including concurrent CLI invocations.
    with FileLock(str(root / "extract.lock"), timeout=0):
        run(root, args, jobs)


def run(root, args, jobs):
    plan = dict(version=2, jobs=jobs, prefix_mib=args.prefix_mib, schema_sha256=sha256_file(SCHEMA))
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != json.loads(json.dumps(plan)):
        raise ValueError("existing data directory has a different frozen extraction plan")
    atomic_json(plan_path, plan)
    access = Access(root, args.prior_probe_bytes, args.auth)
    def worker(job):
        for attempt in range(3):
            try:
                return extract_one(job, root, args.prefix_mib * MIB, access)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    urllib3.exceptions.HTTPError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
            except requests.exceptions.HTTPError as error:
                if attempt == 2 or error.response.status_code not in (429, 500, 502, 503, 504):
                    raise
                time.sleep(2 ** attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = []
        pending = set()
        job_iter = iter(jobs)
        for _ in range(args.workers):
            pending.add(executor.submit(worker, next(job_iter)))
        while pending:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED)
            records.extend(task.result() for task in done)
            for _ in done:
                job = next(job_iter, None)
                if job is not None:
                    pending.add(executor.submit(worker, job))
    records.sort(key=lambda r: (r["source"], r["shard"]))
    summarize(root, records)


if __name__ == "__main__":
    main()
