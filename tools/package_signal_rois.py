"""Package, verify, or inspect the exact RGB inputs to the lossless signal audits."""

import argparse
from itertools import groupby
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from crosswalk import signal_observer
from crosswalk.camera_geometry import CAMERAS
from crosswalk.lamp import project_box
from tools.rescore_signal_rgb import _digest, _native, _pair, _published_reader, _reader, _transform


SCHEMA = "ado-native-signal-rois-v1"
FRONT = CAMERAS["front"]
IMAGE_SHAPE = (FRONT.height, FRONT.width, 3)


def _modules():
    published, published_digest = _published_reader()
    source = Path(signal_observer.__file__).read_bytes().replace(b"\r\n", b"\n")
    return {"published_reader": published, "revised_reader": signal_observer}, {
        "published_reader": published_digest,
        "revised_reader": hashlib.sha256(source).hexdigest(),
        "classifier": _digest(signal_observer.CALIBRATION),
    }


def _layout(observer, row):
    ego = _transform(row["ego_pose"])
    matrix = observer.mounts["front"] @ np.asarray(ego.get_inverse_matrix())
    for head_idx, head in enumerate(observer.heads):
        bounds = project_box(head.vertices, matrix, IMAGE_SHAPE, FRONT.hfov)
        if bounds is not None:
            yield head_idx, bounds


def _pack_capture(path, modules, arrays, counter):
    manifest = json.loads((path / "capture.json").read_text(encoding="utf8"))
    expected = _native(path, modules)
    record = {key: value for key, value in manifest.items() if key not in ("heads", "sequences")}
    record.update(source_manifest_sha256=_digest(path / "capture.json"), clips=[])
    for source in manifest["sequences"]:
        clip = dict(source)
        clip["geometry"] = [head for head in manifest["heads"] if head["map_id"] == clip["target"]]
        clip["arrays"] = f"clip{counter}"
        clip["scores"] = expected[clip["name"]]
        original = path / (clip["name"] + ".npz")
        clip["source_npz_sha256"] = _digest(original)
        observer = _reader(signal_observer, clip["geometry"])
        samples, pixels = [], bytearray()
        with np.load(original, allow_pickle=False) as source_data:
            for frame_index, (row, rgb) in enumerate(zip(clip["rows"], source_data["front"], strict=True)):
                if rgb.dtype != np.uint8 or rgb.shape != IMAGE_SHAPE:
                    raise ValueError(f"Unexpected camera data at frame {row['frame']}")
                for head_index, (x0, y0, x1, y1) in _layout(observer, row):
                    samples.append((frame_index, head_index, x0, y0, x1, y1, len(pixels)))
                    pixels.extend(rgb[y0:y1, x0:x1].tobytes())
        arrays[clip["arrays"] + "_regions"] = np.asarray(samples, dtype=np.int32).reshape(-1, 7)
        arrays[clip["arrays"] + "_pixels"] = np.asarray(np.frombuffer(bytes(pixels), dtype=np.uint8))
        record["clips"].append(clip)
        counter += 1
    return record, counter


def pack(inputs, output):
    modules, digests = _modules()
    manifest = dict(schema=SCHEMA, image_shape=IMAGE_SHAPE, sha256=digests, captures=[])
    arrays, counter = {}, 0
    for path in inputs:
        record, counter = _pack_capture(path, modules, arrays, counter)
        manifest["captures"].append(record)
    if not counter:
        raise ValueError("No clips to package")
    manifest_bytes = json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode("utf8")
    arrays["manifest"] = np.frombuffer(manifest_bytes, dtype=np.uint8)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _archive_manifest(archive):
    manifest = json.loads(archive["manifest"].tobytes())
    if manifest["schema"] != SCHEMA or tuple(manifest["image_shape"]) != IMAGE_SHAPE:
        raise ValueError("Unsupported native signal archive")
    return manifest


def _regions(archive, clip):
    pixels = archive[clip["arrays"] + "_pixels"]
    samples = archive[clip["arrays"] + "_regions"]
    if pixels.dtype != np.uint8 or pixels.ndim != 1 or samples.dtype != np.int32 or samples.shape[1:] != (7,):
        raise ValueError("Unexpected signal storage format")
    size = 0
    previous = None
    for frame_idx, head_idx, x0, y0, x1, y1, offset in samples:
        key = int(frame_idx), int(head_idx)
        if previous is not None and key <= previous:
            raise ValueError("Duplicate or out-of-order signal region")
        previous = key
        if offset != size or not (0 <= x0 < x1 <= FRONT.width and 0 <= y0 < y1 <= FRONT.height):
            raise ValueError("Invalid signal region bounds or byte offset")
        size += (x1 - x0) * (y1 - y0) * 3
        if size > len(pixels) or not (0 <= frame_idx < len(clip["rows"]) and 0 <= head_idx < len(clip["geometry"])):
            raise ValueError("Signal region outside its archive")
        crop = pixels[offset:size].reshape((y1 - y0, x1 - x0, 3))
        yield int(frame_idx), int(head_idx), (int(x0), int(y0), int(x1), int(y1)), crop
    if size != len(pixels):
        raise ValueError("Unexpected bytes after the final signal region")


def _frames(archive, clip):
    observer = _reader(signal_observer, clip["geometry"])
    groups = iter(groupby(_regions(archive, clip), key=lambda item: item[0]))
    group_idx, group = next(groups, (None, None))
    camera = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    written = np.zeros(IMAGE_SHAPE[:2], dtype=bool)
    for index, row in enumerate(clip["rows"]):
        camera.fill(0)
        written.fill(False)
        expected = dict(_layout(observer, row))
        if group_idx == index:
            for _, head_idx, bounds, pixels in group:
                if expected.pop(head_idx, None) != bounds:
                    raise ValueError(f"Signal region differs from the surveyed projection on frame {row['frame']}")
                x0, y0, x1, y1 = bounds
                tile = camera[y0:y1, x0:x1]
                overlap = written[y0:y1, x0:x1]
                if not np.array_equal(tile[overlap], pixels[overlap]):
                    raise ValueError("Overlapping signal regions disagree on RGB bytes")
                tile[:] = pixels
                overlap[:] = True
            group_idx, group = next(groups, (None, None))
        if expected:
            raise ValueError(f"Missing signal regions on frame {row['frame']}")
        reference = {"signal_reference": {clip["target"]: row["state"]}}
        yield _transform(row["ego_pose"]), {"front": camera}, {**row, "observer_map": reference}


def verify(path):
    modules, digests = _modules()
    with np.load(path, allow_pickle=False) as archive:
        manifest = _archive_manifest(archive)
        if manifest["sha256"] != digests:
            raise ValueError("The signal reader and calibration hashes must match the packaged audit")
        exposures, regions = 0, 0
        for capture in manifest["captures"]:
            for clip in capture["clips"]:
                result = _pair(_frames(archive, clip), modules, clip["geometry"], clip["target"])
                if result != clip["scores"]:
                    raise ValueError(f"Reconstructed RGB changed score results for {clip['name']}")
                exposures += len(clip["rows"])
                regions += len(archive[clip["arrays"] + "_regions"])
                print(f"{clip['name']}: {len(clip['rows'])} exposures, verified", flush=True)
    print(f"Verified {exposures} exposures and {regions} exact RGB rectangles. SHA-256 {_digest(path)}")


def extract(path, clip_name, frame, output):
    with np.load(path, allow_pickle=False) as archive:
        manifest = _archive_manifest(archive)
        clips = [clip for capture in manifest["captures"] for clip in capture["clips"] if clip["name"] == clip_name]
        if len(clips) != 1:
            raise ValueError(f"Unknown or ambiguous clip {clip_name}")
        clip = clips[0]
        index = next((i for i, row in enumerate(clip["rows"]) if row["frame"] == frame), None)
        if index is None:
            raise ValueError(f"No frame {frame} in {clip_name}")
        selected = [item for item in _regions(archive, clip) if item[0] == index]
        if not selected:
            raise ValueError(f"No mapped head is projected at frame {frame}")
        for arm, score in clip["scores"].items():
            wrong = [item for item in score["mismatches"] if item["frame"] == frame]
            print(arm, f"reference {clip['rows'][index]['state']}", json.dumps(wrong))
        tiles = []
        for _, head_idx, bounds, rgb in selected:
            tile = np.full((206, 190, 3), 33, dtype=np.uint8)
            h, w = rgb.shape[:2]
            zoom = max(1, min(176 // w, 158 // h))
            tile[8:8 + h * zoom, 8:8 + w * zoom] = cv2.resize(rgb[:, :, ::-1], (w * zoom, h * zoom),
                                                              interpolation=cv2.INTER_NEAREST)
            label = f"Head {head_idx + 1}, {w} x {h} px"
            cv2.putText(tile, label, (8, 184), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(tile, str(bounds), (8, 200), cv2.FONT_HERSHEY_SIMPLEX, .38, (220, 220, 220), 1, cv2.LINE_AA)
            tiles.append(tile)
        if not cv2.imwrite(str(output), np.hstack(tiles)):
            raise RuntimeError(f"Could not write {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_subparsers(dest="action", required=True)
    packing = action.add_parser("pack")
    packing.add_argument("archive", type=Path)
    packing.add_argument("captures", type=Path, nargs="+")
    checking = action.add_parser("verify")
    checking.add_argument("archive", type=Path)
    review = action.add_parser("extract")
    review.add_argument("archive", type=Path)
    review.add_argument("clip")
    review.add_argument("frame", type=int)
    review.add_argument("output", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.action == "pack":
        pack(args.captures, args.archive)
    elif args.action == "verify":
        verify(args.archive)
    else:
        extract(args.archive, args.clip, args.frame, args.output)
