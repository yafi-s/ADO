"""Compare the original and current signal reader on identical saved RGB exposures."""

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import types

import carla
import cv2
import numpy as np

from crosswalk import signal_observer
from crosswalk.camera_geometry import CAMERAS
from crosswalk.replay_pack import _array_items
from tools.score_signal_reads import score


PUBLISHED_READER = "a7a00120ff6e3f2e266db7de1880e31c38289553"


def _head(module, key, map_id, vertices, yaw):
    verts = np.asarray(vertices)
    return module.Head(key, map_id, verts, verts.mean(axis=0), yaw)


def _reader(module, geometry):
    instance = object.__new__(module.SignalObserver)
    instance.classifier = module.HousingClassifier()
    instance.heads = [_head(module, row["id"], row["map_id"], row["vertices"], row["yaw"])
                      for row in geometry]
    instance.candidates = {}
    instance.recent = {}
    instance.mounts = {name: np.asarray(carla.Transform(carla.Location(cam.x, cam.y, cam.z),
                                                        carla.Rotation(yaw=cam.yaw)).get_inverse_matrix())
                       for name, cam in CAMERAS.items()}
    return instance


def _transform(p):
    return carla.Transform(carla.Location(p["x"], p["y"], p["z"]),
                           carla.Rotation(pitch=p["pitch"], yaw=p["yaw"], roll=p["roll"]))


def _metrics(rows, target):
    reads = Counter()
    transitions = []
    mismatches = []

    def observed_frames():
        previous = None
        active = None
        for index, row in enumerate(rows):
            truth = row["observer_map"]["signal_reference"]
            reference = truth.get(target)
            if previous and reference and reference != previous:
                active = dict(frame=row["frame"], source_frame=row.get("source_frame"), from_state=previous,
                              to_state=reference, first_agreement_frames=None, wrong_head_frames=0,
                              abstained_head_frames_before_agreement=0)
                transitions.append(active)
                change = index
            elif not reference or not previous:
                active = None
            colors = []
            for head in row["signals"]:
                state = head["state"]
                ref = truth.get(head["map_id"])
                if state != "unknown" and ref:
                    kind = "retained" if head.get("age_s", 0) > 0 else "fresh"
                    reads[f"{kind}_{'correct' if state == ref else 'wrong'}"] += 1
                    if head.get("camera_support"):
                        reads[f"corroborated_{'correct' if state == ref else 'wrong'}"] += 1
                    if state != ref:
                        mismatches.append(dict(frame=row["frame"], source_frame=row.get("source_frame"), id=head["id"],
                                               map_id=head["map_id"], reference=ref, read=state, age_s=head.get("age_s"),
                                               view=head.get("view"), confidence=head.get("confidence")))
                if active and head["map_id"] == target:
                    colors.append(state)
            if active:
                active["wrong_head_frames"] += sum(color not in ("unknown", reference) for color in colors)
                if active["first_agreement_frames"] is None and reference != "off":
                    if reference in colors:
                        active["first_agreement_frames"] = index - change
                    else:
                        active["abstained_head_frames_before_agreement"] += colors.count("unknown")
            previous = reference
            yield row

    return dict(**score(observed_frames()), evidence=reads, transitions=transitions, mismatches=mismatches)


def _pair(frames, modules, geometry, target):
    readers = {name: _reader(module, geometry) for name, module in modules.items()}
    results = {name: [] for name in modules}
    for pose, images, row in frames:
        for name, reader in readers.items():
            result = reader.read(pose, images, row["sim_time"])
            results[name].append({**row, "signals": result})
    return {name: _metrics(rows, target) for name, rows in results.items()}


def _native(capture, modules):
    manifest = json.loads((capture / "capture.json").read_text(encoding="utf8"))
    out = {}
    for seq in manifest["sequences"]:
        with np.load(capture / (seq["name"] + ".npz"), allow_pickle=False) as data:
            def frames():
                for row, rgb in zip(seq["rows"], data["front"], strict=True):
                    ref = {"signal_reference": {seq["target"]: row["state"]}}
                    yield _transform(row["ego_pose"]), {"front": rgb}, {**row, "observer_map": ref}

            out[seq["name"]] = _pair(frames(), modules, [h for h in manifest["heads"] if h["map_id"] == seq["target"]],
                                     seq["target"])
    return out


def _encoded(replay, video_dir, extension, geometry, modules):
    with replay.open(encoding="utf8") as stream:
        mapped = {h["id"]: h for row in _array_items(stream) for h in row["signals"]}
    surveyed = {head["id"]: head for head in geometry}
    if mapped.keys() - surveyed.keys():
        raise ValueError("The supplied survey omits recorded signal heads")
    geometry = [{**surveyed[key], "yaw": row["governs_yaw"]} for key, row in mapped.items()]
    with ExitStack() as cleanup:
        videos = {}
        for name in CAMERAS:
            path = video_dir / f"{name}.{extension}"
            video = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
            cleanup.callback(video.release)
            if not video.isOpened():
                raise ValueError(f"Could not decode {path}")
            videos[name] = video

        def frames():
            with replay.open(encoding="utf8") as stream:
                for index, row in enumerate(_array_items(stream)):
                    if row["video_index"] != index or row["frame"] != row["camera_frame"]:
                        raise ValueError(f"Expected aligned video exposure at index {index}")
                    images = {}
                    for name, video in videos.items():
                        ok, bgr = video.read()
                        if not ok:
                            raise ValueError(f"Missing {name} exposure at index {index}")
                        images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    truth = {"signal_reference": row["observer_map"]["signal_reference"]}
                    yield _transform(row["ego_pose"]), images, dict(frame=row["frame"], sim_time=index * .1,
                                                                   observer_map=truth)
            if any(video.read()[0] for video in videos.values()):
                raise ValueError("A camera video extends past the aligned replay")

        return _pair(frames(), modules, geometry, "2024")


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _published_reader():
    source = subprocess.check_output(["git", "show", f"{PUBLISHED_READER}:crosswalk/signal_observer.py"])
    weights = subprocess.check_output(["git", "show", f"{PUBLISHED_READER}:models/surround-signals/colors.npz"])
    if hashlib.sha256(weights).hexdigest() != _digest(signal_observer.CALIBRATION):
        raise ValueError("This comparison requires the released signal-classifier weights")
    baseline = types.ModuleType("crosswalk._published_signal_observer")
    baseline.__file__ = signal_observer.__file__
    exec(compile(source, baseline.__file__, "exec"), baseline.__dict__)
    return baseline, hashlib.sha256(source).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path, help="Directory from capture_signal_transitions")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--mjpeg", type=Path)
    parser.add_argument("--mp4", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.replay and (args.mjpeg or args.mp4):
        parser.error("--replay is required to score the encoded exposures")
    cv2.setNumThreads(1)
    baseline, baseline_sha256 = _published_reader()
    modules = {"published_reader": baseline, "revised_reader": signal_observer}
    evidence = dict(native_capture_sha256={p.name: _digest(p) for p in sorted(args.native.glob("*"))},
                    reader_sha256={"published": baseline_sha256,
                                   "revised": _digest(Path(signal_observer.__file__))},
                    classifier_sha256=_digest(signal_observer.CALIBRATION),
                    native=_native(args.native, modules))
    if args.replay:
        with args.replay.open(encoding="utf8") as stream:
            saved = [dict(frame=row["frame"], signals=row["signals"],
                          observer_map={"signal_reference": row["observer_map"]["signal_reference"]})
                     for row in _array_items(stream)]
        evidence["recorded_live_rgb_readings"] = _metrics(saved, "2024")
    geometry = json.loads((args.native / "capture.json").read_text(encoding="utf8"))["heads"]
    for extension in ("mjpeg", "mp4"):
        directory = getattr(args, extension)
        if directory is not None:
            evidence[extension] = dict(input_sha256={p.name: _digest(p) for p in
                                       [args.replay, *(directory / f"{c}.{extension}" for c in CAMERAS)]},
                                       result=_encoded(args.replay, directory, extension, geometry, modules))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
