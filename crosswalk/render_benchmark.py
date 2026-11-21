"""Measure the complete five-view capture and perception loop on the local machine."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import cv2
import numpy as np
import carla

from .control import Navigation
from .policy import features
from .robotaxi import BodyKit
from .runtime import producer_info, simulator
from .sensors import Rig
from .viewio import _jpegs
from .vision import Detector
from .world import Scenario


def gpu():
    try:
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader", "-i", "0"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def trial(world, mode, percent, frames, images):
    os.environ.update(CROSSWALK_RGB_RENDER=mode, CROSSWALK_DLSS_PERCENT=str(percent))
    cv2.setNumThreads(2)
    with ExitStack() as stack:
        scene = Scenario(world, "signal", 31, district="train")
        stack.callback(scene.close)
        scene.ego.apply_control(carla.VehicleControl(brake=1., hand_brake=True))
        kit = BodyKit(world, scene.ego)
        stack.callback(kit.close)
        rig = Rig(world, scene.ego, visuals=True)
        stack.callback(rig.close)
        detector, navigator = Detector(), Navigation(scene.route)
        tick_times, combined = [], []
        detections = []
        for i in range(frames + 16):
            start = perf_counter()
            obs = rig.read(world.tick())
            captured = perf_counter()
            if i % 2 == 0:
                detections = detector.detect(obs["front"])
            features(obs["front"], obs["lidar"], obs["radar"], 0., detections,
                     navigation=navigator.read(scene.ego), side_views=(obs["left"], obs["right"]))
            _jpegs(obs)
            if i >= 16:
                tick_times.append((captured - start)*1000)
                combined.append((perf_counter()-start)*1000)
        images.mkdir(exist_ok=True, parents=True)
        for view in ("front","chase","infrared"):
            cv2.imwrite(str(images / f"{mode}-{percent}-{view}.jpg"), obs[view][:,:,::-1])
        return dict(mode=mode, requested_percent=percent if mode=="dlss" else 100, frames=frames,
                    median_tick_ms=round(float(np.median(tick_times)),2),
                    p95_tick_ms=round(float(np.percentile(tick_times,95)),2),
                    combined_hz=round(float(1000/np.mean(combined)),2),
                    p95_combined_ms=round(float(np.percentile(combined,95)),2),
                    front_std=float(obs["front"].std()), camera_readback=rig.rendering, gpu_sample=gpu())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--port", type=int, default=2468)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("results/rendering.json"))
    parser.add_argument("--images", type=Path, default=Path(".local/render-benchmark"))
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least two")
    with simulator(args.port, args.attach) as world:
        report = dict(producer=producer_info(world), hardware=gpu(), platform=platform.platform(),
                      measurements="Stationary signal approach, five encoded views, lidar, radar, thermal, "
                      "feature extraction and NanoDet every other frame. Sixteen warmup ticks per profile.",
                      profiles=[])
        for mode, percent in (("native",100),("dlss",58),("dlss",50),("mixed",50)):
            row = trial(world, mode, percent, args.frames, args.images)
            report["profiles"].append(row)
            if mode == "mixed":
                report["selected_mode"] = "mixed"
                report["selected_percent"] = 50
                report["selection_reason"] = "Retain native front RGB for small signal lamps and request upscaling on auxiliary RGB views."
            print(mode, percent, row["combined_hz"], "Hz", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.output.with_suffix(".partial.json")
            tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            tmp.replace(args.output)


if __name__ == "__main__":
    main()
