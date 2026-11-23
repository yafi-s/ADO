"""Check sealed native frame clocks, raw camera streams, and policy observations."""

import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np

from tools.score_city_interactions import sha256


def read(path):
    return json.loads(path.read_bytes())


def jpeg_stream(path):
    digest = hashlib.sha256()
    marker = b"\xff\xd9\xff\xd8"
    count, boundary, begin = 0, b"", None
    with path.open("rb") as stream:
        while piece := stream.read(4 * 1024 * 1024):
            if begin is None:
                begin = piece[:2]
            digest.update(piece)
            data = boundary + piece
            count += data.count(marker)
            boundary = data[-3:]
    if begin != b"\xff\xd8" or boundary[-2:] != b"\xff\xd9":
        raise ValueError(f"Incomplete native JPEG stream: {path.name}")
    return digest.hexdigest(), count + 1


def native_journal(path, rows):
    digest = hashlib.sha256()
    count, prior_command = 0, None
    with path.open("rb") as stream:
        for line, row in zip_longest(stream, rows):
            if line is None or row is None:
                raise ValueError("Perception journal length differs from telemetry")
            digest.update(line)
            exposure = json.loads(line)
            keys = ("frame", "sim_time", "progress_m", "speed_mps")
            if any(exposure[key] != row[key] for key in keys):
                raise ValueError(f"Mismatched perception capture at native frame {row['frame']}")
            # The perception journal records this tick's command. The camera
            # telemetry presents the control actually applied on this tick.
            controls = ("steer", "throttle", "brake")
            commanded = row["commanded_control"]
            applied = row["applied_control"]
            if any(exposure[key] != commanded[key] or row[key] != applied[key] for key in controls):
                raise ValueError(f"Native controls do not match perception at frame {row['frame']}")
            if prior_command is not None:
                np.testing.assert_allclose([applied[key] for key in controls],
                                           [prior_command[key] for key in controls], atol=.000006, rtol=0.)
            prior_command = commanded
            count += 1
    return digest.hexdigest(), count


def verify(stage, range_retention):
    source_path = stage / "capture.json"
    capture = read(source_path)
    raw = stage / "raw"
    frame_path = raw / "telemetry.json"
    telemetry = read(frame_path)
    frame_numbers = np.array([r["frame"] for r in telemetry])
    expected = np.arange(capture["first_frame"], capture["last_frame"] + 1)
    np.testing.assert_array_equal(frame_numbers, expected)
    if len(telemetry) != capture["summary"]["frames"]:
        raise ValueError("Native capture and score disagree on frame count")
    if (capture["quality_requested"] != "High" or not capture["world_settings"]["synchronous_mode"]
            or capture["world_settings"]["fixed_delta_seconds"] != .1):
        raise ValueError("The High native recording must use the declared synchronous clock")
    for key in ("sim_time", "video_time"):
        np.testing.assert_allclose([r[key] for r in telemetry], np.arange(len(telemetry)) * .1, atol=.0001, rtol=0.)
    np.testing.assert_array_equal([r["video_index"] for r in telemetry], np.arange(len(telemetry)))
    np.testing.assert_array_equal([r["camera_frame"] for r in telemetry], expected)
    np.testing.assert_allclose([r["camera_timestamp_s"] for r in telemetry],
                               [r["world_elapsed_s"] for r in telemetry], atol=.001, rtol=0.)
    np.testing.assert_allclose(np.diff([r["world_elapsed_s"] for r in telemetry]), .1, atol=.001, rtol=0.)

    verified_raw = {}
    for name, digest in capture["raw_sha256"].items():
        source = raw / name
        if source.suffix == ".mjpeg":
            actual, nr = jpeg_stream(source)
            want = capture["chase_source_frames"] if name == "chase.mjpeg" else len(expected)
            if nr != want:
                raise ValueError(f"Wrong native exposure count in {name}: {nr} != {want}")
            details = {"source_exposures": nr}
        elif name == "perception.jsonl":
            actual, nr = native_journal(source, telemetry)
            details = {"source_exposures": nr}
        else:
            actual = sha256(source)
            details = {}
        if actual != digest:
            raise ValueError(f"Different SHA-256 on sealed source file {name}")
        verified_raw[name] = {"sha256": actual, "bytes": source.stat().st_size, **details}
    expected_views = {"chase.mjpeg", "front.mjpeg", *(f"{v}.mjpeg" for v in capture["perimeter_views"])}
    if set(expected_views) - set(verified_raw):
        raise ValueError("A native camera stream was omitted from the seal")

    policy_tape = stage / "policy-observations.npz"
    if sha256(policy_tape) != capture["policy_observations_sha256"]:
        raise ValueError("Native policy tape digest differs from the capture")
    with np.load(policy_tape, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["frame"], frame_numbers)
        np.testing.assert_allclose(arrays["speed_request_mps"], [r["speed_request_mps"] for r in telemetry],
                                   atol=.001, rtol=0.)
        np.testing.assert_allclose(arrays["posted_limit_kmh"], [r["posted_limit_kmh"] for r in telemetry],
                                   atol=.01, rtol=0.)
        if len(arrays["features"]) != len(telemetry) or not np.isfinite(arrays["features"]).all():
            raise ValueError("Policy observations are incomplete or nonfinite")

    ranges = read(range_retention)
    capture_digest = sha256(source_path)
    witnesses = len(capture["maneuver_sweeps"])
    if (ranges["capture_sha256"] != capture_digest or ranges["first_frame"] != int(expected[0])
            or ranges["last_frame"] != int(expected[-1])
            or ranges["contiguous_range_exposures"] != len(telemetry)
            or ranges["pose_matches"] != len(telemetry)
            or ranges["actual_control_lidar_and_five_radar_witnesses_matched"] != witnesses):
        raise ValueError("The independent raw-range verification refers to different native exposures")
    return dict(
        capture_id=capture["capture_id"], capture_sha256=capture_digest, source_pin=capture["capture_checkout"]["head_at_capture_start"],
        quality=capture["quality_requested"], first_frame=int(expected[0]), last_frame=int(expected[-1]),
        contiguous_camera_telemetry_and_policy_frames=len(telemetry), native_frame_seconds=.1,
        raw=verified_raw, aligned_raw_ranges=ranges,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", type=Path)
    cli.add_argument("range_retention", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    result = verify(args.stage, args.range_retention)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
