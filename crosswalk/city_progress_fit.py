"""Reproduce the small city progress calibration from the retained native demonstration."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from . import city_crossing, city_progress
from .city_policy import FEATURE_NAMES, INDEX
from .city_progress import PARAMETERS, ProgressModel, VERSION, curvature


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(lesson):
    with zipfile.ZipFile(lesson) as archive:
        name, = [name for name in archive.namelist() if name.endswith("/lesson.npz")]
        with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as d:
            if tuple(d["feature_names"]) != FEATURE_NAMES:
                raise ValueError("Incompatible city instructor schema")
            x, target = d["X"], d["teacher_speed"]
        summary = json.loads(archive.read(name.replace("lesson.npz", "summary.json")))
        if summary["outcome"] != "complete" or summary["collision_events"]:
            raise ValueError("A completed collision-free demonstration is required")
    speed = x[:, INDEX["ego_speed_mps"]]
    accel = x[:, INDEX["ego_acceleration"]]
    bend = curvature(x)
    clear = ((x[:, INDEX["path_body_lidar"]] >= 28.) & (x[:, INDEX["path_body_radar"]] >= 28.)
             & (x[:, INDEX["oncoming_gap_recent"]] >= 32.) & (x[:, INDEX["rgb_person_height_recent"]] < .055)
             & (x[:, INDEX["mapped_lamp_gap_m"]] > 25.) & (speed > 2.))
    steady = clear & (abs(accel) < .65)
    # Fit the free-speed plateau on settled, maximal instructor requests;
    # the curvature term uses actual lateral acceleration in settled sharp turns.
    # The recovery term measures positive native acceleration during a sustained speed deficit.
    cruise = steady & (target >= .95 * max(target)) & (bend < .03)
    turning = steady & (bend >= .07)
    recovery = clear & (target > speed + .6) & (accel > .2)
    if min(np.sum(mask) for mask in (cruise, turning, recovery)) < 25:
        raise ValueError("Insufficient measured native support for cruise, turns, or recovery")
    coeff = np.array([np.median(speed[cruise]),
                      np.quantile((speed**2 * bend)[turning], .85),
                      np.quantile(accel[recovery], .85)], dtype=float)
    counts = {"native_frames": len(x), "cruise_fit": int(cruise.sum()),
              "turn_fit": int(turning.sum()), "recovery_fit": int(recovery.sum())}
    return coeff, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("models/city-clearance/candidate.npz"))
    parser.add_argument("--lesson", type=Path, default=Path("data/city-cruise/lessons.zip"))
    parser.add_argument("--stops", type=Path, default=Path("results/roadworks-pace/stopping-calibration.json"))
    parser.add_argument("--output", type=Path, default=Path("models/city-progress"))
    parser.add_argument("--grade", action="store_true", help="Fit revision 4 with measured local pavement confirmation")
    args = parser.parse_args()
    coeff, rows = fit(args.lesson)
    if args.grade:
        from . import city_grade
        model_type = city_grade.GradeModel
    else:
        model_type = ProgressModel
    model = model_type(args.base.read_bytes(), coeff)
    tested = json.loads(args.stops.read_bytes())
    hold = [r for r in tested["trials"] if r["split"] == "held_out"]
    errors = [model.stopping[0]*r["cue_speed_mps"]**2 + model.stopping[1]*r["cue_speed_mps"] +
              model.stopping[2] - r["stopping_distance_m"] for r in hold]
    if not errors or min(errors) < -.1:
        raise ValueError("The inherited stopping fit differs from the independent native stopping trials")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "candidate.npz"
    model.save(checkpoint)
    np.testing.assert_array_equal(model_type.load(checkpoint).coefficients, coeff)
    report = dict(schema=VERSION, checkpoint_sha256=digest(checkpoint), base_sha256=digest(args.base),
                  source_sha256=digest(Path(city_progress.__file__)), parameters=dict(zip(PARAMETERS, coeff)),
                  observed_frames=rows, braking_holdout_trials=len(hold),
                  braking_holdout_signed_errors_m=errors, brake_response_tested_through_mps=float(model.stopping[-1]),
                  demonstration="Town03 native instructor seed 75; fitted speeds, curvature and acceleration are measured.",
                  scope="Offline fit. Controller comparisons require new closed-loop runs.",
                  inputs={str(p.as_posix()):digest(p) for p in (args.base, args.lesson, args.stops, Path(__file__))},
                  crossing_source_sha256=digest(Path(city_crossing.__file__)))
    # Store the portable source path in the report regardless of the caller's checkout location.
    report["inputs"]["crosswalk/city_progress_fit.py"] = report["inputs"].pop(Path(__file__).as_posix())
    if args.grade:
        report.update(revision=4, grade_source_sha256=digest(Path(city_grade.__file__)))
    (args.output/"fit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
