"""Audit the new maneuver tree on archived pre-control lane measurements."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .maneuver_choice import MODEL, ManeuverChoice
from .route_maneuvers import Surface, usable_gap


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ("results/tactical-pass/v04/paired-kiu_oqoh","results/tactical-pass/v04-road67/paired-jo9p4yw0")


def audit(path):
    model = ManeuverChoice()
    ahead, behind = Surface(), Surface()
    passes, holds, first_invitation, first_yield = [], [], None, None
    with gzip.open(path, "rt") as f:
        trace = [json.loads(line) for line in f]
    for step,row in enumerate(trace):
        observed = row["tactical"]
        if not observed or abs(observed["source_lateral_m"]) > .35:
            break
        s, speed = observed["corridor_station_m"], row["speed_mps"]
        front = ahead.update(step,s,observed["target_front_m"],1.)
        rear = behind.update(step,s,observed["target_rear_m"],-1.)
        fc = max(0.,speed-(ahead.speed or 0.)) if front < 60. else 0.
        rc = max(0.,(behind.speed if behind.speed is not None else 15.)-speed) if rear < 60. else 0.
        x = (observed["source_gap_m"],observed["source_speed_mps"],front,rear,fc,rc,speed)
        action,_ = model.predict(x)
        safe = all(observed["adjacent_pavement_seen"]) and usable_gap(front,rear,fc,rc,speed)
        # The actor measurements below compare an invitation to physical traffic.
        # They never enter the four-action classifier or measured safety envelope.
        if action == "pass" and safe:
            passes.append(step)
            if first_invitation is None:
                first_invitation = dict(frame=row["frame"],time_s=row["time_s"],
                                        signed_adjacent_bumper_gap_m=row.get("adjacent_bumper_gap_m"))
        if action=="yield" or (action=="pass" and not safe):
            holds.append(step)
            if first_yield is None:
                first_yield = row["time_s"]
    return dict(source=path.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                case=path.name.split(".")[0],frames_in_source_center=step+(abs(observed["source_lateral_m"])<=.35),
                hold_recommendations=len(holds),first_hold_s=first_yield,
                pass_invitations=len(passes),first_invitation=first_invitation)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    cases = [audit(ROOT/group/(name+".jsonl.gz")) for group in ARCHIVES for name in ("no-lead","occupied","clear")]
    result = dict(scope="Classifier and projected gap gate on the legacy measured stream until ego leaves the source center by 0.35 m; no v0.7 physical trajectory is implied.",
                  model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),holdout=json.loads(MODEL.read_bytes())["holdout"],cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
