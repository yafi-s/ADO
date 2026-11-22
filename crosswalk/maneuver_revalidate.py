"""Audit the new maneuver tree on archived pre-control lane measurements."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .maneuver_choice import MODEL, ManeuverChoice
from .route_maneuvers import Surface, compare_progress, usable_gap


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ("results/tactical-pass/v04/paired-kiu_oqoh","results/tactical-pass/v04-road67/paired-jo9p4yw0")


def audit(path):
    model = ManeuverChoice()
    ahead, behind = Surface(), Surface()
    passes, holds, first_invitation, first_yield = [], [], None, None
    comparisons = []
    with gzip.open(path, "rt") as f:
        trace = [json.loads(line) for line in f]
    for step,row in enumerate(trace):
        observed = row.get("tactical",row.get("tactical_pass"))
        if not observed or abs(observed["source_lateral_m"]) > .35:
            break
        s, speed = observed["corridor_station_m"], row["speed_mps"]
        front = ahead.update(step,s,observed["target_front_m"],1.)
        rear = behind.update(step,s,observed["target_rear_m"],-1.)
        fc = max(0.,speed-(ahead.speed or 0.)) if front < 60. else 0.
        rc = max(0.,(behind.speed if behind.speed is not None else 15.)-speed) if rear < 60. else 0.
        x = (observed["source_gap_m"],observed["source_speed_mps"],front,rear,fc,rc,speed)
        fitted,_ = model.predict(x)
        # Earlier recordings retain only the adjacent pavement coverage bit,
        # not the farthest ground hit. Use an optimistic 60 m for those
        # archives; the two-queue recording retains measured pavement reach.
        target_sight = observed.get("pavement_reach_m",[0.,0.,60.])[2]
        action,gain,horizon = compare_progress(fitted,x[0],x[1],front,rear,ahead.speed or 0.,target_sight)
        if observed.get("maneuver_phase") in ("signal_left","change_left","yield") and (
                not comparisons or comparisons[-1]["prior_phase"]!=observed["maneuver_phase"]):
            comparisons.append(dict(frame=row["frame"],prior_phase=observed["maneuver_phase"],
                                    fitted_choice=fitted,effective_choice=action,source_gap_m=x[0],
                                    target_front_m=front,projected_progress_gain_m=round(gain,3),
                                    reachable_target_surface_m=round(horizon,3)))
        safe = all(observed["adjacent_pavement_seen"]) and usable_gap(front,rear,fc,rc,speed)
        # The actor measurements below compare an invitation to physical traffic.
        # They never enter the four-action classifier or measured safety envelope.
        if action == "pass" and safe:
            passes.append(step)
            if first_invitation is None:
                first_invitation = dict(frame=row["frame"],time_s=row.get("time_s",row.get("sim_time")),
                                        signed_adjacent_bumper_gap_m=row.get("adjacent_bumper_gap_m"))
        if action=="yield" or (action=="pass" and not safe):
            holds.append(step)
            if first_yield is None:
                first_yield = row.get("time_s",row.get("sim_time"))
    return dict(source=path.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                case=path.name.split(".")[0],historical_phase_changes=comparisons,frames_in_source_center=step+(abs(observed["source_lateral_m"])<=.35),
                hold_recommendations=len(holds),first_hold_s=first_yield,
                pass_invitations=len(passes),first_invitation=first_invitation)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    cases = [audit(ROOT/group/(name+".jsonl.gz")) for group in ARCHIVES for name in ("no-lead","occupied","clear")]
    queue = ROOT/"results/city-maneuvers/speed/stationary-queue/trace.jsonl.gz"
    if queue.is_file():
        queue_check = audit(queue)
        queue_check["case"] = "two-stationary-queues"
        cases.append(queue_check)
    result = dict(scope="Classifier and projected gap gate on the legacy measured stream until ego leaves the source center by 0.35 m; no v0.7 physical trajectory is implied.",
                  runtime_sha256=hashlib.sha256(Path(__file__).with_name("route_maneuvers.py").read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),holdout=json.loads(MODEL.read_bytes())["holdout"],cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
