"""Build the compact public speed comparison from native frame-scored summaries."""
import argparse
import json
from pathlib import Path

from .experiment import fingerprint, write_json


def metrics(root, name):
    folder=root/name
    summary_path=folder/"summary.json"
    s=json.loads(summary_path.read_bytes())
    grade=s["grade"]
    section=grade["fixed_section"]
    result=dict(run=name, checkpoint_sha256=s["model_sha256"], observer_sha256=s["implementation"]["city_maneuvers.py"],
                source_sha256=s["implementation"], maneuver_choice_sha256=s["maneuver_choice_sha256"],
                native_summary_sha256=fingerprint(summary_path),native_trace_sha256=fingerprint(folder/"trace.jsonl.gz"),
                observation_sha256=fingerprint(folder/"observations.npz"), native_map=s["producer"]["map"],
                sensor_configuration=s["sensor_configuration"],seed=s["seed"], actual_peak_kmh=grade["peak_kmh"],
                section_m=[section["start_m"],section["end_m"]],actual_section_median_kmh=section["actual_median_kmh"],
                actual_section_p10_kmh=section["actual_p10_kmh"],actual_section_p90_kmh=section["actual_p90_kmh"],
                section_max_half_second_braking_mps2=section["max_half_second_braking_mps2"],
                section_target_drops_5_kmh_one_tick=len(section["target_drops_5_kmh_one_tick"]),
                posted_excess_max_kmh=grade["maximum_posted_excess_kmh"],
                seconds_above_posted_plus_1=grade["over_posted_tolerance_s"], signed_entries=grade["lower_sign_entries"],
                contacts=s["collision_events"],raw_episode_outcome=s["outcome"],duration_s=s["duration_s"],
                route_length_m=s["route_length_m"])
    for limit in (40,45,50,55,60):
        band=grade[f"above_{limit}_kmh_in_90_zone"]
        result[f"seconds_above_{limit}_in_90"]=band["seconds"]
        result[f"longest_above_{limit}_in_90_s"]=band["longest_s"]
    if "queue" in grade:
        result["stationary_queue"]=grade["queue"]
    return result


def build(root):
    cal=json.loads((root/"stopping-envelope.json").read_bytes())
    stress=json.loads((root/"final-limited-sight.json").read_bytes())
    calibration=dict(delay_s=cal["delay_s"],quadratic_s2_per_m=cal["quadratic_s2_per_m"],guard_m=cal["guard_m"],
                     actual_fit_cue_kmh=[r["cue_speed_mps"]*3.6 for r in cal["trials"] if r["split"]=="fit"],
                     measured_stops_m=[r["stopping_distance_m"] for r in cal["trials"] if r["split"]=="fit"],
                     held_out=[dict(name=r["name"], actual_kmh=r["cue_speed_mps"]*3.6,measured_m=r["stopping_distance_m"],
                                    predicted_m=r["prediction_m"],guarded_margin_m=r["guarded_margin_m"])
                               for r in cal["trials"] if r["split"] in ("held_out","grip_stress")],
                     source_sha256=fingerprint(root/"stopping-envelope.json"))
    shorter=dict(native_sensor_cutoff_m=stress["native_sensor_cutoff_m"],high_samples=stress["sampled_high_proposals"],
                 all_withdrew=stress["all_withdrew_high_regime"],largest_fallback_request_kmh=stress["maximum_restricted_request_kmh"])
    result=dict(schema="ado-native-speed-comparison-v1",units="km/h, m, s",
                purpose="Measured native driving and independent stopping calibration",
                stopping_calibration=calibration,limited_visibility_replay=shorter,
                town03=dict(capped=metrics(root,"final-long-matched-40"),higher=metrics(root,"final-long-clear"),
                            occupied=metrics(root,"final-long-queue"),primary_queue=metrics(root,"final-primary-queue")))
    for folder in sorted(root.glob("motorway-transfer*"),reverse=True):
        required=[folder/c/"summary.json" for c in ("matched-40","higher-speed","stationary-queue")]
        if (folder/"host-restored.json").is_file() and all(p.is_file() for p in required):
            result["town04_transfer"]={key:metrics(root,str((folder/run).relative_to(root)).replace("\\","/"))
                                       for key,run in (("capped","matched-40"),("higher","higher-speed"),
                                                       ("occupied","stationary-queue"))}
            break
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("results/city-maneuvers/speed"))
    args=p.parse_args()
    write_json(args.root/"comparison.json",build(args.root))


if __name__=="__main__":
    main()
