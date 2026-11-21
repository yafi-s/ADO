"""Frozen Town01 focused interaction test. Keep a trace and checkpoint for every drive."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from . import experiment as ex, report, runtime, sensors
from .revision_benchmark import read_case, run_case, source_files, writer_lock
from .town01_interaction import BASE, MODEL, ROOT, InteractionModel, digest
from .town01_interaction_gate import GatedInteractionModel, MODEL as GATED_MODEL

SCHEDULE = {
    "known_failure": [(105, "left"), (104, "pedestrian-delayed"), (107, "left"),
                      (2111, "left"), (2112, "left"), (2113, "left")],
    "known_control": [(104, "left"), (103, "pedestrian-delayed"), (104, "turn"), (104, "clear")],
    "fresh_challenge": [(4901, "left"), (4902, "left"), (4901, "pedestrian-delayed"),
                        (4902, "pedestrian-delayed"), (4902, "turn"), (4902, "clear")],
}
GATED_SCHEDULE = dict(SCHEDULE, fresh_challenge=[
    (5261, "left"), (5262, "left"), (5261, "pedestrian-delayed"),
    (5262, "pedestrian-delayed"), (5262, "turn"), (5262, "clear")])


def study(output, port, *, gated=False, attach=False):
    output = Path(output)
    schedule = GATED_SCHEDULE if gated else SCHEDULE
    checkpoint = GATED_MODEL if gated else MODEL
    driver = GatedInteractionModel if gated else InteractionModel
    policy = "interaction-gated-v04" if gated else "interaction-v04"
    sources = source_files("crosswalk.town01_interaction_study:study")
    hashes = {name: digest(path) for name, path in sorted(sources.items())}
    for path in (BASE, checkpoint, ROOT/"data/signals/approach-gate.json", ROOT/"vendor/nanodet/object_detection_nanodet_2022nov.onnx"):
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    manifest = dict(policy=policy, schedule={group: [list(p) for p in pairs] for group, pairs in schedule.items()},
                    hashes=hashes, rgb=sensors.render_profile(), map="Town01_Opt", timestep_s=.1)
    version = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    marker = output/"manifest.json"
    with writer_lock(output):
        preregistration = output/"protocol.json"
        if preregistration.exists():
            frozen = ex.read_json(preregistration)
            if (frozen["schedule"] != manifest["schedule"] or frozen["policy"] != policy
                    or frozen["weights_sha256"] != digest(checkpoint)
                    or any(hashes["crosswalk/" + n] != h for n,h in frozen["planner_sha256"].items())):
                raise ValueError("Model or schedule changed since preregistration.")
        if marker.exists() and ex.read_json(marker) != manifest:
            raise ValueError("Frozen inputs changed. Use a new study directory.")
        ex.write_json(marker, manifest)
        archive = output/"study-source.zip"
        if not archive.exists():
            partial = archive.with_suffix(".partial")
            with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as snapshot:
                for name, source in sorted(sources.items()):
                    contents = source.read_bytes()
                    if hashlib.sha256(contents).hexdigest() != hashes[name]:
                        raise ValueError(f"Source changed while freezing {name}")
                    snapshot.writestr(name, contents)
            partial.replace(archive)
        with zipfile.ZipFile(archive) as snapshot:
            if set(snapshot.namelist()) != set(sources) or any(
                    hashlib.sha256(snapshot.read(name)).hexdigest() != hashes[name] for name in sources):
                raise ValueError("Study source archive differs from the manifest.")
        with runtime.simulator(port=port, attach=attach) as world:
            producer = runtime.producer_info(world)
            if producer != {"server_version":"1.0", "client_version":"1.0", "map":"Town01_Opt"}:
                raise ValueError(f"Wrong producer {producer}")
            rows = []
            for group, pairs in schedule.items():
                for seed, case in pairs:
                    trial = ex.Trial(seed, case, policy, "heldout", group)
                    folder = output/trial.trial_id
                    row = read_case(folder, version, trial)
                    if row is None:
                        print("Starting", trial.trial_id, flush=True)
                        row = run_case(world, trial, folder, driver.load(checkpoint), version,
                                       producer, sensors.render_profile())
                    rows.append(dict(group=group, seed=seed, case=case, clean=report.clean(row),
                        outcome=row["outcome"], collisions=row["collision_events"], yields=row["yield_violations"],
                        reds=row["red_light_violations"], route_id=row["route_id"], duration_s=row["duration_s"]))
                    ex.write_json(output/"results.json", rows)
                    print(rows[-1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--port", type=int, default=2992)
    args = parser.parse_args()
    location = args.output or ROOT/"out"/("town01-yield-gated-v04" if args.gated else "town01-yield-v04")
    study(location, args.port, gated=args.gated, attach=args.attach)
