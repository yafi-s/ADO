"""Record selected scheduled CARLA trials without ticking the simulation or controlling actors."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import carla


def tail(path):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 6000))
        return stream.read().decode("utf-8", errors="replace").splitlines()


def capture(client, world, trial, progress_log, output):
    start = f"Starting primary {trial}"
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        recent = tail(progress_log)
        if recent and recent[-1] == start:
            break
        if any(s.startswith("evaluate ") and f" {trial} " in s for s in recent):
            raise RuntimeError(f"{trial} has already completed")
        time.sleep(.15)
    else:
        raise TimeoutError(f"Did not see {trial} begin")
    path = (output / f"{trial}.rec").resolve()
    if path.exists():
        raise FileExistsError(path)
    first = world.get_snapshot()
    response = client.start_recorder(str(path), True)
    try:
        while time.monotonic() < deadline:
            if any(s.startswith("evaluate ") and f" {trial} " in s for s in tail(progress_log)):
                break
            time.sleep(.15)
        else:
            raise TimeoutError(f"Did not see {trial} finish")
    finally:
        client.stop_recorder()
    last = world.get_snapshot()
    info = client.show_recorder_file_info(str(path), False)
    if not path.exists() or path.stat().st_size < 100 or "Frames" not in info:
        raise RuntimeError(f"CARLA did not finalize {trial}: {info[:160]}")
    result = dict(trial_id=trial, filename=path.name, bytes=path.stat().st_size,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  producer={"server_version":client.get_server_version(),"client_version":client.get_client_version(),
                            "map":world.get_map().name.split("/")[-1]},
                  observed_start_frame=first.frame, observed_stop_frame=last.frame,
                  start_elapsed_s=first.timestamp.elapsed_seconds, stop_elapsed_s=last.timestamp.elapsed_seconds,
                  recorder_summary=info, method="Additional-data CARLA recorder, independent observer with no tick authority")
    (output / f"{trial}.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(f"Saved {path.name} ({result['bytes']} bytes).",flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--port",type=int,default=2468)
    cli.add_argument("--progress-log",type=Path,required=True)
    cli.add_argument("--output",type=Path,default=Path("results/native-replays"))
    cli.add_argument("trials",nargs="+")
    args=cli.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    client=carla.Client("127.0.0.1",args.port,1)
    client.set_timeout(10.)
    if client.get_server_version().split("-")[0] != "1.0":
        raise RuntimeError("A CARLA 1.0 server is required")
    for trial in args.trials:
        capture(client,client.get_world(),trial,args.progress_log,args.output)
