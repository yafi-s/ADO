"""Run the real-data Waymo pilot inside an authenticated Colab GPU runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/content/ado-data"))
    parser.add_argument("--output", type=Path, default=Path("/content/ado-run"))
    parser.add_argument("--train-shards", type=int, default=112)
    parser.add_argument("--val-shards", type=int, default=40)
    parser.add_argument("--prefix-mib", type=int, default=32)
    parser.add_argument("--prior-probe-bytes", type=int, default=0)
    parser.add_argument("--auth", choices=("adc","job-grant"), default="adc")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41,73])
    args = parser.parse_args()
    if not shutil.which("nvidia-smi"):
        raise SystemExit("Select a Colab GPU runtime before starting.")
    gpu = subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                                  text=True).strip()
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True)
    provenance = {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((repo/"waymo_training").rglob("*.py"))}
    meta = dict(gpu=gpu, source_sha256=provenance, source="Waymo Open Dataset End-to-End Driving v1.0.0",
                frame="right-handed, rear axle origin, +x forward, +y left", evaluation="open-loop trajectory prediction")
    # Keep the output folder empty before train() creates its own configuration.
    args.data.mkdir(parents=True,exist_ok=True)
    (args.data/"runtime.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf8")
    env = {**os.environ,"TF_CPP_MIN_LOG_LEVEL":"2","PYTHONUNBUFFERED":"1"}
    def run(module, flags):
        subprocess.run([sys.executable,"-u","-m","waymo_training."+module,*map(str,flags)],
                       cwd=repo,env=env,check=True)
    print("Remote GPU:",gpu,flush=True)
    run("extract",["--out",args.data,"--auth",args.auth,"--workers",2,
                   "--train-shards",args.train_shards,"--val-shards",args.val_shards,
                   "--prefix-mib",args.prefix_mib,"--prior-probe-bytes",args.prior_probe_bytes])
    feature_dir = args.data / "features"
    flags = ["--features",feature_dir,"--output",args.output,"--epochs",args.epochs,
             "--seeds",*args.seeds,"--device","gpu"]
    if (args.output/"config.json").exists():
        flags.append("--resume")
    run("train",flags)
    if not (args.output/"evaluation.json").is_file():
        run("evaluate",["--features",feature_dir,"--run",args.output,"--device","gpu"])
    (args.output/"runtime.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf8")
    shutil.copy2(repo/"waymo_training/NOTICE.md",args.output/"NOTICE.md")
    print("Completed on Colab. Private weights and evaluation:",args.output,flush=True)


if __name__ == "__main__":
    main()
