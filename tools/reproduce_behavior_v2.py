"""Verify, package, or reproduce the CARLA behavior-v2 fit from the public inputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/behavior-v2")
REFERENCE = Path("models/behavior-v2")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify(data, expected, name):
    if sha256(data) != expected:
        raise ValueError(f"SHA-256 mismatch for {name}")


def inputs():
    manifest = json.loads((ROOT / DATA / "provenance.json").read_bytes())
    report = json.loads((ROOT / REFERENCE / "training-report.json").read_bytes())
    for name, item in manifest["artifacts"].items():
        verify((ROOT / name).read_bytes(), item["sha256"], name)
    for name, item in manifest["fit_sources"].items():
        raw = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        verify(raw, item["lf_sha256"], name)

    episodes = manifest["supplement"]
    names = [row["member"] for row in episodes]
    if names != sorted(set(names)) or any(Path(name).name != name for name in names):
        raise ValueError("invalid or duplicate supplemental episode names")
    if len(names) != 32 or sum(row["rows"] for row in episodes) != 7150:
        raise ValueError("incomplete training cohort")
    with zipfile.ZipFile(ROOT / DATA / "lessons.zip") as archive:
        if sorted(archive.namelist()) != names:
            raise ValueError("lesson members differ from the fitted cohort")
        for row in episodes:
            name = row["member"]
            if report["input_sha256"][name] != row["sha256"]:
                raise ValueError(f"training report disagrees with provenance for {name}")
            verify(archive.read(name), row["sha256"], name)
    return manifest


def bundle(path, manifest):
    files = [*manifest["artifacts"], *manifest["fit_sources"], Path("tools/reproduce_behavior_v2.py"),
             DATA / "provenance.json", DATA / "requirements-colab.txt", DATA / "README.md"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for rel in files:
            archive.write(ROOT / rel, Path(rel).as_posix())
    print(f"Colab bundle: {path} ({path.stat().st_size:,} bytes)")


def reproduce(output, manifest):
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as working_directory:
        work = Path(working_directory)
        # Filenames were checked above. Only manifest-listed observations are
        # reconstructed, in the same lexical order as the original fit.
        lessons = []
        with zipfile.ZipFile(ROOT / DATA / "lessons.zip") as archive:
            for row in manifest["supplement"]:
                dest = work / row["member"]
                dest.write_bytes(archive.read(row["member"]))
                lessons.append(str(dest))
        candidate = work / "candidate.npz"
        report = work / "training-report.json"
        args = [sys.executable, "-m", "crosswalk.behavior_v2",
                "--legacy", str(ROOT / "data/training.npz"),
                "--legacy-labels", str(ROOT / DATA / "labels.npz"),
                "--supplement", *lessons, "--base", str(ROOT / "models/fusion.npz"),
                "--out", str(candidate), "--report", str(report)]
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        subprocess.run(args, cwd=ROOT, env=env, check=True)
        # JSON newlines can differ by operating system. Compare decoded report
        # values; verify all fitted model bytes independently.
        verify(candidate.read_bytes(), manifest["repeat"]["model_sha256"], candidate.name)
        if json.loads(report.read_bytes()) != json.loads((ROOT / REFERENCE / report.name).read_bytes()):
            raise ValueError("reconstructed training report differs from the reference")
        for path in (candidate, report):
            dest = output / path.name
            if dest.exists() and dest.read_bytes() != path.read_bytes():
                raise FileExistsError(f"existing output differs, choose a new --output directory: {dest}")
        for path in (candidate, report):
            dest = output / path.name
            if not dest.exists():
                shutil.copyfile(path, dest)
    print(f"Reproduced {output / 'candidate.npz'}")
    print(f"SHA-256 {manifest['repeat']['model_sha256']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify-only", action="store_true", help="check hashes without fitting")
    action.add_argument("--colab-zip", type=Path, help="create a small portable input bundle")
    parser.add_argument("--output", type=Path, default=ROOT / "out/behavior-v2-refit")
    args = parser.parse_args()
    manifest = inputs()
    print("Verified 48 archived demonstrations and 32 supplemental episodes.", flush=True)
    if args.colab_zip:
        bundle(args.colab_zip, manifest)
    elif not args.verify_only:
        reproduce(args.output.resolve(), manifest)


if __name__ == "__main__":
    main()
