"""Run the Waymo pilot in an owned Colab session and download a private archive."""

import argparse
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4


REPO_URL = "https://github.com/yafi-s/ADO.git"
SOURCE_SHA = "80c85dc752665728ad541fcc03885600a5f00cdd"
BUCKET = "waymo_open_dataset_end_to_end_camera_v_1_0_0"
PREFIXES = ("training_202504031202_202504151040.tfrecord-", "val_202504211843.tfrecord-")
TEST_OBJECT = "test_202504211836-202504220845.tfrecord-00000-of-00266"
GRANT_PATH = "/content/.ado-waymo-job-grant.json"
ARCHIVE_PATH = "/content/ado-private-export.zip"
GIB = 1024 ** 3

# The CLI's interactive automation history can record the OAuth code typed into
# the terminal. Invoke its official auth command without that transcript.
AUTH_ENTRYPOINT = """\
import builtins
from getpass import getpass
import logging
from colab_cli.cli import main
from colab_cli.history import HistoryLogger

builtins.input = getpass
logging.disable(logging.CRITICAL)
HistoryLogger.log_event = lambda *args, **kwargs: None
main()
"""

SETUP_SCRIPT = f"""\
from pathlib import Path
import subprocess
import sys

repo = Path('/content/ado-waymo')
subprocess.run(['git', 'clone', '--no-checkout', '--depth=1', '--filter=blob:none',
                '--sparse', {REPO_URL!r}, str(repo)], check=True)
subprocess.run(['git', 'sparse-checkout', 'set', 'waymo_training'], cwd=repo, check=True)
subprocess.run(['git', 'fetch', '--depth=1', 'origin', {SOURCE_SHA!r}], cwd=repo, check=True)
subprocess.run(['git', 'checkout', '--detach', {SOURCE_SHA!r}], cwd=repo, check=True)
if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != {SOURCE_SHA!r}:
    raise RuntimeError('Unexpected training source revision')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                '-r', 'waymo_training/requirements-colab.txt'], cwd=repo, check=True)
print('ADO_SETUP_READY', flush=True)
"""

RUN_SCRIPT = """\
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

repo = Path('/content/ado-waymo')
data = Path('/content/ado-data')
run = Path('/content/ado-run')
archive = Path(ARCHIVE_PATH)
grant = Path(GRANT_PATH)
portable = archive.with_name('ado-waymo-vision.npz')
env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL='2', PYTHONUNBUFFERED='1',
           OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
if AUTH == 'job-grant':
    env['ADO_WAYMO_GRANT_FILE'] = str(grant)

portable_ok = False
try:
    print('Extracting Waymo data and training on the GPU.', flush=True)
    with archive.with_name('ado-job.log').open('w', encoding='utf8') as log:
        trained = subprocess.run(
            [sys.executable, '-u', '-m', 'waymo_training.colab_job',
             '--data', str(data), '--output', str(run), '--auth', AUTH],
            cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
        ).returncode == 0
        if trained:
            print('Exporting the portable vision checkpoint and training state.', flush=True)
            portable_ok = subprocess.run(
                [sys.executable, '-m', 'waymo_training.portable', '--run', str(run),
                 '--features', str(data / 'features'), '--output', str(portable)],
                cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
            ).returncode == 0
finally:
    if AUTH == 'job-grant':
        grant.unlink(missing_ok=True)

members = {}
for path in run.glob('checkpoint-*.weights.h5'):
    if re.fullmatch(r'checkpoint-(history|vision)-[0-9]+[.]weights[.]h5', path.name):
        members['run/' + path.name] = path
for path in (run / 'resume').glob('*/*'):
    if (re.fullmatch(r'(history|vision)-[0-9]+', path.parent.name)
            and re.fullmatch(r'checkpoint|ckpt-[0-9]+[.](index|data-[0-9]+-of-[0-9]+)', path.name)):
        members['run/' + path.relative_to(run).as_posix()] = path
if not members:
    print('ADO_NO_CHECKPOINTS', flush=True)
else:
    for directory, prefix, names in (
        (run, 'run/', ('config.json', 'training_curve.json', 'evaluation.json', 'runtime.json')),
        (data, 'extraction/', ('plan.json', 'manifest.json', 'transfer-budget.json', 'runtime.json')),
    ):
        for name in names:
            path = directory / name
            if path.is_file():
                members[prefix + name] = path
    members['NOTICE.md'] = repo / 'waymo_training' / 'NOTICE.md'
    if portable.is_file():
        members[portable.name] = portable
    state = 'complete' if portable_ok and portable.is_file() else 'partial'
    pending = archive.with_suffix('.zip.part')
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as packed:
        packed.writestr('provenance.json', json.dumps({'source_sha': SOURCE_SHA, 'state': state}) + '\\n')
        for name, path in sorted(members.items()):
            packed.write(path, name)
    pending.replace(archive)
    with archive.open('rb') as stream:
        sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    archive.with_suffix('.sha256').write_text(sha256 + '\\n', encoding='ascii')
    print('ADO_ARCHIVE', state, sha256, flush=True)
"""


def run_colab(exe, args, *, interactive=False, timeout=None):
    command = [sys.executable, "-P", "-c", AUTH_ENTRYPOINT, *args] if interactive else [exe, *args]
    try:
        result = subprocess.run(
            command, text=True, encoding="utf8", errors="replace", timeout=timeout,
            stdin=None if interactive else subprocess.DEVNULL,
            capture_output=not interactive,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Colab {args[0]} timed out.") from None
    if result.returncode:
        raise RuntimeError(f"Colab {args[0]} failed (exit {result.returncode}).")
    return result.stdout or ""


def session_endpoint(exe, name):
    try:
        output = run_colab(exe, ["status", "-s", name], timeout=60)
    except RuntimeError:
        raise RuntimeError("Colab status failed. Run `colab sessions` in your terminal to check sign-in.") from None
    match = re.search(rf"^\[{re.escape(name)}\] (\S+) \| Hardware:", output, re.MULTILINE)
    if match:
        return match[1]
    if f"[colab] Session '{name}' not found." in output.splitlines():
        return None
    raise RuntimeError("Could not confirm the Colab session identity.")


def issue_job_grant(gcloud):
    """Exchange the local Google access token before passing a restricted grant to Colab."""
    try:
        source = subprocess.run(
            [gcloud, "auth", "print-access-token"], capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=30, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        raise RuntimeError("Could not get a Google Cloud access token. Check `gcloud auth login`.") from None
    if not source:
        raise RuntimeError("Google Cloud returned an empty access token.")

    resource = f"projects/_/buckets/{BUCKET}"
    rule = {
        "availableResource": "//storage.googleapis.com/" + resource,
        "availablePermissions": ["inRole:roles/storage.objectViewer"],
        "availabilityCondition": {"expression": " || ".join(
            f"resource.name.startsWith('{resource}/objects/{prefix}')" for prefix in PREFIXES
        )},
    }
    form = urlencode({
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "subject_token": source,
        "options": json.dumps({"accessBoundary": {"accessBoundaryRules": [rule]}}),
    }).encode("ascii")
    request = Request("https://sts.googleapis.com/v1/token", data=form,
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        issued = time.time()
        with urlopen(request, timeout=15) as response:
            grant = json.load(response)
        token = grant["access_token"]
        lifetime = min(2400, int(grant["expires_in"]))
        if not isinstance(token, str) or not re.fullmatch(r"[!-~]+", token) or lifetime <= 0:
            raise ValueError("invalid grant response")
    except (URLError, OSError, ValueError, TypeError, KeyError):
        raise RuntimeError("Google STS could not issue the restricted Waymo grant.") from None

    base = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
    headers = {"Authorization": "Bearer " + token}
    probes = (
        (base + "/" + PREFIXES[0] + "00000-of-00263?fields=size,generation", 200),
        (base + "/" + TEST_OBJECT + "?fields=name", 403),
        (base + "?maxResults=1&fields=items(name)", 403),
    )
    for url, expected in probes:
        try:
            with urlopen(Request(url, headers=headers), timeout=12) as response:
                status = response.status
        except HTTPError as error:
            status = error.code
            error.close()
        except (URLError, OSError):
            raise RuntimeError("Could not verify the Waymo grant's object access.") from None
        if status != expected:
            raise RuntimeError(f"Waymo access scope check failed (HTTP {status}, expected {expected}).")
    return {"access_token": token, "expires_at": issued + lifetime,
            "bucket": BUCKET, "allowed_prefixes": PREFIXES}


def upload_grant(exe, name, gcloud):
    grant = issue_job_grant(gcloud)
    with tempfile.TemporaryDirectory(prefix="ado-waymo-grant-") as directory:
        path = Path(directory) / ".ado-waymo-job-grant.json"
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf8") as stream:
            json.dump(grant, stream)
        run_colab(exe, ["upload", "-s", name, str(path), GRANT_PATH], timeout=60)


def exec_script(exe, name, directory, filename, source, *, timeout):
    path = directory / filename
    path.write_text(source, encoding="ascii")
    return run_colab(exe, ["exec", "-s", name, "-f", str(path), "--timeout", str(timeout)])


def download_archive(exe, name, destination, remote_hash):
    pending = destination.with_suffix(".zip.part")
    try:
        run_colab(exe, ["download", "-s", name, ARCHIVE_PATH, str(pending)], timeout=180)
        with pending.open("rb") as stream:
            local_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if local_hash != remote_hash:
            raise RuntimeError("The Colab archive failed SHA-256 verification.")
        pending.replace(destination)
        destination.with_suffix(".sha256").write_text(local_hash + "\n", encoding="ascii")
    finally:
        pending.unlink(missing_ok=True)


def pilot(args, colab, gcloud):
    name = "ado-waymo-" + uuid4().hex
    if session_endpoint(colab, name) is not None:
        raise RuntimeError(f"Colab session name collision for {name}.")
    destination = args.output.expanduser().resolve() / name
    destination.mkdir(parents=True, mode=0o700)
    endpoint = None
    grant_pending = False
    safe_to_stop = True
    print(f"Creating Colab {args.gpu} session {name}.", flush=True)
    try:
        run_colab(colab, ["new", "-s", name, "--gpu", args.gpu], timeout=180)
        endpoint = session_endpoint(colab, name)
        if endpoint is None:
            raise RuntimeError(f"Colab did not confirm session {name}.")
        with tempfile.TemporaryDirectory(prefix="ado-waymo-colab-") as directory:
            scripts = Path(directory)
            setup = exec_script(colab, name, scripts, "setup.py", SETUP_SCRIPT, timeout=180)
            if "ADO_SETUP_READY" not in setup.splitlines():
                raise RuntimeError("Colab did not finish installing the pinned training source.")
            if session_endpoint(colab, name) != endpoint:
                raise RuntimeError(f"Colab session {name} changed before authentication.")
            if args.auth == "adc":
                run_colab(colab, ["auth", "-s", name], interactive=True)
            else:
                grant_pending = True
                upload_grant(colab, name, gcloud)
            if session_endpoint(colab, name) != endpoint:
                raise RuntimeError(f"Colab session {name} changed before training.")
            safe_to_stop = False
            print("Running the byte-bounded Waymo pilot and packaging private results.", flush=True)
            prelude = (f"AUTH = {args.auth!r}\nSOURCE_SHA = {SOURCE_SHA!r}\n"
                       f"GRANT_PATH = {GRANT_PATH!r}\nARCHIVE_PATH = {ARCHIVE_PATH!r}\n")
            result = exec_script(colab, name, scripts, "run.py", prelude + RUN_SCRIPT, timeout=0)
            if "ADO_NO_CHECKPOINTS" in result.splitlines():
                grant_pending = False  # This marker is emitted after remote grant cleanup.
                safe_to_stop = True
                raise RuntimeError("The Colab job returned without a checkpoint. Check dataset access and GPU availability.")
            receipt = re.search(r"^ADO_ARCHIVE (complete|partial) ([0-9a-f]{64})$", result, re.MULTILINE)
            if not receipt:
                raise RuntimeError("Colab did not confirm an export. The job may still be running or export may have failed.")
            grant_pending = False
            if session_endpoint(colab, name) != endpoint:
                raise RuntimeError(f"Colab session {name} changed before download.")
            archive = destination / ("waymo-colab.zip" if receipt[1] == "complete" else "waymo-colab-partial.zip")
            download_archive(colab, name, archive, receipt[2])
            safe_to_stop = True
            if receipt[1] == "partial":
                raise RuntimeError(f"Training or portable export failed. Verified recovery archive: {archive}")
            return archive
    finally:
        if endpoint is not None:
            job_failed = sys.exc_info()[0] is not None
            try:
                current_endpoint = session_endpoint(colab, name)
                if current_endpoint is not None and current_endpoint != endpoint:
                    raise RuntimeError(f"Colab session {name} no longer matches the created assignment.")
                if current_endpoint is None:
                    if not safe_to_stop:
                        print(f"Colab session {name} expired before checkpoint recovery.", file=sys.stderr)
                else:
                    if grant_pending:
                        try:
                            run_colab(colab, ["rm", "-s", name, GRANT_PATH], timeout=30)
                        except RuntimeError:
                            pass  # The extractor normally consumes the grant before training.
                    if safe_to_stop:
                        run_colab(colab, ["stop", "-s", name], timeout=60)
                    else:
                        print(f"Kept Colab session {name} for checkpoint recovery. "
                              f"Check {ARCHIVE_PATH}, its .sha256 file, or /content/ado-run. "
                              f"After recovery, stop it with `colab stop -s {name}`.", file=sys.stderr)
            except (RuntimeError, OSError) as exc:
                detail = f"Could not confirm cleanup of Colab session {name}. {exc}"
                if not job_failed:
                    raise RuntimeError(f"Verified private archive at {archive}. {detail}") from None
                print(detail, file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Private local directory for verified results")
    parser.add_argument("--gpu", choices=("T4", "L4", "G4", "H100", "A100"), default="T4")
    parser.add_argument("--auth", choices=("job-grant", "adc"), default="job-grant",
                        help="Restricted grant from local gcloud, or interactive Colab ADC")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan locally without starting Colab")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"source_repo": REPO_URL, "source_sha": SOURCE_SHA, "gpu": args.gpu,
                          "dataset_transfer_limit_bytes": 5 * GIB, "auth": args.auth,
                          "output": str(args.output.expanduser().resolve()),
                          "stages": ["create unique Colab session", "sparse checkout", "authenticate",
                                     "extract and train", "package allowlisted artifacts",
                                     "download and verify SHA-256", "stop owned session when safe"]}, indent=2))
        return 0
    colab = shutil.which("colab")
    if not colab:
        parser.error("Install Google's Colab CLI with `python -m pip install google-colab-cli`.")
    gcloud = None
    if args.auth == "job-grant":
        gcloud = shutil.which("gcloud.cmd") or shutil.which("gcloud")
        if not gcloud:
            parser.error("Install and sign in to gcloud for --auth job-grant, or use --auth adc in a terminal.")
    else:
        if not sys.stdin.isatty():
            parser.error("Run --auth adc from a terminal to complete the interactive Google sign-in.")
        if find_spec("colab_cli") is None:
            parser.error("For interactive auth, install the Colab CLI in this Python environment with `python -m pip install google-colab-cli`.")
    try:
        archive = pilot(args, colab, gcloud)
    except (RuntimeError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Waymo Colab run interrupted.", file=sys.stderr)
        return 130
    print(f"Verified private archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
