from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from subprocess import CompletedProcess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs
from zipfile import ZipFile

from waymo_training import colab_cli_run as launcher


class StubColab:
    def __init__(self, *, corrupt=False, collision=False, state="complete", swap=False):
        self.actions = []
        self.corrupt = corrupt
        self.endpoint = "existing" if collision else None
        self.state = state
        self.swap = swap
        self.payload = b"stub archive"
        self.digest = hashlib.sha256(self.payload).hexdigest()

    def __call__(self, exe, args, **options):
        action, _, name, *rest = args
        self.actions.append(action)
        if action == "status":
            if self.endpoint:
                return f"[{name}] {self.endpoint} | Hardware: T4 | Shape: Standard"
            return f"[colab] Session '{name}' not found.\n"
        if action == "new":
            assert rest == ["--gpu", "T4"]
            self.endpoint = "owned"
        elif action == "exec":
            assert rest[0] == "-f"
            script = Path(rest[1]).read_text()
            compile(script, rest[1], "exec")
            if Path(rest[1]).name == "setup.py":
                assert launcher.SOURCE_SHA in script
                return "ADO_SETUP_READY\n"
            if self.swap:
                self.endpoint = "replacement"
            return f"ADO_ARCHIVE {self.state} {self.digest}\n"
        elif action == "auth":
            assert options["interactive"]
        elif action == "upload":
            self.uploaded_path = Path(rest[0])
            assert self.uploaded_path.is_file() and rest[1] == launcher.GRANT_PATH
            assert "stub_grant" not in str(args)
        elif action == "download":
            assert rest[0] == launcher.ARCHIVE_PATH
            Path(rest[1]).write_bytes(b"corrupted" if self.corrupt else self.payload)
        elif action == "stop":
            assert self.endpoint == "owned"
            self.endpoint = None
        return ""


class LauncherTests(unittest.TestCase):
    def test_dry_run_never_contacts_colab_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "new folder"
            with patch.object(launcher.shutil, "which", side_effect=AssertionError), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(launcher.main(["--output", str(destination), "--dry-run"]), 0)
            self.assertEqual(json.loads(output.getvalue())["dataset_transfer_limit_bytes"], 5 * 1024 ** 3)
            self.assertFalse(destination.exists())

    def test_owned_job_grant_session_downloads_verifies_and_stops(self):
        stub = StubColab()
        with tempfile.TemporaryDirectory() as directory, patch.object(launcher, "run_colab", stub):
            with patch.object(launcher, "issue_job_grant", return_value={"access_token": "stub_grant"}):
                archive = launcher.pilot(SimpleNamespace(output=Path(directory), gpu="T4", auth="job-grant"),
                                         "colab", "gcloud")
            self.assertFalse(stub.uploaded_path.exists())
            self.assertEqual(archive.read_bytes(), stub.payload)
            self.assertEqual(archive.with_suffix(".sha256").read_text().strip(), stub.digest)
            self.assertLess(stub.actions.index("upload"), stub.actions.index("download"))
            self.assertEqual(stub.actions[-1], "stop")

    def test_collision_does_not_create_or_stop_a_session(self):
        stub = StubColab(collision=True)
        with patch.object(launcher, "run_colab", stub), self.assertRaisesRegex(RuntimeError, "collision"):
            launcher.pilot(SimpleNamespace(), "colab", None)
        self.assertEqual(stub.actions, ["status"])

    def test_adc_uses_official_auth_with_terminal_and_history_suppression(self):
        with patch.object(launcher.subprocess, "run", return_value=CompletedProcess([], 0)) as child:
            launcher.run_colab("colab", ["auth", "-s", "test-session"], interactive=True)
        command, = child.call_args.args
        self.assertIn("from colab_cli.cli import main", command[3])
        self.assertIn("HistoryLogger.log_event", command[3])
        self.assertEqual(command[-3:], ["auth", "-s", "test-session"])
        self.assertIsNone(child.call_args.kwargs["stdin"])
        self.assertFalse(child.call_args.kwargs["capture_output"])

    def test_hash_mismatch_preserves_the_remote_checkpoint(self):
        stub = StubColab(corrupt=True)
        with tempfile.TemporaryDirectory() as directory, patch.object(launcher, "run_colab", stub):
            with redirect_stderr(io.StringIO()) as output, self.assertRaisesRegex(RuntimeError, "SHA-256"):
                launcher.pilot(SimpleNamespace(output=Path(directory), gpu="T4", auth="adc"), "colab", None)
            self.assertIn("checkpoint recovery", output.getvalue())
            self.assertEqual(list(Path(directory).glob("*/*")), [])
        self.assertIn("auth", stub.actions)
        self.assertNotIn("upload", stub.actions)
        self.assertNotIn("stop", stub.actions)

    def test_session_change_prevents_download_or_stop(self):
        stub = StubColab(swap=True)
        with tempfile.TemporaryDirectory() as directory, patch.object(launcher, "run_colab", stub):
            with redirect_stderr(io.StringIO()), self.assertRaisesRegex(RuntimeError, "changed before download"):
                launcher.pilot(SimpleNamespace(output=Path(directory), gpu="T4", auth="adc"), "colab", None)
        self.assertNotIn("download", stub.actions)
        self.assertNotIn("stop", stub.actions)

    def test_verified_partial_run_is_recoverable_and_stops(self):
        stub = StubColab(state="partial")
        with tempfile.TemporaryDirectory() as directory, patch.object(launcher, "run_colab", stub):
            with self.assertRaisesRegex(RuntimeError, "Verified recovery archive"):
                launcher.pilot(SimpleNamespace(output=Path(directory), gpu="T4", auth="adc"), "colab", None)
            self.assertEqual(next(Path(directory).glob("*/waymo-colab-partial.zip")).read_bytes(), stub.payload)
        self.assertEqual(stub.actions[-1], "stop")

    def test_sts_restrictions_and_all_access_probe_results(self):
        def get(request, **kwargs):
            calls.append(request)
            if request.get_method() == "POST":
                return io.BytesIO(b'{"access_token":"stub_grant", "expires_in":3600}')
            status = 200 if len(calls) == 2 else 403 if len(calls) == 3 else list_status
            if status == 200:
                response = io.BytesIO(b"{}")
                response.status = 200
                return response
            raise HTTPError(request.full_url, status, "", {}, None)

        for list_status in (403, 200):
            calls = []
            with patch.object(launcher.subprocess, "run", return_value=CompletedProcess([], 0, stdout="stub_source\n")):
                with patch.object(launcher, "urlopen", side_effect=get):
                    if list_status == 403:
                        grant = launcher.issue_job_grant("gcloud")
                        self.assertEqual(grant["allowed_prefixes"], launcher.PREFIXES)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "scope check failed"):
                            launcher.issue_job_grant("gcloud")
            form = parse_qs(calls[0].data.decode("ascii"))
            rule, = json.loads(form["options"][0])["accessBoundary"]["accessBoundaryRules"]
            self.assertEqual(rule["availablePermissions"], ["inRole:roles/storage.objectViewer"])
            expression = rule["availabilityCondition"]["expression"]
            self.assertEqual(expression.count("resource.name.startsWith"), 2)
            for prefix in launcher.PREFIXES:
                self.assertIn(prefix, expression)
            self.assertIn(launcher.PREFIXES[0], calls[1].full_url)
            self.assertIn(launcher.TEST_OBJECT, calls[2].full_url)
            self.assertIn("?maxResults=1", calls[3].full_url)

    def test_remote_archive_includes_only_allowed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_names = (
                "ado-waymo/waymo_training/NOTICE.md", "ado-run/checkpoint-vision-41.weights.h5",
                "ado-run/resume/vision-41/ckpt-1.index", "ado-run/resume/vision-41/ckpt-1.data-00000-of-00001",
                "ado-run/resume/vision-41/checkpoint", "ado-run/evaluation.json", "ado-data/manifest.json",
                "ado-data/features/train-00000.npz", "ado-run/predictions.npz", "ado-data/frames.tfrecord",
            )
            for name in artifact_names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("stub")
            grant = root / ".grant"
            grant.touch()
            archive = root / "result.zip"
            source = launcher.RUN_SCRIPT.replace("/content/", root.as_posix() + "/")
            (root / "ado-waymo-vision.npz").touch()
            output = io.StringIO()
            with patch.object(launcher.subprocess, "run", return_value=CompletedProcess([], 0)), redirect_stdout(output):
                exec(source, {"ARCHIVE_PATH": str(archive), "GRANT_PATH": str(grant),
                              "AUTH": "job-grant", "SOURCE_SHA": launcher.SOURCE_SHA})
            self.assertFalse(grant.exists())
            self.assertIn("ADO_ARCHIVE complete " + hashlib.sha256(archive.read_bytes()).hexdigest(), output.getvalue())
            with ZipFile(archive) as packed:
                self.assertEqual(set(packed.namelist()), {
                    "NOTICE.md", "ado-waymo-vision.npz", "provenance.json", "run/checkpoint-vision-41.weights.h5",
                    "run/resume/vision-41/ckpt-1.index", "run/resume/vision-41/ckpt-1.data-00000-of-00001",
                    "run/resume/vision-41/checkpoint", "run/evaluation.json", "extraction/manifest.json",
                })


if __name__ == "__main__":
    unittest.main()
