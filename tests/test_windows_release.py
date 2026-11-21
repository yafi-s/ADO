import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from tools import build_windows_release as release


def json_bytes(data):
    return json.dumps(data).encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fixture_zip(members):
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, contents in members.items():
            archive.writestr(release.PREFIX + name, contents)
    stream.seek(0)
    return ZipFile(stream)


def city_film(complete, surround=False):
    media = release.SURROUND_MEDIA if surround else "web/media/long-city-drive"
    model = b"test checkpoint"
    score = dict(frame=42, camera_frame=42, video_index=0, map=release.CITY_PRODUCER["map"])
    frame = dict(score, run_id="long-city-drive")
    files = {name: b"test artifact" for name in release.CITY_ARTIFACTS}
    files.update({
        "telemetry.json": json_bytes([score]), "replay.json": json_bytes([frame]),
        "town.json": json_bytes(dict(producer=release.CITY_PRODUCER, roads=[[[1, 2, 3], [2, 3, 4]]])),
    })
    milestones = dict(triggered=True, exposed=True, cleared=True, passed=True, handled=True, violation=False)
    kinds = sorted(release.CITY_EVENTS) if complete else ["pedestrian"]
    events = [dict(milestones, kind=kind) for kind in kinds]
    outcome = "complete" if complete else "lane_departure"
    summary = dict(outcome=outcome, events=events, frames=1, progress_m=1420 if complete else 160,
                   route_length_m=1420, duration_s=550 if complete else 80,
                   collision_events=0, red_light_violations=0, yield_violations=0)
    manifest = dict(producer=release.CITY_PRODUCER, benchmark_member=False, teacher=False, map_asset="town.json",
                    source_mode="one closed-loop simulation with simultaneous camera capture",
                    policy=dict(schema="ado-behavior-v2", path="candidate.npz", sha256=sha256(model)),
                    required_event_kinds=sorted(release.CITY_EVENTS), trial_id="long-city-drive",
                    outcome=outcome, events=events, summary=summary,
                    passed=complete, failure_reasons=[] if complete else ["route_incomplete"],
                    first_frame=42, last_frame=42, sha256={name: sha256(data) for name, data in files.items()})
    members = {f"{media}/{name}": data for name, data in files.items()}
    folder = "city-cruise" if surround else "behavior-v2"
    members[f"models/{folder}/candidate.npz"] = model
    if surround:
        manifest.update(perimeter_views=list(release.PERIMETER_VIEWS), camera_videos={})
        for camera in release.SURROUND_CAMERAS:
            name = f"{camera}.mp4"
            contents = b"test artifact"
            members[f"{media}/{name}"] = contents
            manifest["sha256"][name] = sha256(contents)
            manifest["camera_videos"][camera] = dict(nb_read_frames="1", codec_name="h264", pix_fmt="yuv420p",
                                                    r_frame_rate="10/1", width=320, height=180)
        tactical = json_bytes(dict(format="ado-tactical-pass-v1"))
        manifest["policy"].update(schema=release.SURROUND_SCHEMA, tactical_sha256=sha256(tactical))
        members[release.TACTICAL_MODEL] = tactical
        members["models/city-cruise/fit.json"] = json_bytes(dict(
            schema=release.SURROUND_SCHEMA, model_sha256=sha256(model)))
    return media, manifest, members


def benchmark_members(omit_trace=False):
    stream = BytesIO()
    stages = {stage: [dict(seed=seed, case_id="pedestrian", policy="candidate") for seed in range(100)]
              for stage in ("fresh", "regression")}
    with ZipFile(stream, "w") as evidence:
        evidence.writestr("manifest.json", json_bytes(dict(stages=stages)))
        for name in ("SHA256SUMS", "archive-info.json", "protocol.md", "execution/results/primary.json",
                     "published-v0.1/traces.jsonl.gz"):
            evidence.writestr(name, b"evidence")
        for stage, trials in stages.items():
            for name in ("report.json", "verification.json"):
                evidence.writestr(f"{stage}/{name}", b"report")
            for trial in trials:
                root = f"{stage}/{trial['seed']}-{trial['case_id']}-{trial['policy']}"
                for name in ("summary.json", "trace.jsonl.gz", "checkpoint.json"):
                    if not (omit_trace and root == "fresh/0-pedestrian-candidate" and name == "trace.jsonl.gz"):
                        evidence.writestr(f"{root}/{name}", b"trial evidence")
    bundle = stream.getvalue()
    report = json_bytes(dict(complete=True, completed_drives=100, expected_drives=100,
                             matched_pairs=49, all_pairs_verified=False))
    study = dict(archive=dict(file="closed-loop-evidence.zip", bytes=len(bundle), sha256=sha256(bundle),
                              new_trials=200, historical_primary_trials=100),
                 cohorts={stage: dict(report_sha256=sha256(report)) for stage in stages})
    return {f"{release.BEHAVIOR_ROOT}/{name}": data for name, data in {
        "study.json": json_bytes(study), "fresh.json": report, "regression.json": report,
        "closed-loop-evidence.zip": bundle,
    }.items()}


class WindowsReleaseTests(unittest.TestCase):
    def test_original_release_paths_unchanged(self):
        self.assertIs(release.release_paths("v0.1.0"), release.PACKAGE_PATHS)
        self.assertIs(release.required_files("v0.1.0"), release.REQUIRED)
        self.assertIn("waymo_training", release.release_paths("v0.2.0"))
        self.assertNotIn(release.SURROUND_DETECTOR, release.required_files("v0.2.0"))
        self.assertIn(release.SURROUND_DETECTOR, release.required_files("v0.3.0"))

    @patch.object(release, "verify_video")
    def test_city_film_accepts_truthful_outcomes(self, verify_video):
        for complete in (True, False):
            with self.subTest(complete=complete):
                media, manifest, members = city_film(complete)
                with fixture_zip(members) as archive:
                    release.verify_city_film(archive, media, manifest, Path("unused"))
        self.assertEqual(verify_video.call_count, 4)

    @patch.object(release, "verify_video")
    def test_surround_film_verifies_six_videos_and_tactical_model(self, verify_video):
        media, manifest, members = city_film(True, surround=True)
        with fixture_zip(members) as archive:
            release.verify_city_film(archive, media, manifest, Path("unused"))
        self.assertEqual([call.args[1] for call in verify_video.call_args_list],
                         [f"{media}/{camera}.mp4" for camera in release.SURROUND_CAMERAS])

        members[release.TACTICAL_MODEL] += b"\n"
        with fixture_zip(members) as archive, self.assertRaisesRegex(ValueError, "Wrong SHA-256 .*tactical-pass"):
            release.verify_city_film(archive, media, manifest, Path("unused"))
        del manifest["sha256"]["rear_right.mp4"]
        with fixture_zip(members) as archive, self.assertRaisesRegex(ValueError, "Incomplete Town03 film"):
            release.verify_city_film(archive, media, manifest, Path("unused"))

    @patch.object(release, "verify_video_file")
    def test_release_master_matches_the_committed_capture(self, verify_video):
        media, manifest, members = city_film(True, surround=True)
        video = b"native chase master"
        manifest["master"] = dict(chase=dict(file_bytes=len(video), sha256=sha256(video)))
        members[f"{media}/manifest.json"] = json_bytes(manifest)
        with TemporaryDirectory() as directory, fixture_zip(members) as package:
            scratch = Path(directory)
            source, target = scratch / "source.mp4", scratch / release.SURROUND_MASTER
            source.write_bytes(video)
            digest = release.stage_release_master(source, package.fp, target)
            self.assertEqual(digest, sha256(video))
            self.assertEqual(target.read_bytes(), video)
            verify_video.assert_called_once()
            source.write_bytes(b"X" + video[1:])
            with self.assertRaisesRegex(ValueError, "Wrong surround master SHA-256"):
                release.stage_release_master(source, package.fp, target)

    def test_tactical_study_accepts_yield_then_pass_and_rejects_swapped_timing(self):
        score = dict(case="occupied", passed=True, frames=140, collisions=0, yield_s=3.0,
                     first_yield_s=0.4, first_commit_s=4.0, first_alongside_s=8.0, completed_pass_s=10.0,
                     lead_speed_during_pass_mps=1.0, minimum_passing_clearance_m=1.25)
        capture = dict(cohort="tactical-study", case="occupied", count=140, passed=True, outcome="complete",
                       summary=dict(scenario="overtake", frames=140, duration_s=14.0, outcome="complete",
                                    collision_events=0), score=score)
        release.verify_tactical_result(capture, "overtake-yield", "occupied")
        score["first_yield_s"] = 4.1
        with self.assertRaisesRegex(ValueError, "Wrong yield timing"):
            release.verify_tactical_result(capture, "overtake-yield", "occupied")

    def test_city_film_rejects_false_success_and_desynchronized_replay(self):
        media, manifest, members = city_film(False)
        manifest.update(passed=True, failure_reasons=[])
        with fixture_zip(members) as archive, self.assertRaisesRegex(ValueError, "Unsubstantiated"):
            release.verify_city_film(archive, media, manifest, Path("unused"))

        media, manifest, members = city_film(True)
        relative = f"{media}/replay.json"
        replay = json.loads(members[relative])
        replay[0]["camera_frame"] += 1
        members[relative] = json_bytes(replay)
        manifest["sha256"]["replay.json"] = sha256(members[relative])
        with fixture_zip(members) as archive, self.assertRaisesRegex(ValueError, "Unsynchronized"):
            release.verify_city_film(archive, media, manifest, Path("unused"))

    def test_benchmark_accepts_completed_stage_with_unmatched_pairs_and_verifies_hash(self):
        members = benchmark_members()
        with fixture_zip(members) as archive:
            release.verify_behavior_study(archive)
        relative = f"{release.BEHAVIOR_ROOT}/closed-loop-evidence.zip"
        members[relative] += b"X"
        members[relative] = members[relative][1:]
        with fixture_zip(members) as archive, self.assertRaisesRegex(ValueError, "Wrong SHA-256"):
            release.verify_behavior_study(archive)

    def test_benchmark_rejects_a_missing_trial_trace(self):
        with fixture_zip(benchmark_members(omit_trace=True)) as archive, \
                self.assertRaisesRegex(ValueError, "Missing Behavior v2 fresh trial logs"):
            release.verify_behavior_study(archive)


if __name__ == "__main__":
    unittest.main()
