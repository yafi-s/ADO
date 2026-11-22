import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

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


def fixture_trial_sources(sources, second):
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, payload in sources.items():
            member = ZipInfo(name, (2026, 11, 21, 19, 5, second))
            archive.writestr(member, payload)
    return stream.getvalue()


def write_fixture_zip(path, members, commit):
    with fixture_zip(members) as archive:
        path.write_bytes(archive.fp.getvalue())
    with ZipFile(path, "a") as archive:
        archive.comment = commit.encode("ascii")


def city_film(complete, surround=False, pace=False):
    media = release.PACE_MEDIA if pace else release.SURROUND_MEDIA if surround else "web/media/long-city-drive"
    model = b"test checkpoint"
    score = dict(frame=42, camera_frame=42, video_index=0, map=release.CITY_PRODUCER["map"])
    frame = dict(score, run_id="long-city-drive")
    files = {name: b"test artifact" for name in release.CITY_ARTIFACTS}
    files.update({
        "telemetry.json": json_bytes([score]), "replay.json": json_bytes([frame]),
        "town.json": json_bytes(dict(producer=release.CITY_PRODUCER, roads=[[[1, 2, 3], [2, 3, 4]]])),
    })
    milestones = dict(triggered=True, exposed=True, cleared=True, passed=True, handled=True, violation=False)
    required_events = sorted(release.CITY_EVENTS | {"barrier"} if pace else release.CITY_EVENTS)
    kinds = required_events if complete else ["pedestrian"]
    events = [dict(milestones, kind=kind) for kind in kinds]
    outcome = "complete" if complete else "lane_departure"
    summary = dict(outcome=outcome, events=events, frames=1, progress_m=1420 if complete else 160,
                   route_length_m=1420, duration_s=550 if complete else 80,
                   collision_events=0, red_light_violations=0, yield_violations=0)
    manifest = dict(producer=release.CITY_PRODUCER, benchmark_member=False, teacher=False, map_asset="town.json",
                    source_mode="one closed-loop simulation with simultaneous camera capture",
                    policy=dict(schema="ado-behavior-v2", path="candidate.npz", sha256=sha256(model)),
                    required_event_kinds=required_events, trial_id="long-city-drive",
                    outcome=outcome, events=events, summary=summary,
                    passed=complete, failure_reasons=[] if complete else ["route_incomplete"],
                    first_frame=42, last_frame=42, sha256={name: sha256(data) for name, data in files.items()})
    members = {f"{media}/{name}": data for name, data in files.items()}
    folder = "city-pace" if pace else "city-cruise" if surround else "behavior-v2"
    members[f"models/{folder}/candidate.npz"] = model
    if surround or pace:
        manifest.update(perimeter_views=list(release.PERIMETER_VIEWS), camera_videos={})
        for camera in release.SURROUND_CAMERAS:
            name = f"{camera}.mp4"
            contents = b"test artifact"
            members[f"{media}/{name}"] = contents
            manifest["sha256"][name] = sha256(contents)
            manifest["camera_videos"][camera] = dict(nb_read_frames="1", codec_name="h264", pix_fmt="yuv420p",
                                                    r_frame_rate="10/1", width=320, height=180)
        tactical = json_bytes(dict(format="ado-tactical-pass-v1"))
        schema = release.PACE_SCHEMA if pace else release.SURROUND_SCHEMA
        manifest["policy"].update(schema=schema, tactical_sha256=sha256(tactical))
        members[release.TACTICAL_MODEL] = tactical
        members[f"models/{folder}/fit.json"] = json_bytes(dict(schema=schema, model_sha256=sha256(model)))
    if pace:
        members[release.BARRIER_MODEL] = b"test barrier model"
        manifest["policy"]["barrier_sha256"] = sha256(members[release.BARRIER_MODEL])
        pack = gzip.compress(json_bytes(dict(format="ado-replay-pack-v1", replay=[frame],
                                             shared_path=None, path_ranges=[])))
        members[f"{media}/replay-pack.json.gz"] = pack
        manifest["sha256"]["replay-pack.json.gz"] = sha256(pack)
    return media, manifest, members


def populated_city_film(schema, revision=3):
    old_media, manifest, members = city_film(False, pace=True)
    folder = release.CITY_MODEL_FOLDERS[schema]
    media = f"web/media/{'city-life' if schema == release.LIFE_SCHEMA else 'city-progress'}"
    members = {name.replace(old_media, media).replace("models/city-pace/", f"models/{folder}/"): value
               for name, value in members.items()}
    _, source_name = release.POPULATED_POLICIES[schema]
    source = b"populated checkpoint source"
    members[f"crosswalk/{source_name}"] = source
    base = "city-pace" if schema == release.LIFE_SCHEMA else "city-clearance"
    members[f"models/{base}/candidate.npz"] = b"base checkpoint"
    members[release.LIFE_CALIBRATION] = b"native calibration trials"
    run = Path(media).name
    manifest.update(population_profile="city-life", frame_period_s=.1, trial_id=run, replay="replay.json.gz")
    manifest["policy"].update(schema=schema, code_sha256={source_name: sha256(source)})
    summary = manifest["summary"]
    summary.update(policy=schema, scenario="journey", duration_s=.1)
    fit = dict(
        schema=schema, checkpoint_sha256=manifest["policy"]["sha256"], source_sha256=sha256(source),
        base_sha256=sha256(members[f"models/{base}/candidate.npz"]),
        calibration_sha256=sha256(members[release.LIFE_CALIBRATION]))
    if schema == release.PROGRESS_SCHEMA:
        crossing = b"crossing observer source"
        manifest["policy"]["code_sha256"][release.PROGRESS_CROSSING_SOURCE] = sha256(crossing)
        fit["crossing_source_sha256"] = sha256(crossing)
        if revision >= 4:
            grade = b"grade estimator source"
            manifest["policy"]["code_sha256"][release.PROGRESS_GRADE_SOURCE] = sha256(grade)
            fit.update(revision=revision, grade_source_sha256=sha256(grade))
    members[f"models/{folder}/fit.json"] = json_bytes(fit)
    score = json.loads(members[f"{media}/telemetry.json"])[0]
    score.update(sim_time=0., video_time=0., collision_count=0, red_light_violations=0, yield_violations=0,
                 **{key: summary[key] for key in ("progress_m", "route_length_m", "events")})
    replay = [dict(score, run_id=run)]
    original = json_bytes(replay)
    manifest["raw_sha256"] = {"replay.json": sha256(original)}
    del members[f"{media}/replay.json"], manifest["sha256"]["replay.json"]
    artifacts = {"replay.json.gz": gzip.compress(original), "telemetry.json": json_bytes([score]),
                 "replay-pack.json.gz": gzip.compress(json_bytes(dict(format="ado-replay-pack-v1", replay=replay,
                                                                       shared_path=None, path_ranges=[])))}
    for name, data in artifacts.items():
        members[f"{media}/{name}"] = data
        manifest["sha256"][name] = sha256(data)
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
        self.assertNotIn(release.BARRIER_MODEL, release.required_files("v0.3.0"))
        self.assertIn(release.BARRIER_MODEL, release.required_files("v0.4.0"))
        self.assertEqual(release.featured_recording("v0.4.0"),
                         (release.PACE_MEDIA, release.PACE_MASTER, release.PACE_SCHEMA))
        self.assertEqual(release.featured_recording("v0.5.0"),
                         (release.LIFE_MEDIA, release.LIFE_MASTER, release.LIFE_SCHEMA))
        self.assertEqual(release.featured_recording("v0.6.0"),
                         (release.PROGRESS_MEDIA, release.PROGRESS_MASTER, release.PROGRESS_SCHEMA))
        self.assertEqual(release.featured_recording("v0.7.0"),
                         (release.MANEUVERS_MEDIA, release.MANEUVERS_MASTER, release.MANEUVERS_SCHEMA))
        self.assertLess(set(release.required_files("v0.5.0")), set(release.required_files("v0.6.0")))
        self.assertLess(set(release.required_files("v0.6.0")), set(release.required_files("v0.7.0")))
        self.assertIn(f"{release.PACE_MEDIA}/{release.CONE_REVIEW}", release.required_files("v0.5.0"))
        with self.assertRaisesRegex(ValueError, "native 1080p --film-master"):
            release.build_release("unused", "v0.5.0", Path("unused"))

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

    def test_master_size_limit_is_strictly_below_two_gibibytes(self):
        limit = 2**31
        master = dict(path="master/chase.mp4", width=1920, height=1080, file_bytes=limit - 1,
                      sha256="a" * 64, nb_read_frames="1", codec_name="h264", pix_fmt="yuv420p", r_frame_rate="10/1")
        capture = dict(summary=dict(frames=1), master=dict(chase=master))
        release.verify_master_metadata(capture)
        master["file_bytes"] = limit
        with self.assertRaisesRegex(ValueError, "Wrong surround master metadata"):
            release.verify_master_metadata(capture)

    def test_release_oversize_relocates_only_optional_sweeps_and_pins_exact_bytes(self):
        root = release.PROGRESS_RESULTS
        small = f"{root}/first/candidate/sensor-sweeps.zip"
        large = f"{root}/second/candidate/sensor-sweeps.zip"
        linked = f"{root}/linked/candidate/sensor-sweeps.zip"
        build = {"evidence": [{"url": f"../{linked}"}]}
        members = {small: b"s" * 300, large: b"l" * 500, linked: b"t" * 201,
                   f"{root}/first/candidate/front.mp4": b"native video" * 100,
                   "web/media/build.json": json_bytes(build), "docs/release.md": b"Release checks"}
        commit = "a" * 40
        with TemporaryDirectory() as folder, fixture_zip(members) as package:
            out = Path(folder) / "ADO-v0.6.0-Windows.zip"
            out.write_bytes(package.fp.getvalue())
            size = out.stat().st_size
            with patch.object(release, "MAX_UPLOAD_BYTES", size + 1):
                self.assertEqual(release.external_sweep_inventory(out, commit), [])
            with patch.object(release, "MAX_UPLOAD_BYTES", size - 200), patch.object(release, "SWEEP_INDEX_RESERVE", 0):
                omitted = release.external_sweep_inventory(out, commit)
            expected = dict(path=small, url=f"https://raw.githubusercontent.com/yafi-s/ADO/{commit}/{small}",
                            bytes=len(members[small]), sha256=sha256(members[small]))
            self.assertEqual(omitted, [expected])

            del members[small]
            with fixture_zip(members) as compact:
                out.write_bytes(compact.fp.getvalue())
            with ZipFile(out, "a") as compact:
                compact.comment = commit.encode()
            release.add_sweep_index(out, commit, omitted)
            with ZipFile(out) as compact:
                self.assertEqual(compact.comment, commit.encode())
                self.assertEqual(set(compact.namelist()), {release.PREFIX + item for item in members} |
                                 {release.PREFIX + release.EXTERNAL_SWEEPS_INDEX})
                index = compact.read(release.PREFIX + release.EXTERNAL_SWEEPS_INDEX).decode()
                self.assertIn(expected["url"], index)
                self.assertIn(expected["sha256"], index)

    def test_maneuvers_package_archives_only_the_optional_pinned_research_needed_to_fit(self):
        root = release.PROGRESS_RESULTS
        sweep = f"{root}/trials/candidate/sensor-sweeps.zip"
        clip = f"{root}/trials/candidate/front.mp4"
        linked = f"{root}/trials/linked/front.mp4"
        current = f"{root}/trials/current/front.mp4"
        contents = {sweep: b"s" * 300, clip: b"c" * 200, linked: b"l" * 500, current: b"n" * 500,
                    "web/media/build.json": json_bytes({"evidence": [{"url": f"../{linked}"}]})}
        with TemporaryDirectory() as folder, fixture_zip(contents) as package:
            archive = Path(folder) / "ADO-v0.7.0-Windows.zip"
            archive.write_bytes(package.fp.getvalue())
            size = archive.stat().st_size
            with patch.object(release, "historical_research_paths", return_value={sweep, clip, linked}), \
                    patch.object(release, "SWEEP_INDEX_RESERVE", 0):
                with patch.object(release, "MAX_UPLOAD_BYTES", size - 100):
                    selected = release.external_sweep_inventory(archive, "a" * 40, historical=True)
                    self.assertEqual([member["path"] for member in selected], [sweep])
                with patch.object(release, "MAX_UPLOAD_BYTES", size - 400):
                    selected = release.external_sweep_inventory(archive, "a" * 40, historical=True)
                    self.assertEqual({member["path"] for member in selected}, {sweep, clip})
                for member in selected:
                    path = member["path"]
                    self.assertIn(f"/{release.PROGRESS_ARCHIVE_COMMIT}/{path}", member["url"])
                    self.assertEqual(member["sha256"], sha256(contents[path]))

    def test_companion_only_moves_eligible_movies_when_old_archives_are_insufficient(self):
        sweep = f"{release.PROGRESS_RESULTS}/trials/candidate/sensor-sweeps.zip"
        older = f"{release.PROGRESS_RESULTS}/trials/candidate/front.mp4"
        optional = f"{release.MANEUVERS_RESULTS}/pace/reference/front.mp4"
        native = f"{release.MANEUVERS_TRIALS}/candidate/slow/front.mp4"
        linked = f"{release.MANEUVERS_RESULTS}/pace/required/front.mp4"
        hero = f"{release.MANEUVERS_MEDIA}/front.mp4"
        compare = "web/media/city-maneuvers-tests/comparison.mp4"
        traces = f"{release.MANEUVERS_TRIALS}/candidate/slow/trace.jsonl.gz"
        members = {sweep: b"s" * 200, older: b"o" * 300, optional: b"p" * 300, native: b"n" * 600,
                   linked: b"l" * 900, hero: b"h" * 800, compare: b"c" * 800, traces: b"t" * 800,
                   "web/media/build.json": json_bytes({"evidence": [{"url": f"../{linked}"}]})}
        commit = "a" * 40
        with TemporaryDirectory() as folder:
            path = Path(folder) / "ADO-v0.7.0-Windows.zip"
            write_fixture_zip(path, members, commit)
            size = path.stat().st_size
            with patch.object(release, "historical_research_paths", return_value={sweep, older}), \
                    patch.object(release, "SWEEP_INDEX_RESERVE", 0):
                for needed, external_paths, companion_paths in (
                    (-1, set(), set()), (150, {sweep}, set()), (750, {sweep, older}, {optional}),
                    (1000, {sweep, older}, {optional, native}),
                ):
                    with self.subTest(needed=needed), patch.object(release, "MAX_UPLOAD_BYTES", size - needed):
                        external, companion = release.release_evidence_inventory(path, commit, historical=True,
                                                                                 allow_companion=True)
                        self.assertEqual({record["path"] for record in external}, external_paths)
                        self.assertEqual({record["path"] for record in companion}, companion_paths)
                        for record in companion:
                            data = members[record["path"]]
                            self.assertEqual(record, dict(path=record["path"], bytes=len(data), sha256=sha256(data)))
                with patch.object(release, "MAX_UPLOAD_BYTES", size - 1800):
                    with self.assertRaisesRegex(ValueError, "exceeds 2 GiB"):
                        release.release_evidence_inventory(path, commit, historical=True, allow_companion=True)

    def test_companion_archives_share_one_verified_inventory_and_validate_each_payload(self):
        commit = "a" * 40
        movie = f"{release.MANEUVERS_TRIALS}/candidate/slow/front.mp4"
        docs = "docs/release.md"
        index = release.COMPANION_EVIDENCE_INDEX
        camera = b"native camera frames"
        core_members = {docs: b"extract the companion beside the core ZIP"}
        with TemporaryDirectory() as folder:
            main = Path(folder) / "ADO-v0.7.0-Windows.zip"
            companion = main.with_name("ADO-v0.7.0-Maneuver-Evidence.zip")
            write_fixture_zip(companion, {movie: camera}, commit)
            evidence = dict(schema=release.COMPANION_EVIDENCE_SCHEMA, commit=commit, archive=companion.name,
                            bytes=companion.stat().st_size, sha256=sha256(companion.read_bytes()),
                            files=[dict(path=movie, bytes=len(camera), sha256=sha256(camera))])
            write_fixture_zip(main, core_members, commit)
            release.add_release_index(main, index, json_bytes(evidence))
            files = {docs, movie, index}
            with release.verified_release_archives(main, files, commit, companion, evidence) as archive:
                self.assertEqual(archive.read(release.PREFIX + docs), core_members[docs])
                self.assertEqual(archive.getinfo(release.PREFIX + movie).file_size, len(camera))
                release.verify_digest(archive, movie, sha256(camera))
            with self.assertRaisesRegex(ValueError, "inventory differs"), \
                    release.verified_release_archives(main, files, commit):
                self.fail("A core missing its companion passed verification")
            write_fixture_zip(main, {**core_members, movie: camera}, commit)
            release.add_release_index(main, index, json_bytes(evidence))
            with self.assertRaisesRegex(ValueError, "overlaps"), \
                    release.verified_release_archives(main, files, commit, companion, evidence):
                self.fail("Archives with duplicate files passed verification")

            write_fixture_zip(companion, {movie: bytes(reversed(camera))}, commit)
            changed = dict(evidence, bytes=companion.stat().st_size, sha256=sha256(companion.read_bytes()))
            write_fixture_zip(main, core_members, commit)
            release.add_release_index(main, index, json_bytes(changed))
            with self.assertRaisesRegex(ValueError, "Wrong SHA-256"), \
                    release.verified_release_archives(main, files, commit, companion, changed):
                self.fail("A companion whose individual camera checksum changed passed verification")

    def test_companion_build_publishes_both_zip_checksums_with_the_master(self):
        commit = "b" * 40
        tag = "v0.7.0"
        sweep = f"{release.PROGRESS_RESULTS}/trials/candidate/sensor-sweeps.zip"
        movie = f"{release.MANEUVERS_TRIALS}/candidate/slow/front.mp4"
        comparison = "web/media/city-maneuvers-tests/comparison.mp4"
        members = {"docs/release.md": b"Setup and playback", "web/media/build.json": b'{"evidence": []}',
                   sweep: b"s" * 400, movie: b"m" * 1600, comparison: b"c" * 2000}

        def archive_release(path, revision, version, excluded=()):
            removed = {entry["path"] for entry in excluded}
            write_fixture_zip(path, {name: data for name, data in members.items() if name not in removed}, revision)

        def git_archive(*args):
            path = next(arg.removeprefix("--output=") for arg in args if arg.startswith("--output="))
            selected = {arg.removeprefix(":(top,literal)") for arg in args if arg.startswith(":(top,literal)")}
            self.assertEqual(selected, {movie})
            self.assertIn(commit, args)
            write_fixture_zip(Path(path), {name: members[name] for name in selected}, commit)

        def verify_packages(path, files, revision, version, scratch, external, companion, evidence):
            with release.verified_release_archives(path, files, revision, companion, evidence) as archive:
                release.verify_digest(archive, movie, sha256(members[movie]))
                release.verify_digest(archive, comparison, sha256(members[comparison]))

        def stage_master(source, package, target, media):
            target.write_bytes(b"native 1080p")
            return sha256(target.read_bytes())

        with TemporaryDirectory() as folder:
            root = Path(folder)
            preflight = root / "preflight.zip"
            write_fixture_zip(preflight, members, commit)
            with patch.multiple(release, MAX_UPLOAD_BYTES=preflight.stat().st_size - 500, SWEEP_INDEX_RESERVE=0,
                                archive_release=archive_release, git=git_archive, validate_zip=verify_packages,
                                stage_release_master=stage_master), \
                    patch.object(release, "historical_research_paths", return_value={sweep}), \
                    patch.object(release, "resolve_commit", return_value=commit), \
                    patch.object(release, "package_inventory", return_value=set(members)):
                final, _, _ = release.build_release(commit, tag, root / "published", Path("supplied.mp4"))
            output = final.parent
            evidence = json.loads((output / "release-metadata.json").read_bytes())["companion_evidence"]
            companion = output / evidence["archive"]
            self.assertEqual(companion.name, "ADO-v0.7.0-Maneuver-Evidence.zip")
            self.assertEqual(evidence["files"], [dict(path=movie, bytes=len(members[movie]), sha256=sha256(members[movie]))])
            with ZipFile(final) as archive:
                self.assertNotIn(release.PREFIX + movie, archive.namelist())
                index = release.read_json(archive, evidence.pop("index"))
                self.assertEqual(index, evidence)
            checksum_lines = (output / "SHA256SUMS.txt").read_text().splitlines()
            assets = (final, companion, output / release.MANEUVERS_MASTER)
            self.assertEqual(checksum_lines, [f"{sha256(item.read_bytes())}  {item.name}" for item in assets])

    def test_maneuver_witness_requires_actual_return_and_waiting_for_slow_traffic(self):
        witness = dict(before_target_s=1., alternate_alongside_s=5., rear_cleared_target_s=6., returned_to_route_s=8.,
                       passed_s=10., station_m=300., rejoin_station_m=322., rejoin_rear_clear_m=4.,
                       minimum_body_clearance_m=.8, alongside_minimum_body_gap_m=1.2, alongside_ego_lane=[67, -1],
                       source_lane=[67, -2], passing_lane=[67, -1])
        stationary = dict(witness, kind="in-lane-obstruction", target_count=3, maximum_target_displacement_m=.02)
        moving = dict(witness, kind="slow-traffic", occupied_gap_wait_s=2., adjacent_traffic_cleared_s=4.,
                      occupied_gap_frames=2, moving_target_s=1., moving_alongside_s=5., alongside_target_mps=2.,
                      target_traveled_m=14., target_maximum_step_m=.4)
        capture = dict(events=[stationary, moving], summary=dict(duration_s=15.))
        release.verify_maneuver_events(capture)
        moving["adjacent_traffic_cleared_s"] = 5.2
        with self.assertRaisesRegex(ValueError, "wait for the occupied passing lane"):
            release.verify_maneuver_events(capture)
        moving["adjacent_traffic_cleared_s"] = 4.
        stationary["returned_to_route_s"] = 5.7
        with self.assertRaisesRegex(ValueError, "native frame sequence"):
            release.verify_maneuver_events(capture)

    def test_selected_stationary_intent_binds_the_recorded_and_packaged_models_to_the_fit(self):
        field = release.STATIONARY_INTENT_FIELD
        weights = b"native stationary vehicle intent weights"
        digest = sha256(weights)
        capture = {"policy": {field: digest}, "summary": {field: digest}}
        fit = dict(schema=release.STATIONARY_INTENT_SCHEMA, checkpoint_sha256=digest)
        members = {release.STATIONARY_INTENT_MODEL: weights, release.STATIONARY_INTENT_CAPTURE: weights,
                   release.STATIONARY_INTENT_FIT: json_bytes(fit)}
        with fixture_zip(members) as archive:
            release.verify_stationary_intent(archive, capture)
            with self.assertRaisesRegex(ValueError, "different stationary vehicle intent checkpoint"):
                release.verify_stationary_intent(archive, dict(capture, summary={field: "0" * 64}))
            with self.assertRaisesRegex(ValueError, "different stationary vehicle intent checkpoint"):
                release.verify_stationary_intent(archive, dict(capture, policy={}))
            with self.assertRaisesRegex(ValueError, "Invalid declared"):
                release.verify_stationary_intent(archive, {"policy": {field: None}, "summary": {field: None}})
        for path in (release.STATIONARY_INTENT_MODEL, release.STATIONARY_INTENT_CAPTURE):
            with self.subTest(path=path), fixture_zip({**members, path: b"different weights"}) as archive:
                with self.assertRaisesRegex(ValueError, "Wrong SHA-256"):
                    release.verify_stationary_intent(archive, capture)
        fit["checkpoint_sha256"] = "0" * 64
        with fixture_zip({**members, release.STATIONARY_INTENT_FIT: json_bytes(fit)}) as archive:
            with self.assertRaisesRegex(ValueError, "fit describes a different checkpoint"):
                release.verify_stationary_intent(archive, capture)

    def test_city_maneuvers_catalog_requires_intent_checkpoint_only_if_selected(self):
        digest = "d" * 64
        key = release.STATIONARY_INTENT_CATALOG_KEY
        build = {"checkpoints": {"city-maneuvers": {"sha256": "base checkpoint"}}}
        legacy = {"policy": {"sha256": "base checkpoint"}, "summary": {}}
        with fixture_zip({}) as archive:
            release.verify_stationary_intent(archive, legacy)
            release.verify_driver_bundle(archive, legacy)
        release.verify_maneuver_catalog_checkpoints(build, legacy)

        selected = {"policy": {**legacy["policy"], release.STATIONARY_INTENT_FIELD: digest}}
        with self.assertRaisesRegex(ValueError, "identify the selected stationary vehicle intent checkpoint"):
            release.verify_maneuver_catalog_checkpoints(build, selected)
        build["checkpoints"][key] = {"sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "identify the selected stationary vehicle intent checkpoint"):
            release.verify_maneuver_catalog_checkpoints(build, selected)
        build["checkpoints"][key]["sha256"] = digest
        release.verify_maneuver_catalog_checkpoints(build, selected)
        bundle_digest = "b" * 64
        selected["policy"][release.DRIVER_BUNDLE_FIELD] = bundle_digest
        with self.assertRaisesRegex(ValueError, "identify the selected City Maneuvers driver bundle"):
            release.verify_maneuver_catalog_checkpoints(build, selected)
        build["checkpoints"][release.DRIVER_BUNDLE_CATALOG_KEY] = {"sha256": bundle_digest}
        release.verify_maneuver_catalog_checkpoints(build, selected)

    def test_selected_driver_bundle_binds_both_copies_and_its_exact_component_roles(self):
        bundle_field = release.DRIVER_BUNDLE_FIELD
        intent_field = release.STATIONARY_INTENT_FIELD
        pace, tactical, intent = b"original pace weights", b"active lane choice", b"stationary intent weights"
        policy = dict(schema=release.MANEUVERS_SCHEMA, path="candidate.npz", sha256=sha256(pace),
                      tactical_sha256=sha256(tactical), maneuver_choice_sha256=sha256(tactical),
                      driver_bundle_path=release.DRIVER_BUNDLE_NAME, **{intent_field: sha256(intent)})
        manifest = dict(format=release.DRIVER_BUNDLE_FORMAT, schema=release.MANEUVERS_SCHEMA,
                        pace=dict(path="candidate.npz", sha256=sha256(pace)),
                        maneuver_choice=dict(path="../route-maneuvers/choice.json", sha256=sha256(tactical)),
                        stationary_intent=dict(path="stationary-intent.npz", sha256=sha256(intent)))
        components = {"models/city-maneuvers/candidate.npz": pace, release.MANEUVERS_CHOICE: tactical,
                      release.STATIONARY_INTENT_MODEL: intent, release.STATIONARY_INTENT_CAPTURE: intent,
                      release.STATIONARY_INTENT_FIT: json_bytes(dict(schema=release.STATIONARY_INTENT_SCHEMA,
                                                                   checkpoint_sha256=sha256(intent)))}

        def seal(bundle):
            payload = json_bytes(bundle)
            digest = sha256(payload)
            return ({**components, release.DRIVER_BUNDLE_MODEL: payload, release.DRIVER_BUNDLE_CAPTURE: payload},
                    {"policy": {**policy, bundle_field: digest}, "summary": {bundle_field: digest, intent_field: sha256(intent)}})

        members, capture = seal(manifest)
        with fixture_zip(members) as archive:
            release.verify_stationary_intent(archive, capture)
            release.verify_driver_bundle(archive, capture)
            with self.assertRaisesRegex(ValueError, "different driver bundle"):
                release.verify_driver_bundle(archive, dict(capture, summary={bundle_field: "0" * 64}))
            with self.assertRaisesRegex(ValueError, "missing its checksum"):
                release.verify_driver_bundle(archive, dict(capture, policy=policy, summary={}))
            changed = dict(capture["policy"], driver_bundle_path="another-driver.json")
            with self.assertRaisesRegex(ValueError, "path or SHA-256"):
                release.verify_driver_bundle(archive, dict(capture, policy=changed))

        for path in (release.DRIVER_BUNDLE_MODEL, release.DRIVER_BUNDLE_CAPTURE, release.MANEUVERS_CHOICE,
                     "models/city-maneuvers/candidate.npz"):
            with self.subTest(path=path), fixture_zip({**members, path: b"changed file"}) as archive:
                with self.assertRaisesRegex(ValueError, "Wrong SHA-256"):
                    release.verify_driver_bundle(archive, capture)
        for field, changes in (("pace", {"sha256": "0" * 64}), ("maneuver_choice", {"path": "../other/choice.json"}),
                               ("stationary_intent", {"sha256": "0" * 64})):
            changed = {**manifest, field: dict(manifest[field], **changes)}
            altered_members, altered_capture = seal(changed)
            with self.subTest(component=field), fixture_zip(altered_members) as archive:
                with self.assertRaisesRegex(ValueError, "components differ from the native capture"):
                    release.verify_driver_bundle(archive, altered_capture)

    def test_maneuver_audit_verifies_measured_pace_above_forty_and_legal_lane_witnesses(self):
        route = json_bytes(dict(route_id="surveyed route", opendrive_sha256="native geometry"))
        events = [dict(kind=kind, alongside_ego_lane=[67, -1]) for kind in sorted(release.MANEUVER_EVENTS)]
        telemetry = [dict(speed_mps=speed, speed_request_mps=speed, posted_limit_kmh=90) for speed in (10, 12, 15, 8)]
        summary = dict(frames=4, outcome="complete", progress_m=300., route_length_m=300., route_id="surveyed route",
                       destination="market", events=events, **dict.fromkeys(release.SAFETY_COUNTS, 0))
        capture = dict(summary=summary, capture_id="native-receipt", quality_requested="High", policy={"sha256": "model"},
                       sha256={"telemetry.json": "telemetry"}, population_reference_sha256="population",
                       first_frame=100, last_frame=103, frame_period_s=.1)
        moves = []
        for frame, event in zip((100, 102), events, strict=True):
            moves.append(dict(road=67, alternate_lane=-1, whole_body_in_alternate={"frame": frame},
                              restored_source={"frame": frame + 1}, shoulder_or_departure_samples=0,
                              first_native_legality_failure=None,
                              moving_vehicle_clearance={"scored_frames_with_moving_counterpart": 2},
                              alongside={"14": dict(first={"frame": frame, "role": event["kind"]},
                                                    nearest={"body_clearance_m": .8})},
                              ahead_with_sampled_clearance={"14": {"frame": frame + 1}}))
        audit = dict(schema="ado-city-maneuver-audit-v1",
                     source=dict(seal_verified=True, quality="High", producer=release.CITY_PRODUCER,
                                 capture_id=capture["capture_id"], route_id=summary["route_id"], policy_sha256="model",
                                 map_sha256="native geometry", first_frame=100, last_frame=103,
                                 route_sha256=sha256(route), telemetry_sha256="telemetry", population_sha256="population"),
                     whole=dict(entry={"frame": 100}, exit={"frame": 103}, frames=4),
                     grade=dict(summary, participant_contacts=[]),
                     independent_clearance={"cars": {"overlap_samples": 0}}, actual_lateral_departures=moves,
                     straddling_moving_separation=dict(minimum_required_straddle_separation_m=1.,
                                                      scored_frames_with_moving_counterpart=2, sampled_pass=True,
                                                      sub_1m_samples=0, overlap_samples=0),
                     posted_90=dict(frames=4, seconds=.4, kmh_p10_p50_p95_max=[0, 0, 0, 54.],
                                    seconds_at_or_above_35=.3, seconds_at_or_above_37=.2, over_40_kmh_seconds=.2,
                                    higher_thresholds={str(k): dict(actual_seconds=s, requested_seconds=s)
                                                       for k, s in ((45, .1), (50, .1), (55, 0.), (60, 0.))}))
        registered = patch.object(release, "MARKET_ROUTE_SHA256", sha256(route))
        with registered, fixture_zip({release.MANEUVERS_AUDIT: json_bytes(audit), release.MANEUVERS_ROUTE: route}) as archive:
            release.verify_maneuver_audit(archive, capture, telemetry)
            telemetry[3]["posted_limit_kmh"] = 26
            with self.assertRaisesRegex(ValueError, "pace scores"):
                release.verify_maneuver_audit(archive, capture, telemetry)
            telemetry[3]["posted_limit_kmh"] = 90
        moves[0]["first_native_legality_failure"] = {"frame": 100}
        with registered, fixture_zip({release.MANEUVERS_AUDIT: json_bytes(audit), release.MANEUVERS_ROUTE: route}) as archive:
            with self.assertRaisesRegex(ValueError, "legal alternate lane"):
                release.verify_maneuver_audit(archive, capture, telemetry)
        audit["straddling_moving_separation"]["sub_1m_samples"] = 1
        with registered, fixture_zip({release.MANEUVERS_AUDIT: json_bytes(audit), release.MANEUVERS_ROUTE: route}) as archive:
            with self.assertRaisesRegex(ValueError, "measured separation"):
                release.verify_maneuver_audit(archive, capture, telemetry)

    def test_market_loop_route_binds_exact_prefix_and_ordered_native_junctions(self):
        ids = [*range(1, 12), 4, 6, 7]
        samples = [dict(junction=None, heading_deg=0., station_m=0.)]
        for junction in ids:
            samples.extend((dict(junction=junction, heading_deg=0., station_m=float(len(samples))),
                            dict(junction=None, heading_deg=0., station_m=float(len(samples) + 1))))
        prefix = samples[:23]
        original = dict(route_id="Market", seed=212, opendrive_sha256="Town03 geometry", samples=prefix,
                        route_length_m=prefix[-1]["station_m"])
        loop = dict(original, destination="market-loop", route_id="Market Loop", samples=samples,
                    route_length_m=samples[-1]["station_m"], shared_market_prefix_count=len(prefix),
                    shared_market_prefix_to_m=original["route_length_m"])
        original_sha, loop_sha = sha256(json_bytes(original)), sha256(json_bytes(loop))
        receipt = dict(
            proposed_survey=dict(file="route-market-loop-212.json", sha256=loop_sha, route_id=loop["route_id"],
                                 route_length_m=loop["route_length_m"], samples=len(samples), junction_traversals=14,
                                 unique_junction_ids=sorted(set(ids))),
            identical_original_prefix=dict(file="route-212.json", sha256=original_sha, route_id=original["route_id"],
                                           samples=len(prefix), cutoff_m=original["route_length_m"], junction_traversals=11,
                                           every_sample_and_lane_marking_identical=True))
        source = dict(route_id=loop["route_id"], route_sha256=loop_sha, map_sha256=loop["opendrive_sha256"])
        traversals = [dict(junction_id=j, direction="straight", entry_m=2 * i + 1, exit_m=2 * i + 2,
                           entered_s=2 * i + 1, completed_s=2 * i + 2) for i, j in enumerate(ids)]
        summary = dict(seed=212, route_length_m=loop["route_length_m"], duration_s=29., junction_traversals=traversals)
        files = {release.MANEUVERS_ROUTE: json_bytes(original), release.MARKET_LOOP_ROUTE: json_bytes(loop),
                 release.MARKET_LOOP_SURVEY: json_bytes(receipt)}
        registration = dict(MARKET_ROUTE_SHA256=original_sha, MARKET_LOOP_ROUTE_SHA256=loop_sha,
                            MARKET_PREFIX_SAMPLES=len(prefix), MARKET_LOOP_SAMPLES=len(samples))
        with patch.multiple(release, **registration), fixture_zip(files) as archive:
            self.assertEqual(release.verify_market_loop_route(archive, source, summary), (loop, original))
            traversals[-1]["junction_id"] = 8
            with self.assertRaisesRegex(ValueError, "junction sequence"):
                release.verify_market_loop_route(archive, source, summary)
            traversals[-1]["junction_id"] = 7
        changed = dict(loop, samples=[dict(samples[0], heading_deg=7.), *samples[1:]])
        changed_sha = sha256(json_bytes(changed))
        files[release.MARKET_LOOP_ROUTE] = json_bytes(changed)
        registration["MARKET_LOOP_ROUTE_SHA256"] = source["route_sha256"] = changed_sha
        with patch.multiple(release, **registration), fixture_zip(files) as archive:
            with self.assertRaisesRegex(ValueError, "exact surveyed Market route prefix"):
                release.verify_market_loop_route(archive, source, summary)

    def test_market_loop_timeline_keeps_both_native_sign_islands_and_sensor_regions(self):
        telemetry = [dict(frame=40 + i, sim_time=round(i * .1, 1), progress_m=station, posted_limit_kmh=posted,
                          speed_mps=speed, speed_request_mps=speed,
                          tactical_pass={"maneuver_phase": phase})
                     for i, (station, posted, speed, phase) in enumerate((
                         (399., 30, 4., "yield"), (400., 90, 5., "yield"), (401., 90, 7., "yield"),
                         (1450., 30, 4., None), (1584.890, 90, 15., "left"), (1650., 90, 11., "passing"),
                         (1719.890, 30, 4., "rejoined")))]

        def stats(rows):
            def stamp(row):
                return dict(frame=row["frame"], time_s=row["sim_time"], station_m=row["progress_m"],
                            actual_kmh=round(row["speed_mps"] * 3.6, 3), target_kmh=round(row["speed_request_mps"] * 3.6, 3))

            if not rows:
                return {"frames": 0}
            speeds = [row["speed_mps"] * 3.6 for row in rows]
            tiers = {str(level): dict.fromkeys(("actual_seconds", "requested_seconds"), round(
                .1 * sum(speed >= level for speed in speeds), 3)) for level in (45, 50, 55, 60)}
            return dict(frames=len(rows), seconds=round(.1 * len(rows), 3), entry=stamp(rows[0]), exit=stamp(rows[-1]),
                        kmh_p10_p50_p95_max=[round(max(speeds), 3)], higher_thresholds=tiers,
                        over_40_kmh_seconds=round(.1 * sum(speed > 40 for speed in speeds), 3))

        islands = [dict(stats(telemetry[a:b]), posted_kmh=telemetry[a]["posted_limit_kmh"],
                        shared_prefix=stats(telemetry[a:b] if b <= 3 else []),
                        extension=stats(telemetry[a:b] if a >= 3 else []))
                   for a, b in ((0, 1), (1, 3), (3, 4), (4, 6), (6, 7))]
        original = dict(route_id="Market", route_length_m=1416.437782)
        audit = dict(signed_limit_islands=islands,
                     route_comparison=dict(original_route_id="Market", original_route_sha256=release.MARKET_ROUTE_SHA256,
                                           shared_route_to_m=original["route_length_m"],
                                           shared_prefix=dict(whole=stats(telemetry[:3]), posted_90=stats(telemetry[1:3])),
                                           extension=dict(whole=stats(telemetry[3:]), posted_90=stats(telemetry[4:6]))))
        release.verify_market_loop_timeline(audit, telemetry, original, .1)
        islands[3]["higher_thresholds"]["45"]["requested_seconds"] = .5
        with self.assertRaisesRegex(ValueError, "requested speed durations"):
            release.verify_market_loop_timeline(audit, telemetry, original, .1)

        route = dict(samples=[dict(road=67, lane=3, station_m=station) for station in (1584.890, 1719.890)],
                     route_length_m=1719.890)
        frames = {40, 41, 44, 45, 46}
        capture = dict(maneuver_sweep_windows_m=[[399., 548.], [1584.890, 1719.890 + .001]],
                       maneuver_sweeps={f"frame-{frame}.npz": "pinned native sweep" for frame in frames})
        self.assertEqual(release.verify_market_loop_sweep_coverage(capture, telemetry, route), frames)
        del capture["maneuver_sweeps"]["frame-44.npz"]
        with self.assertRaisesRegex(ValueError, "occupied station and maneuver bin"):
            release.verify_market_loop_sweep_coverage(capture, telemetry, route)

    def test_market_loop_lane_move_checks_every_physical_frame_and_its_signed_lane_pair(self):
        exposures = [dict(frame=i, body_in_source=i == 14, center_in_alternate=i in (12, 13), body_in_alternate=i == 13,
                          body_within_legal_union=True) for i in range(11, 15)]
        move = dict(road=67, source_lane=3, alternate_lane=2, side="left", footprint_departs={"frame": 11},
                    center_crosses={"frame": 12}, whole_body_in_alternate={"frame": 13}, last_alternate_sample={"frame": 13},
                    restored_source={"frame": 14}, lane_exposures=exposures)
        event = dict(kind="slow-traffic", source_lane=[67, 3], passing_lane=[67, 2])
        choice = dict(committed=True, source_lane=[67, 3], target_lane=[67, 2])
        rows = {frame: dict(tactical_pass=dict(choice), passing_footprint_in_legal_lanes=True) for frame in range(11, 15)}
        release.verify_market_loop_lane_move(move, event, rows, {12})
        with self.assertRaisesRegex(ValueError, "contiguous legal lane and native sensor"):
            release.verify_market_loop_lane_move(dict(move, lane_exposures=exposures[1:]), event, rows, {12})
        with self.assertRaisesRegex(ValueError, "contiguous legal lane and native sensor"):
            release.verify_market_loop_lane_move(move, event, rows, {20})
        rows[12]["tactical_pass"]["source_lane"] = [67, -2]
        with self.assertRaisesRegex(ValueError, "different lane pair or footprint"):
            release.verify_market_loop_lane_move(move, event, rows, {12})

    def test_maneuver_sequence_matches_the_witnessed_wait_to_actual_controls(self):
        capture = dict(events=[dict(kind="in-lane-obstruction", returned_to_route_s=1.),
                               dict(kind="slow-traffic", occupied_gap_wait_s=2.)],
                       maneuver_sequence=dict(passed=True, obstruction_rejoin_precedes_occupied_wait=True,
                                              actual_wait_policy_frames=1, first_actual_wait_frame=51,
                                              last_actual_wait_frame=51))
        yield_control = dict(maneuver_phase="yield", committed=False)
        frames = [dict(frame=50 + i, events=[dict(kind="slow-traffic", occupied_gap_frames=count)],
                       tactical_pass=yield_control) for i, count in enumerate((0, 1, 1))]
        release.verify_maneuver_sequence(capture, frames)
        frames[1]["tactical_pass"] = dict(maneuver_phase="yield", committed=True)
        with self.assertRaisesRegex(ValueError, "controls do not show a wait"):
            release.verify_maneuver_sequence(capture, frames)

    def test_maneuvers_fit_binds_its_raw_sources_and_retains_inherited_source_checks(self):
        base = "models/city-progress/candidate.npz"
        parent_fit = "models/city-progress/fit.json"
        raw_base = b"City Progress checkpoint"
        fit = dict(base_sha256=sha256(raw_base), inputs={
            base: sha256(raw_base), "crosswalk/city_maneuvers.py": "model", "crosswalk/city_maneuvers_fit.py": "trainer"})
        sources = {"city_maneuvers.py": "model", "city_maneuvers_fit.py": "trainer", "city_progress.py": "current CRLF"}
        parent = dict(revision=4, checkpoint_sha256=sha256(raw_base), source_sha256="historical LF",
                      crossing_source_sha256="crossing", grade_source_sha256="grade")
        with fixture_zip({base: raw_base, parent_fit: json_bytes(parent)}) as archive:
            names = release.maneuver_fit_sources(archive, fit, {"code_sha256": sources})
            self.assertEqual(names, set(sources) | {"city_crossing.py", "city_grade.py"})
            sources["city_maneuvers.py"] = "unsupported source"
            with self.assertRaisesRegex(ValueError, "different execution source"):
                release.maneuver_fit_sources(archive, fit, {"code_sha256": sources})

    @patch.object(release, "verify_video")
    @patch.object(release, "MANEUVERS_TRIAL_CASES", {"board", "slow", "blocked", "inside-curb", "outside-curb"})
    def test_maneuver_trials_require_registered_amendments_and_actual_curb_placement(self, verify_video):
        base_model, candidate_model = b"baseline weights", b"candidate weights"
        artifacts = {"trace.jsonl.gz": gzip.compress(b'{"frame": 21}\n'),
                     "reference.jsonl.gz": gzip.compress(b'{"frame": 21}\n'),
                     "observations.npz": b"inference features"}
        source_path = "crosswalk/maneuver_trials.py"
        pins = {release.TRIAL_EVALUATOR_PIN: {source_path: b"original evaluator\n"},
                release.TRIAL_AMENDMENT_PIN: {source_path: b"registered amended evaluator\n"},
                release.CURB_TRIAL_PIN: {source_path: b"first-exposure curb evaluator\n"}}
        tactical = sha256(b"tactics")
        members = {"models/city-maneuvers/fit.json": json_bytes({"base_sha256": sha256(base_model)}),
                   f"{release.PROGRESS_MEDIA}/manifest.json": json_bytes({"policy": {"tactical_sha256": tactical}})}
        windows = {case: [67, -2, 48., 145.] for case in ("board", "inside-curb", "outside-curb")}
        windows.update(dict.fromkeys(release.AMENDED_TRIAL_CASES, [67, -2, 94., 231.]))
        for arm, schema, model in (("baseline", release.PROGRESS_SCHEMA, base_model),
                                   ("candidate", release.MANEUVERS_SCHEMA, candidate_model)):
            folder = f"{release.MANEUVERS_TRIALS}/{arm}"
            second = 2 if arm == "baseline" else 4
            original = fixture_trial_sources(pins[release.TRIAL_EVALUATOR_PIN], second)
            common = dict(seed=212, policy_schema=schema, model_sha256=sha256(model))
            protocol = dict(common, cases=list(windows), windows=windows, step_s=.1, launch_route_stations_m={},
                            requested_quality="High", map="Town03_Opt", weather="ClearNoon",
                            tactical_model_sha256=tactical, execution_source_sha256=sha256(original))
            source_hashes = dict.fromkeys(windows, sha256(original))
            for name, pin, cases, _ in release.TRIAL_AMENDMENTS:
                amended = fixture_trial_sources(pins[pin], second)
                amendment = dict(protocol, cases=list(cases), windows={case: windows[case] for case in cases},
                                 registration_pin=pin, base_evaluator_pin=release.TRIAL_EVALUATOR_PIN,
                                 production_pin=release.TRIAL_PRODUCTION_PIN, execution_source_sha256=sha256(amended))
                del amendment["launch_route_stations_m"]
                members.update({f"{folder}/{name}.zip": amended, f"{folder}/{name}.json": json_bytes(amendment)})
                source_hashes.update(dict.fromkeys(cases, sha256(amended)))
            qualification = dict(arm=arm, model_sha256=sha256(model), pin=release.TRIAL_PRODUCTION_PIN,
                                 evaluator_pin=release.TRIAL_EVALUATOR_PIN, native_protocol_sha256=sha256(json_bytes(protocol)))
            members.update({f"{folder}/checkpoint.npz": model, f"{folder}/study-source.zip": original,
                            f"{folder}/tactical-model.json": b"tactics", f"{folder}/protocol.json": json_bytes(protocol),
                            f"{folder}/qualification.json": json_bytes(qualification),
                            f"{folder}/capture-status.json": json_bytes(dict.fromkeys(windows, True))})
            for case, (road, lane, start, end) in windows.items():
                trial = f"{folder}/{case}"
                trial_artifacts = dict(artifacts)
                summary = dict(common, first_frame=21, last_frame=21, frames=1, case=case, capture_complete=True,
                               outcome="max_steps" if arm == "baseline" else "complete", producer=release.CITY_PRODUCER,
                               geometry=dict(road_id=road, lane_id=lane, start_s=start, end_s=end, route_start_m=200. + start),
                               front_sha256=sha256(b"camera"), execution_source_sha256=source_hashes[case])
                if case in release.CURB_TRIAL_CASES:
                    extent = [2., 2.85] if case == "inside-curb" else [3.15, 4.]
                    blueprint = "static.prop.streetbarrier"
                    center = [10., 20.]
                    footprint = [[10., 20.], [11., 20.]]
                    summary["geometry"].update(board_native_s=84., driving_edge_m=1.75, sidewalk_edge_m=3.,
                                               board_lateral_extent_m=extent, board_blueprint=blueprint,
                                               requested_center_xy=center)
                    placement = dict(frame=21, actor_id=96 + second, blueprint=blueprint, actual_center_xy=center,
                                     center_error_m=0., physical_lateral_extent_m=extent, polygon_xy=footprint, verified=True)
                    summary["curb_placement"] = placement
                    target = dict(actor_id=placement["actor_id"], role=case, polygon_xy=footprint)
                    reference = dict(frame=21, curb_placement=placement, target_polygons=[target])
                    trial_artifacts["reference.jsonl.gz"] = gzip.compress(json_bytes(reference))
                summary["artifact_sha256"] = {name: sha256(data) for name, data in trial_artifacts.items()}
                members.update({f"{trial}/summary.json": json_bytes(summary), f"{trial}/front.mp4": b"camera"})
                members.update({f"{trial}/{name}": data for name, data in trial_artifacts.items()})
        capture = dict(policy={"sha256": sha256(candidate_model), "tactical_sha256": tactical},
                       quality_requested="High", frame_period_s=.1)
        with patch.object(release, "pinned_trial_sources", side_effect=pins.__getitem__):
            with fixture_zip(members) as archive:
                release.verify_maneuver_trials(archive, capture, Path("unused"))
            self.assertEqual(verify_video.call_count, 10)

            amendment_path = f"{folder}/{release.TRIAL_AMENDMENT}.json"
            curb_amendment_path = f"{folder}/{release.CURB_TRIAL_AMENDMENT}.json"
            board_path = f"{folder}/board/summary.json"
            slow_path = f"{folder}/slow/summary.json"
            curb_path = f"{folder}/inside-curb/summary.json"
            curb_ref = f"{folder}/inside-curb/reference.jsonl.gz"
            amendment = json.loads(members[amendment_path])
            curb_amendment = json.loads(members[curb_amendment_path])
            amended = members[f"{folder}/{release.TRIAL_AMENDMENT}.zip"]
            board = json.loads(members[board_path])
            slow = json.loads(members[slow_path])
            curb = json.loads(members[curb_path])
            first = json.loads(gzip.decompress(members[curb_ref]))
            wrong_window = dict(amendment, windows={**amendment["windows"], "slow": [67, -2, 94., 214.]})
            wrong_geometry = dict(slow["geometry"], route_start_m=999.)
            failures = (
                (amendment_path, dict(amendment, cases=["slow", "board"]), "Wrong registered"),
                (amendment_path, wrong_window, "Wrong registered"),
                (amendment_path, dict(amendment, tactical_model_sha256="different choice"), "Wrong registered"),
                (amendment_path, dict(amendment, registration_pin="unregistered"), "Wrong registered"),
                (curb_amendment_path, dict(curb_amendment, cases=["inside-curb", "board"]), "Wrong registered"),
                (board_path, dict(board, execution_source_sha256=sha256(amended)), "Wrong native trial identity"),
                (slow_path, dict(slow, execution_source_sha256=sha256(original)), "Wrong native trial identity"),
                (curb_path, dict(curb, execution_source_sha256=sha256(amended)), "Wrong native trial identity"),
                (slow_path, dict(slow, geometry=wrong_geometry), "different native fixture geometry"),
            )
            for path, payload, error in failures:
                with self.subTest(path=path, error=error), fixture_zip({**members, path: json_bytes(payload)}) as archive:
                    with self.assertRaisesRegex(ValueError, error):
                        release.verify_maneuver_trials(archive, capture, Path("unused"))

            source_blob = f"{folder}/{release.TRIAL_AMENDMENT}.zip"
            edited = fixture_trial_sources({source_path: b"unregistered evaluator\n"}, 4)
            replacement = {source_blob: edited, amendment_path: json_bytes(dict(amendment, execution_source_sha256=sha256(edited)))}
            with fixture_zip({**members, **replacement}) as archive, self.assertRaisesRegex(ValueError, "Wrong pinned source"):
                release.verify_maneuver_trials(archive, capture, Path("unused"))

            shifted = dict(first["curb_placement"], actual_center_xy=[10.01, 20.], center_error_m=.01)
            wrong_side = dict(first["curb_placement"], physical_lateral_extent_m=[3.15, 4.])
            for reference, message in (
                (dict(first, target_polygons=[]), "Missing the physical curb board"),
                (dict(first, curb_placement=wrong_side), "differs from the registered geometry"),
                (dict(first, curb_placement=shifted), "different native fixture geometry"),
            ):
                data = gzip.compress(json_bytes(reference))
                summary = dict(curb, curb_placement=reference["curb_placement"],
                               artifact_sha256={**curb["artifact_sha256"], "reference.jsonl.gz": sha256(data)})
                with self.subTest(curb=message), fixture_zip({**members, curb_path: json_bytes(summary), curb_ref: data}) as archive:
                    with self.assertRaisesRegex(ValueError, message):
                        release.verify_maneuver_trials(archive, capture, Path("unused"))

    @patch.object(release, "verify_video")
    def test_pace_film_verifies_eleven_events_and_an_exact_replay_pack(self, verify_video):
        media, manifest, members = city_film(True, pace=True)
        with fixture_zip(members) as archive:
            release.verify_city_film(archive, media, manifest, Path("unused"))
        self.assertEqual(verify_video.call_count, 6)

        manifest["required_event_kinds"].remove("barrier")
        with fixture_zip(members) as archive, self.assertRaisesRegex(ValueError, "Inconsistent encounter"):
            release.verify_city_film(archive, media, manifest, Path("unused"))
        manifest["required_event_kinds"].append("barrier")
        pack = gzip.compress(json_bytes(dict(format="ado-replay-pack-v1", replay=[],
                                             shared_path=None, path_ranges=[])))
        members[f"{media}/replay-pack.json.gz"] = pack
        manifest["sha256"]["replay-pack.json.gz"] = sha256(pack)
        with fixture_zip(members) as archive, self.assertRaisesRegex(ValueError, "Replay pack differs"):
            release.verify_city_film(archive, media, manifest, Path("unused"))

    @patch.object(release, "verify_city_life_evidence")
    @patch.object(release, "verify_video")
    def test_city_life_requires_exact_gzip_replay_and_scored_safety(self, verify_video, _verify_evidence):
        media, manifest, members = populated_city_film(release.LIFE_SCHEMA)
        with fixture_zip(members) as archive:
            release.verify_city_film(archive, media, manifest, Path("unused"))
        self.assertEqual(verify_video.call_count, 6)

        manifest["summary"]["collision_events"] = 1
        with fixture_zip(members) as archive, self.assertRaisesRegex(ValueError, "totals differ from the scored telemetry"):
            release.verify_city_film(archive, media, manifest, Path("unused"))
        manifest["raw_sha256"]["replay.json"] = "0" * 64
        with fixture_zip(members) as archive, self.assertRaisesRegex(ValueError, "differs from the native recording"):
            release.verify_city_film(archive, media, manifest, Path("unused"))

    @patch.object(release, "verify_city_life_evidence")
    @patch.object(release, "verify_video")
    def test_city_progress_checks_the_base_checkpoint_and_crossing_observer(self, verify_video, _verify_evidence):
        media, manifest, members = populated_city_film(release.PROGRESS_SCHEMA)
        with fixture_zip(members) as archive:
            release.verify_city_film(archive, media, manifest, Path("unused"))
        self.assertEqual(verify_video.call_count, 6)
        hashes = manifest["policy"]["code_sha256"]
        digest = hashes[release.PROGRESS_CROSSING_SOURCE]
        hashes[release.PROGRESS_CROSSING_SOURCE] = "0" * 64
        with fixture_zip(members) as archive, self.assertRaisesRegex(ValueError, "different fitted module.*city_crossing.py"):
            release.verify_city_film(archive, media, manifest, Path("unused"))
        hashes[release.PROGRESS_CROSSING_SOURCE] = digest
        path = "models/city-clearance/candidate.npz"
        members[path] += b"unexpected edits"
        with fixture_zip(members) as archive, self.assertRaisesRegex(ValueError, f"Wrong SHA-256.*{path}"):
            release.verify_city_film(archive, media, manifest, Path("unused"))

    @patch.object(release, "verify_city_life_evidence")
    @patch.object(release, "verify_video")
    def test_city_progress_grade_revision_requires_the_fitted_grade_module(self, _verify_video, verify_evidence):
        media, capture, members = populated_city_film(release.PROGRESS_SCHEMA, revision=4)
        with fixture_zip(members) as archive:
            release.verify_city_film(archive, media, capture, Path("unused"))
        self.assertIn(release.PROGRESS_GRADE_SOURCE, verify_evidence.call_args.args[-1])
        path = "models/city-progress/fit.json"
        fit = json.loads(members[path])
        del fit["grade_source_sha256"]
        members[path] = json_bytes(fit)
        with fixture_zip(members) as archive, self.assertRaisesRegex(ValueError, "grade fit differs from the checkpoint revision"):
            release.verify_city_film(archive, media, capture, Path("unused"))

    def test_city_progress_preserves_captured_source_and_compares_git_line_endings(self):
        policy_source = "city_progress.py"
        fitted_sources = (policy_source, release.PROGRESS_CROSSING_SOURCE, release.PROGRESS_GRADE_SOURCE)
        names = (*fitted_sources, "city_clearance.py", "city_population.py", "journey_capture.py", "population_score.py")
        recorded = b"model = 'captured source'\r\n"
        capture = dict(policy=dict(schema=release.PROGRESS_SCHEMA, code_sha256=dict.fromkeys(names, sha256(recorded))))
        stream = BytesIO()
        with ZipFile(stream, "w") as snapshot:
            for name in names:
                snapshot.writestr(f"crosswalk/{name}", recorded)
        contents = {f"{release.PROGRESS_RESULTS}/capture-source.zip": stream.getvalue(),
                    **{f"crosswalk/{name}": recorded.replace(b"\r\n", b"\n") for name in fitted_sources}}
        with fixture_zip(contents) as archive:
            release.verify_population_sources(archive, capture, release.PROGRESS_RESULTS, policy_source, fitted_sources)
        for name in fitted_sources:
            changed = {**contents, f"crosswalk/{name}": b"model = 'different source'\n"}
            with self.subTest(name=name), fixture_zip(changed) as archive, \
                    self.assertRaisesRegex(ValueError, "different fit sources"):
                release.verify_population_sources(archive, capture, release.PROGRESS_RESULTS, policy_source, fitted_sources)

    def test_featured_city_progress_requires_a_pass_and_exact_reported_counters(self):
        _, capture, _ = populated_city_film(release.PROGRESS_SCHEMA)
        capture["video"] = capture["camera_videos"]["chase"]
        capture["video"].update(width=1920, height=1080)
        capture["outcome"] = capture["summary"]["outcome"] = "complete"
        entry = dict(replay_url="./media/city-progress/replay.json.gz")
        showcase = dict(capture["summary"], **entry, model_sha256=capture["policy"]["sha256"])
        with self.assertRaisesRegex(ValueError, "must pass the native route grade"):
            release.verify_populated_catalog(capture, showcase, entry, release.PROGRESS_MEDIA)
        capture["passed"] = True
        release.verify_populated_catalog(capture, showcase, entry, release.PROGRESS_MEDIA)
        showcase["yield_violations"] = 1
        with self.assertRaisesRegex(ValueError, "catalog differs from the measured capture"):
            release.verify_populated_catalog(capture, showcase, entry, release.PROGRESS_MEDIA)

    def test_population_counts_simultaneous_motion_only_after_six_consecutive_exposures(self):
        rows = [dict(frame=i, sim_time=i/10, ego=[0, 0],
                     others=[dict(actor_id=actor, type="car" if actor < 2 else "pedestrian", x=actor + .2*i, y=0)
                             for actor in range(4)]) for i in range(6)]
        self.assertEqual(release.population_density(rows)["35"]["two_of_each_frames"], 1)
        rows[2]["others"].pop()
        density = release.population_density(rows)["35"]
        self.assertEqual(density["scored_frames"], 1)
        self.assertEqual(density["both_frames"], 1)
        self.assertEqual(density["two_of_each_frames"], 0)

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
