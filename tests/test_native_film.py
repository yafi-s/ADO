"""Offline tests for native packet identity and playback frame registration."""
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from crosswalk import native_film as film


def sample(number, x):
    frame = film.NativeFrame(number, .1, (number-1)*.1)
    frame.poses[42] = dict(x=x, y=0., z=0., yaw=0., pitch=0., roll=0.)
    return frame


def exposure(x, camera_offset=0., rotation_offset=0.):
    return dict(ego_pose=sample(1, x).poses[42], camera_pose={
        **film.MOUNT, "x":x+film.MOUNT["x"]+camera_offset, "yaw":film.MOUNT["yaw"]+rotation_offset})


class AlignmentTests(unittest.TestCase):
    def test_recover_install_with_original_data_preserved(self):
        with TemporaryDirectory() as tmp:
            stage, output = Path(tmp)/"staged", Path(tmp)/"published"
            stage.mkdir()
            output.mkdir()
            for name in film._artifacts(True):
                (stage/name).write_bytes(name.encode())
            manifest = dict(trial_id=film.TRIAL, cameras=["chase", "front"],
                            sha256={name:film.fingerprint(stage/name) for name in film._artifacts(True)})
            film.write_json(stage/"manifest.json", manifest)
            film.write_json(stage/"install.json", dict(output=str(output), manifest_sha256=film.fingerprint(stage/"manifest.json")))
            (output/"chase.mp4").write_bytes(b"chase.mp4")
            with patch("crosswalk.showcase._commit_file", side_effect=OSError("locked file")):
                with self.assertRaisesRegex(RuntimeError, "crosswalk.native_film"):
                    film.install_staged(stage)
            self.assertFalse((output/"manifest.json").exists())
            self.assertTrue((stage/"chase.mp4").exists())
            self.assertEqual(film.install_staged(stage), manifest)
            self.assertEqual(film.install_staged(stage), manifest)
            self.assertEqual(film.fingerprint(output/"manifest.json"), film.fingerprint(stage/"manifest.json"))

    def test_prefers_unique_measured_camera_and_ego_window(self):
        target = [sample(i+1, x) for i,x in enumerate((0., 0., 1., 2., 4.))]
        images = [exposure(x) for x in (-3., -1., 0., 0., 1., 2., 4., 6.)]
        start, report = film.align(images, target, 42)
        self.assertEqual(start, 2)
        self.assertEqual(report["status"], "pose_verified")
        self.assertLess(report["max_observer_error_m"], .0001)

    def test_moved_camera_or_wrong_orientation_precludes_original_frame_labels(self):
        target = [sample(i+1, i) for i in range(3)]
        for displacement, yaw in ((.1, 0.), (0., 1.)):
            start, report = film.align([exposure(i, displacement, yaw) for i in range(3)], target, 42)
            self.assertIsNone(start)
            self.assertEqual(report["status"], "unverified")

    def test_stationary_duplicates_cannot_establish_time_registration(self):
        target = [sample(1, 0.), sample(2, 0.)]
        start, report = film.align([exposure(0.)]*4, target, 42)
        self.assertIsNone(start)
        self.assertEqual(report["passing_windows"], 3)

    def test_display_uses_real_ego_relative_height_and_signed_doppler(self):
        points, returns = film.display_returns(np.array([[6, 1, -3, .5], [9, 2, -.3, .7]], dtype=float),
                                              np.array([[-2., 0., math.pi/6, 5.]], dtype=float))
        self.assertEqual(points, [[9., 2., 2.]])
        self.assertEqual(returns, [[6.33, 0., -2., 3.4]])


class OriginalRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not film.SOURCE.exists():
            raise unittest.SkipTest("Local data from the locked trial is unavailable")
        try:
            cls.checkpoint = film.default_checkpoint()
        except ValueError as error:
            raise unittest.SkipTest(str(error)) from error
        cls.source = film.load_source(film.SOURCE, cls.checkpoint)

    def test_native_clock_speed_and_applied_controls_match_locked_market_case(self):
        meta, summary, trace, frames, actors, hero, box = self.source
        self.assertEqual((summary["frames"], len(frames), len(trace)), (358, 358, 358))
        self.assertEqual((frames[0].number, frames[-1].number), (9, 366))
        self.assertEqual((trace[0]["frame"], trace[-1]["frame"]), (59369, 59726))
        self.assertEqual(meta["observed_stop_frame"], 59727)
        self.assertEqual(actors[hero]["blueprint"], summary["ego_blueprint"])
        self.assertEqual(len(actors), 7)
        self.assertGreater(box[3], 2.)
        self.assertLess(box[4], 1.)
        self.assertEqual(summary["progress_m"], 184.424)

    def test_rejects_one_frame_displacement_of_command(self):
        meta = self.source[0]
        frames, actors, hero, box = film.decode(film.SOURCE, meta)
        frames[9].control = frames[8].control
        with patch.object(film, "decode", return_value=(frames, actors, hero, box)):
            with self.assertRaisesRegex(ValueError, "59369"):
                film.load_source(film.SOURCE, self.checkpoint)

    def test_rejects_terminal_sentinel_early(self):
        meta = self.source[0]
        frames, actors, hero, box = film.decode(film.SOURCE, meta)
        frames[-2].duration = -1.
        with patch.object(film, "decode", return_value=(frames, actors, hero, box)):
            with self.assertRaisesRegex(ValueError, "clock mismatch"):
                film.load_source(film.SOURCE, self.checkpoint)
