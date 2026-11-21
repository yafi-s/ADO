from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from tools import prepare_pages_film as pages


class PagesFilmTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.web = Path(directory.name) / "web"
        self.compact = b"committed x264 data"
        self.master = b"released x264 data" * 240

    def fixture(self, tag):
        directory, asset = pages.RELEASE_FILMS[tag]
        self.folder = self.web / "media" / directory
        self.folder.mkdir(parents=True)
        self.chase = self.folder / "chase.mp4"
        self.manifest = self.folder / "manifest.json"
        self.build = self.web / "media/build.json"
        self.chase.write_bytes(self.compact)
        self.compact_sha = pages.sha256(self.chase)
        self.chase.write_bytes(self.master)
        self.master_sha = pages.sha256(self.chase)
        self.chase.write_bytes(self.compact)
        self.master_info = dict(width=1920, height=1080, codec_name="h264", pix_fmt="yuv420p",
                                nb_read_frames="420", r_frame_rate="10/1")
        video = dict(self.master_info, width=1280, height=720)
        pages.write_json(self.manifest, {
            "trial_id": directory,
            "video": video,
            "camera_videos": {"chase": video, "front": {}},
            "sha256": {"chase.mp4": self.compact_sha},
            "master": {"chase": dict(self.master_info, file_bytes=len(self.master), sha256=self.master_sha)},
        })
        pages.write_json(self.build, {"release": {"label": tag}, "showcases": [{"trial_id": directory, **video}]})
        self.url = f"{pages.GITHUB_REPO}/releases/download/{tag}/{asset}"

    def test_release_master_replaces_only_the_featured_film(self):
        for tag in pages.RELEASE_FILMS:
            with self.subTest(tag=tag):
                self.fixture(tag)
                with patch.object(pages, "urlopen", return_value=BytesIO(self.master)) as request:
                    pages.prepare(self.web)
                request.assert_called_once_with(self.url, timeout=90)
                manifest = json.loads(self.manifest.read_bytes())
                self.assertEqual(self.chase.read_bytes(), self.master)
                self.assertEqual(manifest["camera_videos"]["chase"], self.master_info)
                self.assertEqual(manifest["sha256"]["chase.mp4"], self.master_sha)
                self.assertEqual(manifest["archive_chase"]["sha256"], self.compact_sha)
                self.assertEqual(json.loads(self.build.read_bytes())["showcases"][0]["height"], 1080)

    def test_missing_or_unverified_release_asset_keeps_committed_film(self):
        self.fixture("v0.3.0")
        original_manifest = self.manifest.read_bytes()
        missing = HTTPError(self.url, 404, "missing release asset", {}, None)
        with patch.object(pages, "urlopen", side_effect=missing):
            pages.prepare(self.web)
        self.assertEqual(self.chase.read_bytes(), self.compact)
        with patch.object(pages, "urlopen", return_value=BytesIO(b"x" * len(self.master))), \
                self.assertRaisesRegex(ValueError, "differs from the original CARLA capture"):
            pages.prepare(self.web)
        self.assertEqual(self.chase.read_bytes(), self.compact)
        self.assertEqual(self.manifest.read_bytes(), original_manifest)

    def test_oversized_release_master_uses_a_verified_1080p_encode(self):
        self.fixture("v0.3.0")
        encoded = b"verified result from the master" * 20

        def encode(source, target, master, budget):
            self.assertEqual(source.read_bytes(), self.master)
            self.assertEqual(master["sha256"], self.master_sha)
            self.assertGreaterEqual(budget, len(encoded))
            self.assertLess(budget, len(self.master))
            target.write_bytes(encoded)
            return self.master_info

        limit = pages.site_size(self.web) + pages.SITE_RESERVE + len(encoded)
        with patch.object(pages, "SITE_LIMIT", limit), \
                patch.object(pages, "encode_for_site", side_effect=encode), \
                patch.object(pages, "urlopen", return_value=BytesIO(self.master)):
            pages.prepare(self.web)
        self.assertEqual(self.chase.read_bytes(), encoded)
        manifest = json.loads(self.manifest.read_bytes())
        self.assertEqual(manifest["playback_chase"]["kind"], "pages-encode")
        self.assertEqual(manifest["playback_chase"]["source_sha256"], self.master_sha)
        self.assertEqual(manifest["sha256"]["chase.mp4"], pages.sha256(self.chase))
        with patch.object(pages, "urlopen") as request:
            pages.prepare(self.web)
        request.assert_not_called()
        self.assertEqual(self.chase.read_bytes(), encoded)

    def test_pages_budget_preserves_the_pack_and_links_exact_archived_data(self):
        self.fixture("v0.4.0")
        base = "./media/urban-pace"
        catalog = self.web / "media/viewer-catalog.json"
        replay = self.folder / "replay.json"
        telemetry = self.folder / "telemetry.json"
        pack = self.folder / "replay-pack.json.gz"
        replay.write_bytes(b"original replay" * 500)
        telemetry.write_bytes(b"original telemetry" * 500)
        pack.write_bytes(b"compressed replay")
        manifest = json.loads(self.manifest.read_bytes())
        manifest["sha256"].update({path.name: pages.sha256(path) for path in (replay, telemetry, pack)})
        pages.write_json(self.manifest, manifest)
        pages.write_json(catalog, [{"path": "urban-pace", "replay_pack_url": f"{base}/{pack.name}",
                                    "replay_pack_sha256": pages.sha256(pack)}])
        build = json.loads(self.build.read_bytes())
        build["showcases"][0].update(recording_path="urban-pace", log_url=f"{base}/{telemetry.name}")
        build["evidence"] = [{"label": "Recorded telemetry", "url": f"{base}/{telemetry.name}"}]
        pages.write_json(self.build, build)
        original_size = pages.site_size(self.web)
        limit = original_size + pages.SITE_RESERVE - replay.stat().st_size - 1000
        revision = "a" * 40
        missing = HTTPError(self.url, 404, "missing release asset", {}, None)
        with patch.object(pages, "SITE_LIMIT", limit), patch.object(pages, "urlopen", side_effect=missing):
            with self.assertRaisesRegex(ValueError, "immutable checkout commit"):
                pages.prepare(self.web)
            self.assertEqual(pages.site_size(self.web), original_size)
            pages.prepare(self.web, archive_revision=revision)

        paths = {name: f"{pages.ARCHIVE_REPO}/{revision}/web/media/urban-pace/{name}"
                 for name in (replay.name, telemetry.name)}
        self.assertLessEqual(pages.site_size(self.web), limit - pages.SITE_RESERVE)
        self.assertFalse(replay.exists())
        self.assertFalse(telemetry.exists())
        self.assertEqual(pack.read_bytes(), b"compressed replay")
        self.assertEqual(self.chase.read_bytes(), self.compact)
        entry, = json.loads(catalog.read_bytes())
        self.assertEqual(entry["replay_pack_url"], f"{base}/{pack.name}")
        self.assertEqual(entry["replay_url"], paths[replay.name])
        self.assertEqual(entry["log_url"], paths[telemetry.name])
        build = json.loads(self.build.read_bytes())
        self.assertEqual(build["showcases"][0]["replay_url"], paths[replay.name])
        self.assertEqual(build["showcases"][0]["log_url"], paths[telemetry.name])
        self.assertEqual(build["evidence"][0]["url"], paths[telemetry.name])
        staged_manifest = json.loads(self.manifest.read_bytes())
        self.assertEqual(staged_manifest["sha256"], manifest["sha256"])
        self.assertEqual(staged_manifest["external_artifacts"], paths)


if __name__ == "__main__":
    unittest.main()
