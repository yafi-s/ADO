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


if __name__ == "__main__":
    unittest.main()
