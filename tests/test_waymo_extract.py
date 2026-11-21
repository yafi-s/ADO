import io
import json
import os
import time
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import requests

from waymo_training import extract as ex


def packed(payload):
    length = struct.pack("<Q", len(payload))
    return length + struct.pack("<I", ex.masked_crc(length)) + payload + struct.pack("<I", ex.masked_crc(payload))


def example(group, index):
    frame = ex.FRAME()
    frame.frame.context.name = f"{group}-{index:03}"
    # The published sample explicitly stores a zero timestamp.
    frame.frame.timestamp_micros = 0
    frame.intent = 2
    for field in ex.PAST:
        getattr(frame.past_states, field).extend(range(16))
    for field in ("pos_x", "pos_y"):
        getattr(frame.future_states, field).extend(range(20))
    for view in reversed(ex.VIEWS):
        camera = frame.frame.images.add()
        camera.name = view
        color = np.full((24, 24, 3), (3, 10, 70 * view), dtype=np.uint8)
        camera.image = cv2.imencode(".jpg", color)[1].tobytes()
    return frame


def response(content, status, headers):
    res = requests.Response()
    res.status_code = status
    res.headers.update(headers)
    res.raw = io.BytesIO(content)
    res._content = content
    return res


class ExtractTests(unittest.TestCase):
    def test_validate_tfrecord_and_ignore_partial_tail(self):
        good = packed(b"waymo")
        self.assertEqual(list(ex.complete_records(good + good[:-2])), [b"waymo"])
        for i in (8, 12, len(good) - 1):
            broken = bytearray(good)
            broken[i] ^= 1
            with self.assertRaisesRegex(ValueError, "checksum"):
                list(ex.complete_records(bytes(broken)))

    def test_real_schema_decode_and_camera_order(self):
        group = "a" * 32
        f = example(group, 7)
        preference = f.preference_trajectories.add()
        preference.preference_score = 8
        preference.pos_x.extend(range(20))
        preference.pos_y.extend(range(20))
        row = ex.decode(f.SerializeToString(), "val")
        self.assertEqual(row["images"].shape, (3, 64, 96, 3))
        np.testing.assert_allclose(row["images"][:, 0, 0, 0], [70, 140, 210], atol=3)
        self.assertEqual(row["group"], group)
        self.assertEqual(row["frame_index"], 7)
        self.assertEqual(row["timestamps"], 0)
        self.assertEqual(row["split"], ex.partition("val", group))
        self.assertEqual(row["future"].shape, (20, 2))
        self.assertEqual(row["past"].shape, (16, 6))
        np.testing.assert_array_equal(row["rater_scores"], [8, -1, -1])
        self.assertTrue(np.isnan(row["rater_trajectories"][1:]).all())
        f.past_states.ClearField("accel_y")
        with self.assertRaisesRegex(ValueError, "incomplete trajectory"):
            ex.decode(f.SerializeToString(), "val")

    def test_checkpoint_resume_crc_and_segment_identity(self):
        group = "b" * 32
        raw = b"".join(packed(example(group, i).SerializeToString()) for i in (1, 2))
        size = len(raw)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for child in ("features", "receipts"):
                (root / child).mkdir()
            auth = ex.Access(root, 0, "adc")
            session = MagicMock()
            session.get.side_effect = [
                response(json.dumps({"generation": "123", "size": size + 20}).encode(), 200, {}),
                response(raw, 206, {"Content-Range": f"bytes 0-{size-1}/{size+20}",
                                    "x-goog-generation": "123"}),
            ]
            with patch.object(auth, "headers", return_value={}), patch("requests.Session") as client:
                client.return_value.__enter__.return_value = session
                receipt = ex.extract_one(("val", 0), root, size, auth)
                self.assertEqual(ex.extract_one(("val", 0), root, size, auth), receipt)
                self.assertEqual(session.get.call_count, 2)
            summary = ex.summarize(root, [receipt])
            self.assertEqual(sum(summary["rows"].values()), 2)
            self.assertEqual(sum(summary["groups"].values()), 1)
            self.assertEqual(summary["zero_timestamp_rows"], 2)
            self.assertEqual(summary["reserved_transfer_bytes"], size)
            with self.assertRaisesRegex(ValueError, "duplicate segment/frame"):
                ex.summarize(root, [receipt, receipt])
            with self.assertRaisesRegex(RuntimeError, "limit"):
                auth.reserve(ex.LIMIT)
            (root / "features" / "val-00000.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                ex.extract_one(("val", 0), root, size, auth)

    def test_application_default_credentials_refresh(self):
        with tempfile.TemporaryDirectory() as folder:
            access = ex.Access(Path(folder), 0, "adc")
            credentials = MagicMock(valid=False, token="test-token")
            with patch("google.auth.default", return_value=(credentials, None)) as default:
                self.assertEqual(access.headers()["Authorization"], "Bearer test-token")
                credentials.valid = True
                self.assertEqual(access.headers()["Authorization"], "Bearer test-token")
            default.assert_called_once_with(scopes=["https://www.googleapis.com/auth/devstorage.read_only"])
            credentials.refresh.assert_called_once()
            self.assertNotIn("test-token", access.ledger.read_text())

    def test_job_grant_is_consumed_once_and_stays_scoped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / ex.GRANT_FILENAME
            grant = dict(access_token="test-only-token", expires_at=time.time() + 30,
                         bucket=ex.BUCKET, allowed_prefixes=sorted(ex.GRANT_PREFIXES))
            file.write_text(json.dumps(grant))
            with patch.object(ex, "COLAB_CONTENT", root), patch.dict(
                    os.environ, {"ADO_WAYMO_GRANT_FILE": str(file)}):
                client = ex.Access(root, 0, "job-grant")
            self.assertFalse(file.exists())
            for prefix in ex.GRANT_PREFIXES:
                self.assertEqual(client.headers(prefix + "00000-of-00093")["Authorization"],
                                 "Bearer test-only-token")
            with self.assertRaisesRegex(ValueError, "outside"):
                client.headers("test_undisclosed.tfrecord")
            with patch("time.time", return_value=grant["expires_at"] + 1), self.assertRaisesRegex(
                    RuntimeError, "expired"):
                client.headers(sorted(ex.GRANT_PREFIXES)[0] + "00000-of-00263")
            self.assertNotIn("test-only-token", client.ledger.read_text())

    def test_job_grant_rejects_wrong_scope_expiry_and_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / ex.GRANT_FILENAME
            sample = dict(access_token="test-only-token", expires_at="2100-01-01T00:00:00Z",
                          bucket=ex.BUCKET, allowed_prefixes=sorted(ex.GRANT_PREFIXES))
            with patch.object(ex, "COLAB_CONTENT", root), patch.dict(
                    os.environ, {"ADO_WAYMO_GRANT_FILE": str(file)}):
                for changed in ({"bucket": "other"}, {"allowed_prefixes": ["training", "val"]},
                                {"expires_at": 1}, {"expires_at": float("nan")}):
                    file.write_text(json.dumps({**sample, **changed}))
                    with self.assertRaisesRegex(ValueError, "invalid or expired"):
                        ex.load_job_grant()
                    self.assertFalse(file.exists())
                file.write_text(json.dumps(sample))
                token, expiry = ex.load_job_grant()
                self.assertEqual(token, "test-only-token")
                self.assertGreater(expiry, time.time())
            wrong = root / "ordinary-file.json"
            wrong.write_text("preserve")
            with patch.object(ex, "COLAB_CONTENT", root), patch.dict(
                    os.environ, {"ADO_WAYMO_GRANT_FILE": str(wrong)}), self.assertRaisesRegex(
                    ValueError, "dedicated"):
                ex.load_job_grant()
            self.assertEqual(wrong.read_text(), "preserve")

    def test_range_precondition_prevents_full_object_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            auth = ex.Access(root, 0, "adc")
            source = response(b"{}", 200, {})
            source.raw = MagicMock()
            session = MagicMock()
            session.get.side_effect = [response(b'{"size":10000,"generation":"7"}', 200, {}), source]
            with patch.object(auth, "headers", return_value={}), patch("requests.Session") as client:
                client.return_value.__enter__.return_value = session
                with self.assertRaisesRegex(ValueError, "pinned object range"):
                    ex.extract_one(("train", 0), root, 100, auth)
            source.raw.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
