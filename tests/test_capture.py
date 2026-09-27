"""Capture server over real HTTPS on localhost, plus photo import. Run: python -m unittest discover tests"""
from __future__ import annotations

import io
import json
import shutil
import ssl
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture import server as capture  # noqa: E402
from core import photos  # noqa: E402

TEST_PORT = 18443


def jpeg_bytes(seed: int = 0, blur: float = 0) -> bytes:
    img = Image.fromarray((np.random.default_rng(seed).random((240, 320, 3)) * 255).astype(np.uint8))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


class CaptureServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._old_captures = capture.CAPTURES
        capture.CAPTURES = Path(cls._tmp.name) / "captures"
        capture.CERT_DIR = capture.CAPTURES / ".cert"
        cls.logs: list[str] = []
        cls.srv = capture.CaptureServer(lambda folder, mode: None, cls.logs.append, port=TEST_PORT)
        cls.srv.ip = "127.0.0.1"
        cls.srv.start()
        cls.ctx = ssl.create_default_context()
        cls.ctx.check_hostname = False
        cls.ctx.verify_mode = ssl.CERT_NONE

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()
        capture.CAPTURES = cls._old_captures
        capture.CERT_DIR = cls._old_captures / ".cert"
        cls._tmp.cleanup()

    def request(self, path: str, data: bytes | None = None, token: bool = True):
        sep = "&" if "?" in path else "?"
        url = f"https://127.0.0.1:{TEST_PORT}{path}" + (f"{sep}t={self.srv.token}" if token else "")
        req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, context=self.ctx, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_page_requires_token(self):
        self.assertEqual(self.request("/", token=False)[0], 403)
        status, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("Tarama Rehberi".encode(), body)

    def test_upload_flow_accepts_jpeg_and_rejects_other_bytes(self):
        status, body = self.request("/api/session", json.dumps({"mode": "object"}).encode())
        self.assertEqual(status, 200)
        sid = json.loads(body)["id"]
        self.assertEqual(self.request(f"/api/session/{sid}/photo", b"not a jpeg")[0], 400)
        status, body = self.request(f"/api/session/{sid}/photo", jpeg_bytes())
        self.assertEqual((status, json.loads(body)["count"]), (200, 1))
        self.assertTrue((capture.CAPTURES / sid / "00001.jpg").exists())
        status, body = self.request(f"/api/session/{sid}/status")
        self.assertEqual(json.loads(body)["state"], "capturing")

    def test_unknown_mode_and_session_are_rejected(self):
        self.assertEqual(self.request("/api/session", json.dumps({"mode": "space"}).encode())[0], 400)
        self.assertEqual(self.request("/api/session/20990101-000000/photo", jpeg_bytes())[0], 404)

    def test_private_files_are_not_served(self):
        for path in ("/core/tools.py", "/captures/.cert/key.pem", "/viewer/../app.py", "/projects/"):
            self.assertEqual(self.request(path)[0], 404, path)


class PhotoImportTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_drops_blurry_and_renumbers(self):
        src = self.dir / "src"
        src.mkdir()
        for i in range(10):
            (src / f"p{i}.jpg").write_bytes(jpeg_bytes(i))
        (src / "blurry.jpg").write_bytes(jpeg_bytes(99, blur=6))
        (src / "notes.txt").write_text("ignored")

        class QuietRunner:
            def check(self): ...
            def log(self, _msg): ...

        kept, dropped = photos.import_photos(src, self.dir / "out", QuietRunner(), lambda _p: None)
        self.assertEqual((kept, dropped), (10, 1))
        names = sorted(p.name for p in (self.dir / "out").glob("*.jpg"))
        self.assertEqual(names, [f"img_{n:05d}.jpg" for n in range(1, 11)])

    def test_too_few_photos_is_a_clear_error(self):
        src = self.dir / "few"
        src.mkdir()
        (src / "a.jpg").write_bytes(jpeg_bytes())
        with self.assertRaises(ValueError):
            photos.import_photos(src, self.dir / "out", None, lambda _p: None)


if __name__ == "__main__":
    unittest.main()
