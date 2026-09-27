"""Local HTTPS server the phone connects to over Wi-Fi: it serves the guided capture page, receives
photos straight onto this PC and runs the pipeline on them. Nothing leaves the local network.

Android Chrome only exposes the camera and motion sensors to secure pages, hence HTTPS with a
self-signed certificate generated here on first start.
"""
from __future__ import annotations

import datetime as dt
import http.server
import ipaddress
import json
import re
import secrets
import socket
import ssl
import threading
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from core import pipeline
from core.tools import ROOT, Runner

CAPTURES = ROOT / "captures"
CERT_DIR = CAPTURES / ".cert"
WEB_DIR = Path(__file__).resolve().parent / "web"
DEFAULT_PORT = 8443
MAX_PHOTO_BYTES = 25 * 2**20
MAX_PHOTOS = 400
SESSION_ID = re.compile(r"^\d{8}-\d{6}$")
MODES = ("object", "room")
COOKIE = "capture_token"
# Phone-to-subject height used for scale: tabletop objects are shot from ~40 cm above the table.
CAMERA_HEIGHT = {"object": 0.4, "room": 1.5}

OptionsFactory = Callable[[Path, str], pipeline.Options]
LogFn = Callable[[str], None]


def lan_ip() -> str:
    """The address other devices on the Wi-Fi reach this PC at (no packet is actually sent)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def ensure_certificate(ip: str) -> tuple[Path, Path]:
    """Self-signed cert for this PC's LAN IP; regenerated when the IP changes."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    cert_path, key_path, ip_path = CERT_DIR / "cert.pem", CERT_DIR / "key.pem", CERT_DIR / "ip.txt"
    if cert_path.exists() and key_path.exists() and ip_path.exists() and ip_path.read_text() == ip:
        return cert_path, key_path
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "video-3d-map yerel")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=730))
            .add_extension(x509.SubjectAlternativeName([
                x509.IPAddress(ipaddress.ip_address(ip)), x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                x509.DNSName("localhost")]), critical=False)
            .sign(key, hashes.SHA256()))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    ip_path.write_text(ip)
    return cert_path, key_path


def load_token() -> str:
    """Access token kept across restarts, so a phone that connected once keeps working."""
    path = CERT_DIR / "token.txt"
    try:
        token = path.read_text(encoding="ascii").strip()
        if len(token) >= 16:
            return token
    except OSError:
        pass
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(16)
    path.write_text(token, encoding="ascii")
    return token


@dataclass
class Job:
    state: str = "capturing"          # capturing -> processing -> done | failed
    progress: float = 0.0
    label: str = ""
    error: str | None = None
    outputs: dict = field(default_factory=dict)


class CaptureServer:
    def __init__(self, make_options: OptionsFactory, log: LogFn, port: int = DEFAULT_PORT):
        self._make_options = make_options
        self._log = log
        self.port = port
        self.ip = lan_ip()
        self.token = load_token()
        self.jobs: dict[str, Job] = {}
        self.runner: Runner | None = None
        self._lock = threading.Lock()
        self._httpd: http.server.ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        return f"https://{self.ip}:{self.port}/?t={self.token}"

    def start(self) -> None:
        cert, key = ensure_certificate(self.ip)
        server = self

        class Handler(_Handler):
            owner = server

        self._httpd = http.server.ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        self._httpd.socket = context.wrap_socket(self._httpd.socket, server_side=True)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        self._log(f"Telefon çekim sunucusu açık: {self.url}")

    def stop(self) -> None:
        if self.runner:
            self.runner.cancel()
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # --- session lifecycle -------------------------------------------------------------------
    def new_session(self, mode: str) -> str:
        if mode not in MODES:
            raise ValueError("bilinmeyen mod")
        with self._lock:
            while True:
                sid = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                folder = CAPTURES / sid
                if not folder.exists():
                    break
                threading.Event().wait(1.0)
            folder.mkdir(parents=True)
            (folder / "mode.txt").write_text(mode, encoding="ascii")
            self.jobs[sid] = Job()
        self._log(f"Yeni çekim oturumu: {sid} ({'obje' if mode == 'object' else 'oda'})")
        return sid

    def add_photo(self, sid: str, data: bytes) -> int:
        folder = self._folder(sid)
        if self.jobs[sid].state != "capturing":
            raise ValueError("bu oturum kapandı")
        if not data.startswith(b"\xff\xd8"):
            raise ValueError("sadece JPEG kabul ediliyor")
        with self._lock:
            count = len(list(folder.glob("*.jpg")))
            if count >= MAX_PHOTOS:
                raise ValueError(f"en fazla {MAX_PHOTOS} fotoğraf")
            (folder / f"{count + 1:05d}.jpg").write_bytes(data)
        return count + 1

    def finish(self, sid: str) -> None:
        folder = self._folder(sid)
        job = self.jobs[sid]
        if job.state != "capturing":
            return
        mode = (folder / "mode.txt").read_text(encoding="ascii")
        job.state = "processing"
        threading.Thread(target=self._process, args=(sid, folder, mode), daemon=True).start()

    def _process(self, sid: str, folder: Path, mode: str) -> None:
        job = self.jobs[sid]

        def report(p: float, label: str) -> None:
            job.progress, job.label = p, label

        self.runner = Runner(self._log)
        try:
            opts = self._make_options(folder, mode)
            result = pipeline.run(opts, self.runner, report)
            job.outputs = {k: "/" + v.resolve().relative_to(ROOT).as_posix()
                           for k, v in (("mesh", result.model), ("splat", result.splat)) if v}
            job.state, job.progress = "done", 1.0
        except Exception as exc:  # reported to the phone; the full trace went to the desktop log
            job.state, job.error = "failed", str(exc).splitlines()[0]
            self._log(f"Çekim {sid} işlenemedi: {exc}")

    def _folder(self, sid: str) -> Path:
        if not SESSION_ID.match(sid) or sid not in self.jobs:
            raise KeyError("oturum yok")
        return CAPTURES / sid


class _Handler(http.server.SimpleHTTPRequestHandler):
    owner: CaptureServer
    # Results the phone may open once a scan finishes; everything else stays private.
    READ_ONLY_ROOTS = ("/viewer/", "/projects/")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, *args) -> None:
        pass

    def _authorised(self) -> bool:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        if secrets.compare_digest(query.get("t", [""])[0], self.owner.token):
            return True
        cookies = dict(part.strip().split("=", 1) for part in self.headers.get("Cookie", "").split(";") if "=" in part)
        return secrets.compare_digest(cookies.get(COOKIE, ""), self.owner.token)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _deny(self) -> None:
        # send_error() puts its message in the latin-1 status line; Turkish text there drops the connection.
        body = ("<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
                "<body style='font:18px system-ui;padding:24px;background:#111;color:#eee'>"
                "Bağlantı eski. Bilgisayardaki programda <b>📱 Telefonla çek</b> düğmesine bas; açılan QR kodu telefonla tekrar okut.</body>").encode("utf-8")
        self.send_response(403)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self._authorised():
            self._deny()
            return
        path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
        if path in ("/", "/index.html"):
            body = (WEB_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Set-Cookie", f"{COOKIE}={self.owner.token}; Secure; HttpOnly; SameSite=Strict; Path=/")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        match = re.fullmatch(r"/api/session/([\d-]+)/status", path)
        if match:
            job = self.owner.jobs.get(match[1])
            if not job:
                self._json(404, {"error": "oturum yok"})
                return
            self._json(200, {"state": job.state, "progress": job.progress, "label": job.label,
                             "error": job.error, "outputs": job.outputs})
            return
        if path.startswith(self.READ_ONLY_ROOTS) and ".." not in path and not path.endswith("/"):
            super().do_GET()
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if not self._authorised():
            self._json(403, {"error": "yetkisiz"})
            return
        path = urllib.parse.urlsplit(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_PHOTO_BYTES:
            self._json(413, {"error": "fotoğraf çok büyük"})
            return
        body = self.rfile.read(length) if length else b""
        try:
            if path == "/api/session":
                mode = json.loads(body or b"{}").get("mode", "object")
                self._json(200, {"id": self.owner.new_session(mode)})
            elif match := re.fullmatch(r"/api/session/([\d-]+)/photo", path):
                self._json(200, {"count": self.owner.add_photo(match[1], body)})
            elif match := re.fullmatch(r"/api/session/([\d-]+)/finish", path):
                self.owner.finish(match[1])
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "yok"})
        except KeyError:
            self._json(404, {"error": "oturum yok"})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
