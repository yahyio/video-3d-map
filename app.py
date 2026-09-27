"""Video -> 3D map desktop app. Run: python app.py [video]"""
from __future__ import annotations

import collections
import functools
import http.server
import os
import sys
import threading
import time
import traceback
import urllib.parse
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QDoubleSpinBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QVBoxLayout, QWidget)

from capture.server import CAMERA_HEIGHT, CaptureServer
from core import frames, pipeline, primitives
from core.tools import ROOT, Cancelled, Runner, missing_tools

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
LOG_FLUSH_MS = 200
LOG_MAX_LINES = 4000
THREAD_STOP_MS = 10_000
STOPPED = "Durduruldu."
WARN_COLOR = "#ff8a3d"
QUALITY_HINTS = {"fast": "~5–15 dk · taslak", "normal": "~15–40 dk · önerilen", "high": "~40–90 dk · en detaylı"}

STYLE = """
* { font-family: 'Segoe UI Variable Text', 'Segoe UI'; font-size: 13px; color: #d9d6cf; }
QMainWindow, #root { background: #111214; }
#title { font-size: 20px; font-weight: 700; letter-spacing: 1px; }
#hint, #stage, #sub { color: #7d7a74; }
#drop { border: 1px dashed #3a3b3e; background: #151619; }
#drop[active="true"] { border: 1px solid #ffb000; background: #1a1812; }
#dropMain { font-size: 15px; font-weight: 600; }
QPushButton { background: #1a1b1e; border: 1px solid #2a2b2e; padding: 7px 14px; }
QPushButton:hover { border-color: #7d7a74; }
QPushButton:checked { border-color: #ffb000; color: #ffb000; }
QPushButton:disabled { color: #4a4945; border-color: #1f2023; }
#start { background: #ffb000; color: #111214; border: none; font-weight: 700; padding: 10px 22px; }
#start:hover { background: #ffc233; }
#start:disabled { background: #3a3220; color: #7d7a74; }
#stop { border-color: #6b2b24; color: #ff6b5b; padding: 10px 22px; }
QDoubleSpinBox { background: #1a1b1e; border: 1px solid #2a2b2e; padding: 5px; min-width: 70px; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #3a3b3e; background: #1a1b1e; }
QCheckBox::indicator:checked { background: #ffb000; border-color: #ffb000; }
QProgressBar { background: #1f2023; border: none; max-height: 4px; }
QProgressBar::chunk { background: #ffb000; }
QPlainTextEdit { background: #0b0c0d; border: 1px solid #1f2023;
                 font-family: 'Cascadia Mono', Consolas, monospace; font-size: 11px; color: #9a968e; }
#rule { background: #1f2023; max-height: 1px; }
"""


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    ALLOWED = ("/viewer/", "/projects/")

    def log_message(self, *args) -> None:
        pass

    def send_head(self):
        path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
        if not path.startswith(self.ALLOWED) or ".." in path or path.endswith("/"):
            self.send_error(404)
            return None
        return super().send_head()


class ViewerServer:
    """Serves the app folder on localhost so the browser viewer can load OBJ + textures."""

    def __init__(self) -> None:
        handler = functools.partial(_QuietHandler, directory=str(ROOT))
        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def url_for(self, model: Path) -> str:
        rel = model.resolve().relative_to(ROOT).as_posix()
        return (f"http://127.0.0.1:{self._httpd.server_port}/viewer/index.html?"
                + urllib.parse.urlencode({"model": "/" + rel}))


class Worker(QObject):
    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, job, log_queue: collections.deque):
        super().__init__()
        self._job = job                     # (runner, report) -> pipeline.Result
        self.runner = Runner(log_queue.append)

    def run(self) -> None:
        try:
            self.finished.emit(self._job(self.runner, self.progress.emit))
        except Cancelled:
            self.failed.emit(STOPPED)
        except Exception as exc:  # shown to the user; the full trace goes to the log panel
            self.runner.log(traceback.format_exc())
            self.failed.emit(str(exc))


class DropZone(QFrame):
    def __init__(self, on_video):
        super().__init__(objectName="drop", acceptDrops=True, minimumHeight=130)
        self.setCursor(Qt.PointingHandCursor)
        self._on_video = on_video
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        self.main = QLabel("Videoyu buraya bırak", objectName="dropMain", alignment=Qt.AlignCenter)
        self.sub = QLabel("ya da tıklayıp seç · mp4, mov, mkv…", objectName="sub", alignment=Qt.AlignCenter)
        lay.addWidget(self.main)
        lay.addWidget(self.sub)

    def _set_active(self, on: bool) -> None:
        self.setProperty("active", "true" if on else "false")
        self.style().polish(self)

    def mousePressEvent(self, _event) -> None:
        pattern = "Video (" + " ".join(f"*{e}" for e in sorted(VIDEO_EXT)) + ")"
        path, _ = QFileDialog.getOpenFileName(self, "Video seç", str(Path.home()), pattern)
        if path:
            self._on_video(Path(path))

    def dragEnterEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls and Path(urls[0].toLocalFile()).suffix.lower() in VIDEO_EXT:
            event.acceptProposedAction()
            self._set_active(True)

    def dragLeaveEvent(self, _event) -> None:
        self._set_active(False)

    def dropEvent(self, event) -> None:
        self._set_active(False)
        self._on_video(Path(event.mimeData().urls()[0].toLocalFile()))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video → 3D Map")
        self.resize(780, 720)
        self.video: Path | None = None
        self.result: pipeline.Result | None = None
        self.log_queue: collections.deque[str] = collections.deque()
        self.server: ViewerServer | None = None
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self.capture: CaptureServer | None = None
        self.started_at = 0.0
        self._build()
        QTimer(self, interval=LOG_FLUSH_MS, timeout=self._flush_log).start()
        QTimer.singleShot(0, self._check_tools)

    def _build(self) -> None:
        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(14)
        lay.addWidget(QLabel("<span style='color:#ffb000'>●</span> VIDEO → 3D MAP", objectName="title"))
        lay.addWidget(QLabel("Mekânın içinde ya da etrafında yavaşça yürüyerek çektiğin videodan dokulu 3D model çıkarır.",
                             objectName="hint", wordWrap=True))
        self.drop = DropZone(self._set_video)
        lay.addWidget(self.drop)
        lay.addLayout(self._quality_row())
        lay.addLayout(self._height_row())
        lay.addLayout(self._options_row())
        lay.addWidget(QFrame(objectName="rule"))
        lay.addLayout(self._action_row())
        self.bar = QProgressBar(textVisible=False, maximum=1000)
        lay.addWidget(self.bar)
        self.stage = QLabel("Hazır.", objectName="stage")
        lay.addWidget(self.stage)
        self.log = QPlainTextEdit(readOnly=True, maximumBlockCount=LOG_MAX_LINES)
        lay.addWidget(self.log, 1)

    def _quality_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Kalite"))
        self.quality = QButtonGroup(self)
        for key, q in pipeline.QUALITIES.items():
            b = QPushButton(q.label, checkable=True, checked=key == "normal")
            b.setProperty("key", key)
            self.quality.addButton(b)
            row.addWidget(b)
        self.quality.buttonToggled.connect(self._update_hint)
        self.quality_hint = QLabel(objectName="hint")
        row.addWidget(self.quality_hint)
        row.addStretch()
        self._update_hint()
        return row

    def _height_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Kamera yüksekliği"))
        self.height = QDoubleSpinBox(decimals=1, minimum=0.1, maximum=500, singleStep=0.1, value=1.5, suffix=" m")
        self.height.setToolTip("Modelin gerçek boyutu bundan hesaplanır.")
        row.addWidget(self.height)
        row.addWidget(QLabel("oda ≈ 1,5 · masadaki obje ≈ 0,3 (masadan yükseklik) · drone: uçuş yüksekliği", objectName="hint"))
        row.addStretch()
        return row

    def _options_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.splat = QCheckBox("Gaussian Splat", checked=True)
        self.splat.setToolTip("Fotoğraf gibi görünür, en hızlı yol. Beğenirsen sonra Mesh'e çevirebilirsin.")
        self.mesh = QCheckBox("Mesh (yavaş, CPU'yu yorar)", checked=False)
        self.mesh.setToolTip("Dokulu model.obj üretir (Blender, oyun motorları); süreyi 2–3 katına çıkarır.")
        self.keep = QCheckBox("Ara dosyaları sakla")
        self.keep.setToolTip("GB'larca yer tutar")
        for i, w in enumerate((self.splat, self.mesh, self.keep)):
            if i:
                row.addSpacing(18)
            row.addWidget(w)
        row.addStretch()
        return row

    def _action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.start = QPushButton("Başlat", objectName="start", enabled=False, clicked=self._start)
        self.stop = QPushButton("Durdur", objectName="stop", visible=False, clicked=self._stop)
        self.preview = QPushButton("Mesh'i aç", enabled=False, clicked=lambda: self._preview(self.result.model))
        self.preview_splat = QPushButton("Splat'i aç", enabled=False,
                                         clicked=lambda: self._preview(self.result.splat))
        self.open_dir = QPushButton("Klasörü aç", enabled=False, clicked=self._open_folder)
        self.to_mesh = QPushButton("Mesh'e çevir", enabled=False, clicked=self._convert_to_mesh)
        self.to_mesh.setToolTip("Beğendiğin splat'ten en yüksek kalitede dokulu mesh (OBJ) üretir. "
                                "Kareler ve kamera konumları tekrar hesaplanmaz.")
        self.to_shapes = QPushButton("Geometriye çevir", enabled=False, clicked=self._convert_to_shapes)
        self.to_shapes.setToolTip("Splat'teki nesneleri dokulu silindir ve kutulara çevirir (saniyeler sürer).")
        self.phone = QPushButton("📱 Telefonla çek", clicked=self._start_phone)
        self.phone.setToolTip("Telefonda rehberli çekim: fotoğraflar Wi-Fi ile bu bilgisayara gelir, dışarı çıkmaz.")
        row.addWidget(self.start)
        row.addWidget(self.stop)
        row.addWidget(self.phone)
        row.addStretch()
        row.addWidget(self.preview)
        row.addWidget(self.preview_splat)
        row.addWidget(self.to_mesh)
        row.addWidget(self.to_shapes)
        row.addWidget(self.open_dir)
        return row

    def _check_tools(self) -> None:
        missing = missing_tools()
        if missing:
            QMessageBox.critical(self, "Eksik araçlar", "Şunlar bulunamadı:\n" + "\n".join(missing))

    def _update_hint(self, *_args) -> None:
        self.quality_hint.setText(QUALITY_HINTS[self.quality.checkedButton().property("key")])

    def _set_video(self, path: Path) -> None:
        if path.suffix.lower() not in VIDEO_EXT or not path.is_file():
            QMessageBox.warning(self, "Video değil", f"{path.name} desteklenen bir video değil.")
            return
        self.video = path
        self.drop.main.setText(path.name)
        size = frames.video_size(path)
        dims = f"{size[0]}×{size[1]} · " if size else ""
        warning = frames.low_resolution_warning(size)
        self.drop.sub.setText(warning or f"{dims}{path.stat().st_size / 1e6:.0f} MB · değiştirmek için tıkla")
        self.drop.sub.setStyleSheet(f"color: {WARN_COLOR}" if warning else "")
        self.drop.sub.setWordWrap(True)
        self.start.setEnabled(self.thread is None)

    def _start(self) -> None:
        opts = pipeline.Options(self.video, quality=self.quality.checkedButton().property("key"),
                                camera_height_m=self.height.value(), mesh=self.mesh.isChecked(),
                                splat=self.splat.isChecked(), keep_work=self.keep.isChecked())
        self.log.clear()
        self.result = None
        self._launch(lambda runner, report: pipeline.run(opts, runner, report))

    def _launch(self, job) -> None:
        self.started_at = time.time()
        self._set_running(True)
        self.thread = QThread(self)
        self.worker = Worker(job, self.log_queue)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.thread.start()

    def _stop(self) -> None:
        if self.worker:
            self.stage.setText("Durduruluyor…")
            self.worker.runner.cancel()

    def _set_running(self, running: bool) -> None:
        self.start.setVisible(not running)
        self.stop.setVisible(running)
        self.drop.setEnabled(not running)
        for w in (self.height, self.mesh, self.splat, self.keep, *self.quality.buttons()):
            w.setEnabled(not running)
        done = not running and self.result is not None
        self.preview.setEnabled(done and self.result.model is not None)
        self.preview_splat.setEnabled(done and self.result.splat is not None)
        self.to_mesh.setEnabled(done and self.result.model is None and pipeline.can_convert_to_mesh(self.result.folder))
        self.to_shapes.setEnabled(done and self.result.splat is not None)
        self.open_dir.setEnabled(done)

    def _elapsed(self) -> str:
        m, s = divmod(int(time.time() - self.started_at), 60)
        return f"{m:02d}:{s:02d}"

    def _on_progress(self, value: float, label: str) -> None:
        self.bar.setValue(int(value * 1000))
        self.stage.setText(f"{label} · %{value * 100:.0f} · {self._elapsed()}")

    def _finish_thread(self) -> None:
        self._flush_log()
        if self.thread:
            self.thread.quit()
            self.thread.wait(THREAD_STOP_MS)
        self.thread = self.worker = None
        self._set_running(False)

    def _on_done(self, result: pipeline.Result) -> None:
        self.result = result
        self._finish_thread()
        info = result.info
        size = " × ".join(f"{s:g}" for s in info["size_m"])
        parts = [f"{info['registered']}/{info['frames']} kare eşleşti"]
        if info["triangles"]:
            parts.append(f"{info['triangles']:,} üçgen".replace(",", "."))
        self.stage.setText(f"Bitti · {self._elapsed()} · " + " · ".join(parts) + f" · yaklaşık {size} m")
        if info.get("splat_error"):
            QMessageBox.warning(self, "Splat üretilemedi",
                                "Mesh hazır ve kaydedildi, ama Gaussian splat üretilemedi:\n\n"
                                f"{info['splat_error']}")
        self._preview(result.splat or result.model)

    def _on_failed(self, message: str) -> None:
        self._finish_thread()
        self.bar.setValue(0)
        if message == STOPPED:
            self.stage.setText(STOPPED)
            return
        self.stage.setText(f"Hata · {self._elapsed()}")
        QMessageBox.warning(self, "Olmadı", message)

    def _flush_log(self) -> None:
        lines = []
        while self.log_queue:
            lines.append(self.log_queue.popleft())
        if lines:
            self.log.appendPlainText("\n".join(lines))

    def _preview(self, target: Path | None) -> None:
        if not target:
            return
        self.server = self.server or ViewerServer()
        QDesktopServices.openUrl(QUrl(self.server.url_for(target)))

    def _phone_options(self, folder: Path, mode: str) -> pipeline.Options:
        """Scans shot on the phone use the window's current output choices."""
        return pipeline.Options(folder, quality=self.quality.checkedButton().property("key"),
                                camera_height_m=CAMERA_HEIGHT[mode], mesh=self.mesh.isChecked(),
                                splat=self.splat.isChecked(), keep_work=self.keep.isChecked())

    def _start_phone(self) -> None:
        if self.capture is None:
            capture = CaptureServer(self._phone_options, self.log_queue.append)
            try:
                capture.start()
            except OSError as exc:
                QMessageBox.warning(self, "Telefon sunucusu açılamadı",
                                    f"Port {capture.port} kullanımda olabilir.\n\n{exc}")
                return
            self.capture = capture
        self.server = self.server or ViewerServer()
        qr = self.server.url_for(ROOT / "viewer" / "qr.html").split("?")[0]
        QDesktopServices.openUrl(QUrl(qr + "?" + urllib.parse.urlencode({"u": self.capture.url})))
        self.stage.setText("Telefon bekleniyor · QR kod tarayıcıda açıldı. Windows izin sorarsa 'İzin ver' de.")

    def _convert_to_mesh(self) -> None:
        folder, info = self.result.folder, self.result.info
        opts = pipeline.Options(Path(info.get("video", folder)), quality="high",
                                camera_height_m=info.get("options", {}).get("camera_height_m", self.height.value()),
                                mesh=True, splat=False)
        self.stage.setText("Mesh'e çevriliyor (maksimum kalite)…")
        self._launch(lambda runner, report: pipeline.resume_from_cameras(folder, opts, runner, report))

    def _convert_to_shapes(self) -> None:
        out = self.result.splat.parent
        self.setCursor(Qt.WaitCursor)
        try:
            shapes = primitives.export(self.result.splat, out / "splat_frame.json", out / "geometry")
        except (ValueError, IndexError, OSError) as exc:
            QMessageBox.warning(self, "Geometri çıkarılamadı", str(exc))
            return
        finally:
            self.unsetCursor()
        self.stage.setText(f"Geometri: {len(shapes)} şekil · geometry/primitives.obj")
        self._preview(out / "geometry" / "primitives.obj")

    def _open_folder(self) -> None:
        if self.result:
            os.startfile(self.result.folder / "output")

    def closeEvent(self, event) -> None:
        if self.capture:
            self.capture.stop()
        if self.worker:
            self.worker.runner.cancel()
            self._finish_thread()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win._set_video(Path(sys.argv[1]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
