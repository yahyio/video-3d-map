"""Locates the bundled COLMAP/OpenMVS/ffmpeg binaries and runs them with live log streaming."""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
COLMAP = TOOLS / "colmap" / "bin" / "colmap.exe"
OPENMVS_DIR = TOOLS / "openmvs" / "vc17" / "x64" / "Release"
BRUSH = TOOLS / "brush" / "brush_app.exe"

# Scans used every core at normal priority and made the PC (and games) stutter. Half the cores at
# below-normal priority keeps the machine usable; Windows hands the tools any CPU left idle.
WORKER_THREADS = max(2, (os.cpu_count() or 4) // 2)
_TOOL_PRIORITY = subprocess.BELOW_NORMAL_PRIORITY_CLASS

LogFn = Callable[[str], None]


class Cancelled(Exception):
    """Raised when the user stops the pipeline."""


class ToolError(RuntimeError):
    """Raised when an external tool exits with a non-zero code."""


def openmvs(name: str) -> Path:
    return OPENMVS_DIR / f"{name}.exe"


def ffmpeg() -> Path:
    local = TOOLS / "ffmpeg" / "ffmpeg.exe"
    if local.exists():
        return local
    found = shutil.which("ffmpeg")
    if not found:
        raise ToolError("ffmpeg bulunamadı. ffmpeg'i kurun veya tools/ffmpeg/ffmpeg.exe olarak koyun.")
    return Path(found)


def missing_tools() -> list[str]:
    required = [COLMAP] + [openmvs(n) for n in
                           ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh")]
    required.append(BRUSH)
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    try:
        ffmpeg()
    except ToolError:
        missing.append("ffmpeg")
    return missing


class Runner:
    """Runs one external process at a time; cancel() kills the active one."""

    def __init__(self, log: LogFn):
        self._log = log
        self._proc: subprocess.Popen | None = None
        self._cancelled = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()
        proc = self._proc
        if proc and proc.poll() is None:
            proc.kill()

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled()

    def log(self, message: str) -> None:
        self._log(message)

    def run(self, exe: Path, args: Sequence[str], cwd: Path | None = None,
            on_line: LogFn | None = None) -> None:
        self.check()
        self._log(f"$ {exe.name} {' '.join(map(str, args))}")
        tail: list[str] = []
        emit_lock = threading.Lock()

        def emit(line: str) -> None:
            with emit_lock:   # stdout reader and log tailer both call this
                tail.append(line)
                del tail[:-30]
                self._log(line)
                if on_line:
                    on_line(line)

        # OpenMVS writes its progress to <Tool>-<stamp>.log in the working folder, not to stdout.
        log_dir = cwd if exe.parent == OPENMVS_DIR and cwd else None
        before = set(log_dir.glob(f"{exe.stem}-*.log")) if log_dir else set()
        done = threading.Event()
        tailer = threading.Thread(target=_tail_new_log, args=(log_dir, exe.stem, before, done, emit),
                                  daemon=True) if log_dir else None

        self._proc = subprocess.Popen(
            [str(exe), *map(str, args)], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW | _TOOL_PRIORITY,
        )
        if self.cancelled:   # cancel() may have run before _proc was set
            self._proc.kill()
        if tailer:
            tailer.start()
        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            if raw.strip():
                emit(raw.rstrip())
        code = self._proc.wait()
        self._proc = None
        done.set()
        if tailer:
            tailer.join(timeout=5)
        self.check()
        if code != 0:
            raise ToolError(_describe_failure(exe.name, code) + "\n\n" + "\n".join(tail[-8:]))


# Windows NTSTATUS codes seen as process exit codes when a tool crashes rather than failing cleanly.
_CRASH_CODES = {0xC0000005: "bellek erişim hatası", 0xC00000FD: "yığın taşması", 0xC0000409: "bellek bozulması"}


def _describe_failure(name: str, code: int) -> str:
    unsigned = code & 0xFFFFFFFF
    if unsigned in _CRASH_CODES:
        return (f"{name} çöktü ({_CRASH_CODES[unsigned]}, 0x{unsigned:08X}). Bu genelde videodan yeterince "
                "eşleşen kare çıkmadığında olur: daha yavaş, bol örtüşmeli çekim ya da Yüksek kalite deneyin.")
    return f"{name} hata ile bitti (kod {code})."


def _tail_new_log(folder: Path, stem: str, before: set[Path], done: threading.Event, emit: LogFn) -> None:
    """Streams lines from the log file the tool creates, then deletes it once the tool exits."""
    path = None
    while path is None and not done.is_set():
        fresh = set(folder.glob(f"{stem}-*.log")) - before
        path = max(fresh, key=lambda p: p.stat().st_mtime) if fresh else None
        if path is None:
            done.wait(0.3)
    if path is None:
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            if line:
                if line.strip():
                    emit(line.rstrip())
                continue
            if done.is_set():
                break
            done.wait(0.3)
    path.unlink(missing_ok=True)
