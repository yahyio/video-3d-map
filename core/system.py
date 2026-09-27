"""Machine checks that stop a run early instead of letting a tool crash half an hour in."""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import time
from pathlib import Path

# Brush crashed with AllocError when Windows' commit charge was 39.7 of 40.9 GB (a game and a
# browser were open); a run needs a few GB of headroom for depth maps and splat training.
MIN_FREE_COMMIT_GB = 3.0
DISK_RESERVE_GB = 5.0      # page-file growth never counted into the last few GB of the disk
GROW_CHUNK_GB = 0.25
GROW_TIMEOUT_S = 60.0
_MEM_COMMIT, _MEM_RESERVE, _MEM_RELEASE, _PAGE_READWRITE = 0x1000, 0x2000, 0x8000, 0x04
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def parse_paging_files(entries: list[str]) -> list[tuple[Path, float]]:
    """PagingFiles registry lines ("d:\\pagefile.sys 8192 24576") -> (path, max size GB).
    Entries without an explicit maximum (system-managed) are skipped: their growth is unknown."""
    parsed = []
    for entry in entries:
        parts = entry.split()
        if len(parts) == 3 and parts[2].isdigit() and int(parts[2]) > 0:
            parsed.append((Path(parts[0].replace("?:", "C:")), int(parts[2]) / 1024))
    return parsed


def page_file_growth_gb() -> float:
    """How much further the page files may grow before hitting their configured maximum.

    Windows reports commit against the page file's *current* size (10 GB here with a 24 GB cap),
    which made a machine with ~17 GB to spare look like it had 3.7 GB.
    """
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management")
        entries, _ = winreg.QueryValueEx(key, "PagingFiles")
    except OSError:
        return 0.0
    growth = 0.0
    for path, max_gb in parse_paging_files(list(entries)):
        try:
            current = path.stat().st_size / 2**30
            disk_free = shutil.disk_usage(path.anchor).free / 2**30
        except OSError:
            continue
        growth += max(0.0, min(max_gb - current, disk_free - DISK_RESERVE_GB))
    return growth


def committable_now_gb() -> float:
    """Memory that can be committed immediately, without waiting for the page file to grow."""
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Bellek durumu okunamadı")
    return status.ullAvailPageFile / 2**30


def grow_commit(target_gb: float, timeout_s: float = GROW_TIMEOUT_S) -> float:
    """Commits and releases memory until `target_gb` is committable (or the page file / disk limit
    is hit), so Windows extends the page file *before* a tool needs it. Windows keeps the larger
    page file until reboot; Brush's own gradual allocations failed while it was still growing.
    Returns the committable amount afterwards."""
    want = min(target_gb, committable_now_gb() + page_file_growth_gb())
    if committable_now_gb() >= want:
        return committable_now_gb()
    k = ctypes.windll.kernel32
    k.VirtualAlloc.restype = ctypes.c_void_p
    k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    k.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    blocks, deadline = [], time.monotonic() + timeout_s
    try:
        # Keep committing until the free commit (on top of what we hold) reaches the target.
        while committable_now_gb() + len(blocks) * GROW_CHUNK_GB < want and time.monotonic() < deadline:
            p = k.VirtualAlloc(None, int(GROW_CHUNK_GB * 2**30), _MEM_COMMIT | _MEM_RESERVE, _PAGE_READWRITE)
            if p:
                blocks.append(p)
            else:
                time.sleep(0.5)          # Windows is extending the page file
    finally:
        for p in blocks:
            k.VirtualFree(ctypes.c_void_p(p), 0, _MEM_RELEASE)
    return committable_now_gb()


def free_memory_gb() -> tuple[float, float]:
    """(free physical RAM, free commit incl. page-file growth still allowed), in GB."""
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Bellek durumu okunamadı")
    return status.ullAvailPhys / 2**30, status.ullAvailPageFile / 2**30 + page_file_growth_gb()


def memory_problem() -> str | None:
    """A user-facing message when there is not enough memory to run safely, else None."""
    physical, commit = free_memory_gb()
    if commit >= MIN_FREE_COMMIT_GB:
        return None
    return (f"Bellek neredeyse dolu: sadece {commit:.1f} GB ayrılabilir (boş RAM {physical:.1f} GB). "
            f"Tarama en az {MIN_FREE_COMMIT_GB:.0f} GB ister ve yarıda çöker. "
            "Oyunları (ör. VALORANT), tarayıcıyı ve Discord gibi uygulamaları kapatıp tekrar deneyin.")


_TOP_USERS_PS = ("Get-Process | Group-Object Name | ForEach-Object { '{0}|{1}' -f $_.Name, "
                 "(($_.Group | Measure-Object PrivateMemorySize64 -Sum).Sum) }")


def top_memory_users(count: int = 4) -> list[tuple[str, float]]:
    """Biggest committed-memory users by program name, in GB; empty if Windows won't say."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", _TOP_USERS_PS], capture_output=True,
                             text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        name, _, size = line.rpartition("|")
        if name and size.strip().isdigit():
            rows.append((name, int(size) / 2**30))
    return sorted(rows, key=lambda r: r[1], reverse=True)[:count]


def describe_top_users() -> str:
    users = top_memory_users()
    return ", ".join(f"{name} {gb:.1f} GB" for name, gb in users) if users else "bilinmiyor"


def _pid_alive(pid: int) -> bool:
    handle = ctypes.windll.kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == _STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


class RunLock:
    """One scan at a time: two runs share 16 GB of RAM and the GPU, and both crash."""

    def __init__(self, path: Path):
        self._path = path

    def __enter__(self) -> "RunLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                holder = int(self._path.read_text(encoding="ascii").strip() or 0)
            except (OSError, ValueError):
                holder = 0
            if holder and holder != os.getpid() and _pid_alive(holder):
                raise RuntimeError("Başka bir tarama zaten çalışıyor. Bitmesini bekleyin ya da onu durdurun.")
            self._path.unlink(missing_ok=True)   # left behind by a crashed run
        self._path.write_text(str(os.getpid()), encoding="ascii")
        return self

    def __exit__(self, *_exc) -> None:
        self._path.unlink(missing_ok=True)
