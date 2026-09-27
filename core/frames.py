"""Video -> sharp still frames. Extracts densely with ffmpeg, then keeps the sharpest frame per window."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .tools import WORKER_THREADS, Runner, ffmpeg

EXTRACT_FPS = 6          # candidate density; the best of each window survives
MAX_LONG_SIDE = 1920     # frames larger than this add runtime but little detail
SHARPNESS_PREVIEW = 480  # sharpness is measured on a downscaled copy for speed
MIN_SHORT_SIDE = 1080    # below this, messenger-compressed footage loses most matchable detail


def sharpness(img: Image.Image) -> float:
    """Variance of the Laplacian: higher means crisper edges, low means motion blur."""
    gray = img.convert("L")
    gray.thumbnail((SHARPNESS_PREVIEW, SHARPNESS_PREVIEW))
    a = np.asarray(gray, dtype=np.float32)
    lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
    return float(lap.var())


@dataclass(frozen=True)
class FrameBudget:
    """How many stills to keep: proportional to video length, clamped to a range."""
    per_second: float
    minimum: int
    maximum: int


def frames_for_duration(seconds: float, budget: FrameBudget) -> int:
    # A fixed count starved long walkthroughs: 120 frames over 57 s left 0.5 s gaps and
    # COLMAP split the desk scan into three pieces (67/120 registered).
    return int(min(budget.maximum, max(budget.minimum, round(seconds * budget.per_second))))


def pick_sharpest(scores: list[float], target: int) -> list[int]:
    """Split the sequence into `target` equal windows and return the sharpest index of each."""
    n = len(scores)
    if n == 0 or target <= 0:
        return []
    if target >= n:
        return list(range(n))
    window = n / target
    picked = []
    for w in range(target):
        start, end = int(round(w * window)), int(round((w + 1) * window))
        picked.append(max(range(start, max(end, start + 1)), key=lambda i: scores[i]))
    return picked


def video_size(video: Path) -> tuple[int, int] | None:
    """(width, height) of the first video stream, or None if ffprobe cannot read it."""
    probe = ffmpeg().with_name("ffprobe" + ffmpeg().suffix)
    try:
        out = subprocess.run([str(probe), "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
                             capture_output=True, text=True, timeout=30,
                             creationflags=subprocess.CREATE_NO_WINDOW).stdout
        w, h = (int(x) for x in out.strip().split(",")[:2])
        return w, h
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def low_resolution_warning(size: tuple[int, int] | None) -> str | None:
    if size is None or min(size) >= MIN_SHORT_SIDE:
        return None
    return (f"Video sadece {size[0]}×{size[1]} piksel. WhatsApp/Telegram gibi uygulamalar videoyu "
            "küçültür ve detayın çoğu kaybolur. Telefondaki orijinal dosyayı kablo ya da "
            "Google Drive ile aktarıp onu kullan.")


def _timecode(match: re.Match) -> float:
    return int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])


def extract(video: Path, out_dir: Path, budget: FrameBudget, runner: Runner, progress) -> int:
    """Writes the chosen frames to out_dir as frame_00001.jpg...; returns how many were kept."""
    cand_dir = out_dir.parent / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in [*cand_dir.glob("*.jpg"), *out_dir.glob("*.jpg")]:
        old.unlink()

    side = MAX_LONG_SIDE
    scale = f"scale='if(gt(iw,ih),min({side},iw),-2)':'if(gt(iw,ih),-2,min({side},ih))'"
    duration = [0.0]

    def on_line(line: str) -> None:
        if m := re.search(r"Duration: (\d+):(\d+):([\d.]+)", line):
            duration[0] = _timecode(m)
        if (m := re.search(r"time=(\d+):(\d+):([\d.]+)", line)) and duration[0] > 0:
            progress(min(_timecode(m) / duration[0], 1.0) * 0.7)

    runner.run(ffmpeg(), ["-hide_banner", "-y", "-threads", WORKER_THREADS, "-i", video,
                          "-vf", f"fps={EXTRACT_FPS},{scale}", "-q:v", "2",
                          cand_dir / "c_%05d.jpg"], on_line=on_line)

    candidates = sorted(cand_dir.glob("c_*.jpg"))
    if len(candidates) < 3:
        raise ValueError("Videodan yeterli kare çıkmadı. Video çok kısa veya bozuk olabilir.")
    scores = []
    for i, path in enumerate(candidates):
        runner.check()
        with Image.open(path) as im:
            scores.append(sharpness(im))
        progress(0.7 + 0.25 * (i + 1) / len(candidates))

    keep = pick_sharpest(scores, frames_for_duration(len(candidates) / EXTRACT_FPS, budget))
    for n, idx in enumerate(keep, start=1):
        candidates[idx].replace(out_dir / f"frame_{n:05d}.jpg")
    for leftover in cand_dir.glob("*.jpg"):
        leftover.unlink()
    cand_dir.rmdir()
    progress(1.0)
    return len(keep)
