"""Photo input: a folder of stills instead of a video. Phone photos are ~12 MP and free of motion
blur, which is why professional photogrammetry is shot as photos rather than video."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from .frames import sharpness

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_LONG_SIDE = 3200      # more pixels than this slows every stage for little extra detail
MIN_PHOTOS = 8
BLUR_RATIO = 0.35         # drop shots far blurrier than the set's median


def list_photos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_EXT and p.is_file())


def import_photos(src: Path, out_dir: Path, runner, progress) -> tuple[int, int]:
    """Copies photos to out_dir as upright JPEGs (EXIF rotation applied), dropping blurry ones.
    Returns (kept, dropped)."""
    photos = list_photos(src)
    if len(photos) < MIN_PHOTOS:
        raise ValueError(f"Klasörde en az {MIN_PHOTOS} fotoğraf olmalı, {len(photos)} bulundu.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    staged: list[tuple[Path, float]] = []
    for i, path in enumerate(photos):
        runner.check()
        try:
            with Image.open(path) as im:
                upright = ImageOps.exif_transpose(im).convert("RGB")
        except OSError as exc:
            runner.log(f"Atlandı (okunamadı): {path.name}: {exc}")
            continue
        upright.thumbnail((MAX_LONG_SIDE, MAX_LONG_SIDE), Image.LANCZOS)
        target = out_dir / f"stage_{len(staged) + 1:05d}.jpg"
        upright.save(target, quality=95)
        staged.append((target, sharpness(upright)))
        progress(0.9 * (i + 1) / len(photos))

    if len(staged) < MIN_PHOTOS:
        raise ValueError(f"Okunabilen fotoğraf sayısı çok az ({len(staged)}).")
    scores = sorted(s for _, s in staged)
    median = scores[len(scores) // 2]
    kept = [path for path, score in staged if score >= BLUR_RATIO * median]
    for path, score in staged:
        if score < BLUR_RATIO * median:
            path.unlink()
    # COLMAP only needs stable names; keep them contiguous after dropping blurry shots.
    for n, path in enumerate(kept, start=1):
        path.rename(out_dir / f"img_{n:05d}.jpg")
    progress(1.0)
    return len(kept), len(staged) - len(kept)
