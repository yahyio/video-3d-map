"""Downloads the external engines into tools/ (Windows x64):

    COLMAP 4.2.0     camera poses (structure from motion)
    OpenMVS 2.4.0    dense reconstruction, mesh and texture
    Brush 0.3.0      Gaussian splat training
    ffmpeg           frame extraction (skipped when already on PATH)

Usage:  python setup_tools.py [--no-cuda] [--force]

Everything comes over HTTPS from the projects' own GitHub releases; only Brush publishes checksums,
so only Brush is verified against one.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
GITHUB = "https://github.com"
COLMAP_URL = GITHUB + "/colmap/colmap/releases/download/4.2.0/colmap-x64-windows-{variant}.zip"
OPENMVS_URL = GITHUB + "/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Windows_x64.zip"
BRUSH_URL = GITHUB + "/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-x86_64-pc-windows-msvc.zip"
FFMPEG_URL = GITHUB + "/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
CHUNK = 1 << 20


def fetch(url: str):
    try:
        return urllib.request.urlopen(url)
    except urllib.error.URLError as exc:
        raise SystemExit(f"Download failed: {url}\n  {exc}") from None


def download(url: str, dest: Path) -> Path:
    print(f"  {url}")
    with fetch(url) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MB", end="", flush=True)
    print()
    return dest


def verify_sha256(path: Path, url: str) -> None:
    with fetch(url) as resp:
        expected = resp.read().decode("ascii").split()[0].lower()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit(f"Checksum mismatch for {path.name}; delete it and run again.")


def extract_dir_of(archive: Path, marker: str, work: Path) -> Path:
    """Extracts `archive` and returns the folder that directly contains `marker` (e.g. colmap.exe)."""
    target = work / archive.stem
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    found = next(target.rglob(marker), None)
    if found is None:
        raise SystemExit(f"{marker} not found in {archive.name}; the release layout may have changed.")
    return found.parent


def install(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)


def setup_colmap(work: Path, cuda: bool) -> None:
    variant = "cuda" if cuda else "nocuda"
    print(f"COLMAP 4.2.0 ({variant})")
    archive = download(COLMAP_URL.format(variant=variant), work / f"colmap-{variant}.zip")
    bin_dir = extract_dir_of(archive, "colmap.exe", work)
    install(bin_dir.parent, TOOLS / "colmap")          # keeps bin/ and plugins/ side by side


def setup_openmvs(work: Path) -> None:
    print("OpenMVS 2.4.0")
    archive = download(OPENMVS_URL, work / "openmvs.zip")
    install(extract_dir_of(archive, "InterfaceCOLMAP.exe", work), TOOLS / "openmvs" / "vc17" / "x64" / "Release")


def setup_brush(work: Path) -> None:
    print("Brush 0.3.0")
    archive = download(BRUSH_URL, work / "brush.zip")
    verify_sha256(archive, BRUSH_URL + ".sha256")
    install(extract_dir_of(archive, "brush_app.exe", work), TOOLS / "brush")


def setup_ffmpeg(work: Path) -> None:
    print("ffmpeg")
    archive = download(FFMPEG_URL, work / "ffmpeg.zip")
    bin_dir = extract_dir_of(archive, "ffmpeg.exe", work)
    dest = TOOLS / "ffmpeg"
    dest.mkdir(parents=True, exist_ok=True)
    for exe in ("ffmpeg.exe", "ffprobe.exe"):
        shutil.copy2(bin_dir / exe, dest / exe)


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("The bundled engines are Windows x64 builds; run this on Windows.")
    parser = argparse.ArgumentParser(description="Download COLMAP, OpenMVS, Brush and ffmpeg into tools/.")
    parser.add_argument("--no-cuda", action="store_true", help="CPU build of COLMAP (no NVIDIA GPU)")
    parser.add_argument("--force", action="store_true", help="download again even if already installed")
    args = parser.parse_args()
    cuda = not args.no_cuda and shutil.which("nvidia-smi") is not None

    steps = [
        (TOOLS / "colmap" / "bin" / "colmap.exe", lambda w: setup_colmap(w, cuda)),
        (TOOLS / "openmvs" / "vc17" / "x64" / "Release" / "InterfaceCOLMAP.exe", setup_openmvs),
        (TOOLS / "brush" / "brush_app.exe", setup_brush),
    ]
    if not shutil.which("ffmpeg"):
        steps.append((TOOLS / "ffmpeg" / "ffmpeg.exe", setup_ffmpeg))

    with tempfile.TemporaryDirectory(prefix="v3m-setup-") as tmp:
        for marker, run in steps:
            if marker.exists() and not args.force:
                print(f"{marker.relative_to(ROOT)} already installed, skipping")
                continue
            run(Path(tmp))
    print("Done. Start the app with Baslat.bat or: python app.py")


if __name__ == "__main__":
    main()
