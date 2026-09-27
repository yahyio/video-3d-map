"""Dense reconstruction with OpenMVS: undistorted COLMAP workspace -> textured OBJ mesh."""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import mesh_clean
from .tools import WORKER_THREADS, Runner, openmvs

# Faces no camera saw get this colour. OpenMVS defaults to bright orange (0xFF7F27), which drew the
# eye to every hole in the first phone scan; a neutral grey reads as "unscanned" instead.
EMPTY_RGB = (110, 110, 110)
EMPTY_COLOR_ARG = (EMPTY_RGB[0] << 16) | (EMPTY_RGB[1] << 8) | EMPTY_RGB[2]
# Seam leveling can crush whole patches to black; above this share we re-texture without it.
MAX_BLACK_SHARE = 0.05
# DensifyPointCloud logs one estimate per view for each patch-match pass (photometric + geometric refinements).
DEPTH_PASSES = 3


# Measured on a failed 1910x1072 run: 401 .dmap files took 16 GB, ~19 bytes per pixel per file,
# and each view writes a depth map plus its geometric-refined copy.
DMAP_BYTES_PER_PIXEL = 2 * 19
DISK_RESERVE_GB = 4.0     # the mesh, textures and Windows' growing page file share the same disk

# RefineMesh auto-decimates by default (83k -> 15k faces on a desk scan); cap only to protect memory.
REFINE_MAX_FACES = 1_500_000


@dataclass(frozen=True)
class MvsSettings:
    dense_px: int           # target long side for depth maps; downscaling only kicks in above it
    refine: bool            # RefineMesh is slow but sharpens geometry noticeably
    refine_px: int
    texture_level: int
    max_texture: int        # atlas page size in pixels


def resolution_level(long_side: int, target_px: int) -> int:
    """OpenMVS halves images per level; pick the first level at or below the target.

    Rounding to the nearest level kept 1910 px frames at full size for a 1600 px target, and 269
    such depth maps filled 16 GB of disk in 55 minutes before the run died.
    """
    if long_side <= target_px:
        return 0
    return max(0, math.ceil(float(np.log2(long_side / target_px))))


def depth_disk_gb(views: int, width: int, height: int, level: int) -> float:
    """Disk the depth maps will take (depth + geometric-refined copy per view)."""
    pixels = (width >> level) * (height >> level)
    return views * pixels * DMAP_BYTES_PER_PIXEL / 2**30


def _image_size(images: Path) -> tuple[int, int]:
    first = next(iter(sorted(images.glob("*"))), None)
    if first is None:
        return 0, 0
    with Image.open(first) as im:
        return im.size


def _long_side(images: Path) -> int:
    first = next(iter(sorted(images.glob("*"))), None)
    if first is None:
        return 0
    with Image.open(first) as im:
        return max(im.size)


def _face_count(ply: Path) -> int:
    with ply.open("rb") as f:
        for raw in f:
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("element face"):
                return int(line.split()[2])
            if line == "end_header":
                break
    return 0


@dataclass(frozen=True)
class MvsResult:
    textured_obj: Path
    mesh_ply: Path          # untextured final mesh, used to level and scale the scan
    scene_mvs: Path
    seam_leveling: bool


def black_share(texture: Path) -> float:
    """Share of near-black pixels among texels that belong to real patches."""
    with Image.open(texture) as im:
        im.thumbnail((1024, 1024))
        a = np.asarray(im.convert("RGB"), dtype=np.int16)
    empty = (np.abs(a - np.array(EMPTY_RGB)) < 12).all(axis=2)
    used = ~empty
    if not used.any():
        return 0.0
    return float((a.sum(axis=2) < 25)[used].mean())


def _textures(obj: Path) -> list[Path]:
    mtl = obj.with_suffix(".mtl")
    if not mtl.exists():
        return []
    return [obj.parent / ln.split(maxsplit=1)[1]
            for ln in mtl.read_text(encoding="utf-8").splitlines() if ln.startswith("map_Kd ")]


def run(dense: Path, out: Path, s: MvsSettings, runner: Runner, progress) -> MvsResult:
    mvs = dense.resolve()
    cfg = ["-w", mvs, "--max-threads", WORKER_THREADS]

    def step(tool: str, args: list, done: float, on_line=None) -> None:
        runner.run(openmvs(tool), [*args, *cfg], cwd=mvs, on_line=on_line)
        progress(done)

    step("InterfaceCOLMAP", ["-i", mvs, "-o", "scene.mvs", "--image-folder", mvs / "images"], 0.05)
    views = max(1, len(list((mvs / "images").glob("*"))))
    depth_maps = [0]

    def densify_line(line: str) -> None:
        # Depth estimation dominates densify time; each pass logs one line per finished view.
        if "Depth-map for image" in line and "estimated" in line:
            depth_maps[0] += 1
            progress(0.05 + 0.45 * min(1.0, depth_maps[0] / (views * DEPTH_PASSES)))

    long_side = _long_side(mvs / "images")
    dense_level = resolution_level(long_side, s.dense_px)
    runner.log(f"Görüntüler {long_side}px; derinlik haritaları {long_side >> dense_level}px'te hesaplanacak.")
    need = depth_disk_gb(views, *_image_size(mvs / "images"), dense_level) + DISK_RESERVE_GB
    free = shutil.disk_usage(mvs).free / 2**30
    if free < need:
        raise RuntimeError(f"Diskte yer yetmiyor: derinlik haritaları için ~{need:.0f} GB gerekli, "
                           f"{mvs.anchor} sürücüsünde {free:.0f} GB boş. Eski taramaları (projects klasörü) "
                           "silip ya da daha düşük kalite seçip tekrar deneyin.")
    step("DensifyPointCloud", ["-i", "scene.mvs", "-o", "scene_dense.mvs",
                               "--resolution-level", dense_level,
                               # Tower mode kicks in when views lack neighbours and crashed (0xC0000005)
                               # on a real desk scan; plain densification of the same scene succeeded.
                               "--tower-mode", 0,
                               # OpenMVS guesses a region of interest and drops views outside it: on a
                               # room walkthrough 171 of 269 frames got no neighbours and the whole bed
                               # and door side of the room was missing from the mesh.
                               "--estimate-roi", 0, "--crop-to-roi", 0], 0.55, densify_line)
    step("ReconstructMesh", ["-i", "scene_dense.mvs", "-o", "scene_mesh.ply", "--crop-to-roi", 0], 0.7)
    mesh = "scene_mesh.ply"
    if s.refine:
        faces = _face_count(mvs / mesh)
        keep = min(1.0, REFINE_MAX_FACES / faces) if faces else 1.0
        step("RefineMesh", ["-i", "scene_dense.mvs", "-m", mesh, "-o", "scene_refined.ply",
                            "--resolution-level", resolution_level(long_side, s.refine_px),
                            "--decimate", f"{keep:.4f}"], 0.85)
        mesh = "scene_refined.ply"

    report = mesh_clean.clean(mvs / mesh, mvs / "scene_clean.ply")
    runner.log(f"Mesh temizlendi: {report.pieces_before} parçadan {report.pieces_kept} ana parça kaldı, "
               f"{report.faces_before - report.faces_after:,} üçgenlik kopuk parça atıldı.")
    mesh = "scene_clean.ply"

    out.mkdir(parents=True, exist_ok=True)
    textured = out.resolve() / "model.obj"

    def texture(leveling: bool, done: float) -> None:
        for old in _textures(textured):
            old.unlink(missing_ok=True)
        step("TextureMesh", ["-i", "scene_dense.mvs", "-m", mesh, "-o", textured,
                             "--export-type", "obj", "--resolution-level", s.texture_level,
                             "--max-texture-size", s.max_texture, "--empty-color", EMPTY_COLOR_ARG,
                             "--global-seam-leveling", int(leveling), "--local-seam-leveling", int(leveling)],
             done)

    # Seam leveling blackened 23-46% of the texture on every real scan so far (5 of 5), and the
    # retry cost a full extra texturing pass (5.5 min on a room), so it is not attempted.
    texture(False, 1.0)
    leveling = False
    progress(1.0)
    if not textured.exists():
        raise RuntimeError("TextureMesh çıktı üretmedi; logu kontrol edin.")
    return MvsResult(textured, mvs / mesh, mvs / "scene_dense.mvs", leveling)
