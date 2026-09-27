"""Gaussian splat training with Brush on the same undistorted COLMAP workspace the mesh uses.

Brush prints nothing while training, so progress comes from counting its periodic checkpoints.
Splats stay in COLMAP coordinates; the viewer applies the mesh's upright/scale frame as a matrix,
which keeps view-dependent (SH) colour correct without rotating coefficients here.
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .orient import Frame, camera_views
from .tools import BRUSH, Runner, ToolError

CHECKPOINTS = 10          # exports during training; each one advances the progress bar
CHECKPOINT_POLL_S = 2.0
BOUNDS_PERCENTILE = 1.0   # far background floaters would otherwise make the viewer start zoomed out
START_LOOK_M = 1.5        # orbit pivot distance in front of the starting camera
MAX_SPLATS = 1_000_000    # desk/room scans land at 0.2-0.3 M; the 10 M default only lets memory run away
LEAN_MAX_SPLATS = 500_000
MIN_FREE_GB = 2.0         # Brush's measured floor plus headroom; below it training cannot finish
BYTES_PER_PIXEL = 12      # float32 RGB per training frame, held in RAM for the whole run
BRUSH_BASE_GB = 1.0       # runtime, splats and GPU staging (measured floor ~1.3 GB incl. a small dataset)
MEMORY_SAFETY = 0.8       # only this share of the currently committable memory is planned for
SPLAT_GB_PER_100K = 0.15  # growth during training: 1.4 -> 2.0 GB over 4000 steps as splats multiply
ACCEPT_PARTIAL = 0.6      # a crash after this share of the steps keeps the last checkpoint
# (resolution or 0 for the quality's own maximum, keep every Nth frame), best first. More viewpoints
# beat more pixels: a room trained on 129 of 259 frames at 1600 px came out smeared in the foreground.
PLAN_ORDER = [(0, 1), (1280, 1), (960, 1), (0, 2), (1280, 2), (960, 2), (720, 1), (720, 2), (960, 3), (720, 3), (720, 4)]


def _watch(folder: Path, total: int, done: threading.Event, progress) -> None:
    while not done.wait(CHECKPOINT_POLL_S):
        progress(min(0.99, len(list(folder.glob("ckpt_*.ply"))) / total))


@dataclass(frozen=True)
class MemoryPlan:
    max_resolution: int
    subsample_frames: int     # 1 = every frame, 2 = every other frame
    max_splats: int


def brush_memory_gb(frames: int, width: int, height: int, max_resolution: int, subsample: int,
                    max_splats: int = LEAN_MAX_SPLATS) -> float:
    """Brush keeps every training frame in RAM as float RGB: a failed 1600 px allocation was exactly
    1600 x 899 x 12 bytes. 259 such frames need ~4.5 GB on top of Brush's own ~1 GB."""
    scale = min(1.0, max_resolution / max(width, height, 1))
    per_frame = (width * scale) * (height * scale) * BYTES_PER_PIXEL
    return (BRUSH_BASE_GB + max_splats / 100_000 * SPLAT_GB_PER_100K
            + math.ceil(frames / subsample) * per_frame / 2**30)


def plan_for_memory(budget_gb: float, frames: int, width: int, height: int, max_resolution: int) -> MemoryPlan | None:
    """Sharpest resolution / most frames that fit `budget_gb`, or None if even the leanest does not.
    Frames are kept before resolution: missing viewpoints smear the splat, lower resolution only
    softens it; that is the order tried."""
    for res, sub in PLAN_ORDER:
        res = min(res, max_resolution) if res else max_resolution
        splats = MAX_SPLATS if sub == 1 and res >= max_resolution else LEAN_MAX_SPLATS
        if brush_memory_gb(frames, width, height, res, sub, splats) <= budget_gb:
            return MemoryPlan(res, sub, splats)
    return None


def run(dense: Path, out: Path, steps: int, plan: MemoryPlan, runner: Runner, progress) -> Path:
    """Trains a splat from `dense` (images/ + sparse/) and returns out/splat.ply."""
    ckpt_dir = out / "_splat_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for old in ckpt_dir.glob("*.ply"):
        old.unlink()

    every = max(1, steps // CHECKPOINTS)
    done = threading.Event()
    watcher = threading.Thread(target=_watch, args=(ckpt_dir, CHECKPOINTS, done, progress), daemon=True)
    watcher.start()
    failure = None
    try:
        runner.run(BRUSH, [dense.resolve(), "--total-steps", steps, "--export-every", every,
                           "--max-resolution", plan.max_resolution, "--max-splats", plan.max_splats,
                           "--subsample-frames", plan.subsample_frames,
                           "--export-path", ckpt_dir.resolve(), "--export-name", "ckpt_{iter}.ply"])
    except ToolError as exc:
        failure = exc
    finally:
        done.set()
        watcher.join(timeout=5)

    checkpoints = sorted(ckpt_dir.glob("ckpt_*.ply"), key=lambda p: int(p.stem.split("_")[1]))
    if failure is not None:
        reached = int(checkpoints[-1].stem.split("_")[1]) if checkpoints else 0
        # Training mostly sharpens detail late on; a checkpoint past this point is kept rather than
        # throwing away many minutes when memory runs out near the end.
        if reached < ACCEPT_PARTIAL * steps:
            raise failure
        runner.log(f"UYARI: Brush {reached}/{steps} adımda durdu (bellek); son ara kayıt kullanılıyor.")
    if not checkpoints:
        raise RuntimeError("Brush splat dosyası üretmedi; logu kontrol edin.")
    final = out / "splat.ply"
    checkpoints[-1].replace(final)
    for leftover in ckpt_dir.glob("*"):
        leftover.unlink()
    ckpt_dir.rmdir()
    progress(1.0)
    return final


def splat_centers(ply: Path) -> np.ndarray:
    """Gaussian centres from a 3DGS PLY (all-float vertex properties, any order)."""
    data = ply.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    props = [ln.split()[-1] for ln in header if ln.startswith("property")]
    if not all(ln.split()[1] in ("float", "float32") for ln in header if ln.startswith("property")):
        raise ValueError(f"{ply.name}: beklenmeyen splat PLY alanları")
    count = int(next(ln for ln in header if ln.startswith("element vertex")).split()[-1])
    table = np.frombuffer(data, "<f4", count * len(props), end).reshape(count, len(props))
    return table[:, [props.index(k) for k in ("x", "y", "z")]].astype(np.float64)


def start_view(frame: Frame, centers: np.ndarray, forwards: np.ndarray) -> dict:
    """A real capture viewpoint: splats only look right near where the video was shot."""
    pick = int(np.argmin(np.linalg.norm(centers - np.median(centers, axis=0), axis=1)))
    rotation = frame.matrix()[:3, :3] / frame.scale
    position = frame.apply(centers[pick:pick + 1])[0]
    target = position + rotation @ forwards[pick] * START_LOOK_M
    return {"position": position.round(4).tolist(), "target": target.round(4).tolist()}


def capture_views(frame: Frame, centers: np.ndarray, forwards: np.ndarray) -> list[dict]:
    """Every capture viewpoint (in capture order), so the viewer can step through the shoot."""
    rotation = frame.matrix()[:3, :3] / frame.scale
    positions = frame.apply(centers)
    targets = positions + forwards @ rotation.T * START_LOOK_M
    return [{"position": p.round(3).tolist(), "target": t.round(3).tolist()} for p, t in zip(positions, targets)]


def write_frame(frame: Frame, splat_ply: Path, images_bin: Path, path: Path) -> None:
    """Viewer metadata: COLMAP -> upright-metres matrix (row-major), robust bounds and a start view."""
    placed = frame.apply(splat_centers(splat_ply))
    lo, hi = np.percentile(placed, BOUNDS_PERCENTILE, axis=0), np.percentile(placed, 100 - BOUNDS_PERCENTILE, axis=0)
    path.write_text(json.dumps({
        "matrix": frame.matrix().round(8).ravel().tolist(),
        "bounds": [lo.round(4).tolist(), hi.round(4).tolist()],
        "view": start_view(frame, *camera_views(images_bin)),
        "views": capture_views(frame, *camera_views(images_bin)),
        "splats": len(placed),
    }), encoding="utf-8")


SH_C0 = 0.28209479177387814      # degree-0 spherical-harmonic constant: colour = 0.5 + C0 * f_dc


def read_ply(ply: Path) -> tuple[str, list[str], np.ndarray]:
    """(header text, property names, float32 table) of a standard 3DGS PLY (all-float properties)."""
    data = ply.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii")
    lines = header.splitlines()
    if any(ln.startswith("property") and ln.split()[1] not in ("float", "float32") for ln in lines):
        raise ValueError(f"{ply.name}: beklenmeyen splat PLY alanları")
    props = [ln.split()[-1] for ln in lines if ln.startswith("property")]
    count = int(next(ln for ln in lines if ln.startswith("element vertex")).split()[-1])
    table = np.frombuffer(data, "<f4", count * len(props), end).reshape(count, len(props))
    return header, props, table


def write_ply(ply: Path, header: str, table: np.ndarray) -> None:
    lines = [f"element vertex {len(table)}" if ln.startswith("element vertex") else ln
             for ln in header.splitlines()]
    ply.write_bytes(("\n".join(lines) + "\n").encode("ascii") + table.astype("<f4").tobytes())


def splat_points(ply: Path, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """World-space centres (m), RGB 0..1, opacity and largest axis size (m) per splat."""
    _, props, table = read_ply(ply)
    col = {name: i for i, name in enumerate(props)}
    xyz = table[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
    world = np.c_[xyz, np.ones(len(xyz))] @ matrix.T
    scale = np.cbrt(abs(np.linalg.det(matrix[:3, :3])))
    rgb = np.clip(0.5 + SH_C0 * table[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]], 0, 1)
    opacity = 1.0 / (1.0 + np.exp(-table[:, col["opacity"]]))
    size = np.exp(table[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]).max(axis=1) * scale
    return world[:, :3], rgb, opacity, size
