"""Tiled splat training: the scene is cut into floor-plan tiles, each tile is trained by Brush on
every frame that sees it, at full resolution, and the tiles' own regions are merged into one splat.

One Brush run has to fit all frames in RAM (float RGB per frame), so a 458-frame room had to train
at 960 px with a 500 k splat cap and came out soft. Tiles keep full resolution and give every
region its own splat budget; overlapping margins hide the seams, and each tile keeps only the
splats whose centres fall inside its own region.
"""
from __future__ import annotations

import _winapi
import json
import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import splat
from .orient import Frame
from .splat import read_ply, write_ply
from .tools import Runner, ToolError

MARGIN_SHARE = 0.25          # tile margin as a share of its size (overlap hides seams)
MIN_MARGIN_M = 0.3
MIN_SHARED_POINTS = 20       # an image joins a tile when it sees this many of the tile's points
MIN_SCORE_SHARE = 0.05       # ...and at least this share of the best-covering image's count
TILE_COUNTS = (1, 2, 3, 4, 6, 8, 9, 12)
TILE_SPLATS = 750_000
STEPS_PER_FRAME = 150         # training steps per tile frame; small tiles need far fewer than 30 k
MIN_TILE_STEPS = 8_000
MIN_TILE_FRAMES = 20
PLAN_FILE = "plan.json"
RETRY_FRAME_SHARE = 0.6      # a tile that still runs out of memory retries with fewer frames


# --- COLMAP binary models ------------------------------------------------------------------
@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    raw: bytes                   # the full record, written back unchanged into tile models
    point_ids: np.ndarray        # observed 3-D point ids (-1 entries dropped)


def read_images(path: Path) -> list[ImageRecord]:
    data = path.read_bytes()
    (count,), pos = struct.unpack_from("<Q", data, 0), 8
    records = []
    for _ in range(count):
        start = pos
        image_id = struct.unpack_from("<I", data, pos)[0]
        pos = data.index(b"\0", pos + 64) + 1                     # id, qvec, tvec, camera id, name
        (npts,) = struct.unpack_from("<Q", data, pos)
        obs = np.frombuffer(data, np.dtype([("x", "<f8"), ("y", "<f8"), ("pid", "<i8")]), npts, pos + 8)
        pos += 8 + npts * 24
        records.append(ImageRecord(image_id, data[start:pos], obs["pid"][obs["pid"] >= 0].copy()))
    return records


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray, list[bytes]]:
    """(point ids, xyz, raw records)."""
    data = path.read_bytes()
    (count,), pos = struct.unpack_from("<Q", data, 0), 8
    ids, xyz, raws = np.empty(count, np.int64), np.empty((count, 3)), []
    for i in range(count):
        start = pos
        ids[i] = struct.unpack_from("<Q", data, pos)[0]
        xyz[i] = struct.unpack_from("<3d", data, pos + 8)
        pos += 8 + 24 + 3 + 8
        (track,) = struct.unpack_from("<Q", data, pos)
        pos += 8 + track * 8
        raws.append(data[start:pos])
    return ids, xyz, raws


def write_records(path: Path, raws: list[bytes]) -> None:
    path.write_bytes(struct.pack("<Q", len(raws)) + b"".join(raws))


# --- partition -----------------------------------------------------------------------------
def grid_shape(extent_a: float, extent_b: float, parts: int) -> tuple[int, int]:
    """Rows x cols close to the footprint's aspect ratio, with rows * cols >= parts."""
    if parts <= 1:
        return 1, 1
    ratio = max(extent_a, 1e-9) / max(extent_b, 1e-9)
    rows = min(parts, max(1, round(math.sqrt(parts * ratio))))
    return rows, math.ceil(parts / rows)


@dataclass(frozen=True)
class Tile:
    lo: np.ndarray               # own region on the floor plan (x, z); outer tiles extend to +-inf
    hi: np.ndarray
    image_ids: list[int]


def _edges(values: np.ndarray, parts: int) -> np.ndarray:
    cuts = np.quantile(values, np.linspace(0, 1, parts + 1)[1:-1]) if parts > 1 else np.array([])
    return np.concatenate([[-np.inf], cuts, [np.inf]])


def _expanded(lo: np.ndarray, hi: np.ndarray, core_lo: np.ndarray, core_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite_lo = np.where(np.isfinite(lo), lo, core_lo)
    finite_hi = np.where(np.isfinite(hi), hi, core_hi)
    margin = np.maximum((finite_hi - finite_lo) * MARGIN_SHARE, MIN_MARGIN_M)
    return lo - margin, hi + margin


def partition(points_xz: np.ndarray, point_ids: np.ndarray, images: list[ImageRecord], tiles: int,
              frame_cap: int) -> list[Tile]:
    """Split the floor plan into `tiles` equal-point cells and pick the images that see each one."""
    core_lo, core_hi = np.percentile(points_xz, 2, axis=0), np.percentile(points_xz, 98, axis=0)
    nx, nz = grid_shape(*(core_hi - core_lo), tiles)
    index = {int(pid): k for k, pid in enumerate(point_ids)}
    rows_per_image = [np.array([index[p] for p in img.point_ids.tolist() if p in index], int) for img in images]
    x_edges = _edges(points_xz[:, 0], nx)
    out = []
    for i in range(nx):
        in_col = (points_xz[:, 0] >= x_edges[i]) & (points_xz[:, 0] < x_edges[i + 1])
        z_edges = _edges(points_xz[in_col, 1] if in_col.any() else points_xz[:, 1], nz)
        for j in range(nz):
            lo = np.array([x_edges[i], z_edges[j]])
            hi = np.array([x_edges[i + 1], z_edges[j + 1]])
            elo, ehi = _expanded(lo, hi, core_lo, core_hi)
            inside = np.all((points_xz >= elo) & (points_xz <= ehi), axis=1)
            scores = np.array([int(inside[rows].sum()) for rows in rows_per_image])
            need = max(MIN_SHARED_POINTS, MIN_SCORE_SHARE * scores.max()) if scores.size else 0
            # A single tile is the whole scene: every registered frame is a viewpoint worth keeping.
            chosen = np.arange(len(images)) if tiles == 1 else np.flatnonzero(scores >= need)
            chosen = chosen[np.argsort(-scores[chosen], kind="stable")][:frame_cap]
            out.append(Tile(lo, hi, sorted(images[k].image_id for k in chosen)))
    return out


def choose_partition(points_xz, point_ids, images, frame_cap: int) -> list[Tile]:
    """Fewest tiles whose image sets all fit `frame_cap` (the most frames one Brush run can hold)."""
    for n in TILE_COUNTS:
        tiles = partition(points_xz, point_ids, images, n, frame_cap + 1)
        if max(len(t.image_ids) for t in tiles) <= frame_cap:
            return tiles
    return partition(points_xz, point_ids, images, TILE_COUNTS[-1], frame_cap)


# --- training ------------------------------------------------------------------------------
def frame_cap(budget_gb: float, width: int, height: int, resolution: int) -> int:
    scale = min(1.0, resolution / max(width, height))
    per_frame = width * scale * height * scale * splat.BYTES_PER_PIXEL / 2**30
    room = budget_gb - splat.BRUSH_BASE_GB - TILE_SPLATS / 100_000 * splat.SPLAT_GB_PER_100K
    return max(0, int(room / per_frame))


def _tile_dataset(dense: Path, tile: Tile, images: list[ImageRecord], raws: list[bytes],
                  points_xz: np.ndarray, core_lo, core_hi, folder: Path) -> None:
    sparse = folder / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dense / "sparse" / "cameras.bin", sparse / "cameras.bin")
    wanted = set(tile.image_ids)
    write_records(sparse / "images.bin", [img.raw for img in images if img.image_id in wanted])
    elo, ehi = _expanded(tile.lo, tile.hi, core_lo, core_hi)
    inside = np.all((points_xz >= elo) & (points_xz <= ehi), axis=1)
    write_records(sparse / "points3D.bin", [raw for raw, keep in zip(raws, inside) if keep])
    link = folder / "images"
    if not link.exists():
        _winapi.CreateJunction(str((dense / "images").resolve()), str(link))   # no copies of the frames


def merge(parts: list[tuple[Path, Tile]], matrix: np.ndarray, out: Path) -> int:
    """Concatenate each tile's splats that fall inside its own region; returns the splat count."""
    header, props, tables = None, None, []
    for ply, tile in parts:
        h, p, table = read_ply(ply)
        if props is None:
            header, props = h, p
        table = table[:, [p.index(name) for name in props]]
        col = {name: i for i, name in enumerate(props)}
        xyz = table[:, [col["x"], col["y"], col["z"]]].astype(np.float64)
        world = (np.c_[xyz, np.ones(len(xyz))] @ matrix.T)[:, [0, 2]]
        own = np.all((world >= tile.lo) & (world < tile.hi), axis=1)
        tables.append(table[own])
    merged = np.concatenate(tables)
    write_ply(out, header, merged)
    return len(merged)


def save_plan(path: Path, tiles: list[Tile]) -> None:
    """The partition is kept so an interrupted run (a closed session, a crash) resumes with the same
    tiles and skips the ones already trained, instead of starting hours of training over."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"lo": t.lo.tolist(), "hi": t.hi.tolist(), "images": t.image_ids}
                                for t in tiles]), encoding="utf-8")


def load_plan(path: Path) -> list[Tile] | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Tile(np.array(r["lo"], float), np.array(r["hi"], float), list(r["images"])) for r in rows]


def _remove_links(work: Path) -> None:
    if not work.exists():
        return
    for folder in work.iterdir():
        link = folder / "images"
        if link.exists():
            link.rmdir()                     # removes the junction only, never the frames


def train(dense: Path, out: Path, frame: Frame, steps: int, resolution: int, budget_gb: float,
          runner: Runner, progress: Callable[[float], None]) -> Path:
    from .mvs import _image_size

    images = read_images(dense / "sparse" / "images.bin")
    ids, xyz, raws = read_points(dense / "sparse" / "points3D.bin")
    points_xz = frame.apply(xyz)[:, [0, 2]]
    width, height = _image_size(dense / "images")
    work = out / "_tiles"
    tiles = load_plan(work / PLAN_FILE)
    if tiles is None:
        cap = frame_cap(budget_gb, width, height, resolution)
        if cap < MIN_TILE_FRAMES:
            raise RuntimeError(f"Tam çözünürlükte eğitim için bellek yetmiyor: {budget_gb:.1f} GB ayrılabilir.")
        tiles = choose_partition(points_xz, ids, images, cap)
        _remove_links(work)
        shutil.rmtree(work, ignore_errors=True)
        save_plan(work / PLAN_FILE, tiles)
    else:
        runner.log("Yarım kalan bölgeli eğitim bulundu; biten bölgeler yeniden eğitilmeyecek.")
    runner.log(f"Splat: {len(tiles)} bölge, {resolution}px, bölge başına {TILE_SPLATS:,} splat, "
               f"bölge kare sayıları {[len(t.image_ids) for t in tiles]}.")
    core_lo, core_hi = np.percentile(points_xz, 2, axis=0), np.percentile(points_xz, 98, axis=0)
    parts = []
    try:
        for k, tile in enumerate(tiles):
            folder = work / f"tile_{k}"
            if (folder / "splat.ply").exists():
                parts.append((folder / "splat.ply", tile))      # finished before an interruption
                progress((k + 1) / len(tiles))
                continue
            _tile_dataset(dense, tile, images, raws, points_xz, core_lo, core_hi, folder)
            report = lambda p, k=k: progress((k + p) / len(tiles))  # noqa: E731
            plan = splat.MemoryPlan(resolution, 1, TILE_SPLATS)
            tile_steps = int(np.clip(len(tile.image_ids) * STEPS_PER_FRAME, MIN_TILE_STEPS, steps))
            try:
                ply = splat.run(folder, folder, tile_steps, plan, runner, report)
            except ToolError:
                keep = max(MIN_TILE_FRAMES, int(len(tile.image_ids) * RETRY_FRAME_SHARE))
                runner.log(f"UYARI: bölge {k + 1} için bellek yetmedi; {keep} kareyle yeniden deneniyor.")
                tile = Tile(tile.lo, tile.hi, tile.image_ids[:keep])
                _tile_dataset(dense, tile, images, raws, points_xz, core_lo, core_hi, folder)
                ply = splat.run(folder, folder, tile_steps, plan, runner, report)
            parts.append((ply, tile))
            runner.log(f"Bölge {k + 1}/{len(tiles)} bitti ({len(tile.image_ids)} kare, {tile_steps:,} adım).")
        total = merge(parts, frame.matrix(), out / "splat.ply")
    finally:
        _remove_links(work)
    runner.log(f"{len(tiles)} bölge birleştirildi: {total:,} splat.")
    shutil.rmtree(work, ignore_errors=True)
    progress(1.0)
    return out / "splat.ply"
