"""End-to-end run: video or photos -> frames -> camera poses -> Gaussian splat and/or textured mesh."""
from __future__ import annotations

import json
import shutil
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from . import frames, meshio, mvs, orient, photos, sfm, splat, system, tiles
from .tools import ROOT, Runner, ToolError

PROJECTS = ROOT / "projects"
RUN_LOCK = PROJECTS / ".running"
# Page file is pre-grown to this much free commit before splat training (disk and page-file cap permitting).
SPLAT_COMMIT_TARGET_GB = 12.0
# COLMAP/OpenMVS builds may open files through narrow-char APIs, so project paths stay ASCII.
_TR_ASCII = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")


@dataclass(frozen=True)
class Quality:
    label: str
    frames: frames.FrameBudget
    image_size: int
    mvs: mvs.MvsSettings
    splat_steps: int
    splat_px: int


QUALITIES = {
    # dense_px / refine_px are targets: small (e.g. messenger-compressed) videos run at full size.
    # Normal skips RefineMesh: it took 26 of 93 minutes on a room walkthrough; High keeps it.
    # splat_px: Brush holds every frame as float RGB (12 bytes/pixel) in RAM; 1280 instead of 1600
    # cuts that by a third, which kept it from running out of memory next to a game and a browser.
    "fast": Quality("Hızlı", frames.FrameBudget(1.5, 40, 100), 1280, mvs.MvsSettings(640, False, 640, 1, 4096), 7000, 960),
    "normal": Quality("Normal", frames.FrameBudget(2.0, 80, 300), 1600, mvs.MvsSettings(1000, False, 640, 0, 8192), 15000, 1280),
    "high": Quality("Yüksek", frames.FrameBudget(4.0, 120, sfm.MAX_FRAMES), 2400, mvs.MvsSettings(1600, True, 1200, 0, 8192), 30000, 1920),
}

# Share of the progress bar each stage owns.
STAGES = [("frames", "Kareler seçiliyor", 0.05), ("sfm", "Kamera konumları çözülüyor", 0.15),
          ("mvs", "Yoğun model ve doku", 0.60), ("orient", "Model hizalanıyor", 0.03),
          ("splat", "Gaussian splat eğitiliyor", 0.30)]


@dataclass(frozen=True)
class Options:
    video: Path
    quality: str = "normal"
    camera_height_m: float = 1.5
    mesh: bool = True       # OpenMVS textured mesh (OBJ); costs most of the runtime and CPU
    splat: bool = True
    keep_work: bool = False


@dataclass(frozen=True)
class Result:
    folder: Path
    model: Path | None
    splat: Path | None
    info: dict


class Progress:
    """Maps per-stage 0..1 progress onto one overall bar and reports the active stage name."""

    def __init__(self, report: Callable[[float, str], None], skip: frozenset[str] = frozenset()):
        self._stages = [s for s in STAGES if s[0] not in skip]
        self._total = sum(w for _, _, w in self._stages)
        self._report = report

    def stage(self, key: str) -> Callable[[float], None]:
        start = 0.0
        for k, label, weight in self._stages:
            if k == key:
                self._report(start / self._total, label)
                return lambda p, s=start, w=weight, lb=label: self._report((s + w * p) / self._total, lb)
            start += weight
        raise KeyError(key)


def new_project(video: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ascii_stem = unicodedata.normalize("NFKD", video.stem.translate(_TR_ASCII)).encode("ascii", "ignore").decode()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in ascii_stem)[:40].strip("_") or "tarama"
    folder = PROJECTS / f"{safe}_{stamp}"
    folder.mkdir(parents=True)
    return folder


def _check_memory() -> None:
    if problem := system.memory_problem():
        raise RuntimeError(problem)


def run(opts: Options, runner: Runner, report: Callable[[float, str], None]) -> Result:
    if opts.quality not in QUALITIES:
        raise ValueError(f"Bilinmeyen kalite: {opts.quality}")
    if not (opts.video.is_file() or opts.video.is_dir()):
        raise FileNotFoundError(f"Video ya da fotoğraf klasörü bulunamadı: {opts.video}")
    if not (opts.mesh or opts.splat):
        raise ValueError("En az bir çıktı seçin: Mesh ya da Gaussian Splat.")
    _check_memory()
    with system.RunLock(RUN_LOCK):
        return _run(opts, runner, report)


def _train_splat(dense: Path, out: Path, frame: orient.Frame, q: Quality, runner: Runner,
                 progress, fatal: bool) -> tuple[Path | None, str | None]:
    """Returns (splat.ply, error). When a mesh already exists the splat is optional, so a
    failure is reported instead of throwing away the finished mesh."""
    try:
        # Only memory committable right now counts: Windows grows the page file too slowly for
        # Brush's burst of frame allocations, and planning with that growth crashed a 259-frame run.
        committable = system.grow_commit(SPLAT_COMMIT_TARGET_GB)
        # Brush reads every frame each few steps, so frames must fit in physical RAM, not just commit.
        free = min(committable, system.free_memory_gb()[0])
        budget = free * splat.MEMORY_SAFETY
        runner.log(f"Splat: {free:.1f} GB ayrılabilir bellek; sahne gerekirse bölgelere ayrılıp tam "
                   f"çözünürlükte ({q.splat_px}px) eğitilecek.")
        if tiles.frame_cap(budget, *mvs._image_size(dense / "images"), q.splat_px) < tiles.MIN_TILE_FRAMES:
            raise RuntimeError(
                f"Splat için yeterli bellek yok: şu an {free:.1f} GB ayrılabilir. "
                f"En çok bellek kullananlar: {system.describe_top_users()}.")
        ply = tiles.train(dense, out, frame, q.splat_steps, q.splat_px, budget, runner, progress)
        splat.write_frame(frame, ply, dense / "sparse" / "images.bin", out / "splat_frame.json")
        return ply, None
    except (RuntimeError, OSError) as exc:
        if fatal:
            raise
        runner.log(f"UYARI: Gaussian splat üretilemedi, mesh sonuçları korundu.\n{exc}")
        return None, str(exc).splitlines()[0]


def _run(opts: Options, runner: Runner, report: Callable[[float, str], None]) -> Result:
    q = QUALITIES[opts.quality]
    wanted = {"mvs": opts.mesh, "orient": opts.mesh, "splat": opts.splat}
    skip = frozenset(k for k, on in wanted.items() if not on)
    prog = Progress(report, skip)
    folder = new_project(opts.video)
    work, out = folder / "work", folder / "output"
    started = time.time()
    runner.log(f"Proje klasörü: {folder}")
    if opts.video.is_dir():
        kept, dropped = photos.import_photos(opts.video, work / "frames", runner, prog.stage("frames"))
        runner.log(f"{kept} fotoğraf alındı" + (f", {dropped} bulanık fotoğraf atıldı." if dropped else "."))
    else:
        if warning := frames.low_resolution_warning(frames.video_size(opts.video)):
            runner.log(f"UYARI: {warning}")
        kept = frames.extract(opts.video, work / "frames", q.frames, runner, prog.stage("frames"))
        runner.log(f"{kept} net kare seçildi.")
    dense, registered = sfm.run(work / "frames", work, q.image_size, runner, prog.stage("sfm"))
    runner.log(f"{registered}/{kept} kare için kamera konumu bulundu.")
    return _reconstruct(opts, folder, dense, kept, registered, started, runner, report, prog)


def resume_from_cameras(folder: Path, opts: Options, runner: Runner,
                        report: Callable[[float, str], None]) -> Result:
    """Reruns everything after camera solving for a scan whose later stage failed (disk, memory),
    reusing work/dense so the frame and SfM stages are not repeated."""
    dense = folder / "work" / "dense"
    if not (dense / "sparse" / "images.bin").exists():
        raise FileNotFoundError(f"{folder.name}: kamera verisi yok, bu tarama baştan yapılmalı.")
    if not (opts.mesh or opts.splat):
        raise ValueError("En az bir çıktı seçin: Mesh ya da Gaussian Splat.")
    _check_memory()
    for stale in dense.glob("*.dmap*"):
        stale.unlink()
    registered = sfm.registered_images(dense / "sparse")
    kept = max(len(list((folder / "work" / "frames").glob("*.jpg"))), registered)   # frames/ may be pruned
    wanted = {"frames": False, "sfm": False, "mvs": opts.mesh, "orient": opts.mesh, "splat": opts.splat}
    prog = Progress(report, frozenset(k for k, on in wanted.items() if not on))
    with system.RunLock(RUN_LOCK):
        return _reconstruct(opts, folder, dense, kept, registered, time.time(), runner, report, prog)


def _reconstruct(opts: Options, folder: Path, dense: Path, kept: int, registered: int, started: float,
                 runner: Runner, report: Callable[[float, str], None], prog: Progress) -> Result:
    q = QUALITIES[opts.quality]
    work, out = folder / "work", folder / "output"
    cams = orient.camera_centers(dense / "sparse" / "images.bin")

    result = faces = None
    if opts.mesh:
        result = mvs.run(dense, out, q.mvs, runner, prog.stage("mvs"))
        step = prog.stage("orient")
        verts, faces = meshio.read_ply(result.mesh_ply)
        frame = orient.estimate(verts, cams, opts.camera_height_m, faces)
        meshio.transform_obj(result.textured_obj, frame.apply)
        step(1.0)
    else:
        # Splat-only: COLMAP's sparse points are enough to find the floor, level and scale the scan.
        verts = orient.sparse_points(dense / "sparse" / "points3D.bin")
        frame = orient.estimate(verts, cams, opts.camera_height_m)

    splat_ply = splat_error = None
    if opts.splat:
        splat_ply, splat_error = _train_splat(dense, out, frame, q, runner, prog.stage("splat"),
                                              fatal=result is None)
    elif (out / "splat.ply").exists():
        # A splat from an earlier preview run: re-align it to this (mesh-derived) frame so both line up.
        splat_ply = out / "splat.ply"
        splat.write_frame(frame, splat_ply, dense / "sparse" / "images.bin", out / "splat_frame.json")

    # Percentiles keep a few floating fragments from inflating the reported footprint.
    placed = frame.apply(verts)
    size = np.percentile(placed, 99, axis=0) - np.percentile(placed, 1, axis=0)
    info = {
        "video": str(opts.video), "options": {**asdict(opts), "video": str(opts.video)},
        "frames": kept, "registered": registered, "triangles": int(len(faces)) if faces is not None else 0,
        "scale": frame.scale, "size_m": [round(float(s), 2) for s in sorted(size, reverse=True)],
        "seam_leveling": result.seam_leveling if result else None,
        "splat_error": splat_error,
        "minutes": round((time.time() - started) / 60, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    # A failed splat keeps work/ so resume_splat() can retry just that step later.
    if not opts.keep_work and splat_error is None:
        if result is None:
            _keep_cameras_only(work)
        else:
            shutil.rmtree(work, ignore_errors=True)
    report(1.0, "Bitti")
    return Result(folder, result.textured_obj if result else None, splat_ply, info)


def _keep_cameras_only(work: Path) -> None:
    """Splat previews keep work/dense (undistorted images + cameras, a few hundred MB) so that
    'Mesh'e çevir' can build the mesh later without redoing frames and camera solving."""
    for child in work.iterdir():
        if child.name != "dense":
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)


def can_convert_to_mesh(folder: Path) -> bool:
    return (folder / "work" / "dense" / "sparse" / "images.bin").exists()


def resume_splat(folder: Path, quality: str, runner: Runner, report: Callable[[float, str], None]) -> Result:
    """Retrains only the splat for a scan whose mesh finished but whose splat failed."""
    dense, out = folder / "work" / "dense", folder / "output"
    if not (dense / "sparse" / "images.bin").exists():
        raise FileNotFoundError(f"{folder.name}: kamera verisi yok, bu tarama baştan yapılmalı.")
    info_path = out / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    height = info.get("options", {}).get("camera_height_m", Options.camera_height_m)
    cams = orient.camera_centers(dense / "sparse" / "images.bin")
    mesh_ply = next((dense / n for n in ("scene_refined.ply", "scene_mesh.ply") if (dense / n).exists()), None)
    if mesh_ply:   # same inputs as the original run, so the splat lines up with the saved mesh
        verts, faces = meshio.read_ply(mesh_ply)
        frame = orient.estimate(verts, cams, height, faces)
    else:
        frame = orient.estimate(orient.sparse_points(dense / "sparse" / "points3D.bin"), cams, height)
    _check_memory()
    with system.RunLock(RUN_LOCK):
        ply, _ = _train_splat(dense, out, frame, QUALITIES[quality], runner,
                              Progress(report, frozenset({"frames", "sfm", "mvs", "orient"})).stage("splat"),
                              fatal=True)
    info = {**info, "splat_error": None}
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    model = out / "model.obj"
    report(1.0, "Bitti")
    return Result(folder, model if model.exists() else None, ply, info)
