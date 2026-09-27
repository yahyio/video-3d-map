"""Structure-from-Motion with COLMAP: frames -> camera poses -> undistorted images for dense MVS."""
from __future__ import annotations

import shutil
import struct
from pathlib import Path

from . import retrieval
from .tools import COLMAP, WORKER_THREADS, Runner

# Small sets (phone photo captures) are matched all-pairs: cheap, and nothing can be missed.
# Above this, all-pairs cost 473 s for 300 frames, so retrieval picks the pairs instead.
EXHAUSTIVE_MAX_FRAMES = 120
# Frame budgets cap captures at this many frames. 300 turned a detailed 6.6-minute walkthrough
# into one frame per 1.3 s; retrieval matching keeps 500 affordable.
MAX_FRAMES = 500
GLOBAL_MIN_SHARE = 0.5


def registered_images(model_dir: Path) -> int:
    """images.bin starts with a uint64 count of registered images."""
    path = model_dir / "images.bin"
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def largest_model(sparse_dir: Path) -> Path | None:
    models = [d for d in sparse_dir.iterdir() if d.is_dir()] if sparse_dir.exists() else []
    models = [m for m in models if registered_images(m) > 0]
    return max(models, key=registered_images, default=None)


def _reset(*paths: Path) -> None:
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def _map(frames: Path, db: Path, sparse: Path, total: int, runner: Runner) -> Path | None:
    """Global mapper first (157 s vs 203 s and 269 vs 247 of 300 frames on a room walkthrough);
    the incremental mapper is the fallback when the global solve leaves most frames out."""
    runner.run(COLMAP, ["global_mapper", "--database_path", db, "--image_path", frames, "--output_path", sparse,
                        "--GlobalMapper.num_threads", WORKER_THREADS])
    best = largest_model(sparse)
    if best is not None and registered_images(best) >= GLOBAL_MIN_SHARE * total:
        return best
    runner.log("Hızlı yöntem sahnenin çoğunu bağlayamadı; adım adım yönteme geçiliyor.")
    _reset(sparse)
    sparse.mkdir(parents=True)
    runner.run(COLMAP, ["mapper", "--database_path", db, "--image_path", frames, "--output_path", sparse,
                        "--Mapper.num_threads", WORKER_THREADS])
    return largest_model(sparse)


def run(frames: Path, work: Path, max_image_size: int, runner: Runner, progress) -> tuple[Path, int]:
    """Returns (undistorted COLMAP workspace, number of registered frames)."""
    db, sparse = work / "database.db", work / "sparse"
    aligned, dense = work / "sparse_aligned", work / "dense"
    _reset(db, sparse, aligned, dense)
    sparse.mkdir(parents=True)
    aligned.mkdir(parents=True)

    runner.run(COLMAP, ["feature_extractor", "--database_path", db, "--image_path", frames,
                        "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "OPENCV",
                        "--FeatureExtraction.use_gpu", 1, "--FeatureExtraction.num_threads", WORKER_THREADS])
    progress(0.2)
    total = len(list(frames.glob("*.jpg")))
    match_opts = ["--FeatureMatching.use_gpu", 1, "--FeatureMatching.num_threads", WORKER_THREADS]
    if total <= EXHAUSTIVE_MAX_FRAMES:
        runner.run(COLMAP, ["exhaustive_matcher", "--database_path", db, *match_opts])
    else:
        # Neighbours plus look-alike frames: 6.2k instead of 45k pairs, same registration (271 vs 269).
        pairs = work / "pairs.txt"
        runner.log(f"{retrieval.write_pairs(db, pairs):,} kare çifti seçildi (tümü: {total * (total - 1) // 2:,}).")
        runner.run(COLMAP, ["matches_importer", "--database_path", db, "--match_list_path", pairs,
                            "--match_type", "pairs", *match_opts])
    progress(0.4)
    best = _map(frames, db, sparse, total, runner)
    progress(0.8)

    if best is None:
        raise RuntimeError("Kamera hareketi çözülemedi. Video çok hızlı, bulanık ya da dokusuz olabilir. "
                           "Daha yavaş ve yana kayarak çekmeyi deneyin.")
    count = registered_images(best)
    if count < max(8, total // 4):
        raise RuntimeError(f"Sadece {count}/{total} kare eşleşti; sahne yeniden oluşturulamaz. "
                           "Daha yavaş ve bol örtüşmeli çekim deneyin.")

    # Put gravity on a fixed axis so exports stand upright (phone footage is shot upright).
    runner.run(COLMAP, ["model_orientation_aligner", "--image_path", frames, "--input_path", best,
                        "--output_path", aligned, "--method", "IMAGE-ORIENTATION"])
    progress(0.9)
    runner.run(COLMAP, ["image_undistorter", "--image_path", frames, "--input_path", aligned,
                        "--output_path", dense, "--output_type", "COLMAP", "--num_threads", WORKER_THREADS,
                        "--max_image_size", max_image_size])
    progress(1.0)
    return dense, count
