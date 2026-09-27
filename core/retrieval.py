"""Picks which image pairs to match, so large captures skip the all-pairs cost.

Exhaustive matching of 300 frames took 473 s on a real 3-minute room video. Neighbours in capture
order plus a handful of look-alike frames (revisits, loop closures) cover what the mapper needs.
Look-alikes come from a small bag-of-visual-words built from the SIFT descriptors COLMAP already
stored, which takes seconds where COLMAP's vocab_tree_builder took over ten minutes.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

NEIGHBOURS = 12          # consecutive frames on each side always matched
LOOKALIKES = 8           # extra pairs per image from visual similarity
VOCAB_WORDS = 512
VOCAB_SAMPLE = 60_000    # descriptors used to fit the vocabulary
KMEANS_ITERS = 8
ASSIGN_CHUNK = 50_000


def _load(db: Path) -> tuple[list[str], list[np.ndarray]]:
    with sqlite3.connect(db) as con:
        names = dict(con.execute("SELECT image_id, name FROM images"))
        rows = con.execute("SELECT image_id, rows, cols, data FROM descriptors").fetchall()
    rows.sort(key=lambda r: names[r[0]])
    order = [names[r[0]] for r in rows]
    descs = [np.frombuffer(data, np.uint8).reshape(n, c).astype(np.float32) if n else np.empty((0, 128), np.float32)
             for _, n, c, data in rows]
    return order, descs


def _nearest(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    labels = np.empty(len(points), np.int64)
    centre_sq = (centres ** 2).sum(1)
    for start in range(0, len(points), ASSIGN_CHUNK):
        block = points[start:start + ASSIGN_CHUNK]
        labels[start:start + len(block)] = np.argmin(centre_sq[None, :] - 2 * block @ centres.T, axis=1)
    return labels


def _vocabulary(descs: list[np.ndarray], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pool = np.concatenate([d for d in descs if len(d)])
    sample = pool[rng.choice(len(pool), min(VOCAB_SAMPLE, len(pool)), replace=False)]
    centres = sample[rng.choice(len(sample), min(VOCAB_WORDS, len(sample)), replace=False)].copy()
    for _ in range(KMEANS_ITERS):
        labels = _nearest(sample, centres)
        sums = np.zeros_like(centres)
        np.add.at(sums, labels, sample)
        counts = np.bincount(labels, minlength=len(centres))[:, None]
        centres = np.where(counts > 0, sums / np.maximum(counts, 1), centres)
    return centres


def similarity(descs: list[np.ndarray]) -> np.ndarray:
    """Cosine similarity of TF-IDF word histograms, one row per image."""
    centres = _vocabulary(descs)
    hist = np.zeros((len(descs), len(centres)))
    for i, d in enumerate(descs):
        if len(d):
            hist[i] = np.bincount(_nearest(d, centres), minlength=len(centres))
    idf = np.log(len(descs) / np.maximum((hist > 0).sum(0), 1))
    vec = hist / np.maximum(hist.sum(1, keepdims=True), 1) * idf
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    return vec @ vec.T


def choose_pairs(sim: np.ndarray, neighbours: int = NEIGHBOURS, lookalikes: int = LOOKALIKES) -> set[tuple[int, int]]:
    n = len(sim)
    pairs = {(i, j) for i in range(n) for j in range(i + 1, min(n, i + neighbours + 1))}
    step = 2 * neighbours
    while step < n:                                  # sparse long-range links, like quadratic overlap
        pairs |= {(i, i + step) for i in range(n - step)}
        step *= 2
    ranked = np.argsort(-sim, axis=1)
    for i in range(n):
        picked = 0
        for j in ranked[i]:
            if abs(int(j) - i) <= neighbours:
                continue
            pairs.add((min(i, int(j)), max(i, int(j))))
            picked += 1
            if picked == lookalikes:
                break
    return pairs


def write_pairs(db: Path, out: Path) -> int:
    """Writes COLMAP's `matches_importer --match_type pairs` file; returns the pair count."""
    names, descs = _load(db)
    pairs = choose_pairs(similarity(descs))
    out.write_text("".join(f"{names[i]} {names[j]}\n" for i, j in sorted(pairs)), encoding="utf-8")
    return len(pairs)
