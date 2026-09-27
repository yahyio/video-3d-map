"""Mesh cleanup before texturing: drops floating fragments (small disconnected pieces) that
reconstruction leaves around a scan, keeping the connected surfaces that make up the scene."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import meshio

# A piece survives if it holds at least this share of the largest piece's faces. Walls, floor and
# furniture are connected to the main surface or large on their own; floaters are tiny islands.
MIN_PIECE_SHARE = 0.02


@dataclass(frozen=True)
class MeshCleanReport:
    faces_before: int
    faces_after: int
    pieces_before: int
    pieces_kept: int


def face_components(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    """Component label per face; faces sharing a vertex belong to the same piece."""
    rows = np.repeat(faces[:, 0], 2)          # edges v0-v1 and v0-v2 connect all three corners
    cols = faces[:, 1:].ravel()
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(vertex_count, vertex_count))
    _, vertex_label = connected_components(graph, directed=False)
    return vertex_label[faces[:, 0]]


def clean(src: Path, dst: Path) -> MeshCleanReport:
    verts, faces = meshio.read_ply(src)
    labels = face_components(faces, len(verts))
    sizes = np.bincount(labels)
    keep_labels = np.flatnonzero(sizes >= MIN_PIECE_SHARE * sizes.max())
    kept_verts, kept_faces = meshio.submesh(verts, faces, np.isin(labels, keep_labels))
    meshio.write_ply(dst, kept_verts, kept_faces)
    return MeshCleanReport(len(faces), len(kept_faces), int((sizes > 0).sum()), len(keep_labels))
