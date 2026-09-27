"""Minimal mesh I/O for the formats OpenMVS emits: binary triangle PLY and textured OBJ."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_FACE_DTYPE = np.dtype([("n", "u1"), ("i", "<u4", (3,))])


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Reads OpenMVS' binary PLY (float32 xyz + uint8/uint32 triangle lists) -> (vertices, faces)."""
    data = path.read_bytes()
    end = data.find(b"end_header\n")
    if end < 0:
        raise ValueError(f"{path.name}: PLY başlığı bulunamadı")
    header = data[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path.name}: sadece binary little-endian PLY destekleniyor")
    counts = {ln.split()[1]: int(ln.split()[2]) for ln in header if ln.startswith("element")}
    vstart = next(i for i, ln in enumerate(header) if ln.startswith("element vertex"))
    vprops = []
    for ln in header[vstart + 1:]:
        if not ln.startswith("property"):
            break
        vprops.append(ln.split()[-1] if ln.split()[1] in ("float", "float32") else "?" + ln.split()[-1])
    if vprops != ["x", "y", "z"]:
        raise ValueError(f"{path.name}: beklenmeyen köşe alanları {vprops}")
    offset = end + len(b"end_header\n")
    nv, nf = counts["vertex"], counts.get("face", 0)
    verts = np.frombuffer(data, "<f4", nv * 3, offset).reshape(nv, 3).astype(np.float64)
    faces = np.frombuffer(data, _FACE_DTYPE, nf, offset + nv * 12)
    if nf and not (faces["n"] == 3).all():
        raise ValueError(f"{path.name}: sadece üçgen yüzler destekleniyor")
    return verts, faces["i"].astype(np.int64)


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {len(faces)}\nproperty list uchar uint vertex_indices\nend_header\n")
    packed = np.empty(len(faces), _FACE_DTYPE)
    packed["n"] = 3
    packed["i"] = faces
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.astype("<f4").tobytes())
        f.write(packed.tobytes())


def submesh(verts: np.ndarray, faces: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Faces selected by boolean mask `keep`, with unused vertices dropped and indices remapped."""
    used, inverse = np.unique(faces[keep].ravel(), return_inverse=True)
    return verts[used], inverse.reshape(-1, 3)


def _vertex_rows(lines: list[str]) -> list[int]:
    return [i for i, ln in enumerate(lines) if ln.startswith("v ")]


def read_obj_vertices(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [lines[i][2:].split()[:3] for i in _vertex_rows(lines)]
    return np.array(rows, dtype=np.float64).reshape(-1, 3)


def transform_obj(path: Path, fn) -> None:
    """Rewrites every `v x y z` line through fn(Nx3) -> Nx3; UVs, faces and materials stay untouched."""
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = _vertex_rows(lines)
    pts = np.array([lines[i][2:].split()[:3] for i in idx], dtype=np.float64).reshape(-1, 3)
    for i, (x, y, z) in zip(idx, fn(pts)):
        lines[i] = f"v {x:.6f} {y:.6f} {z:.6f}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
