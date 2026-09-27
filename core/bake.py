"""Bakes splat colours onto clean primitives as textures, so smooth shapes still show labels,
prints and carpet patterns (a single colour per shape came out wrong: a jar's white label with
black print averaged to dark grey).

Every splat near a primitive surface is written into that surface's image at its unwrapped
position, spread by a blur matching splat size, and remaining holes take the nearest colour.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .primitives import CYL_SEGMENTS, Primitive

BAND = 0.015            # metres from a surface that a splat may be to colour it
PX_PER_M = 1000         # texture resolution (1 mm per pixel)
MAX_PX = 2048
MIN_PX = 16
FILL_SIGMAS = (1.2, 2.5, 5.0, 10.0, 20.0)   # px; finest level with enough samples wins
MIN_WEIGHT = 0.3


def _raster(u: np.ndarray, v: np.ndarray, rgb: np.ndarray, w: np.ndarray, width: int, height: int,
            fallback: np.ndarray) -> Image.Image:
    """Weighted splat colours at (u, v) in [0,1]^2 -> RGB image; v=0 is the image bottom."""
    if not len(u):
        return Image.new("RGB", (width, height), tuple(int(c) for c in fallback))
    x = np.clip((u * width).astype(int), 0, width - 1)
    y = np.clip(((1 - v) * height).astype(int), 0, height - 1)
    acc = np.zeros((height, width, 3))
    wsum = np.zeros((height, width))
    np.add.at(acc, (y, x), rgb * w[:, None])
    np.add.at(wsum, (y, x), w)
    # Push-pull fill: each pixel takes the finest blur level that has enough samples. Filling
    # gaps with the nearest sample instead made a crystal-like mosaic of flat cells.
    img = np.zeros((height, width, 3))
    todo = np.ones((height, width), bool)
    for sigma in FILL_SIGMAS:
        a = np.stack([ndimage.gaussian_filter(acc[..., c], sigma) for c in range(3)], axis=-1)
        s = ndimage.gaussian_filter(wsum, sigma)
        # One sample of MIN_WEIGHT within about sigma of the pixel is enough at this level.
        ok = todo & (s > MIN_WEIGHT / (2 * np.pi * sigma * sigma))
        img[ok] = a[ok] / s[ok, None]
        todo &= ~ok
        if not todo.any():
            break
    if todo.any():
        _, (iy, ix) = ndimage.distance_transform_edt(todo, return_indices=True)
        img = img[iy, ix]
    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def _px(metres: float) -> int:
    return int(np.clip(round(metres * PX_PER_M), MIN_PX, MAX_PX))


# --- per-shape unwraps ---------------------------------------------------------------------
def _cylinder(p: Primitive, pts, rgb, w, fallback, stem: str, out_dir: Path) -> tuple[list, list, list]:
    """Side (angle x height) and top (disc seen from above) textures plus UV-mapped geometry."""
    cx, cy, cz = p.center
    r, h = p.size[0] / 2, p.size[1]
    y0, y1 = cy - h / 2, cy + h / 2
    dx, dz = pts[:, 0] - cx, pts[:, 2] - cz
    radial = np.hypot(dx, dz)
    theta = np.mod(np.arctan2(dz, dx), 2 * np.pi)
    side = (np.abs(radial - r) < BAND) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    top = (np.abs(pts[:, 1] - y1) < BAND) & (radial < r + BAND)
    side_img = _raster(theta[side] / (2 * np.pi), (pts[side, 1] - y0) / h, rgb[side], w[side],
                       _px(2 * np.pi * r), _px(h), fallback)
    top_img = _raster(dx[top] / (2 * r) + 0.5, 0.5 - dz[top] / (2 * r), rgb[top], w[top], _px(2 * r), _px(2 * r), fallback)
    side_name, top_name = f"{stem}_side.jpg", f"{stem}_top.jpg"
    side_img.save(out_dir / side_name, quality=92)
    top_img.save(out_dir / top_name, quality=92)

    n = CYL_SEGMENTS
    ang = np.linspace(0, 2 * np.pi, n + 1)                       # duplicate seam column for UVs
    ring = np.c_[np.cos(ang) * r + cx, np.zeros(n + 1), np.sin(ang) * r + cz]
    bottom, upper = ring + (0, y0, 0), ring + (0, y1, 0)
    verts = np.vstack([bottom, upper])
    uvs = [(a / (2 * np.pi), 0.0) for a in ang] + [(a / (2 * np.pi), 1.0) for a in ang]
    side_faces = [((i, i), (n + 1 + i, n + 1 + i), (n + 2 + i, n + 2 + i), (i + 1, i + 1)) for i in range(n)]
    base_v, base_t = len(verts), len(uvs)
    cap = np.vstack([upper[:n], [[cx, y1, cz]]])
    verts = np.vstack([verts, cap])
    uvs += [(np.cos(a) / 2 + 0.5, 0.5 - np.sin(a) / 2) for a in ang[:n]] + [(0.5, 0.5)]
    top_faces = [((base_v + n, base_t + n), (base_v + (i + 1) % n, base_t + (i + 1) % n), (base_v + i, base_t + i))
                 for i in range(n)]
    return verts, uvs, [(side_name, side_faces), (top_name, top_faces)]


_BOX_FACES = [  # axis, sign, (u axis, v axis) in local box coordinates
    (0, 1, (2, 1)), (0, -1, (2, 1)), (1, 1, (0, 2)), (1, -1, (0, 2)), (2, 1, (0, 1)), (2, -1, (0, 1)),
]


def _box(p: Primitive, pts, rgb, w, fallback, stem: str, out_dir: Path) -> tuple[list, list, list]:
    c, s = np.cos(p.yaw), np.sin(p.yaw)
    rot = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])        # local -> world
    local = (pts - p.center) @ rot
    half = np.array(p.size) / 2
    verts, uvs, groups = [], [], []
    for k, (axis, sign, (ua, va)) in enumerate(_BOX_FACES):
        on = (np.abs(local[:, axis] - sign * half[axis]) < BAND)
        on &= (np.abs(local[:, ua]) <= half[ua] + BAND) & (np.abs(local[:, va]) <= half[va] + BAND)
        u = local[on, ua] / (2 * half[ua]) + 0.5
        v = local[on, va] / (2 * half[va]) + 0.5
        name = f"{stem}_f{k}.jpg"
        _raster(u, v, rgb[on], w[on], _px(2 * half[ua]), _px(2 * half[va]), fallback).save(out_dir / name, quality=92)
        corners = []
        for cu, cv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            q = np.zeros(3)
            q[axis], q[ua], q[va] = sign * half[axis], cu * half[ua], cv * half[va]
            corners.append(q @ rot.T + p.center)
        b = len(verts)
        verts += corners
        uvs += [(0, 0), (1, 0), (1, 1), (0, 1)]
        idx = [(b, b), (b + 1, b + 1), (b + 2, b + 2), (b + 3, b + 3)]
        # Winding faces outward: flip when the (u, v, n) basis is left-handed.
        normal = np.zeros(3)
        normal[axis] = sign
        if np.dot(np.cross(np.eye(3)[ua], np.eye(3)[va]), normal) < 0:
            idx = idx[::-1]
        groups.append((name, [tuple(idx)]))
    return np.array(verts), uvs, groups


def write_textured_obj(prims: list[Primitive], pts: np.ndarray, rgb: np.ndarray, w: np.ndarray, obj: Path) -> None:
    out_dir = obj.parent
    lines, mtl = [f"mtllib {obj.stem}.mtl"], []
    v_base = t_base = 1
    for p in prims:
        fallback = np.array(p.color)
        make = _cylinder if p.shape == "cylinder" else _box
        verts, uvs, groups = make(p, pts, rgb, w, fallback, f"{obj.stem}_{p.name}", out_dir)
        lines.append(f"o {p.name}")
        lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts]
        lines += [f"vt {u:.5f} {v:.5f}" for u, v in uvs]
        for tex, faces in groups:
            mat = Path(tex).stem
            lines.append(f"usemtl {mat}")
            lines += ["f " + " ".join(f"{v_base + vi}/{t_base + ti}" for vi, ti in face) for face in faces]
            mtl += [f"newmtl {mat}", "Kd 1 1 1", f"map_Kd {tex}", ""]
        v_base += len(verts)
        t_base += len(uvs)
    obj.write_text("\n".join(lines) + "\n", encoding="utf-8")
    obj.with_suffix(".mtl").write_text("\n".join(mtl), encoding="utf-8")
