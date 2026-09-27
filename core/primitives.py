"""Clean geometric primitives from a splat: floor plane, then per object stacked cylinders and boxes.

Each object above the floor is cut into thin horizontal slices; every slice gets both a circle
and a rectangle fit, and runs of slices that agree become one cylinder or box. Things lying on an
object's top in a different colour (a packet on a jar lid) become their own thin box.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .splat import splat_points

MIN_OPACITY = 0.1
MAX_SPLAT = 0.05            # metres
FLOOR_BIN = 0.005
FLOOR_GAP = 0.015           # object points start this far above the floor
FOOTPRINT_CELL = 0.02
FOOTPRINT_MASS = 0.5        # summed opacity for a footprint cell to count
MIN_OBJECT_MASS = 50.0
SLICE = 0.01
MIN_SLICE_POINTS = 25
TRIM = 0.8                  # fits keep the best 80% of points (splats scatter off surfaces)
SAME_SIZE = 0.15            # adjacent slices within 15% size belong to one primitive
MIN_RUN = 2                 # slices
CIRCLE_BIAS = 1.15          # prefer a box only when it fits clearly better than a circle
TOP_BAND = 0.02             # depth below an object's top searched for things lying on it
COLOR_SPLIT = 60.0          # RGB distance separating a top item from the object's own colour
MIN_TOP_POINTS = 40
TOP_OPACITY = 0.3           # only solid splats decide what lies on top
TOP_MAX_SHARE = 0.6         # a 'top item' covering most of the top is just the top
TOP_MIN_SHARE = 0.05
TOP_MIN_HEIGHT = 0.005
SURFACE_BAND = 0.01         # colour is sampled this close to a fitted surface
CYL_SEGMENTS = 64


@dataclass(frozen=True)
class Primitive:
    name: str
    shape: str                          # cylinder | box
    center: tuple[float, float, float]  # metres, Y up
    size: tuple[float, float, float]    # cylinder: (diameter, height, diameter); box: (x, y, z) before yaw
    yaw: float                          # radians about Y (boxes)
    color: tuple[int, int, int]


# --- 2-D fits ------------------------------------------------------------------------------
def fit_circle(xz: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(centre, radius, trimmed rms residual) by algebraic least squares, refit on inliers."""
    keep = np.ones(len(xz), bool)
    for _ in range(3):
        x, z = xz[keep, 0], xz[keep, 1]
        a = np.c_[x, z, np.ones(keep.sum())]
        sol, *_ = np.linalg.lstsq(a, x * x + z * z, rcond=None)
        centre = sol[:2] / 2
        radius = float(np.sqrt(max(sol[2] + centre @ centre, 1e-12)))
        resid = np.abs(np.linalg.norm(xz - centre, axis=1) - radius)
        keep = resid <= np.quantile(resid, TRIM)
    return centre, radius, float(np.sqrt(np.mean(resid[keep] ** 2)))


def fit_rect(xz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(centre, (width, depth), yaw, trimmed rms distance to the rectangle outline)."""
    mean = xz.mean(axis=0)
    _, _, vt = np.linalg.svd(xz - mean, full_matrices=False)
    yaw = float(np.arctan2(vt[0, 1], vt[0, 0]))
    rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    local = (xz - mean) @ rot.T
    lo, hi = np.quantile(local, 0.03, axis=0), np.quantile(local, 0.97, axis=0)
    centre_local = (lo + hi) / 2
    half = (hi - lo) / 2
    d = np.abs(local - centre_local) - half
    outside = np.linalg.norm(np.maximum(d, 0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0)
    resid = np.abs(outside + inside)
    keep = resid <= np.quantile(resid, TRIM)
    centre = mean + centre_local @ rot
    return centre, hi - lo, yaw, float(np.sqrt(np.mean(resid[keep] ** 2)))


# --- scene decomposition -------------------------------------------------------------------
def floor_height(pts: np.ndarray, w: np.ndarray) -> float:
    edges = np.arange(pts[:, 1].min(), pts[:, 1].max() + FLOOR_BIN, FLOOR_BIN)
    hist, edges = np.histogram(pts[:, 1], bins=edges, weights=w)
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) / 2)


def object_masks(pts: np.ndarray, w: np.ndarray, floor: float) -> list[np.ndarray]:
    """Boolean point masks, one per object standing on the floor (largest first)."""
    above = pts[:, 1] > floor + FLOOR_GAP
    xz = pts[:, [0, 2]]
    lo = xz[above].min(axis=0)
    ij = np.floor((xz - lo) / FOOTPRINT_CELL).astype(int)
    valid = above & np.all(ij >= 0, axis=1)
    dims = ij[valid].max(axis=0) + 1
    grid = np.zeros(dims)
    np.add.at(grid, tuple(ij[valid].T), w[valid])
    labels, count = ndimage.label(ndimage.binary_dilation(grid > FOOTPRINT_MASS))
    mass = ndimage.sum(grid, labels, range(1, count + 1))
    inside_grid = valid & np.all(ij < dims, axis=1)
    point_label = np.zeros(len(pts), int)
    point_label[inside_grid] = labels[tuple(ij[inside_grid].T)]
    masks = []
    for k in np.argsort(-mass):
        if mass[k] < MIN_OBJECT_MASS:
            break
        masks.append(point_label == k + 1)
    return masks


def _slices(pts: np.ndarray, floor: float) -> list[dict]:
    top = np.quantile(pts[:, 1], 0.995)
    out = []
    for y0 in np.arange(floor, top, SLICE):
        sel = (pts[:, 1] >= y0) & (pts[:, 1] < y0 + SLICE)
        if sel.sum() < MIN_SLICE_POINTS:
            continue
        xz = pts[sel][:, [0, 2]]
        c_centre, radius, c_err = fit_circle(xz)
        r_centre, dims, yaw, r_err = fit_rect(xz)
        is_box = r_err * CIRCLE_BIAS < c_err
        out.append({"y0": y0, "y1": y0 + SLICE, "box": is_box, "size": float(max(dims)) if is_box else 2 * radius,
                    "centre": r_centre if is_box else c_centre, "dims": dims, "yaw": yaw, "radius": radius})
    return out


def _runs(slices: list[dict]) -> list[list[dict]]:
    runs: list[list[dict]] = []
    for s in slices:
        last = runs[-1][-1] if runs else None
        if (last and s["box"] == last["box"] and abs(s["y0"] - last["y1"]) < SLICE * 1.5
                and abs(s["size"] - last["size"]) <= SAME_SIZE * max(s["size"], last["size"])):
            runs[-1].append(s)
        else:
            runs.append([s])
    return [r for r in runs if len(r) >= MIN_RUN]


def _color(rgb: np.ndarray, w: np.ndarray | None = None) -> tuple[int, int, int]:
    if not len(rgb):
        return (150, 150, 150)
    if w is None:
        return tuple(int(c) for c in np.clip(np.median(rgb, axis=0) * 255, 0, 255))
    # Opacity-weighted median per channel: faint haze and shadow splats barely count.
    order = np.argsort(rgb, axis=0)
    out = []
    for c in range(3):
        cum = np.cumsum(w[order[:, c]])
        out.append(rgb[order[np.searchsorted(cum, cum[-1] / 2), c], c])
    return tuple(int(v) for v in np.clip(np.array(out) * 255, 0, 255))


def _surface(pts: np.ndarray, p: Primitive) -> np.ndarray:
    """Points within SURFACE_BAND of a primitive's side/top (their colour is the object's colour)."""
    cx, cy, cz = p.center
    local = pts - (cx, cy, cz)
    in_height = np.abs(local[:, 1]) <= p.size[1] / 2 + SURFACE_BAND
    if p.shape == "cylinder":
        radial = np.hypot(local[:, 0], local[:, 2])
        side = np.abs(radial - p.size[0] / 2) < SURFACE_BAND
        top = (np.abs(local[:, 1] - p.size[1] / 2) < SURFACE_BAND) & (radial < p.size[0] / 2)
        return in_height & (side | top)
    c, s = np.cos(p.yaw), np.sin(p.yaw)
    lx, lz = local[:, 0] * c + local[:, 2] * s, -local[:, 0] * s + local[:, 2] * c
    d = np.maximum(np.abs(lx) - p.size[0] / 2, np.abs(lz) - p.size[2] / 2)
    return in_height & (np.abs(d) < SURFACE_BAND)


def object_primitives(name: str, pts: np.ndarray, rgb: np.ndarray, w: np.ndarray, floor: float) -> list[Primitive]:
    prims = []
    for i, run in enumerate(_runs(_slices(pts, floor)), start=1):
        y0, y1 = run[0]["y0"], run[-1]["y1"]
        if i == 1 and y0 - floor < 3 * SLICE:
            y0 = floor                                  # stands on the floor
        centre = np.median([s["centre"] for s in run], axis=0)
        if run[0]["box"]:
            dims = np.median([s["dims"] for s in run], axis=0)
            yaw = float(np.median([s["yaw"] for s in run]))
            prim = Primitive(f"{name}_box{i}", "box", _r((centre[0], (y0 + y1) / 2, centre[1])),
                             _r((dims[0], y1 - y0, dims[1])), round(yaw, 4), (0, 0, 0))
        else:
            d = 2 * float(np.median([s["radius"] for s in run]))
            prim = Primitive(f"{name}_cyl{i}", "cylinder", _r((centre[0], (y0 + y1) / 2, centre[1])),
                             _r((d, y1 - y0, d)), 0.0, (0, 0, 0))
        on = _surface(pts, prim)
        prims.append(Primitive(prim.name, prim.shape, prim.center, prim.size, prim.yaw, _color(rgb[on], w[on])))
    # Things lying on top (a packet on a lid) show up in the baked top texture; a separate box for
    # them picked the wrong colour cluster on a real jar, so none is added.
    return prims


def _two_colours(rgb: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2-means on colour; returns (label per point, the two centres)."""
    centres = np.stack([rgb[np.argmin(rgb.sum(1))], rgb[np.argmax(rgb.sum(1))]])
    for _ in range(10):
        label = np.argmin(((rgb[:, None] - centres[None]) ** 2).sum(-1), axis=1)
        for k in (0, 1):
            if (label == k).any():
                centres[k] = np.average(rgb[label == k], axis=0, weights=w[label == k])
    return label, centres


def _top_items(name: str, pts: np.ndarray, rgb: np.ndarray, w: np.ndarray, top: Primitive) -> list[Primitive]:
    """A differently coloured thing lying on the object's top (a packet on a jar lid): the minority
    colour cluster of the opaque top surface, when it is distinct and covers only part of the top."""
    top_y = top.center[1] + top.size[1] / 2
    local = pts[:, [0, 2]] - (top.center[0], top.center[2])
    over = np.hypot(local[:, 0], local[:, 1]) < max(top.size[0], top.size[2]) / 2
    near = over & (pts[:, 1] > top_y - TOP_BAND) & (w >= TOP_OPACITY)
    if near.sum() < 2 * MIN_TOP_POINTS:
        return []
    label, centres = _two_colours(rgb[near], w[near])
    minority = int(np.argmin(np.bincount(label, minlength=2)))
    share = (label == minority).mean()
    if np.linalg.norm(centres[0] - centres[1]) * 255 < COLOR_SPLIT or not TOP_MIN_SHARE <= share <= TOP_MAX_SHARE:
        return []
    idx = np.flatnonzero(near)[label == minority]
    centre, dims, yaw, _ = fit_rect(pts[idx][:, [0, 2]])
    height = max(float(np.quantile(pts[idx, 1], 0.9) - top_y), TOP_MIN_HEIGHT)
    return [Primitive(f"{name}_ontop", "box", _r((centre[0], top_y + height / 2, centre[1])),
                      _r((dims[0], height, dims[1])), round(yaw, 4), _color(rgb[idx], w[idx]))]


def floor_primitive(pts: np.ndarray, rgb: np.ndarray, w: np.ndarray, floor: float) -> Primitive:
    near = np.abs(pts[:, 1] - floor) < 2 * FLOOR_BIN
    xz = pts[near][:, [0, 2]]
    lo, hi = np.quantile(xz, 0.05, axis=0), np.quantile(xz, 0.95, axis=0)
    c = (lo + hi) / 2
    return Primitive("floor", "box", _r((c[0], floor - 0.005, c[1])), _r((hi[0] - lo[0], 0.01, hi[1] - lo[1])),
                     0.0, _color(rgb[near], w[near]))


def _r(v) -> tuple[float, float, float]:
    return tuple(round(float(x), 4) for x in v)


def fit(pts: np.ndarray, rgb: np.ndarray, opacity: np.ndarray, size: np.ndarray):
    """(primitives, points, colours, weights), everything shifted so the floor top is y = 0."""
    keep = (opacity >= MIN_OPACITY) & (size <= MAX_SPLAT)
    pts, rgb, w = pts[keep], rgb[keep], opacity[keep]
    floor = floor_height(pts, w)
    prims = [floor_primitive(pts, rgb, w, floor)]
    for n, mask in enumerate(object_masks(pts, w, floor), start=1):
        prims += object_primitives(f"object{n}", pts[mask], rgb[mask], w[mask], floor)
    shifted = [Primitive(p.name, p.shape, (p.center[0], round(p.center[1] - floor, 4), p.center[2]), p.size, p.yaw, p.color)
               for p in prims]
    return shifted, pts - (0, floor, 0), rgb, w


def extract(pts: np.ndarray, rgb: np.ndarray, opacity: np.ndarray, size: np.ndarray) -> list[Primitive]:
    return fit(pts, rgb, opacity, size)[0]


# --- output --------------------------------------------------------------------------------
def _mesh(p: Primitive) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    cx, cy, cz = p.center
    if p.shape == "cylinder":
        r, h = p.size[0] / 2, p.size[1]
        ang = np.linspace(0, 2 * np.pi, CYL_SEGMENTS, endpoint=False)
        ring = np.c_[np.cos(ang) * r, np.zeros(CYL_SEGMENTS), np.sin(ang) * r]
        verts = np.vstack([ring + (cx, cy - h / 2, cz), ring + (cx, cy + h / 2, cz),
                           [[cx, cy - h / 2, cz], [cx, cy + h / 2, cz]]])
        n = CYL_SEGMENTS
        faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        faces += [(2 * n, (i + 1) % n, i) for i in range(n)] + [(2 * n + 1, n + i, n + (i + 1) % n) for i in range(n)]
        return verts, faces
    sx, sy, sz = p.size
    c, s = np.cos(p.yaw), np.sin(p.yaw)
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * (sx, sy, sz)
    rot = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    verts = corners @ rot.T + (cx, cy, cz)
    return verts, [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def write_obj(prims: list[Primitive], obj: Path) -> None:
    lines, mtl, base = [f"mtllib {obj.stem}.mtl"], [], 1
    for p in prims:
        verts, faces = _mesh(p)
        lines += [f"o {p.name}", f"usemtl {p.name}"]
        lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts]
        lines += ["f " + " ".join(str(base + i) for i in f) for f in faces]
        base += len(verts)
        mtl += [f"newmtl {p.name}", "Kd " + " ".join(f"{c / 255:.3f}" for c in p.color), ""]
    obj.write_text("\n".join(lines) + "\n", encoding="utf-8")
    obj.with_suffix(".mtl").write_text("\n".join(mtl), encoding="utf-8")


def export(splat_ply: Path, frame_json: Path, out_dir: Path) -> list[Primitive]:
    """primitives.json and a texture-baked primitives.obj (+ its images)."""
    from .bake import write_textured_obj   # bake imports this module

    matrix = np.array(json.loads(frame_json.read_text(encoding="utf-8"))["matrix"]).reshape(4, 4)
    prims, pts, rgb, w = fit(*splat_points(splat_ply, matrix))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "primitives.json").write_text(json.dumps([asdict(p) for p in prims], indent=1), encoding="utf-8")
    write_textured_obj(prims, pts, rgb, w, out_dir / "primitives.obj")
    return prims
