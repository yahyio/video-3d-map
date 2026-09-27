"""Puts reconstructions into a usable frame: Y-up, floor at y=0, centred, metric-ish scale.

COLMAP's orientation aligner leaves gravity along +Y (camera "down"), so up is -Y there.
Scale is unknown from a single video; it is estimated from how high the camera was held.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FLOOR_PERCENTILE = 1.0  # robust against a few stray vertices under the floor
FLOOR_MAX_PERCENTILE = 30.0   # a support plane above this height share is not treated as the floor
PLANE_SAMPLE = 50_000         # RANSAC works on a random subset; more adds time, not accuracy
PLANE_ITERATIONS = 300
PLANE_TOLERANCE = 0.005       # inlier distance as a share of the model's diagonal
PLANE_MAX_TILT_DEG = 45.0     # candidates steeper than this are walls, not floors
PLANE_MIN_SHARE = 0.15        # below this, no plane is trusted and the COLMAP up stays
MANHATTAN_ITERATIONS = 4
MANHATTAN_FLAT = 0.85         # |n.up| above this: floor, table top, shelf board
MANHATTAN_WALL = 0.35         # |n.up| below this: wall, cabinet side
MANHATTAN_MIN_SHARE = 0.05    # each set needs this share of the surface to be trusted
MANHATTAN_MAX_TILT_DEG = 35.0


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def camera_views(images_bin: Path) -> tuple[np.ndarray, np.ndarray]:
    """World-space camera centres and unit viewing directions from a COLMAP binary images.bin."""
    data = images_bin.read_bytes()
    (count,), pos = struct.unpack_from("<Q", data, 0), 8
    centers, forwards = [], []
    for _ in range(count):
        _image_id, qw, qx, qy, qz, tx, ty, tz, _cam = struct.unpack_from("<I7dI", data, pos)
        pos = data.index(b"\0", pos + 64) + 1          # skip null-terminated file name
        (npts,) = struct.unpack_from("<Q", data, pos)
        pos += 8 + npts * 24                           # x, y (double) + point3D id (int64)
        rot = _quat_to_rot(qw, qx, qy, qz)
        centers.append(-rot.T @ np.array([tx, ty, tz]))
        forwards.append(rot.T @ np.array([0.0, 0.0, 1.0]))   # COLMAP cameras look down +Z
    return np.array(centers).reshape(-1, 3), np.array(forwards).reshape(-1, 3)


def sparse_points(points_bin: Path) -> np.ndarray:
    """XYZ of COLMAP's sparse points; enough to level and scale a scan when no mesh is built."""
    data = points_bin.read_bytes()
    (count,), pos = struct.unpack_from("<Q", data, 0), 8
    xyz = np.empty((count, 3))
    for i in range(count):
        xyz[i] = struct.unpack_from("<3d", data, pos + 8)   # after the uint64 point id
        pos += 8 + 24 + 3 + 8                               # id, xyz, rgb, reprojection error
        (track,) = struct.unpack_from("<Q", data, pos)
        pos += 8 + track * 8                                # (image id, point2D idx) per observation
    return xyz


def camera_centers(images_bin: Path) -> np.ndarray:
    return camera_views(images_bin)[0]


def flip_up(p: np.ndarray) -> np.ndarray:
    """180° about X: COLMAP-aligned (-Y up) -> Y up, keeping a right-handed frame."""
    return p * np.array([1.0, -1.0, -1.0])


def dominant_yaw(verts: np.ndarray, faces: np.ndarray) -> float:
    """Angle (radians, Y-up frame) of the main wall direction, found modulo 90° from face normals."""
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])   # length = 2 * area
    horizontal = np.hypot(n[:, 0], n[:, 2])
    theta = np.arctan2(n[:, 2], n[:, 0])
    # Weight by the horizontal part of the area so floors/ceilings do not vote.
    s, c = (horizontal * np.sin(4 * theta)).sum(), (horizontal * np.cos(4 * theta)).sum()
    return float(np.arctan2(s, c) / 4) if horizontal.sum() > 0 else 0.0


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _align_to_up(normal: np.ndarray) -> np.ndarray:
    """Rotation taking unit `normal` onto +Y (Rodrigues about their common perpendicular)."""
    up = np.array([0.0, 1.0, 0.0])
    axis = np.cross(normal, up)
    sin, cos = np.linalg.norm(axis), float(normal @ up)
    if sin < 1e-9:
        return np.eye(3)
    k = axis / sin
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + sin * kx + (1 - cos) * (kx @ kx)


def support_plane(v: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float, float] | None:
    """RANSAC for the biggest roughly-horizontal plane (floor or table) in a Y-up cloud.

    Returns (upward unit normal, offset d of the plane n·x = d, share of points on it),
    or None when no plane is convincing.
    """
    rng = np.random.default_rng(seed)
    pts = v[rng.choice(len(v), min(len(v), PLANE_SAMPLE), replace=False)] if len(v) > 3 else v
    if len(pts) < 3:
        return None
    tol = PLANE_TOLERANCE * np.linalg.norm(np.ptp(pts, axis=0))
    best_count, best = 0, None
    for _ in range(PLANE_ITERATIONS):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-12:
            continue
        n = n / length * np.sign(n[1] or 1.0)
        if n[1] < np.cos(np.radians(PLANE_MAX_TILT_DEG)):
            continue
        inliers = np.abs((pts - a) @ n) < tol
        if inliers.sum() > best_count:
            best_count, best = int(inliers.sum()), (n, pts[inliers])
    if best is None or best_count / len(pts) < PLANE_MIN_SHARE:
        return None
    normal, members = best
    # Refit the normal on all inliers (smallest principal axis) so three noisy samples don't set the tilt.
    centred = members - members.mean(axis=0)
    refined = np.linalg.svd(centred, full_matrices=False)[2][-1]
    refined *= np.sign(refined[1] or 1.0)
    return refined, float(members.mean(axis=0) @ refined), best_count / len(pts)


@dataclass(frozen=True)
class Frame:
    offset: np.ndarray   # subtracted after flipping and levelling
    scale: float         # multiplied last
    yaw: float = 0.0     # walls are rotated by -yaw so they line up with X/Z
    level: np.ndarray = field(default_factory=lambda: np.eye(3))   # tilts the support plane flat

    def apply(self, p: np.ndarray) -> np.ndarray:
        # Rotating (x, z) by -yaw about Y maps a normal at angle `yaw` onto +X.
        return ((flip_up(p) @ self.level.T - self.offset) @ _rot_y(self.yaw).T) * self.scale

    def matrix(self) -> np.ndarray:
        """Same transform as apply() as a 4x4 homogeneous matrix (column-vector convention)."""
        m = np.eye(4)
        m[:3, :3] = self.scale * (_rot_y(self.yaw) @ self.level @ np.diag([1.0, -1.0, -1.0]))
        m[:3, 3] = -self.scale * (_rot_y(self.yaw) @ self.offset)
        return m


def manhattan_up(v: np.ndarray, faces: np.ndarray) -> np.ndarray | None:
    """Up direction that makes flat surfaces horizontal and walls vertical at the same time.

    Maximises sum(a_h (n.u)^2) - sum(a_v (n.u)^2) over near-horizontal (h) and near-vertical (v)
    faces, re-picking both sets around the new estimate a few times. A room scan had no single
    dominant floor plane, so support_plane found nothing and a 10° tilt from image-orientation
    alignment survived; desk, bed, shelves and walls all agreed on the correction.
    """
    tri = v[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    keep = area > 1e-12
    n, a = cross[keep] / area[keep, None], area[keep]
    up = np.array([0.0, 1.0, 0.0])
    for _ in range(MANHATTAN_ITERATIONS):
        dot = n @ up
        horizontal, vertical = np.abs(dot) > MANHATTAN_FLAT, np.abs(dot) < MANHATTAN_WALL
        if a[horizontal].sum() < MANHATTAN_MIN_SHARE * a.sum() or a[vertical].sum() < MANHATTAN_MIN_SHARE * a.sum():
            return None
        m = (n[horizontal] * a[horizontal, None]).T @ n[horizontal] - (n[vertical] * a[vertical, None]).T @ n[vertical]
        vals, vecs = np.linalg.eigh(m)
        up = vecs[:, -1] * np.sign(vecs[1, -1] or 1.0)
    tilt = np.degrees(np.arccos(np.clip(up[1], -1, 1)))
    return up if tilt <= MANHATTAN_MAX_TILT_DEG else None


def estimate(mesh_verts: np.ndarray, cams: np.ndarray, camera_height_m: float,
             faces: np.ndarray | None = None) -> Frame:
    v = flip_up(mesh_verts)
    # Image-orientation alignment assumes upright footage and was off by 10-24° on real scans.
    # With a mesh, floors, tables and walls together fix "up"; otherwise the biggest flat plane does.
    up = manhattan_up(v, faces) if faces is not None and len(faces) else None
    plane = support_plane(v)
    if up is not None:
        level = _align_to_up(up)
    else:
        level = _align_to_up(plane[0]) if plane else np.eye(3)
    v = v @ level.T
    if up is not None:
        plane = support_plane(v)   # floor height is measured in the levelled frame
    floor = float(np.percentile(v[:, 1], FLOOR_PERCENTILE))
    if plane:
        # After levelling, the plane sits at height d. Trust it as the floor only when it is
        # low in the model: in a room the biggest flat surface can be the ceiling.
        low_band = np.percentile(v[:, 1], FLOOR_MAX_PERCENTILE)
        if plane[1] <= low_band:
            floor = plane[1]
    centre = np.median(v, axis=0)
    cam_heights = (flip_up(cams) @ level.T)[:, 1] if len(cams) else np.empty(0)
    held = np.median(cam_heights) - floor if len(cam_heights) else 0.0
    scale = camera_height_m / held if held > 1e-6 and camera_height_m > 0 else 1.0
    yaw = dominant_yaw(v, faces) if faces is not None and len(faces) else 0.0
    return Frame(offset=np.array([centre[0], floor, centre[2]]), scale=float(scale), yaw=yaw, level=level)
