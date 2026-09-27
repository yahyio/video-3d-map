"""Renders a synthetic walkthrough of a textured room (world Y-up, metres) to an mp4 for end-to-end tests.

Usage: python tests/fixtures/make_room_video.py out.mp4 [frames]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H, FOCAL = 960, 540, 620.0
ROOM = (np.array([-4.0, 0.0, -3.0]), np.array([4.0, 3.0, 3.0]))
BOXES = [  # furniture-like solids so the scene has depth, not just walls
    (np.array([-1.0, 0.0, -0.6]), np.array([0.2, 0.9, 0.4])),
    (np.array([1.8, 0.0, 1.4]), np.array([3.0, 1.8, 2.4])),
    (np.array([-3.2, 0.0, 1.6]), np.array([-2.4, 1.2, 2.6])),
]
TEX = 2048


def make_texture(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((TEX, TEX, 3), np.float32)
    for octave, amp in ((8, 0.5), (32, 0.3), (128, 0.2)):
        small = Image.fromarray((rng.random((octave, octave, 3)) * 255).astype(np.uint8))
        img += amp * np.asarray(small.resize((TEX, TEX), Image.BICUBIC), np.float32) / 255
    pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    for _ in range(1500):
        x, y = rng.integers(0, TEX, 2).tolist()
        r = int(rng.integers(4, 40))
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)((x, y, x + r, y + r), fill=colour)
    return np.asarray(pil, np.float32) / 255


def cast(origins: np.ndarray, dirs: np.ndarray, tex: np.ndarray) -> np.ndarray:
    n = dirs.shape[0]
    rows = np.arange(n)
    inv = 1.0 / np.where(np.abs(dirs) < 1e-9, 1e-9, dirs)

    exit_t = np.maximum((ROOM[0] - origins) * inv, (ROOM[1] - origins) * inv)
    axis = exit_t.argmin(1)
    best_t = exit_t.min(1)
    face_id = axis * 2 + (dirs[rows, axis] > 0)

    for b, (lo, hi) in enumerate(BOXES, start=1):
        t0, t1 = (lo - origins) * inv, (hi - origins) * inv
        near, far = np.minimum(t0, t1), np.maximum(t0, t1)
        t_in, t_out = near.max(1), far.min(1)
        hit = (t_in < t_out) & (t_in > 1e-4) & (t_in < best_t)
        best_t = np.where(hit, t_in, best_t)
        ax = near.argmax(1)
        face_id = np.where(hit, b * 6 + ax * 2 + (dirs[rows, ax] < 0), face_id)

    p = origins + dirs * best_t[:, None]
    ax = (face_id % 6) // 2
    uv_axes = np.array([[1, 2], [0, 2], [0, 1]])[ax]
    u, v = p[rows, uv_axes[:, 0]], p[rows, uv_axes[:, 1]]
    offset = (face_id * 977) % TEX
    px = ((u * 240 + offset) % TEX).astype(np.int64)
    py = ((v * 240 + offset * 3) % TEX).astype(np.int64)
    shade = np.array([0.8, 1.0, 0.65])[ax][:, None]
    return tex[py, px] * shade


def camera_pose(k: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    a = 2 * np.pi * k / count
    pos = np.array([2.2 * np.cos(a), 1.6 + 0.1 * np.sin(3 * a), 1.5 * np.sin(a)])
    yaw = a + np.pi / 2 + 0.9  # tangent to the path, turned outward toward the walls
    fwd = np.array([np.cos(yaw), -0.12, np.sin(yaw)])
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, np.stack([right, -up, fwd], 1)  # columns: camera x (right), y (down), z (forward)


def main(out: Path, count: int = 240) -> None:
    tex = make_texture()
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    cam_dirs = np.stack([(xs - W / 2) / FOCAL, (ys - H / 2) / FOCAL, np.ones_like(xs)], -1).reshape(-1, 3)
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(count):
            pos, rot = camera_pose(k, count)
            dirs = cam_dirs @ rot.T
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            rgb = cast(np.broadcast_to(pos, dirs.shape), dirs, tex).reshape(H, W, 3)
            Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8)).save(Path(tmp) / f"f_{k:04d}.png")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "30",
                        "-i", str(Path(tmp) / "f_%04d.png"), "-c:v", "libx264", "-crf", "16",
                        "-pix_fmt", "yuv420p", str(out)], check=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 240)
