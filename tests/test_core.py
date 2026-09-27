"""Unit tests for the pure-Python parts of the pipeline. Run: python -m unittest discover tests"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import frames, mesh_clean, meshio, mvs, orient, pipeline, retrieval, sfm, splat, system, tiles  # noqa: E402


def cube() -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


class TmpDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class FramesTest(unittest.TestCase):
    def test_blurred_image_scores_lower_than_sharp(self):
        rng = np.random.default_rng(0)
        sharp = Image.fromarray((rng.random((300, 400)) * 255).astype(np.uint8))
        blurred = sharp.filter(ImageFilter.GaussianBlur(4))
        self.assertGreater(frames.sharpness(sharp), 10 * frames.sharpness(blurred))

    def test_pick_sharpest_returns_best_per_window(self):
        scores = [1, 9, 2, 3, 4, 8, 7, 1, 0]
        self.assertEqual(frames.pick_sharpest(scores, 3), [1, 5, 6])

    def test_pick_sharpest_keeps_everything_when_target_exceeds_count(self):
        self.assertEqual(frames.pick_sharpest([3, 1], 10), [0, 1])

    def test_pick_sharpest_handles_empty_input(self):
        self.assertEqual(frames.pick_sharpest([], 5), [])

    def test_frames_for_duration_scales_with_length_and_clamps(self):
        budget = frames.FrameBudget(3.0, 80, 220)
        self.assertEqual(frames.frames_for_duration(57.4, budget), 172)   # the desk walkthrough
        self.assertEqual(frames.frames_for_duration(10, budget), 80)
        self.assertEqual(frames.frames_for_duration(600, budget), 220)

    def test_low_resolution_warning_flags_messenger_sized_video(self):
        self.assertIn("478×850", frames.low_resolution_warning((478, 850)))
        self.assertIsNone(frames.low_resolution_warning((1080, 1920)))
        self.assertIsNone(frames.low_resolution_warning(None))


class MeshIoTest(TmpDirCase):
    def test_ply_roundtrip_preserves_geometry(self):
        v, f = cube()
        path = self.dir / "c.ply"
        meshio.write_ply(path, v, f)
        v2, f2 = meshio.read_ply(path)
        np.testing.assert_allclose(v2, v)
        np.testing.assert_array_equal(f2, f)

    def test_read_ply_rejects_ascii(self):
        path = self.dir / "a.ply"
        path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
        with self.assertRaises(ValueError):
            meshio.read_ply(path)

    def test_submesh_drops_unused_vertices_and_remaps(self):
        v, f = cube()
        keep = np.zeros(len(f), bool)
        keep[2] = True
        sv, sf = meshio.submesh(v, f, keep)
        self.assertEqual(len(sv), 3)
        np.testing.assert_allclose(sv[sf[0]], v[f[2]])

    def test_transform_obj_only_touches_vertex_lines(self):
        path = self.dir / "m.obj"
        path.write_text("mtllib m.mtl\nv 1 2 3\nvt 0.5 0.5\nf 1/1 1/1 1/1\n", encoding="utf-8")
        meshio.transform_obj(path, lambda p: p * 2)
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[1], "v 2.000000 4.000000 6.000000")
        self.assertEqual(lines[2:], ["vt 0.5 0.5", "f 1/1 1/1 1/1"])
        np.testing.assert_allclose(meshio.read_obj_vertices(path), [[2, 4, 6]])


def write_images_bin(path: Path, poses: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """poses: (quaternion wxyz, translation) per image, in COLMAP's world-to-camera convention."""
    out = struct.pack("<Q", len(poses))
    for i, (q, t) in enumerate(poses, start=1):
        out += struct.pack("<I7dI", i, *q, *t, 1) + f"img{i}.jpg".encode() + b"\0"
        out += struct.pack("<Q", 2) + b"\0" * 48
    path.write_bytes(out)


class OrientTest(TmpDirCase):
    def test_camera_centers_inverts_pose(self):
        q = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])   # 90° about Y
        rot = orient._quat_to_rot(*q)
        centre = np.array([1.0, 2.0, 3.0])
        path = self.dir / "images.bin"
        write_images_bin(path, [(q, -rot @ centre), (np.array([1.0, 0, 0, 0]), np.zeros(3))])
        np.testing.assert_allclose(orient.camera_centers(path), [centre, [0, 0, 0]], atol=1e-9)

    def test_flip_up_negates_y_and_z(self):
        np.testing.assert_allclose(orient.flip_up(np.array([[1.0, 2.0, 3.0]])), [[1, -2, -3]])

    def test_estimate_puts_floor_at_zero_and_scales_by_camera_height(self):
        # COLMAP-aligned space has up = -Y: floor at y=+4, cameras 2 units above it at y=+2.
        rng = np.random.default_rng(1)
        floor = np.c_[rng.uniform(-5, 5, 500), np.full(500, 4.0), rng.uniform(-5, 5, 500)]
        cams = np.c_[rng.uniform(-1, 1, 20), np.full(20, 2.0), rng.uniform(-1, 1, 20)]
        frame = orient.estimate(floor, cams, camera_height_m=1.5)
        self.assertAlmostEqual(frame.scale, 0.75)
        np.testing.assert_allclose(frame.apply(floor)[:, 1], 0, atol=1e-9)
        self.assertAlmostEqual(float(np.median(frame.apply(cams)[:, 1])), 1.5)

    def test_dominant_yaw_squares_up_a_rotated_box(self):
        v, f = cube()
        v = (v - 0.5) * [8.0, 3.0, 6.0]
        angle = np.radians(30)
        c, s = np.cos(angle), np.sin(angle)
        rotated = v @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]).T      # about Y by +30°
        colmap_space = orient.flip_up(rotated)                             # estimate expects -Y-up input
        frame = orient.estimate(colmap_space, np.empty((0, 3)), 1.5, f)
        placed = frame.apply(colmap_space)
        extent = sorted((placed.max(0) - placed.min(0)).round(6))
        np.testing.assert_allclose(extent, [3, 6, 8], atol=1e-6)

    def test_estimate_levels_a_tilted_tabletop(self):
        rng = np.random.default_rng(7)
        table = np.c_[rng.uniform(-1, 1, 4000), np.zeros(4000), rng.uniform(-1, 1, 4000)]
        jar = np.c_[rng.normal(0, 0.05, 1500), rng.uniform(0, 0.3, 1500), rng.normal(0, 0.05, 1500)]
        tilt = orient._align_to_up(np.array([np.sin(0.42), np.cos(0.42), 0.0])).T   # ~24° lean
        scene = np.vstack([table, jar]) @ tilt.T
        frame = orient.estimate(orient.flip_up(scene), np.empty((0, 3)), 1.5)
        flat = frame.apply(orient.flip_up(table @ tilt.T))
        self.assertLess(np.ptp(flat[:, 1]), 0.02)          # table is level again
        self.assertAlmostEqual(float(np.median(flat[:, 1])), 0.0, places=2)

    def test_manhattan_up_undoes_a_tilt_without_a_dominant_floor(self):
        # Box room (walls + floor + ceiling) tilted 12° about X, like the real room scan.
        v, f = box_mesh([0, 0, 0], [4, 2.5, 3], inward=True)
        tilt = orient._align_to_up(np.array([0.0, np.cos(np.radians(12)), np.sin(np.radians(12))])).T
        up = orient.manhattan_up(v @ tilt.T, f)
        corrected = (v @ tilt.T) @ orient._align_to_up(up).T
        ys = sorted(set(np.round(corrected[:, 1], 6)))
        self.assertEqual(len(ys), 2)                      # floor and ceiling corners level again

    def test_matrix_matches_apply(self):
        frame = orient.Frame(offset=np.array([0.3, -1.2, 2.0]), scale=0.4, yaw=0.7)
        pts = np.random.default_rng(5).normal(size=(50, 3))
        homogeneous = np.c_[pts, np.ones(len(pts))] @ frame.matrix().T
        np.testing.assert_allclose(homogeneous[:, :3], frame.apply(pts), atol=1e-12)

    def test_estimate_falls_back_to_unit_scale_without_cameras(self):
        v, _ = cube()
        self.assertEqual(orient.estimate(v, np.empty((0, 3)), 1.5).scale, 1.0)


class SfmTest(TmpDirCase):
    def test_largest_model_picks_most_registered(self):
        for name, count in (("0", 5), ("1", 40), ("2", 12)):
            (self.dir / name).mkdir()
            (self.dir / name / "images.bin").write_bytes(struct.pack("<Q", count))
        self.assertEqual(sfm.largest_model(self.dir).name, "1")

    def test_largest_model_none_when_empty(self):
        self.assertIsNone(sfm.largest_model(self.dir))


class MvsTest(TmpDirCase):
    def test_resolution_level_keeps_small_videos_at_full_size(self):
        self.assertEqual(mvs.resolution_level(850, 1600), 0)     # WhatsApp portrait clip
        self.assertEqual(mvs.resolution_level(2400, 1200), 1)
        self.assertEqual(mvs.resolution_level(3840, 1000), 2)

    def test_face_count_reads_ply_header(self):
        v, f = cube()
        meshio.write_ply(self.dir / "c.ply", v, f)
        self.assertEqual(mvs._face_count(self.dir / "c.ply"), 12)

    def test_black_share_ignores_empty_colour(self):
        a = np.zeros((100, 100, 3), np.uint8)
        a[:, :50] = mvs.EMPTY_RGB          # unused atlas space
        a[:, 50:75] = (120, 90, 60)        # real texture; the remaining quarter stays black
        path = self.dir / "t.png"
        Image.fromarray(a).save(path)
        self.assertAlmostEqual(mvs.black_share(path), 0.5, places=2)


class ReviewFixesTest(TmpDirCase):
    def test_project_folder_is_ascii(self):
        name = pipeline.new_project(Path("Şişli çarşı ğüİö.mp4"))
        try:
            self.assertTrue(name.name.isascii())
            self.assertTrue(name.name.startswith("Sisli_carsi_guIo_"))
        finally:
            name.rmdir()

    def test_read_ply_rejects_extra_vertex_properties(self):
        path = self.dir / "c.ply"
        path.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\n"
                         b"property float y\nproperty float z\nproperty uchar red\nelement face 0\n"
                         b"property list uchar uint vertex_indices\nend_header\n")
        with self.assertRaises(ValueError):
            meshio.read_ply(path)


class SplatTest(TmpDirCase):
    def test_start_view_uses_central_camera_looking_forward(self):
        frame = orient.Frame(offset=np.zeros(3), scale=1.0)
        centers = np.array([[0.0, 0, 0], [1, 0, 0], [5, 0, 0]])
        forwards = np.array([[0.0, 0, 1], [0, 0, 1], [0, 0, 1]])   # COLMAP +Z
        view = splat.start_view(frame, centers, forwards)
        self.assertEqual(view["position"], [1.0, 0.0, 0.0])
        # flip_up turns +Z into -Z, so the pivot sits START_LOOK_M in front along -Z
        np.testing.assert_allclose(view["target"], [1.0, 0.0, -splat.START_LOOK_M])

    def test_memory_plan_fits_frames_into_budget(self):
        # The failed run: 259 frames of 1072x1910 at 1600 px need ~5.2 GB.
        need = splat.brush_memory_gb(259, 1072, 1910, 1600, 1, max_splats=0)   # frames + runtime only
        self.assertAlmostEqual(need, 5.16, delta=0.1)
        roomy = splat.plan_for_memory(8.0, 259, 1072, 1910, 1600)
        self.assertEqual((roomy.max_resolution, roomy.subsample_frames), (1600, 1))
        tight = splat.plan_for_memory(3.0, 259, 1072, 1910, 1600)
        self.assertLessEqual(splat.brush_memory_gb(259, 1072, 1910, tight.max_resolution, tight.subsample_frames), 3.0)
        self.assertIsNone(splat.plan_for_memory(1.0, 259, 1072, 1910, 1600))

    def test_top_memory_users_lists_real_processes(self):
        users = system.top_memory_users(3)
        self.assertTrue(users)
        self.assertTrue(all(gb >= 0 for _, gb in users))

    def test_splat_centers_reads_xyz_from_any_property_order(self):
        props = ["f_dc_0", "opacity", "x", "y", "z", "rot_0"]
        header = "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        header += "".join(f"property float {p}\n" for p in props) + "end_header\n"
        rows = np.array([[9, 9, 1, 2, 3, 9], [9, 9, 4, 5, 6, 9]], "<f4")
        path = self.dir / "s.ply"
        path.write_bytes(header.encode("ascii") + rows.tobytes())
        np.testing.assert_allclose(splat.splat_centers(path), [[1, 2, 3], [4, 5, 6]])


def box_mesh(lo, hi, inward=False):
    """Axis-aligned box as 12 triangles; inward=True flips normals (room walls seen from inside)."""
    v, f = cube()
    v = np.asarray(lo) + v * (np.asarray(hi) - np.asarray(lo))
    return v, (f[:, ::-1] if inward else f)


class TilesTest(TmpDirCase):
    def test_grid_shape_covers_requested_tiles(self):
        for a, b, n in ((10, 10, 4), (30, 5, 6), (1, 1, 1), (5, 20, 7)):
            rows, cols = tiles.grid_shape(a, b, n)
            self.assertGreaterEqual(rows * cols, n)
        self.assertEqual(tiles.grid_shape(10, 10, 4), (2, 2))

    def test_tiles_cover_the_plane_and_merge_keeps_each_splat_once(self):
        rng = np.random.default_rng(9)
        xz = rng.uniform(-2, 2, (4000, 2))
        images = [tiles.ImageRecord(i, b"", np.arange(i * 500, i * 500 + 500)) for i in range(8)]
        parts = tiles.partition(xz, np.arange(len(xz)), images, 4, 100)
        probe = rng.uniform(-5, 5, (2000, 2))                   # includes points far outside the core
        owners = sum(np.all((probe >= t.lo) & (probe < t.hi), axis=1).astype(int) for t in parts)
        np.testing.assert_array_equal(owners, 1)
        # Merge: identical splat sets per tile -> each splat survives exactly once.
        props = ["x", "y", "z", "opacity"]
        header = "ply\nformat binary_little_endian 1.0\nelement vertex {n}\n" + "".join(
            f"property float {p}\n" for p in props) + "end_header\n"
        table = np.c_[probe[:, 0], np.zeros(len(probe)), probe[:, 1], np.ones(len(probe))].astype("<f4")
        plys = []
        for k, t in enumerate(parts):
            ply = self.dir / f"t{k}.ply"
            ply.write_bytes(header.format(n=len(table)).encode() + table.tobytes())
            plys.append((ply, t))
        self.assertEqual(tiles.merge(plys, np.eye(4), self.dir / "merged.ply"), len(probe))

    def test_record_subsets_round_trip(self):
        path = self.dir / "images.bin"
        write_images_bin(path, [(np.array([1.0, 0, 0, 0]), np.zeros(3)), (np.array([1.0, 0, 0, 0]), np.ones(3))])
        records = tiles.read_images(path)
        tiles.write_records(self.dir / "subset.bin", [records[1].raw])
        self.assertEqual([r.image_id for r in tiles.read_images(self.dir / "subset.bin")], [2])


class RetrievalTest(unittest.TestCase):
    def test_pairs_cover_neighbours_and_a_revisit_but_not_everything(self):
        n = 100
        sim = np.eye(n)
        sim[0, 80] = sim[80, 0] = 0.9          # frame 80 revisits the start of the walk
        pairs = retrieval.choose_pairs(sim, neighbours=5, lookalikes=2)
        self.assertIn((0, 1), pairs)
        self.assertIn((0, 5), pairs)
        self.assertIn((0, 80), pairs)
        self.assertLess(len(pairs), n * (n - 1) // 2 // 4)


class MeshCleanTest(TmpDirCase):
    def test_drops_small_floating_pieces_and_keeps_the_main_surface(self):
        # Main surface: a 40x40 grid (3200 triangles). Floater: one lone triangle far away.
        n = 40
        xs, zs = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
        grid = np.c_[xs.ravel(), np.zeros(xs.size), zs.ravel()].astype(float)
        idx = lambda i, j: i * (n + 1) + j  # noqa: E731
        faces = [[idx(i, j), idx(i + 1, j), idx(i, j + 1)] for i in range(n) for j in range(n)]
        faces += [[idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for i in range(n) for j in range(n)]
        floater = np.random.default_rng(4).uniform(100, 101, (3, 3))
        verts = np.vstack([grid, floater])
        faces = np.array(faces + [[len(grid), len(grid) + 1, len(grid) + 2]])
        meshio.write_ply(self.dir / "in.ply", verts, faces)
        report = mesh_clean.clean(self.dir / "in.ply", self.dir / "out.ply")
        self.assertEqual((report.pieces_before, report.pieces_kept), (2, 1))
        self.assertEqual(report.faces_after, 2 * n * n)
        v, _ = meshio.read_ply(self.dir / "out.ply")
        self.assertLess(v.max(), 50)


class SystemTest(TmpDirCase):
    def test_run_lock_blocks_a_second_live_run(self):
        lock = self.dir / ".running"
        lock.write_text(str(os.getppid()), encoding="ascii")   # a process that is certainly alive
        with self.assertRaises(RuntimeError):
            with system.RunLock(lock):
                pass

    def test_run_lock_clears_a_stale_lock_and_releases(self):
        lock = self.dir / ".running"
        lock.write_text("999999", encoding="ascii")             # no such process
        with system.RunLock(lock):
            self.assertEqual(lock.read_text(encoding="ascii"), str(os.getpid()))
        self.assertFalse(lock.exists())

    def test_paging_files_keep_only_entries_with_a_fixed_maximum(self):
        parsed = system.parse_paging_files(["d:\\pagefile.sys 8192 24576", "?:\\pagefile.sys", "c:\\pagefile.sys 0 0"])
        self.assertEqual([(p.name, gb) for p, gb in parsed], [("pagefile.sys", 24.0)])

    def test_free_memory_reports_positive_numbers(self):
        physical, commit = system.free_memory_gb()
        self.assertGreater(physical, 0)
        self.assertGreaterEqual(commit, 0)


class ProgressTest(unittest.TestCase):
    def test_stages_map_to_monotonic_overall_progress(self):
        seen = []
        prog = pipeline.Progress(lambda p, _label: seen.append(p), frozenset({"splat"}))
        for key in ("frames", "sfm", "mvs", "orient"):
            step = prog.stage(key)
            step(0.5)
            step(1.0)
        self.assertEqual(seen, sorted(seen))
        self.assertAlmostEqual(seen[-1], 1.0)

    def test_unknown_stage_raises(self):
        with self.assertRaises(KeyError):
            pipeline.Progress(lambda *_: None, frozenset({"splat"})).stage("splat")


if __name__ == "__main__":
    unittest.main()
