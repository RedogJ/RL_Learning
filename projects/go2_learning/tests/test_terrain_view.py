"""地面显示层的 UV 对齐与物理几何隔离，无需 GL/Isaac Sim。"""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from go2_learning.ui.terrain_view import stair_surface, style_rough_terrain, terrain_texture
from go2_learning.simulation.terrain import stair_sections


class TerrainViewTests(unittest.TestCase):
    def points(self):
        x, y = np.meshgrid(np.arange(0., 1.25, .25), np.arange(0., 1.25, .25))
        return np.column_stack((x.ravel(), y.ravel(), (x.ravel() - .5) * .16)).astype(np.float32)

    def test_height_colors_match_vertices_and_survive_vertex_reordering(self):
        points = self.points()
        reference = points.copy()
        uvs, image = terrain_texture(points)
        order = np.random.default_rng(42).permutation(len(points))
        reordered_uvs, reordered_image = terrain_texture(points[order])
        np.testing.assert_array_equal(reordered_image, image)
        np.testing.assert_array_equal(reordered_uvs, uvs[order])
        np.testing.assert_array_equal(points, reference)
        self.assertEqual(image.dtype, np.uint8)
        self.assertTrue((uvs > 0).all() and (uvs < 1).all())
        # 纹理上传会翻转图像；UV 取样应仍命中原来的 x/y 像素。
        texture_x = np.floor(uvs[:, 0] * image.shape[1]).astype(int)
        texture_y = np.floor((1. - uvs[:, 1]) * image.shape[0]).astype(int)
        np.testing.assert_array_equal(texture_x, np.tile(np.arange(5), 5))
        np.testing.assert_array_equal(texture_y, np.repeat(np.arange(5), 5))
        # 取非格线处：低洼蓝色占优，凸起暖色占优。
        self.assertLess(image[1, 1, 0], image[1, 1, 2])
        self.assertGreater(image[1, 3, 0], image[1, 3, 2])

    def test_visual_overlay_preserves_collider_and_global_render_settings(self):
        points = self.points()
        indices = np.array([0, 1, 5], dtype=np.int32)
        original = points.copy()
        attr = lambda value: NS(Get=lambda: value)
        mesh = NS(GetPointsAttr=lambda: attr(points), GetFaceVertexIndicesAttr=lambda: attr(indices))
        renderer = NS(draw_edges=False)
        calls = []
        surface = NS(material=(1., 0., 0., 0.), texture_id=7)
        frame_events = []
        viewer = NS(device="cpu", renderer=renderer,
                    log_mesh=lambda *args, **kwargs: calls.append((args, kwargs)),
                    objects={"/go2/rough_terrain_surface": surface}, _qualify=lambda name: name,
                    begin_frame=lambda dt: frame_events.append("begin"),
                    end_frame=lambda: frame_events.append("end"))
        gl = NS(cfg=NS(visualizer_type="newton_gl"), _viewer=viewer)
        sim = NS(visualizers=[gl], stage=NS(GetPrimAtPath=lambda path: path))
        warp = NS(array=lambda value, **kwargs: np.array(value, copy=True), vec3=object(),
                  vec2=object(), int32=object())
        with patch.dict("sys.modules", {"warp": warp, "pxr": NS(UsdGeom=NS(Mesh=lambda path: mesh))}):
            style_rough_terrain(sim)
        np.testing.assert_array_equal(points, original)
        self.assertFalse(renderer.draw_edges)
        self.assertEqual(surface.material, (1., 0., 0., 1.))
        self.assertEqual(frame_events, ["begin", "end"])
        self.assertEqual(len(calls), 1)
        args, data = calls[0]
        self.assertEqual(args, ("/go2/rough_terrain_surface",))
        np.testing.assert_array_equal(data["points"][:, :2], original[:, :2])
        np.testing.assert_allclose(data["points"][:, 2], original[:, 2] + .001)
        np.testing.assert_array_equal(data["indices"], indices)

    def test_no_gl_visualizer_requires_no_render_backend(self):
        style_rough_terrain(NS(visualizers=[]))

    def test_stairs_start_on_flat_and_have_continuous_up_platform_down_sections(self):
        sections = np.array(stair_sections())
        self.assertAlmostEqual(sections[0, 0], 2.)
        np.testing.assert_allclose(sections[1:, 0], sections[:-1, 1])
        np.testing.assert_allclose(sections[:6, 2], np.arange(1, 7) * .05)
        np.testing.assert_allclose(sections[7:, 2], np.arange(5, 0, -1) * .05)
        self.assertAlmostEqual(sections[6, 1] - sections[6, 0], 1.4)
        self.assertAlmostEqual(sections[:, 2].max(), .3)

    def test_stair_vertical_and_horizontal_faces_offset_outward_without_changing_physics(self):
        # 一个向上的踏面和一个朝 -X 的立面，不能仅用 Z 偏移显示层。
        points = np.array([[2., 0., .05], [2.35, 0., .05], [2., 1., .05],
                           [2., 0., 0.], [2., 0., .05], [2., 1., 0.]], dtype=np.float32)
        reference = points.copy()
        indices = np.arange(6, dtype=np.int32)
        display, faces, uvs, image = stair_surface(points, indices)
        np.testing.assert_array_equal(points, reference)
        np.testing.assert_allclose(display[:3, 2], points[:3, 2] + .001)
        np.testing.assert_allclose(display[3:, 0], points[3:, 0] - .001)
        np.testing.assert_array_equal(faces, indices)
        self.assertEqual(uvs.shape, (6, 2))
        self.assertEqual(image.shape, (128, 128, 3))
        self.assertFalse(np.array_equal(image[10, 20], image[10, 90]))


if __name__ == "__main__":
    unittest.main()
