"""只给崎岖地面添加高度配色与网格，不改变机器人显示或物理碰撞。"""

import numpy as np


def terrain_texture(points):
    """规则高度网格的 UV 与 RGB 纹理；蓝色低洼、黄色凸起，格线间距 1 m。"""
    x, xi = np.unique(points[:, 0], return_inverse=True)
    y, yi = np.unique(points[:, 1], return_inverse=True)
    height = np.empty((len(y), len(x)), dtype=np.float32)
    height[yi, xi] = points[:, 2]
    fraction = np.clip((height + 0.08) / 0.16, 0., 1.)
    low = np.array([0.18, 0.36, 0.55])
    mid = np.array([0.56, 0.66, 0.54])
    high = np.array([0.94, 0.77, 0.39])
    lower = low + (mid - low) * (2 * fraction[..., None])
    upper = mid + (high - mid) * (2 * fraction[..., None] - 1)
    rgb = np.where(fraction[..., None] < .5, lower, upper)
    grid = (np.isclose(np.remainder(x - x[0], 1.), 0.)[None, :]
            | np.isclose(np.remainder(y - y[0], 1.), 0.)[:, None])
    rgb[grid] *= .55
    # Newton 上传图像时上下翻转，V 反向后颜色才对应真实地形高度。
    uvs = np.column_stack(((xi + .5) / len(x), 1. - (yi + .5) / len(y))).astype(np.float32)
    return uvs, np.ascontiguousarray((rgb * 255).round().astype(np.uint8))


def stair_surface(points, indices):
    """楼梯的逐面显示网格与交替踏面纹理，适用于带竖直面的非高度网格。"""
    from go2_learning.simulation.terrain import STAIR_START_X, STAIR_STEP_DEPTH

    triangles = points[indices.reshape(-1, 3)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    # 竖直面也沿自身法线偏移，避免显示层与碰撞网格深度闪烁。
    display_points = (triangles + .001 * normals[:, None, :]).reshape(-1, 3)
    original_points = triangles.reshape(-1, 3)
    uvs = np.column_stack(((original_points[:, 0] - STAIR_START_X) / (2 * STAIR_STEP_DEPTH),
                           -original_points[:, 1])).astype(np.float32)
    image = np.empty((128, 128, 3), dtype=np.uint8)
    image[:, :64] = [170, 196, 211]
    image[:, 64:] = [100, 137, 163]
    image[:, [0, 1, 64, 65]] = [30, 45, 60]
    image[:2] = [30, 45, 60]
    return display_points, np.arange(len(display_points), dtype=np.int32), uvs, image


def style_rough_terrain(sim, terrain="rough"):
    visualizers = [v for v in sim.visualizers if v.cfg.visualizer_type == "newton_gl"]
    if not visualizers:
        return
    import warp as wp
    from pxr import UsdGeom

    # 从真正的碰撞地形读同一份几何，GL 的 log_mesh 只注册显示网格。
    mesh = UsdGeom.Mesh(sim.stage.GetPrimAtPath("/World/Ground/terrain/mesh"))
    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    if terrain == "stairs":
        display_points, indices, uvs, texture = stair_surface(points, indices)
    else:
        uvs, texture = terrain_texture(points)
        # 此地面没有额外的父级位姿，点本身已在世界坐标。
        display_points = points.copy()
        display_points[:, 2] += .001  # 仅显示层抬高 1 mm，避免与原地面发生深度闪烁。
    for visualizer in visualizers:
        viewer = visualizer._viewer
        device = viewer.device
        name = "/go2/rough_terrain_surface"
        # 一次性注册静态显示网格；GL 只允许在帧内提交几何。
        viewer.begin_frame(0.)
        try:
            viewer.log_mesh(
                name,
                points=wp.array(display_points, dtype=wp.vec3, device=device),
                indices=wp.array(indices, dtype=wp.int32, device=device),
                uvs=wp.array(uvs, dtype=wp.vec2, device=device),
                texture=texture,
                color=(1., 1., 1.),
                roughness=1.,
                metallic=0.,
            )
            surface = viewer.objects[viewer._qualify(name)]
            if surface.texture_id is None:
                raise RuntimeError("崎岖地面纹理上传失败，不能显示地形高度配色")
            # MeshGL 上传纹理后仍默认 material.w=0；着色器因此只绘制白色底色。
            roughness, metallic, checker, _ = surface.material
            surface.material = (roughness, metallic, checker, 1.)
        finally:
            viewer.end_frame()
    print("地面显示：GL 纹理已启用；" + (
        "楼梯踏面交替配色、深色台阶边缘" if terrain == "stairs" else "蓝色低洼、黄色凸起"
    ))
