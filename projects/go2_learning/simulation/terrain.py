"""现有策略的崎岖地形测试；本地生成，无需外部资源。"""

STAIR_STEP_HEIGHT = 0.05
STAIR_STEP_DEPTH = 0.35
STAIR_STEPS = 6
STAIR_START_X = 2.0
STAIR_PLATFORM_LENGTH = 1.4
STAIR_WIDTH = 4.0


def stair_sections():
    """楼梯各水平区间 (起点 X, 终点 X, 高度)，坐标相对机器人出生点。"""
    sections = []
    x = STAIR_START_X
    for level in range(1, STAIR_STEPS + 1):
        sections.append((x, x + STAIR_STEP_DEPTH, level * STAIR_STEP_HEIGHT))
        x += STAIR_STEP_DEPTH
    sections.append((x, x + STAIR_PLATFORM_LENGTH, STAIR_STEPS * STAIR_STEP_HEIGHT))
    x += STAIR_PLATFORM_LENGTH
    for level in range(STAIR_STEPS - 1, 0, -1):
        sections.append((x, x + STAIR_STEP_DEPTH, level * STAIR_STEP_HEIGHT))
        x += STAIR_STEP_DEPTH
    return sections


def straight_stairs_terrain(difficulty, cfg):
    """生成平地 → 上楼 → 平台 → 下楼的固定三角网格。"""
    import numpy as np
    import trimesh

    cx, cy = cfg.size[0] / 2, cfg.size[1] / 2
    base_depth = 0.15

    def box(size, center):
        return trimesh.creation.box(size, trimesh.transformations.translation_matrix(center))

    meshes = [box((*cfg.size, base_depth), (cx, cy, -base_depth / 2))]
    for start, end, height in stair_sections():
        meshes.append(box((end - start, STAIR_WIDTH, height + base_depth),
                          (cx + (start + end) / 2, cy, (height - base_depth) / 2)))
    # 生成器随后会平移 XY，使出生点落在世界原点的平地区域。
    return meshes, np.array([cx, cy, 0.])


def make_stairs_terrain_cfg(seed=42):
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg

    return TerrainImporterCfg(
        prim_path="/World/Ground",
        terrain_type="generator",
        use_terrain_origins=False,
        terrain_generator=TerrainGeneratorCfg(
            seed=seed, size=(24., 12.), num_rows=1, num_cols=1, use_cache=False,
            sub_terrains={"stairs": MeshPlaneTerrainCfg(function=straight_stairs_terrain)},
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5, dynamic_friction=0.5, restitution=0.,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.60, 0.68), roughness=1.),
        debug_vis=False,
    )


def bounded_random_terrain(difficulty, cfg):
    from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain

    meshes, origin = random_uniform_terrain(difficulty, cfg)
    # 三次插值可能超出采样范围；限制最终碰撞几何的高度。
    for mesh in meshes:
        mesh.vertices[:, 2] = mesh.vertices[:, 2].clip(-0.08, 0.08)
    origin[2] = origin[2].clip(-0.08, 0.08)
    return meshes, origin


def make_rough_terrain_cfg(seed=42):
    import isaaclab.sim as sim_utils
    from isaaclab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg

    return TerrainImporterCfg(
        prim_path="/World/Ground",
        terrain_type="generator",
        # 保留平地模型使用的世界高度基准与环境原点。
        use_terrain_origins=False,
        terrain_generator=TerrainGeneratorCfg(
            seed=seed,
            size=(64.0, 64.0),
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.125,
            vertical_scale=0.002,
            slope_threshold=None,
            use_cache=False,
            sub_terrains={
                "rough": HfRandomUniformTerrainCfg(
                    function=bounded_random_terrain,
                    noise_range=(-0.06, 0.06),
                    noise_step=0.002,
                    downsampled_scale=0.5,
                    # PhysX 与 GL 共用三角网格，避免引入后端高度场转换。
                    convert_to_heightfield=False,
                ),
            },
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5, dynamic_friction=0.5, restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.48, 0.60, 0.68), roughness=1.0,
        ),
        debug_vis=False,
    )
