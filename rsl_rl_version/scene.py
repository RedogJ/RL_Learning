# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""双杆场景：世界 → X 轴滑块 → 第一根杆 → 第二根杆。"""

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

@configclass
class PoleSpawnCfg(sim_utils.CuboidCfg):
    # 本环境内的父刚体，以及父刚体局部坐标系中的连接点高度 [m]。
    parent_body: str = "Cart"
    parent_anchor_z: float = 0.0


@sim_utils.clone
def spawn_cart(
    prim_path,
    cfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    prim = sim_utils.spawn_cuboid(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    stage = sim_utils.get_current_stage()

    joint = UsdPhysics.PrismaticJoint.Define(
        stage,
        f"{prim_path}/SlideJoint",
    )

    # 世界连接小车
    joint.CreateBody1Rel().SetTargets([Sdf.Path(prim_path)])

    anchor = translation if translation is not None else (0.0, 0.0, 0.0)

    joint.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(-2.0)
    joint.CreateUpperLimitAttr(2.0)

    return prim

@sim_utils.clone
def spawn_pole(
    prim_path: str,
    cfg: PoleSpawnCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """创建杆，并通过 Y 轴转动关节连接到父刚体的上端。"""
    prim = sim_utils.spawn_cuboid(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    stage = sim_utils.get_current_stage()

    joint = UsdPhysics.RevoluteJoint.Define(
        stage,
        f"{prim_path}/HingeJoint",
    )

    # 第一根杆的父刚体是 Cart，第二根杆的父刚体是 Ball。
    env_path = prim_path.rsplit("/", 1)[0]
    parent_path = f"{env_path}/{cfg.parent_body}"

    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(prim_path)])

    half_length = cfg.size[2] / 2

    # 父刚体端：小车上表面或第一根杆的上端。
    joint.CreateLocalPos0Attr(
        Gf.Vec3f(0.0, 0.0, cfg.parent_anchor_z)
    )

    # 杆端：杆的下端
    joint.CreateLocalPos1Attr(
        Gf.Vec3f(0.0, 0.0, -half_length)
    )


    # 绕 Y 轴转动，所以杆在 X-Z 平面内摆动
    joint.CreateAxisAttr("Y")

    joint.CreateLowerLimitAttr(-90.0)
    joint.CreateUpperLimitAttr(90.0)
    # 接头附近几何可能交叠，连接的两刚体不相互碰撞。
    joint.CreateCollisionEnabledAttr(False)

    return prim


def configure_cartpole_scene_cfg(scene_cfg):
    """根据刚体长度设置双杆初始位置与局部锚点 [m]，创建场景前调用。"""
    cart_pos = scene_cfg.cart.init_state.pos
    cart_half_height = scene_cfg.cart.spawn.size[2] / 2
    lower_half_length = scene_cfg.ball.spawn.size[2] / 2
    upper_half_length = scene_cfg.ball2.spawn.size[2] / 2

    scene_cfg.ball.spawn.parent_body = "Cart"
    scene_cfg.ball.spawn.parent_anchor_z = cart_half_height
    scene_cfg.ball.init_state.pos = (
        cart_pos[0], cart_pos[1], cart_pos[2] + cart_half_height + lower_half_length
    )
    scene_cfg.ball2.spawn.parent_body = "Ball"
    scene_cfg.ball2.spawn.parent_anchor_z = lower_half_length
    scene_cfg.ball2.init_state.pos = (
        cart_pos[0], cart_pos[1],
        scene_cfg.ball.init_state.pos[2] + lower_half_length + upper_half_length,
    )


def configure_cartpole_joints(scene):
    """修正每个环境的关节连接和世界锚点；必须在 sim.reset() 前调用。"""
    stage = sim_utils.get_current_stage()
    transforms = UsdGeom.XformCache()

    for env_path in scene.env_prim_paths:
        cart_path = f"{env_path}/Cart"
        slide = UsdPhysics.PrismaticJoint.Get(stage, f"{cart_path}/SlideJoint")
        if not slide:
            raise RuntimeError(f"环境关节缺失：{env_path}")

        # 小车的世界位置包含环境原点的偏移。
        cart_transform = transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(cart_path))
        anchor = cart_transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        slide.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
        slide.CreateBody1Rel().SetTargets([Sdf.Path(cart_path)])

        # 两个转动关节分别连接 Cart→Ball、Ball→Ball2。
        # 复制资产后，显式绑定本环境的两端，避免跨环境连接。
        for parent_name, pole_name in (("Cart", "Ball"), ("Ball", "Ball2")):
            pole_path = f"{env_path}/{pole_name}"
            hinge = UsdPhysics.RevoluteJoint.Get(stage, f"{pole_path}/HingeJoint")
            if not hinge:
                raise RuntimeError(f"环境关节缺失：{pole_path}/HingeJoint")
            hinge.CreateBody0Rel().SetTargets([Sdf.Path(f"{env_path}/{parent_name}")])
            hinge.CreateBody1Rel().SetTargets([Sdf.Path(pole_path)])
            hinge.CreateCollisionEnabledAttr(False)


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.1),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.05),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
        ),
    )

    cart = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cart",
        spawn=sim_utils.CuboidCfg(
            func=spawn_cart,
            size=(0.4, 0.3, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.6)),
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=PoleSpawnCfg(
            func=spawn_pole,
            size=(0.08, 0.08, 1.0),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),
        ),
    )

    ball2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball2",
        spawn=PoleSpawnCfg(
            func=spawn_pole,
            parent_body="Ball",
            size=(0.08, 0.08, 1.0),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.4, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 4.2)),
    )
