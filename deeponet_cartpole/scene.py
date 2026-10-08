# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import torch

@configclass
class PoleSpawnCfg(sim_utils.CuboidCfg):
    # 创建场景前，由 main.py 根据小车尺寸填写
    cart_half_height: float = 0.0


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
def spawn_slider(
    prim_path: str,
    cfg:  PoleSpawnCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """创建杆，并通过 Y 轴转动关节连接到小车上表面。"""
    prim = sim_utils.spawn_cuboid(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    stage = sim_utils.get_current_stage()
    jstage = sim_utils.get_current_stage()

    joint = UsdPhysics.RevoluteJoint.Define(
        stage,
        f"{prim_path}/HingeJoint",
    )

    # 找到同一环境下的小车路径
    env_path = prim_path.rsplit("/", 1)[0]
    cart_path = f"{env_path}/Cart"

    # body0 是小车，body1 是杆
    joint.CreateBody0Rel().SetTargets([Sdf.Path(cart_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(prim_path)])

    half_length = cfg.size[2] / 2

    # 小车端：根据小车配置计算上表面中心。
    cart_half_height = cfg.cart_half_height
    joint.CreateLocalPos0Attr(
        Gf.Vec3f(0.0, 0.0, cart_half_height)
    )

    # 杆端：杆的下端
    joint.CreateLocalPos1Attr(
        Gf.Vec3f(0.0, 0.0, -half_length)
    )


    # 绕 Y 轴转动，所以杆在 X-Z 平面内摆动
    joint.CreateAxisAttr("Y")

    joint.CreateLowerLimitAttr(-90.0)
    joint.CreateUpperLimitAttr(90.0)

    return prim


def configure_cartpole_joints(scene):
    """修正每个环境的关节连接和世界锚点；必须在 sim.reset() 前调用。"""
    stage = sim_utils.get_current_stage()
    transforms = UsdGeom.XformCache()

    for env_path in scene.env_prim_paths:
        cart_path = f"{env_path}/Cart"
        pole_path = f"{env_path}/Ball"
        slide = UsdPhysics.PrismaticJoint.Get(stage, f"{cart_path}/SlideJoint")
        hinge = UsdPhysics.RevoluteJoint.Get(stage, f"{pole_path}/HingeJoint")
        if not slide or not hinge:
            raise RuntimeError(f"环境关节缺失：{env_path}")

        # 小车的世界位置包含环境原点的偏移。
        cart_transform = transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(cart_path))
        anchor = cart_transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        slide.CreateLocalPos0Attr(Gf.Vec3f(*anchor))
        slide.CreateBody1Rel().SetTargets([Sdf.Path(cart_path)])

        # 复制资产后，显式绑定本环境的两端，避免跨环境连接。
        hinge.CreateBody0Rel().SetTargets([Sdf.Path(cart_path)])
        hinge.CreateBody1Rel().SetTargets([Sdf.Path(pole_path)])


@configclass
class DropSceneCfg(InteractiveSceneCfg):
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
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.6),
    ),
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=PoleSpawnCfg(
            func=spawn_slider,
            size=(0.08, 0.08, 1.0),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),
        ),
    )

def read_cartpole_state(scene, scene_cfg):
    """返回各环境的小车位置 [m]、速度 [m/s]、杆角度 [rad]、角速度 [rad/s]。"""
    cart = scene["cart"]
    pole = scene["ball"]

    cart_position = cart.data.root_link_pos_w.torch
    pole_position = pole.data.root_com_pos_w.torch

    # 小车相对初始中心的位置
    x = (
        cart_position[:, 0]
        - scene.env_origins[:, 0]
        - scene_cfg.cart.init_state.pos[0]
    )
    x_dot = cart.data.root_link_lin_vel_w.torch[:, 0]

    # 小车上表面的连接点
    pivot_x = cart_position[:, 0]
    pivot_z = (
        cart_position[:, 2]
        + scene_cfg.ball.spawn.cart_half_height
    )

    theta = torch.atan2(
        pole_position[:, 0] - pivot_x,
        pole_position[:, 2] - pivot_z,
    )
    theta_dot = pole.data.root_ang_vel_w.torch[:, 1]

    return x, x_dot, theta, theta_dot
