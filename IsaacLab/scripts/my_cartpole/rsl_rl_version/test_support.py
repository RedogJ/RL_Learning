# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""本项目的 CPU 假物理夹具；只验证数据接口，不模拟倒立摆动力学。"""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

# 仅加载本项目的 USD 场景函数；测试用假 USD 对象验证几何和克隆绑定。
scene_path = Path(__file__).with_name("scene.py")
scene_ast = ast.parse(scene_path.read_text())
functions = [node for node in scene_ast.body if isinstance(node, ast.FunctionDef)
             and node.name in ("configure_cartpole_scene_cfg", "configure_cartpole_joints")]
isolated_ast = ast.Module(body=functions, type_ignores=[])
scene_functions = {}
exec(compile(ast.fix_missing_locations(isolated_ast), str(scene_path), "exec"), scene_functions)
scene_module = SimpleNamespace(**scene_functions)


def make_cfg(num_envs=2):
    def asset_cfg(size, pos):
        return SimpleNamespace(spawn=SimpleNamespace(size=size), init_state=SimpleNamespace(pos=pos))

    cfg = SimpleNamespace(
        num_envs=num_envs,
        cart=asset_cfg((0.4, 0.3, 0.2), (0.0, 0.0, 2.6)),
        ball=asset_cfg((0.08, 0.08, 1.0), (0.0, 0.0, 0.0)),
        ball2=asset_cfg((0.08, 0.08, 1.0), (0.0, 0.0, 0.0)),
    )
    scene_module.configure_cartpole_scene_cfg(cfg)
    return cfg


def y_quaternion(angles):
    result = torch.zeros(len(angles), 4)
    result[:, 1] = torch.sin(angles / 2)
    result[:, 3] = torch.cos(angles / 2)
    return result


class FakeAsset:
    def __init__(self, scene, cfg):
        self.scene = scene
        pose = torch.zeros(len(scene.env_origins), 7)
        pose[:, :3] = torch.tensor(cfg.init_state.pos)
        pose[:, 6] = 1.0  # xyzw
        world_pose = pose.clone()
        world_pose[:, :3] += scene.env_origins
        self.data = SimpleNamespace(
            default_root_pose=SimpleNamespace(torch=pose),
            default_root_vel=SimpleNamespace(torch=torch.zeros(len(pose), 6)),
            root_link_pos_w=SimpleNamespace(torch=world_pose[:, :3]),
            root_link_quat_w=SimpleNamespace(torch=world_pose[:, 3:]),
            root_link_lin_vel_w=SimpleNamespace(torch=torch.zeros(len(pose), 3)),
            root_ang_vel_w=SimpleNamespace(torch=torch.zeros(len(pose), 3)),
        )
        self.instantaneous_wrench_composer = self

    def set_forces_and_torques_index(self, forces, is_global):
        self.scene.forces = forces.clone()

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        self.data.root_link_pos_w.torch[env_ids] = root_pose[:, :3]
        self.data.root_link_quat_w.torch[env_ids] = root_pose[:, 3:]

    def write_root_velocity_to_sim_index(self, root_velocity, env_ids):
        self.data.root_link_lin_vel_w.torch[env_ids] = root_velocity[:, :3]
        self.data.root_ang_vel_w.torch[env_ids] = root_velocity[:, 3:]


class FakeScene:
    def __init__(self, cfg):
        self.env_origins = torch.zeros(cfg.num_envs, 3)
        self.env_origins[:, 0] = torch.arange(cfg.num_envs) * 2.0
        self.forces = torch.zeros(cfg.num_envs, 1, 3)
        self.assets = {name: FakeAsset(self, getattr(cfg, name)) for name in ("cart", "ball", "ball2")}

    def __getitem__(self, name):
        return self.assets[name]

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass

    def reset(self, env_ids):
        pass


class FakeSim:
    device = "cpu"

    def __init__(self, scene):
        self.scene = scene

    def get_physics_dt(self):
        return 1 / 120

    def step(self, render):
        # This fixture checks data flow, not double-pendulum dynamics.
        cart = self.scene["cart"].data
        cart.root_link_lin_vel_w.torch[:, 0] += self.scene.forces[:, 0, 0] / 120
        cart.root_link_pos_w.torch[:, 0] += cart.root_link_lin_vel_w.torch[:, 0] / 120


class FakeJoint:
    def __init__(self):
        self.values = {}

    def CreateBody0Rel(self):
        return SimpleNamespace(SetTargets=lambda targets: self.values.update(body0=targets))

    def CreateBody1Rel(self):
        return SimpleNamespace(SetTargets=lambda targets: self.values.update(body1=targets))

    def CreateLocalPos0Attr(self, value):
        self.values["anchor0"] = value

    def CreateCollisionEnabledAttr(self, value):
        self.values["collision"] = value


