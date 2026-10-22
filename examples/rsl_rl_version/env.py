# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""直接实现 RSL-RL VecEnv：观测、动作、奖励、结束和自动重置。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    from .scene import CartpoleSceneCfg


def make_env(sim: SimulationContext, num_envs: int) -> CartpoleEnv:
    """创建场景并返回环境；在 launch_simulation 内调用。"""
    from isaaclab.scene import InteractiveScene

    from .scene import CartpoleSceneCfg, configure_cartpole_joints, configure_cartpole_scene_cfg

    cfg = CartpoleSceneCfg(num_envs=num_envs, env_spacing=2.0, replicate_physics=False)
    configure_cartpole_scene_cfg(cfg)
    scene = InteractiveScene(cfg)
    configure_cartpole_joints(scene)
    sim.set_camera_view(eye=(5.0, 5.0, 5.0), target=(0.0, 0.0, 3.5))
    sim.reset()
    return CartpoleEnv(sim, scene, cfg)


class CartpoleEnv(VecEnv):
    """6 维观测、1 维动作；物理状态由 Isaac Lab 的 scene 管理。"""

    def __init__(self, sim: SimulationContext, scene: InteractiveScene, scene_cfg: CartpoleSceneCfg):
        self.sim, self.scene, self.scene_cfg = sim, scene, scene_cfg
        self.device = sim.device
        self.num_envs = scene_cfg.num_envs
        self.num_actions = 1
        self.cfg = {}
        self.dt = sim.get_physics_dt()
        self.force_limit = 20.0  # N，动作 [-1, 1] 对应 [-20, 20] N
        self.max_cart_position = 1.8  # m
        self.max_pole_angle = math.radians(30.0)
        self.max_episode_length = math.ceil(5.0 / self.dt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.reset()

    def get_observations(self) -> TensorDict:
        """返回 [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]，单位 m、s、rad。"""
        cart = self.scene["cart"].data
        x = cart.root_link_pos_w.torch[:, 0] - self.scene.env_origins[:, 0] - self.scene_cfg.cart.init_state.pos[0]
        states = [x, cart.root_link_lin_vel_w.torch[:, 0]]
        for name in ("ball", "ball2"):
            pole = self.scene[name].data
            # Isaac Lab 3 的四元数顺序为 xyzw。把杆的局部 Z 轴转到世界，求绝对倾角。
            qx, qy, qz, qw = pole.root_link_quat_w.torch.unbind(dim=-1)
            theta = torch.atan2(2 * (qx * qz + qw * qy), 1 - 2 * (qx.square() + qy.square()))
            states.extend((theta, pole.root_ang_vel_w.torch[:, 1]))
        # stack 创建独立快照，下一步/reset 不会覆盖 Runner 留存的旧观测。
        return TensorDict({"policy": torch.stack(states, dim=-1)}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """施加小车推力，推进一步，并自动重置本步结束的环境。"""
        applied = actions.detach().clamp(-1.0, 1.0)
        self.forces[:, 0, 0] = applied[:, 0] * self.force_limit
        self.scene["cart"].instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=self.forces, is_global=True,
        )
        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(self.dt)
        self.episode_length_buf += 1

        obs = self.get_observations()
        x, _, theta1, _, theta2, _ = obs["policy"].unbind(dim=-1)
        rewards = (
            1 - theta1.square() - theta2.square() - 0.1 * x.square() - 0.001 * applied[:, 0].square()
        ) * self.dt
        failed = (
            (x.abs() > self.max_cart_position)
            | (theta1.abs() > self.max_pole_angle)
            | (theta2.abs() > self.max_pole_angle)
        )
        timed_out = (self.episode_length_buf >= self.max_episode_length) & ~failed
        dones = failed | timed_out
        extras = {
            "time_outs": timed_out,  # 标准 PPO 用于超时价值补偿；失败不补偿。
            "log": {
                "Task/failures_per_step": failed.float().sum(),
                "Task/timeouts_per_step": timed_out.float().sum(),
            },
        }
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel():
            self.reset(done_ids)
            obs = self.get_observations()
        return obs, rewards, dones.long(), extras

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """恢复指定环境中三个刚体的初始位姿和速度。"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        for name in ("cart", "ball", "ball2"):
            asset = self.scene[name]
            pose = asset.data.default_root_pose.torch[env_ids].clone()
            pose[:, :3] += self.scene.env_origins[env_ids]
            velocity = asset.data.default_root_vel.torch[env_ids].clone()
            asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        self.scene.reset(env_ids)
        self.forces[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
