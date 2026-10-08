# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import time

import torch

from scene import read_cartpole_state


class CartpoleEnv:
    def __init__(self, sim, scene, scene_cfg, history_steps=5):
        self.sim = sim
        self.scene = scene
        self.scene_cfg = scene_cfg

        # 一条底层控制指令推进一个物理步；一个策略决策包含多条指令。
        self.step_dt = sim.get_physics_dt()

        # 在小车撞到 ±2 米的行程限制之前结束
        self.max_cart_position = 1.8

        # 我们先训练竖直附近的平衡
        self.max_pole_angle = math.radians(30.0)


        self.cart = scene["cart"]
        self.pole = scene["ball"]
        self.num_envs = scene_cfg.num_envs
        self.history_steps = history_steps
        self.history = torch.zeros((self.num_envs, history_steps, 4), device=sim.device)

        self.force_limit = 20.0  # N
        self.max_episode_steps = math.ceil(5.0 / self.step_dt)

        self.episode_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=sim.device,
        )

        self.forces = torch.zeros(
            (self.num_envs, 1, 3),
            device=sim.device,
        )

    def get_observation(self):
        x, x_dot, theta, theta_dot = read_cartpole_state(
            self.scene,
            self.scene_cfg,
        )

        return torch.stack(
            (x, x_dot, theta, theta_dot),
            dim=-1,
        )

    def get_history(self):
        """返回从旧到新的状态历史，最后一帧为当前状态。"""
        return self.history.clone()

    def step(self, actions, active_envs=None):
        if active_envs is None:
            active_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.sim.device)
        # 输入形状：[环境数量, 1]
        # 网络动作先限制到 [-1, 1]
        applied_actions = actions.detach().clamp(-1.0, 1.0)
        applied_actions = applied_actions * active_envs.unsqueeze(-1)

        # 归一化动作转换为实际推力
        self.forces[:, 0, 0] = (
            applied_actions[:, 0] * self.force_limit
        )

        self.cart.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=self.forces,
            is_global=True,
        )

        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(self.step_dt)

        self.episode_steps += active_envs.long()

        # 读取动作执行后的状态
        observation = self.get_observation()
        next_history = torch.cat((self.history[:, 1:], observation.unsqueeze(1)), dim=1)
        self.history = torch.where(active_envs[:, None, None], next_history, self.history)
        rewards = self.compute_reward(observation, applied_actions) * active_envs
        terminated = self.compute_terminated(observation) & active_envs

        # 没有失败，但达到时间上限
        truncated = (
            self.episode_steps >= self.max_episode_steps
        ) & ~terminated & active_envs

        return observation, rewards, terminated, truncated, {}

    def step_sequence(self, actions, gamma=0.99, real_time=False):
        """逐个物理步执行一段动作；结束的环境在段内停止贡献奖励和历史。"""
        if actions.ndim != 3 or actions.shape[0] != self.num_envs or actions.shape[2] != 1:
            raise ValueError("动作序列应为 [环境数量, 控制步数, 1]")
        if actions.shape[1] < 1:
            raise ValueError("动作序列不能为空")
        rewards = torch.zeros(self.num_envs, device=self.sim.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)
        truncated = torch.zeros_like(terminated)
        action_masks = torch.zeros(actions.shape[:2], dtype=torch.bool, device=self.sim.device)
        observation = self.history[:, -1].clone()

        for index in range(actions.shape[1]):
            step_start = time.perf_counter() if real_time else 0.0
            active_envs = ~(terminated | truncated)
            if not active_envs.any().item():
                break
            action_masks[:, index] = active_envs
            next_observation, step_rewards, step_terminated, step_truncated, _ = self.step(
                actions[:, index], active_envs=active_envs
            )
            rewards += gamma**index * step_rewards
            observation = torch.where(active_envs[:, None], next_observation, observation)
            terminated |= step_terminated
            truncated |= step_truncated
            if real_time:
                time.sleep(max(0.0, self.step_dt - (time.perf_counter() - step_start)))

        info = {"action_masks": action_masks, "durations": action_masks.sum(dim=-1)}
        return observation, rewards, terminated, truncated, info

    def compute_reward(self, observation, applied_actions):
        x = observation[:, 0]
        theta = observation[:, 2]

        # applied_actions 是限幅后实际使用的归一化动作，
        # 形状为 [环境数量, 1]，不是以 N 为单位的推力。
        action = applied_actions.squeeze(-1)

        reward_rate = (
            1.0
            - theta.square()
            - 0.1 * x.square()
            - 0.001 * action.square()
        )

        return reward_rate * self.step_dt

    def compute_terminated(self, observation):
        x = observation[:, 0]
        theta = observation[:, 2]

        cart_out_of_bounds = x.abs() > self.max_cart_position
        pole_fallen = theta.abs() > self.max_pole_angle

        return cart_out_of_bounds | pole_fallen

    def reset(self, env_ids=None):
        # 不指定编号，就重置全部环境
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs,
                device=self.sim.device,
            )

        # 同时恢复小车和杆的位置、姿态、速度
        for asset in (self.cart, self.pole):
            root_pose = asset.data.default_root_pose.torch[env_ids].clone()
            root_pose[:, :3] += self.scene.env_origins[env_ids]

            root_velocity = (
                asset.data.default_root_vel.torch[env_ids].clone()
            )

            asset.write_root_pose_to_sim_index(
                root_pose=root_pose,
                env_ids=env_ids,
            )
            asset.write_root_velocity_to_sim_index(
                root_velocity=root_velocity,
                env_ids=env_ids,
            )

        self.scene.reset(env_ids)
        self.forces[env_ids] = 0.0
        self.episode_steps[env_ids] = 0

        observation = self.get_observation()
        # 新回合没有真实历史，用其初始状态填充；不继承上一回合的帧。
        self.history[env_ids] = observation[env_ids].unsqueeze(1)
        return observation
