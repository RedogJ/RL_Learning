# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""环境适配：原 CartpoleEnv 的五项返回值 → RSL-RL 的 VecEnv 接口。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

if TYPE_CHECKING:
    from env import CartpoleEnv


class CartpoleVecEnv(VecEnv):
    """复用原环境的物理、动作、奖励和重置，只转换训练接口。"""

    def __init__(self, env: CartpoleEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = 1
        self.device = env.sim.device
        self.max_episode_length = env.max_episode_steps
        # 使用可序列化的配置，供 Runner 的日志和 checkpoint 校验使用。
        self.cfg = {
            "task": "my_double_cartpole",
            "observation_order": ["x", "x_dot", "theta1", "theta1_dot", "theta2", "theta2_dot"],
            "num_actions": self.num_actions,
            "step_dt": env.step_dt,
            "force_limit": env.force_limit,
            "max_cart_position": env.max_cart_position,
            "max_pole_angle": env.max_pole_angle,
            "max_episode_length": self.max_episode_length,
        }
        # Runner 初始化时会读取观测，先完成首次 reset。
        self._obs = self._pack(env.reset())

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_steps

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_steps.copy_(value)

    def _pack(self, observation: torch.Tensor) -> TensorDict:
        # 独立快照：不能让下一步/reset 原地改掉 rollout 里的旧观测。
        return TensorDict({"policy": observation.clone()}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # 原环境内部负责 [-1, 1] 限幅；这里不原地修改策略采样的动作。
        observation, rewards, terminated, truncated, info = self.env.step(actions)
        dones = terminated | truncated
        extras = dict(info)
        # RSL-RL 根据 time_outs 对超时进行价值补偿，失败不补偿。
        # 5.5.1 使用当前状态 V(s_t)，与原手写实现 V(s_{t+1}) 略有不同。
        extras["time_outs"] = truncated.clone()
        extras["log"] = {
            "Task/failures_per_step": terminated.float().sum(),
            "Task/timeouts_per_step": truncated.float().sum(),
        }
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel():
            # Runner 不会替原环境 reset；在适配器中只重置已结束的环境。
            observation = self.env.reset(done_ids)
        self._obs = self._pack(observation)
        return self._obs, rewards, dones.long(), extras
