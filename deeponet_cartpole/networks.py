# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    """DeepONet 策略：历史状态经 Branch 编码，Trunk 按段内时间查询控制。"""

    def __init__(self, history_steps=5, control_steps=5, step_dt=1 / 120, feature_dim=32):
        super().__init__()
        if min(history_steps, control_steps, feature_dim) < 1 or step_dt <= 0:
            raise ValueError("历史帧数、控制步数、特征维数和物理步长必须为正")
        self.history_steps = history_steps
        self.control_steps = control_steps
        self.control_period = control_steps * step_dt
        self.feature_dim = feature_dim

        self.branch = nn.Sequential(
            nn.Linear(history_steps * 4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, feature_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, feature_dim),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.log_std = nn.Parameter(torch.full((1,), math.log(0.5)))
        self.register_buffer("control_times", torch.arange(control_steps, dtype=torch.float32) * step_dt)

    def encode_history(self, history):
        """每个控制段只调用一次；history 形状为 [N, 历史帧数, 4]。"""
        if history.ndim != 3 or history.shape[1:] != (self.history_steps, 4):
            raise ValueError(f"历史输入应为 [N, {self.history_steps}, 4]")
        return self.branch(history.flatten(start_dim=1))

    def evaluate_control(self, branch_features, query_times):
        """查询控制均值；query_times 是段起点后的秒数，返回 [N, 查询数, 1]。"""
        times = torch.as_tensor(query_times, device=branch_features.device, dtype=branch_features.dtype)
        trunk_features = self.trunk(times.reshape(-1, 1) / self.control_period)
        mean = branch_features @ trunk_features.T / math.sqrt(self.feature_dim) + self.bias
        return mean.unsqueeze(-1)

    def forward(self, history):
        features = self.encode_history(history)
        return self.evaluate_control(features, self.control_times)

    def get_distribution(self, history):
        # PPO 在实际执行的时间点上采样；播放时直接使用连续均值函数。
        mean = self(history)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)


class ValueNetwork(nn.Module):
    def __init__(self, history_steps=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(history_steps * 4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, history):
        return self.net(history.flatten(start_dim=1)).squeeze(-1)
