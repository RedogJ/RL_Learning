"""每个环境独立的姿态与高度误差积分，与仿真器、PPO 和奖励公式解耦。"""

import math
from dataclasses import dataclass

import torch


@dataclass
class IntegralCfg:
    limit: float = 0.5  # 每轴积分限幅，rad·s；限制历史误差的积累。
    height_limit: float = 0.2  # 高度误差积分限幅，m·s；与姿态积分分别设置。
    decay_seconds: float = 5.0  # 衰减时间常数，s；旧偏差逐渐遗忘，减少反向补偿。

    def __post_init__(self):
        if not math.isfinite(self.limit) or self.limit <= 0:
            raise ValueError("积分 limit 必须为正数且有限")
        if not math.isfinite(self.height_limit) or self.height_limit <= 0:
            raise ValueError("积分 height_limit 必须为正数且有限")
        if not math.isfinite(self.decay_seconds) or self.decay_seconds <= 0:
            raise ValueError("积分 decay_seconds 必须为正数且有限")


class AttitudeIntegral:
    def __init__(self, num_envs, device, dt, cfg):
        self.cfg, self.dt = cfg, dt
        self.decay = math.exp(-dt / cfg.decay_seconds)
        self.values = torch.zeros((num_envs, 3), device=device)  # roll / pitch / height。
        self.limits = torch.tensor([cfg.limit, cfg.limit, cfg.height_limit], device=device)

    @torch.no_grad()
    def update(self, roll, pitch, height_error, active=None):
        """I = decay * I + error * dt；高度误差为实际高度减目标高度。"""
        errors = torch.stack((roll, pitch, height_error), dim=-1)
        updated = (self.values * self.decay + errors * self.dt).clamp(-self.limits, self.limits)
        if active is None:
            self.values.copy_(updated)
        else:
            self.values[active] = updated[active]

    def observations(self):
        """各轴按自己的限幅归一化到 [-1, 1]；奖励使用原始 rad·s / m·s 数值。"""
        return self.values / self.limits

    def reset(self, env_ids):
        """仅清零指定环境，其他环境的历史保留。"""
        self.values[env_ids] = 0.0
