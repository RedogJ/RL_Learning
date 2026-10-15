"""机身坐标下的速度指令；当前阶段全部为零，先训练纯站立。"""

import math
from dataclasses import dataclass

import torch

TASK_VERSION = "go2_standing_height_integral_full_range_v2"


@dataclass
class CommandCfg:
    forward_range: tuple[float, float] = (0.0, 0.0)  # m/s；纯站立不要求前后移动。
    lateral_range: tuple[float, float] = (0.0, 0.0)  # m/s；纯站立不要求侧移。
    yaw_range: tuple[float, float] = (0.0, 0.0)  # rad/s；纯站立不要求转向。
    resample_seconds: float | None = None  # None：只在回合重置时采样，回合内指令不变。
    standing_probability: float = 1.0  # 当前所有环境均练习零速度站立。
    keyboard_forward: float = 0.4
    keyboard_backward: float = 0.3
    keyboard_turn: float = 0.6


class VelocityCommands:
    def __init__(self, num_envs, device, dt, cfg, random_sampling=False, seed=42):
        self.cfg, self.random_sampling = cfg, random_sampling
        self.device = device
        self.values = torch.zeros((num_envs, 3), device=device)  # vx, vy, yaw_rate
        # 保留定时模式以兼容旧配置；当前默认不启动回合内重采样。
        self.period_steps = (max(1, math.ceil(cfg.resample_seconds / dt))
                             if cfg.resample_seconds is not None else None)
        self.remaining = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def reset(self, env_ids):
        if self.random_sampling:
            self._sample(env_ids)
        # 播放重置不更换操作者的指令；下一控制步继续从键盘读取。

    def _sample(self, env_ids):
        count = env_ids.numel()
        samples = torch.rand((count, 4), device=self.device, generator=self.generator)
        for axis, bounds in enumerate((self.cfg.forward_range, self.cfg.lateral_range, self.cfg.yaw_range)):
            low, high = bounds
            self.values[env_ids, axis] = low + (high - low) * samples[:, axis]
        self.values[env_ids[samples[:, 3] < self.cfg.standing_probability]] = 0.0
        self.remaining[env_ids] = self.period_steps or 0

    def step(self, dones, active=None):
        """默认回合内保持不变；只有显式设置采样周期时才定时更换。"""
        if self.random_sampling and self.period_steps is not None:
            if active is None:
                active = torch.ones_like(dones, dtype=torch.bool)
            self.remaining -= active.long()
            ids = ((self.remaining <= 0) & ~dones & active).nonzero().flatten()
            if ids.numel():
                self._sample(ids)

    def set_keyboard(self, axes):
        """按住时移动，松开或窗口失焦后归零；左右键控制转向。"""
        forward, turn = axes
        self.values[:, 0] = forward * (self.cfg.keyboard_forward if forward >= 0 else self.cfg.keyboard_backward)
        self.values[:, 1] = 0.0
        self.values[:, 2] = turn * self.cfg.keyboard_turn
