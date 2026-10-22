"""通用机身坐标速度指令：随机采样、回合内更新和键盘输入。"""

import math

import torch


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
        magnitude = getattr(self.cfg, "forward_fixed_magnitude", None)
        if magnitude is not None:
            self.values[env_ids, 0] = torch.where(samples[:, 0] < 0.5, -magnitude, magnitude)
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
