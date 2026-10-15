"""初始姿态随机化和定时推力；不涉及 PPO、奖励或鼠标交互。"""

import math
from dataclasses import dataclass

import torch


@dataclass
class RandomizationCfg:
    joint_position_noise: float = 0.1  # 默认站姿附近 ±0.1 rad。
    push_force_range: tuple[float, float] = (40.0, 100.0)  # 水平合力大小，N；较原 20～50 N 加倍。
    push_interval_range: tuple[float, float] = (2.0, 4.0)  # 重置/上次推力结束后的等待时间，s。
    push_duration: float = 0.4  # 每次持续施力时间，s；120 Hz 下持续 48 个物理步。
    push_body: str = "base"
    push_point_half_extent: tuple[float, float, float] = (0.15, 0.04, 0.04)  # m；质心周围、沿机身 XYZ 轴的作用点偏移范围。

    def __post_init__(self):
        if not math.isfinite(self.joint_position_noise) or self.joint_position_noise < 0:
            raise ValueError("joint_position_noise 必须非负且有限")
        for name in ("push_force_range", "push_interval_range"):
            low, high = getattr(self, name)
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
                raise ValueError(f"{name} 必须满足 0 <= 最小值 <= 最大值，且为有限值")
        if not math.isfinite(self.push_duration) or self.push_duration <= 0:
            raise ValueError("push_duration 必须为正数且有限")
        if len(self.push_point_half_extent) != 3 or any(
            not math.isfinite(value) or value < 0 for value in self.push_point_half_extent
        ):
            raise ValueError("push_point_half_extent 必须包含三个非负且有限的数值")


def push_positions_world(com_pos_w, link_quat_w, offsets_b):
    """将质心周围、沿机身坐标轴的偏移变换成世界作用点；四元数为 (x,y,z,w)。

    与 Isaac Lab 的 quat_apply 使用同一旋转公式；保持模块仅依赖 torch，便于独立检查。
    com_pos_w/offsets_b 的形状为 (环境, 选中刚体, 3)，link_quat_w 的末维为 4。
    """
    xyz = link_quat_w[..., :3]
    cross = 2 * torch.cross(xyz, offsets_b, dim=-1)
    rotated = offsets_b + link_quat_w[..., 3:] * cross + torch.cross(xyz, cross, dim=-1)
    return com_pos_w + rotated


class DomainRandomizer:
    """每个环境各自保存推力和计时器，以仿真物理步计时。"""

    def __init__(self, robot, num_envs, device, physics_dt, cfg, seed):
        self.robot, self.cfg = robot, cfg
        self.device, self.physics_dt = device, physics_dt
        # 独立随机数流，避免扰动采样改变 PPO 的动作采样随机数序列。
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.body_ids, _ = robot.find_bodies(cfg.push_body)
        if len(self.body_ids) != 1:
            raise ValueError(f"随机推力需要唯一机身刚体，{cfg.push_body!r} 匹配到 {self.body_ids}")
        self.duration_steps = math.ceil(cfg.push_duration / physics_dt)
        self.wait_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.remaining_steps = torch.zeros_like(self.wait_steps)
        self.forces = torch.zeros((num_envs, 1, 3), device=device)
        self.point_offsets_b = torch.zeros_like(self.forces)
        self.point_half_extent = torch.tensor(cfg.push_point_half_extent, device=device)

    def _uniform(self, shape, low, high):
        return low + (high - low) * torch.rand(shape, device=self.device, generator=self.generator)

    def _sample_wait_steps(self, count):
        seconds = self._uniform((count,), *self.cfg.push_interval_range)
        return torch.ceil(seconds / self.physics_dt).long()

    def reset_timers(self, env_ids):
        """只重置指定环境的推力和计时器，不改变关节姿态。"""
        self.forces[env_ids] = 0.0
        self.point_offsets_b[env_ids] = 0.0
        self.remaining_steps[env_ids] = 0
        self.wait_steps[env_ids] = self._sample_wait_steps(env_ids.numel())

    def reset(self, env_ids, default_joint_pos):
        """只重置指定环境的扰动，返回随机初始角；不修改默认站姿。"""
        self.reset_timers(env_ids)
        noise = self._uniform(default_joint_pos.shape, -self.cfg.joint_position_noise,
                              self.cfg.joint_position_noise)
        limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        return (default_joint_pos + noise).clamp(min=limits[..., 0], max=limits[..., 1])

    def apply(self, active=None):
        """每个物理步调用：到点开始推，在整段脉冲内保持同一方向和大小。"""
        if active is None:
            active = torch.ones_like(self.wait_steps, dtype=torch.bool)
        start_ids = ((self.wait_steps == 0) & (self.remaining_steps == 0) & active).nonzero().flatten()
        if start_ids.numel():
            angle = self._uniform((start_ids.numel(),), -math.pi, math.pi)
            magnitude = self._uniform((start_ids.numel(),), *self.cfg.push_force_range)
            self.forces[start_ids, 0, 0] = magnitude * torch.cos(angle)
            self.forces[start_ids, 0, 1] = magnitude * torch.sin(angle)
            self.point_offsets_b[start_ids] = self._uniform((start_ids.numel(), 1, 3), -1., 1.) * self.point_half_extent
            self.remaining_steps[start_ids] = self.duration_steps
            self.wait_steps[start_ids] = self._sample_wait_steps(start_ids.numel())

        pushing = (self.remaining_steps > 0) & active
        forces = self.forces * pushing[:, None, None]
        data = self.robot.data
        positions = push_positions_world(
            data.body_com_pos_w.torch[:, self.body_ids],
            data.body_link_quat_w.torch[:, self.body_ids], self.point_offsets_b,
        )
        # 世界 XY 力作用于固定在机身上的随机点；positions 自动产生相对质心的 r×F 力矩。
        # 偏移的坐标轴取 link 朝向，参考原点取真实质心，不混用 link 原点或惯性主轴朝向。
        # instantaneous 力每次写入仿真后被清空，所以持续推力需要逐物理步写入。
        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=forces, positions=positions, body_ids=self.body_ids, is_global=True,
        )
        self.wait_steps.sub_((~pushing & (self.wait_steps > 0) & active).long())
        self.remaining_steps.sub_(pushing.long())
