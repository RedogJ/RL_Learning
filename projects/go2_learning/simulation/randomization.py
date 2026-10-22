"""初始姿态随机化和定时推力；不涉及 PPO、奖励或鼠标交互。"""

import math

import torch


def push_positions_world(com_pos_w, link_quat_w, offsets_b):
    """将质心周围、沿机身坐标轴的偏移变换成世界作用点；四元数为 (x,y,z,w)。

    与 Isaac Lab 的 quat_apply 使用同一旋转公式；保持模块仅依赖 torch，便于独立检查。
    com_pos_w/offsets_b 的形状为 (环境, 选中刚体, 3)，link_quat_w 的末维为 4。
    """
    return com_pos_w + push_offsets_world(link_quat_w, offsets_b)


def push_offsets_world(link_quat_w, offsets_b):
    """绕机身质心的偏移转到世界方向；无需读取质心的世界位置。"""
    xyz = link_quat_w[..., :3]
    cross = 2 * torch.cross(xyz, offsets_b, dim=-1)
    rotated = offsets_b + link_quat_w[..., 3:] * cross + torch.cross(xyz, cross, dim=-1)
    return rotated


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
        offsets_w = push_offsets_world(
            data.body_link_quat_w.torch[:, self.body_ids], self.point_offsets_b,
        )
        # 刚体等效力系：在偏移 r 处施加 F，等价于质心处 F 加 r×F。
        # 每个物理步仍更新旋转后的力臂；世界 XY 力与随机点的采样不变。
        # 不传 positions，允许框架直接使用世界坐标的质心力/力矩路径。
        torques = torch.cross(offsets_w, forces, dim=-1)
        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=forces, torques=torques, body_ids=self.body_ids, is_global=True,
        )
        self.wait_steps.sub_((~pushing & (self.wait_steps > 0) & active).long())
        self.remaining_steps.sub_(pushing.long())
