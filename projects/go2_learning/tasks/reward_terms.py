"""通用张量计算：姿态角、关节偏差和动作正则项。"""

import torch


def hip_centering_penalty(hip_joint_pos):
    """四个髋侧摆关节靠近 0 rad，减少相对机身的左右外撇/内收。

    输入形状 (num_envs, 4)，输出形状 (num_envs,)。
    这是相对机身的关节角约束，不等同于相对世界重力方向的几何竖直。
    """
    return hip_joint_pos.square().sum(dim=-1)

def roll_pitch_from_gravity(gravity):
    """由机身坐标下的重力方向恢复横滚/俯仰角，单位 rad；不需要全局 yaw。"""
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    pitch = torch.atan2(gravity[:, 0], gravity[:, 1:].norm(dim=-1))
    return roll, pitch

def action_rate_penalty(actions, previous_actions):
    """逐环境计算 sum((a_t - a_{t-1})**2)，不是电机速度或力矩。"""
    return (actions - previous_actions).square().sum(dim=-1)

def action_limit_penalty(raw_actions):
    """只惩罚原始动作超界部分；范围内为零，恒定超界也会持续扣分。"""
    return (raw_actions.abs() - 1.0).clamp_min(0.0).square().sum(dim=-1)

def foot_slide_penalty(state):
    """触地足部的世界水平速度模长之和；空中摆腿不扣分。"""
    speed = state["foot_velocity_w"][..., :2].norm(dim=-1)
    return (speed * state["foot_contact"]).sum(dim=-1)
