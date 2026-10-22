"""站立任务的奖励公式；通用环境通过 StandingTask 调用。"""

import torch

from .reward_cfg import RewardCfg
from go2_learning.tasks.reward_terms import (
    hip_centering_penalty, roll_pitch_from_gravity, action_rate_penalty, action_limit_penalty,
)


def roll_reward(state, cfg):
    """目标 roll=0；exp 模式下越水平正收益越高。"""
    if cfg.tracking_reward_type == "exp":
        return cfg.roll_weight * torch.exp(-(state["roll"] / cfg.roll_sigma).square())
    return -cfg.roll_weight * state["roll"].square()

def pitch_reward(state, cfg):
    """目标 pitch=0；quadratic 分支兼容旧检查点。"""
    if cfg.tracking_reward_type == "exp":
        return cfg.pitch_weight * torch.exp(-(state["pitch"] / cfg.pitch_sigma).square())
    return -cfg.pitch_weight * state["pitch"].square()

def height_reward(state, cfg):
    """目标高度处正收益最大，过高或过低均逐渐减小。"""
    if cfg.tracking_reward_type == "exp":
        return cfg.height_weight * torch.exp(-((state["height"] - cfg.target_height) / cfg.height_sigma).square())
    return -cfg.height_weight * (state["height"] - cfg.target_height).square()

def roll_velocity_reward(state, cfg):
    """机身角速度 p 的平方惩罚；权重为 0 时不要求角速度状态。"""
    if cfg.roll_velocity_weight == 0.0:
        return torch.zeros_like(state["roll"])
    return -cfg.roll_velocity_weight * state["angular_velocity"][:, 0].square()

def pitch_velocity_reward(state, cfg):
    """机身角速度 q 的平方惩罚，不包含绕 Z 轴的角速度。"""
    if cfg.pitch_velocity_weight == 0.0:
        return torch.zeros_like(state["pitch"])
    return -cfg.pitch_velocity_weight * state["angular_velocity"][:, 1].square()

def horizontal_velocity_reward(state, cfg):
    """纯站立目标：世界 XY 平面速度趋于零；竖直下落速度不进入此项。"""
    if cfg.horizontal_velocity_weight == 0.0:
        return torch.zeros_like(state["roll"])
    speed_squared = state["linear_velocity_w"][:, :2].square().sum(dim=-1)
    return cfg.horizontal_velocity_weight * torch.exp(-speed_squared / cfg.horizontal_velocity_sigma**2)

def yaw_velocity_reward(state, cfg):
    """抑制绕世界竖直轴转动；不要求回到出生朝向，不使用机身 Z 分量。"""
    if cfg.yaw_velocity_weight == 0.0:
        return torch.zeros_like(state["roll"])
    return cfg.yaw_velocity_weight * torch.exp(
        -(state["angular_velocity_w"][:, 2] / cfg.yaw_velocity_sigma).square()
    )

def foot_support_reward(state, cfg):
    """仅站立指令下奖励四足支撑；实际运动不会关闭奖励，不禁止恢复性迈步。"""
    if cfg.foot_support_weight == 0.0:
        return torch.zeros_like(state["roll"])
    commands = state["commands"]
    standing = (commands[:, :2].norm(dim=-1) <= cfg.standing_linear_threshold) & (
        commands[:, 2].abs() <= cfg.standing_yaw_threshold
    )
    return cfg.foot_support_weight * state["foot_support_fraction"] * standing.float()

def reward_components(state, actions, failed, dt, cfg):
    """逐环境的实际奖励贡献：已乘权重和 dt，跌倒项不乘 dt。"""
    terms = {
        "alive": torch.full_like(actions[:, 0], cfg.alive * dt),
        "roll": roll_reward(state, cfg) * dt,
        "pitch": pitch_reward(state, cfg) * dt,
        "roll_velocity": roll_velocity_reward(state, cfg) * dt,
        "pitch_velocity": pitch_velocity_reward(state, cfg) * dt,
        "horizontal_velocity": horizontal_velocity_reward(state, cfg) * dt,
        "yaw_velocity": yaw_velocity_reward(state, cfg) * dt,
        "foot_support": foot_support_reward(state, cfg) * dt,
        "height": height_reward(state, cfg) * dt,
        "posture": -cfg.posture_weight * state["joint_offset"].square().sum(dim=-1) * dt,
        "action": -cfg.action_weight * actions.square().sum(dim=-1) * dt,
        "action_rate": -cfg.action_rate_weight * action_rate_penalty(actions, state["previous_actions"]) * dt,
        "hip": -cfg.hip_weight * hip_centering_penalty(state["hip_joint_pos"]) * dt,
        "action_limit": torch.zeros_like(actions[:, 0]),
    }
    limit_cost = torch.zeros_like(actions[:, 0])
    if cfg.action_limit_weight != 0.0:
        limit_cost = -cfg.action_limit_weight * action_limit_penalty(state["raw_actions"]) * dt
    if not cfg.action_limit_outside_floor:
        terms["action_limit"] = limit_cost  # 兼容旧正收益模型的总奖励截断语义。
    if cfg.only_positive_rewards:
        # 记录下限修正，使日志分项之和仍精确等于实际奖励；不能把失败罚分一起截掉。
        terms["positive_floor"] = (-sum(terms.values())).clamp_min(0.0)
    if cfg.action_limit_outside_floor:
        terms["action_limit"] = limit_cost
    # 持续接触约束独立扣分，避免常规奖励的下限修正把它抵消；不触发终止。
    terms["undesired_contact"] = torch.zeros_like(actions[:, 0])
    if cfg.undesired_contact_weight != 0.0:
        terms["undesired_contact"] = -cfg.undesired_contact_weight * state["undesired_contact_count"] * dt
    terms["failure"] = -cfg.failure_penalty * failed.float()
    return terms

def locomotion_reward(state, actions, failed, dt, cfg):
    return sum(reward_components(state, actions, failed, dt, cfg).values())
