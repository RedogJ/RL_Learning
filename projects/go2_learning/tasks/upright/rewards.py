"""直立奖励：用重力投影评价机身前向轴，避开竖直姿态处的欧拉角奇异。"""

import torch

from go2_learning.tasks.reward_terms import action_limit_penalty, foot_slide_penalty
from .tracking_errors import (
    continuous_errors, positive_error_rewards, rear_posture_error_squared,
    rear_symmetry_error, rear_symmetry_error_squared, upright_alignment,
)


def rear_support_fraction(state, cfg):
    """RL/RR 的有效支撑比例；版本 4 扣除同腿膝和脚掌在同一物理帧支撑的比例。"""
    support = state["foot_support_per_foot"][:, 2:]
    if getattr(cfg, "reward_schema_version", 8) == 4:
        # P(脚掌支撑且膝未支撑) = P(脚掌支撑) - P(脚掌与膝同时支撑)。
        # 后者已经由接触模块逐物理帧统计，不能把两者的平均比例直接相乘。
        support = support - state["rear_leg_supported_fraction"]
    return support


def reward_components(state, actions, failed, dt, cfg, error_buffer=None, tracking=None):
    schema = getattr(cfg, "reward_schema_version", 16)
    if schema < 8:
        return _legacy_reward_components(state, actions, failed, dt, cfg)
    squared, valid = tracking if tracking is not None else continuous_errors(state, cfg)
    if error_buffer is not None:
        error_buffer.observe(squared, valid)
    terms = positive_error_rewards(squared, valid, dt, cfg, actions[:, 0])
    foot_slide = torch.zeros_like(actions[:, 0])
    if cfg.foot_slide_weight != 0:
        error = squared["foot_slide"]
        foot_slide = (cfg.foot_slide_weight * torch.exp(-error / cfg.foot_slide_sigma**2) * dt
                      if schema < 16 else -cfg.foot_slide_weight * error * dt)
        foot_slide = foot_slide * valid["foot_slide"]
    for prefix, quadratic_schema in (("linear_velocity", 14), ("angular_velocity", 15)):
        for axis in "xyz":
            name = f"{prefix}_{axis}"
            if name not in squared:
                terms[name] = torch.zeros_like(actions[:, 0])
                continue
            weight = getattr(cfg, f"{name}_weight")
            terms[name] = (-weight * squared[name] * dt if schema >= quadratic_schema or schema == 10 else
                           weight * torch.exp(-squared[name] / getattr(cfg, f"{name}_sigma")**2) * dt)
    front_target = torch.zeros_like(actions[:, 0])
    if cfg.front_target_weight != 0:
        error = squared["front_target"]  # 左右前腿各三个关节，相对配置目标的均方误差。
        front_target = (cfg.front_target_weight * torch.exp(-error / cfg.front_target_sigma**2) * dt
                        if schema < 13 else -cfg.front_target_weight * error * dt)
    hip_rear = torch.zeros_like(actions[:, 0])
    if cfg.hip_rear_weight != 0:
        error = squared["hip_rear"]
        hip_rear = (cfg.hip_rear_weight * torch.exp(-error / cfg.hip_rear_sigma**2) * dt
                    if schema < 12 else -cfg.hip_rear_weight * error * dt)
    action_rate = torch.zeros_like(actions[:, 0])
    if getattr(cfg, "action_rate_weight", 0.0) != 0:
        # 新训练惩罚物理目标角的变化，与观测中的上一条执行动作对应。
        # 历史模型保留原始输出差口径，避免回放旧记录时改变奖励定义。
        delta = (state["joint_target_delta"] if schema >= 11 else
                 state["raw_actions"] - state["previous_raw_actions"])
        action_rate = -cfg.action_rate_weight * delta.square().mean(-1) * dt
    action_rate_derivative = torch.zeros_like(actions[:, 0])
    if getattr(cfg, "action_rate_derivative_weight", 0.0) != 0:
        action_rate_derivative = -cfg.action_rate_derivative_weight * state["joint_target_delta_derivative"].square().mean(-1) * dt
    joint_velocity = torch.zeros_like(actions[:, 0])
    if cfg.joint_velocity_weight != 0:
        error = squared["joint_velocity"]  # 实际关节速度的均方，不是观测中的缩放值。
        joint_velocity = (cfg.joint_velocity_weight * torch.exp(-error / cfg.joint_velocity_sigma**2) * dt
                          if schema == 8 else -cfg.joint_velocity_weight * error * dt)
    support = state["foot_support_per_foot"]
    knee_clearance = torch.zeros_like(actions[:, 0])
    if cfg.rear_knee_clearance_weight != 0:
        heights = state["rear_knee_height"]
        if heights.dim() == 1:
            heights = heights.unsqueeze(-1)
        # 这是离地收益，不是零目标误差；保持原固定尺度，不参与自动校准。
        knee_clearance = cfg.rear_knee_clearance_weight * (
            1.0 - torch.exp(-heights.clamp_min(0.0) / cfg.rear_knee_clearance_sigma)
        ).min(-1).values * dt
    return {
        **terms,
        "foot_slide": foot_slide,
        "front_target": front_target,
        "hip_rear": hip_rear,
        "joint_velocity": joint_velocity,
        "action_rate": action_rate,
        "action_rate_derivative": action_rate_derivative,
        "alive": torch.full_like(actions[:, 0], cfg.alive * dt),
        "rear_support": cfg.rear_support_weight * rear_support_fraction(state, cfg).mean(-1) * dt,
        "front_lift": cfg.front_lift_weight * (1.0 - support[:, :2].mean(-1)) * dt,
        "rear_knee_clearance": knee_clearance,
        # 接触事件、动作合法边界及终止不是连续跟踪误差，保留固定惩罚。
        "rear_feet_airborne": -cfg.rear_feet_airborne_weight * state["rear_feet_airborne_fraction"] * dt,
        "undesired_contact": -cfg.undesired_contact_weight * (
            state["undesired_contact_count"] - state["rear_knee_support_count"]
        ) * dt,
        "rear_knee_contact": -cfg.rear_knee_contact_weight * state["rear_knee_contact_count"] * dt,
        "hip_contact": -cfg.hip_contact_weight * state["hip_contact_count"] * dt,
        "action_limit": -cfg.action_limit_weight * action_limit_penalty(state["raw_actions"]) * dt,
        "failure": -cfg.failure_penalty * failed.float(),
    }


def _legacy_reward_components(state, actions, failed, dt, cfg):
    """仅供历史模型播放；保留 schema 1–7 的计分口径。"""
    front_target = torch.zeros_like(actions[:, 0])
    if cfg.front_target_weight != 0.0:
        error_squared = sum(
            (state[f"{part}_joint_pos"][:, :2] - getattr(cfg, f"front_target_{part}")).square().sum(-1)
            for part in ("hip", "thigh", "calf")
        ) / 6.0
        front_target = -cfg.front_target_weight * error_squared * dt
    alignment = upright_alignment(state)
    hold = torch.exp(-((1.0 - alignment) / cfg.upright_sigma).square())
    support = state["foot_support_per_foot"]  # 固定顺序 FL、FR、RL、RR。
    height = torch.exp(-((state["height"] - cfg.target_height) / cfg.height_sigma).square())
    # Part 1：新框架每个轴独立计分；历史模型保留原来的计算方式。
    if getattr(cfg, "reward_schema_version", 7) >= 6:
        motion = {}
        for axis_id, axis in enumerate("xyz"):
            motion[f"linear_velocity_{axis}"] = (
                -getattr(cfg, f"linear_velocity_{axis}_weight")
                * state["linear_velocity_w"][:, axis_id].square() * dt
            )
            motion[f"angular_velocity_{axis}"] = (
                -getattr(cfg, f"angular_velocity_{axis}_weight")
                * state["angular_velocity"][:, axis_id].square() * dt
            )
    else:
        horizontal_speed_squared = state["linear_velocity_w"][:, :2].square().sum(-1)
        if cfg.reward_schema_version < 3:
            balance = cfg.balance_weight * torch.exp(
                -horizontal_speed_squared / cfg.horizontal_velocity_sigma**2) * hold * dt
        else:
            balance = -cfg.balance_weight * horizontal_speed_squared * dt
    # 使用较低的后小腿 link 原点高度评价离地程度。
    rear_knee_clearance = torch.zeros_like(actions[:, 0])
    if cfg.rear_knee_clearance_weight != 0.0:
        knee_height = state["rear_knee_height"]
        if knee_height.dim() == 1:
            knee_height = knee_height.unsqueeze(-1)
        rear_knee_clearance = cfg.rear_knee_clearance_weight * (
            1.0 - torch.exp(-knee_height.clamp_min(0.0) / cfg.rear_knee_clearance_sigma)
        ).min(dim=-1).values * dt
    if getattr(cfg, "reward_schema_version", 7) >= 7:
        # 当前公式全程使用均方负惩罚，不乘 hold，也不使用 sigma。
        hip_rear = -cfg.hip_rear_weight * state["hip_joint_pos"][:, 2:].square().mean(-1) * dt
        rear_symmetry = -cfg.rear_symmetry_weight * rear_symmetry_error_squared(state) * dt
    else:
        # 历史模型播放保留原来的指数正收益。
        hip_rear = cfg.hip_rear_weight * torch.exp(
            -(state["hip_joint_pos"][:, 2:] / cfg.hip_rear_sigma).square()).prod(-1) * dt
        rear_symmetry = cfg.rear_symmetry_weight * torch.exp(
            -(rear_symmetry_error(state) / cfg.rear_symmetry_sigma).square()) * dt
    rear_posture = torch.zeros_like(actions[:, 0])
    if cfg.rear_posture_weight != 0.0:
        # 作为软参考，不覆盖策略输出，不改变 PD 目标或重置角。
        rear_posture = -cfg.rear_posture_weight * rear_posture_error_squared(state, cfg) * dt
    foot_slide = torch.zeros_like(actions[:, 0])
    if cfg.foot_slide_weight != 0.0:
        foot_slide = -cfg.foot_slide_weight * foot_slide_penalty(state) * dt
    rear_knee_contact = torch.zeros_like(actions[:, 0])
    if cfg.rear_knee_contact_weight != 0.0:
        # 整个后小腿刚体接触即扣分，支撑接触也不豁免，不按直立程度门控。
        rear_knee_contact = -cfg.rear_knee_contact_weight * state["rear_knee_contact_count"] * dt
    joint_velocity = torch.zeros_like(actions[:, 0])
    if cfg.joint_velocity_weight != 0.0:
        # 控制步末读取的实际关节角速度；直接使用 rad/s，不由动作差或关节角差估算。
        joint_velocity = -cfg.joint_velocity_weight * state["joint_velocity"].square().sum(-1) * dt
    # 版本 5 只替换水平速度项，角速度与接触仍使用版本 1 的方式。
    if getattr(cfg, "reward_schema_version", 7) in (1, 5, 6, 7):
        angular_speed_squared = state["angular_velocity_w"].square().sum(-1)
        other_contact_count = state["undesired_contact_count"] - state["rear_knee_support_count"]
    else:
        # Y 轴只由 pitch_velocity 负责。
        angular_speed_squared = state["angular_velocity"][:, [0, 2]].square().sum(-1)
        other_contact_count = (state["undesired_contact_count"]
                               - state["hip_contact_count"] - state["rear_knee_contact_count"])
    if getattr(cfg, "reward_schema_version", 7) < 6:
        angular_velocity = -cfg.angular_velocity_weight * angular_speed_squared * dt
        if cfg.reward_schema_version in (1, 2, 5):
            angular_velocity = angular_velocity * hold
        pitch_velocity = -cfg.pitch_velocity_weight * state["angular_velocity"][:, 1].square() * dt
        motion = {"balance": balance, "angular_velocity": angular_velocity, "pitch_velocity": pitch_velocity}
    return {
        # 1. 基础存活
        "alive": torch.full_like(actions[:, 0], cfg.alive * dt),

        # 2. 直立方向
        "upright_hold": cfg.upright_hold_weight * hold * dt,

        # 3. 机身高度
        "height": cfg.height_weight * height * dt,

        # 4. 前后脚分工：后脚掌支撑，前脚抬起并收在胸前
        "rear_feet_airborne": -cfg.rear_feet_airborne_weight * state["rear_feet_airborne_fraction"] * dt,
        "rear_support": cfg.rear_support_weight * rear_support_fraction(state, cfg).mean(-1) * dt,
        "rear_knee_clearance": rear_knee_clearance,
        "front_target": front_target,
        "front_lift": cfg.front_lift_weight * (1.0 - support[:, :2].mean(-1)) * dt,

        # 5. 机身平衡
        **motion,

        # 6. 后腿姿态
        "hip_rear": hip_rear,
        "rear_symmetry": rear_symmetry,
        "rear_posture": rear_posture,

        # 7. 非足部接触：版本 1/5 保留原叠加；版本 2/3/4 按部位独立计分。
        "undesired_contact": -cfg.undesired_contact_weight * other_contact_count * dt,
        "rear_knee_contact": rear_knee_contact,
        "hip_contact": -cfg.hip_contact_weight * state["hip_contact_count"] * dt,

        # 8. 足部接触质量
        "foot_slide": foot_slide,

        # 9. 关节速度与动作超界
        "joint_velocity": joint_velocity,
        "action_limit": -cfg.action_limit_weight * action_limit_penalty(state["raw_actions"]) * dt,

        # 10. 回合失败
        "failure": -cfg.failure_penalty * failed.float(),
    }
