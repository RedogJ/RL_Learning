"""移动任务的奖励公式；通用环境通过 RunningTask 调用。"""

import torch

from .reward_cfg import RewardCfg
from go2_learning.tasks.reward_terms import (
    action_limit_penalty, action_rate_penalty, foot_slide_penalty, hip_centering_penalty,
    roll_pitch_from_gravity,
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

def tracking_linear_reward(state, cfg):
    """机身 XY 速度跟踪机身坐标指令；转弯后前进仍沿狗自身的 X 轴。"""
    if cfg.tracking_linear_weight == 0.0:
        return torch.zeros_like(state["roll"])
    error_squared = (state["linear_velocity"][:, :2] - state["commands"][:, :2]).square().sum(dim=-1)
    sigma_squared = cfg.linear_sigma**2
    if cfg.stationary_linear_sigma is not None:
        # 按目标指令判断停车；实际正在漂移时仍必须受到严格约束。
        stationary = (state["commands"] == 0.0).all(dim=-1)
        sigma_squared = torch.where(stationary, cfg.stationary_linear_sigma**2, sigma_squared)
    return cfg.tracking_linear_weight * torch.exp(-error_squared / sigma_squared)

def tracking_yaw_reward(state, cfg):
    """机身 Z 角速度跟踪转向指令；近水平时对应 yaw 角速度。"""
    if cfg.tracking_yaw_weight == 0.0:
        return torch.zeros_like(state["roll"])
    return cfg.tracking_yaw_weight * torch.exp(
        -((state["angular_velocity"][:, 2] - state["commands"][:, 2]) / cfg.yaw_sigma).square()
    )

def posture_reward(state, cfg):
    """零指令且已接近停稳时加强站姿约束；保留原关节偏差平方和公式。"""
    penalty = state["joint_offset"].square().sum(dim=-1)
    if cfg.posture_stand_still_scale != 1.0 and cfg.posture_weight != 0.0:
        stationary = ((state["commands"] == 0.0).all(dim=-1)
                      & (state["linear_velocity"][:, :2].norm(dim=-1) <= cfg.posture_velocity_threshold))
        penalty = penalty * torch.where(stationary, cfg.posture_stand_still_scale, 1.0)
    return -cfg.posture_weight * penalty


def foot_time_variance_penalty(state):
    """比较已完成的腾空/触地阶段，使用四足样本方差，不截顶时长。"""
    return (state["foot_last_air_time"].var(dim=-1, unbiased=True)
            + state["foot_last_contact_time"].var(dim=-1, unbiased=True))


def reward_components(state, actions, failed, dt, cfg):
    """逐环境的实际奖励贡献：已乘权重和 dt，跌倒项不乘 dt。"""
    terms = {
        "alive": torch.full_like(actions[:, 0], cfg.alive * dt),
        "roll": roll_reward(state, cfg) * dt,
        "pitch": pitch_reward(state, cfg) * dt,
        "roll_velocity": roll_velocity_reward(state, cfg) * dt,
        "pitch_velocity": pitch_velocity_reward(state, cfg) * dt,
        "tracking_linear": tracking_linear_reward(state, cfg) * dt,
        "tracking_yaw": tracking_yaw_reward(state, cfg) * dt,
        "height": height_reward(state, cfg) * dt,
        "posture": posture_reward(state, cfg) * dt,
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
    # 独立扣除，避免被常规奖励的非负截断抵消。
    terms["foot_air_time"] = torch.zeros_like(actions[:, 0])
    if cfg.foot_air_time_weight != 0.0:
        terms["foot_air_time"] = -cfg.foot_air_time_weight * state["foot_air_time"].square().sum(dim=-1) * dt
    terms["foot_contact_ratio"] = torch.zeros_like(actions[:, 0])
    if cfg.foot_contact_ratio_weight != 0.0:
        error = 1.0 - state["foot_contact_ratio"]
        terms["foot_contact_ratio"] = -cfg.foot_contact_ratio_weight * error.square().sum(dim=-1) * dt
    terms["foot_force_balance"] = torch.zeros_like(actions[:, 0])
    if cfg.foot_force_balance_weight != 0.0:
        terms["foot_force_balance"] = (-cfg.foot_force_balance_weight
                                      * state["foot_mean_force_z"].var(dim=-1, unbiased=False) * dt)
    terms["action_activity"] = torch.zeros_like(actions[:, 0])
    if cfg.action_activity_weight != 0.0:
        terms["action_activity"] = -cfg.action_activity_weight * state["action_activity_variance"] * dt
    terms["all_feet_airborne"] = torch.zeros_like(actions[:, 0])
    if cfg.all_feet_airborne_weight != 0.0:
        terms["all_feet_airborne"] = -cfg.all_feet_airborne_weight * state["all_feet_airborne_fraction"] * dt
    terms["foot_slide"] = torch.zeros_like(actions[:, 0])
    if cfg.foot_slide_weight != 0.0:
        terms["foot_slide"] = -cfg.foot_slide_weight * foot_slide_penalty(state) * dt
    terms["foot_time_variance"] = torch.zeros_like(actions[:, 0])
    if cfg.foot_time_variance_weight != 0.0:
        terms["foot_time_variance"] = -cfg.foot_time_variance_weight * foot_time_variance_penalty(state) * dt
    terms["failure"] = -cfg.failure_penalty * failed.float()
    return terms

def locomotion_reward(state, actions, failed, dt, cfg):
    return sum(reward_components(state, actions, failed, dt, cfg).values())
