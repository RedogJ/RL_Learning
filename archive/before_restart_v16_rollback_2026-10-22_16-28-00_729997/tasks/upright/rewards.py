"""直立奖励：用重力投影评价机身前向轴，避开竖直姿态处的欧拉角奇异。"""

import torch

from go2_learning.tasks.reward_terms import action_limit_penalty, action_rate_penalty, foot_slide_penalty


def upright_alignment(state):
    """机身 +X 与世界 +Z 的点积：四足水平为 0，头朝上直立为 1，头朝下为 -1。"""
    return (-state["gravity"][:, 0]).clamp(-1.0, 1.0)


def rear_symmetry_error(state):
    """RL/RR 后腿的左右对称误差 RMS，单位 rad。

    髋侧摆左腿外撇为负、右腿为正，对称时两角之和为 0；大腿和小腿同一转轴，
    对称时左右角相等。三项各取平方后求 RMS。
    """
    hip = state["hip_joint_pos"][:, 2:]
    thigh = state["thigh_joint_pos"][:, 2:]
    calf = state["calf_joint_pos"][:, 2:]
    return torch.sqrt((
        hip.sum(dim=-1).square()
        + (thigh[:, 0] - thigh[:, 1]).square()
        + (calf[:, 0] - calf[:, 1]).square()
    ) / 3.0)


def rear_support_fraction(state, cfg):
    """RL/RR 的有效支撑比例；版本 4 扣除同腿膝和脚掌在同一物理帧支撑的比例。"""
    support = state["foot_support_per_foot"][:, 2:]
    if cfg.reward_schema_version >= 4:
        # P(脚掌支撑且膝未支撑) = P(脚掌支撑) - P(脚掌与膝同时支撑)。
        # 后者已经由接触模块逐物理帧统计，不能把两者的平均比例直接相乘。
        support = support - state["rear_leg_supported_fraction"]
    return support


def reward_components(state, actions, failed, dt, cfg):
    alignment = upright_alignment(state)
    hold = torch.exp(-((1.0 - alignment) / cfg.upright_sigma).square())
    support = state["foot_support_per_foot"]  # 固定顺序 FL、FR、RL、RR。
    height = torch.exp(-((state["height"] - cfg.target_height) / cfg.height_sigma).square())
    horizontal_speed_squared = state["linear_velocity_w"][:, :2].square().sum(-1)
    if cfg.reward_schema_version < 3:
        balance = cfg.balance_weight * torch.exp(
            -horizontal_speed_squared / cfg.horizontal_velocity_sigma**2) * hold * dt
    else:
        balance = -cfg.balance_weight * horizontal_speed_squared * dt
    vertical_velocity = torch.zeros_like(actions[:, 0])
    if cfg.vertical_velocity_weight != 0.0:
        vertical_velocity = -cfg.vertical_velocity_weight * state["linear_velocity_w"][:, 2].square() * dt
    # 后髋越接近 0 收益越高；组内取乘积，任意一条后腿外撇都会压低该项。
    hip_rear = cfg.hip_rear_weight * torch.exp(
        -(state["hip_joint_pos"][:, 2:] / cfg.hip_rear_sigma).square()).prod(-1) * dt
    # 后腿左右对称的指数收益；误差为三项 RMS，对称时最大，偏差越大越小但始终非负。
    rear_symmetry = cfg.rear_symmetry_weight * torch.exp(
        -(rear_symmetry_error(state) / cfg.rear_symmetry_sigma).square()) * dt
    pitch_velocity = torch.zeros_like(actions[:, 0])
    if cfg.pitch_velocity_weight != 0.0:
        # 使用机身真实角速度，不对竖直姿态附近的欧拉角求导；不按姿态门控。
        pitch_velocity = -cfg.pitch_velocity_weight * state["angular_velocity"][:, 1].square() * dt
    foot_slide = torch.zeros_like(actions[:, 0])
    if cfg.foot_slide_weight != 0.0:
        foot_slide = -cfg.foot_slide_weight * foot_slide_penalty(state) * dt
    rear_knee_contact = torch.zeros_like(actions[:, 0])
    if cfg.rear_knee_contact_weight != 0.0:
        # 整个后小腿刚体接触即扣分，支撑接触也不豁免，不按直立程度门控。
        rear_knee_contact = -cfg.rear_knee_contact_weight * state["rear_knee_contact_count"] * dt
    front_foot_position = torch.zeros_like(actions[:, 0])
    if cfg.front_foot_position_weight != 0.0:
        # 机身坐标下两前脚的距离平方取均值；全程生效，0 误差不扣分。
        front_foot_position = -cfg.front_foot_position_weight * (
            state["front_foot_error_b"].square().sum(dim=-1).mean(dim=-1)
        ) * dt
    # 版本 1 仅用于旧模型播放，保留当时的计分；新训练不再重复计分。
    if cfg.reward_schema_version == 1:
        angular_speed_squared = state["angular_velocity_w"].square().sum(-1)
        other_contact_count = state["undesired_contact_count"] - state["rear_knee_support_count"]
    else:
        # Y 轴只由 pitch_velocity 负责。
        angular_speed_squared = state["angular_velocity"][:, [0, 2]].square().sum(-1)
        other_contact_count = (state["undesired_contact_count"]
                               - state["hip_contact_count"] - state["rear_knee_contact_count"])
    angular_velocity = -cfg.angular_velocity_weight * angular_speed_squared * dt
    if cfg.reward_schema_version < 3:
        angular_velocity = angular_velocity * hold
    return {
        # 1. 基础存活
        "alive": torch.full_like(actions[:, 0], cfg.alive * dt),

        # 2. 直立方向
        "upright_hold": cfg.upright_hold_weight * hold * dt,

        # 3. 机身高度
        "height": cfg.height_weight * height * dt,

        # 4. 前后脚分工：后脚掌支撑，前脚抬起并收在胸前
        "rear_support": cfg.rear_support_weight * rear_support_fraction(state, cfg).mean(-1) * dt,
        "front_lift": cfg.front_lift_weight * (1.0 - support[:, :2].mean(-1)) * dt,
        "front_foot_position": front_foot_position,

        # 5. 机身平衡
        "balance": balance,
        "vertical_velocity": vertical_velocity,
        "angular_velocity": angular_velocity,
        "pitch_velocity": pitch_velocity,

        # 6. 后腿姿态
        "hip_rear": hip_rear,
        "rear_symmetry": rear_symmetry,

        # 7. 非足部接触：版本 2 及以上按其余部位 / 后小腿 / 髋部独立计分。
        "undesired_contact": -cfg.undesired_contact_weight * other_contact_count * dt,
        "rear_knee_contact": rear_knee_contact,
        "hip_contact": -cfg.hip_contact_weight * state["hip_contact_count"] * dt,

        # 8. 足部接触质量
        "foot_slide": foot_slide,

        # 9. 动作正则
        "action_rate": -cfg.action_rate_weight * action_rate_penalty(actions, state["previous_actions"]) * dt,
        "action_limit": -cfg.action_limit_weight * action_limit_penalty(state["raw_actions"]) * dt,

        # 10. 回合失败
        "failure": -cfg.failure_penalty * failed.float(),
    }
