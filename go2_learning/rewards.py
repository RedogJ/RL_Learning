"""站立正收益：姿态/高度达标奖励、动作正则与独立失败罚分。"""

from dataclasses import dataclass
import math

import torch

@dataclass
class RewardCfg:
    # 除 failure_penalty 外，以下各项先组成奖励率，最后统一乘控制周期 dt。
    # 权重设为 0 即可关闭对应项；正奖励权重表示最大每秒收益。
    tracking_reward_type: str = "exp"  # exp 为正达标奖励；quadratic 仅用于兼容旧模型回放。
    only_positive_rewards: bool = True  # 任务与常规正则合计下限为 0；独立约束在此之后扣除。
    action_limit_outside_floor: bool = True  # 超界罚分独立扣除，避免被非负截断抵消后失去约束。
    alive: float = 1.0  # 每步基础奖励率，鼓励持续运行；当前公式在终止步也会加入此项。
    roll_weight: float = 20.0  # roll=0 时最大正奖励率；exp 模式下不是平方惩罚系数。
    pitch_weight: float = 20.0  # pitch=0 时最大正奖励率；当前俯仰误差较大，优先改善。
    roll_velocity_weight: float = 1.0  # 机身坐标系绕 X 轴角速度平方惩罚，rad/s；不严格等于欧拉 roll 导数。
    pitch_velocity_weight: float = 1.0  # 机身坐标系绕 Y 轴角速度平方惩罚，rad/s；近水平时近似 pitch 导数。
    horizontal_velocity_weight: float = 20.0  # 水平速度为零时的最大正奖励率，抑制前后/左右漂移。
    yaw_velocity_weight: float = 10.0  # 绕世界竖直轴角速度为零时的最大正奖励率，抑制持续转动。
    horizontal_velocity_sigma: float = 0.5  # m/s；水平速度大小等于此值时，得到最大收益的约 36.8%。
    yaw_velocity_sigma: float = 1.0  # rad/s；世界 Z 轴角速度绝对值等于此值时，得到最大收益的约 36.8%。
    height_weight: float = 50.0  # 达到目标高度时最大正奖励率。
    roll_sigma: float = 0.30  # rad；奖励斜率最大处约 12.2°，对应日志中的典型横滚误差。
    pitch_sigma: float = 0.40  # rad；斜率最大处约 16.2°，先覆盖当前俯仰误差，再逐渐收紧。
    height_sigma: float = 0.10  # m；斜率最大处为高度偏差约 7.1 cm，覆盖当前偏低约 7 cm 的站姿。
    target_height: float = 0.3  # 期望机身根节点距环境原点的 Z 高度，m；当前平地原点位于地面。
    posture_weight: float = 20.0  # 全部关节相对默认站姿的角度偏差平方和惩罚，减少前后过度伸展及不对称姿态。
    action_weight: float = 0.0  # 完整机械范围内的归一化动作平方和惩罚；0 对应范围中点，不是电机力矩。
    action_rate_weight: float = 5.0  # 相邻控制步动作变化惩罚；先避免压过姿态修正，稳定后可增大。
    action_limit_weight: float = 10.0  # 独立超界惩罚不再被截断抵消，先用 10 约束原始动作。
    hip_weight: float = 4.0  # 四个髋侧摆角相对 0 rad 的额外平方惩罚；与 posture 中的髋角惩罚叠加。
    undesired_contact_weight: float = 20.0  # 每个非足部接触刚体每秒扣分；独立于非负截断，设 0 关闭传感器。
    contact_force_threshold: float = 1.0  # N；法向接触合力模长超过此值才计数，抑制微小数值接触。
    foot_support_weight: float = 20.0  # 站立指令下四足持续支撑的最大正奖励率；三足持续支撑得 75%。
    standing_linear_threshold: float = 0.05  # m/s；目标水平速度大小不超过此值才启用四足支撑奖励。
    standing_yaw_threshold: float = 0.1  # rad/s；目标转速绝对值也需不超过此值，避免抑制原地转向。
    failure_penalty: float = 50.0  # 跌倒终止时一次性扣该数值，不乘 dt；正常超时不扣此项。

    def __post_init__(self):
        if self.tracking_reward_type not in ("exp", "quadratic"):
            raise ValueError("tracking_reward_type 必须为 exp 或 quadratic")
        for name in ("roll_sigma", "pitch_sigma", "height_sigma", "horizontal_velocity_sigma", "yaw_velocity_sigma"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为正数且有限")
        for name in ("undesired_contact_weight", "contact_force_threshold", "horizontal_velocity_weight", "yaw_velocity_weight",
                     "foot_support_weight", "standing_linear_threshold", "standing_yaw_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须非负且有限")


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


def action_rate_penalty(actions, previous_actions):
    """逐环境计算 sum((a_t - a_{t-1})**2)，不是电机速度或力矩。"""
    return (actions - previous_actions).square().sum(dim=-1)


def foot_support_reward(state, cfg):
    """仅站立指令下奖励四足支撑；实际运动不会关闭奖励，不禁止恢复性迈步。"""
    if cfg.foot_support_weight == 0.0:
        return torch.zeros_like(state["roll"])
    commands = state["commands"]
    standing = (commands[:, :2].norm(dim=-1) <= cfg.standing_linear_threshold) & (
        commands[:, 2].abs() <= cfg.standing_yaw_threshold
    )
    return cfg.foot_support_weight * state["foot_support_fraction"] * standing.float()


def action_limit_penalty(raw_actions):
    """只惩罚原始动作超界部分；范围内为零，恒定超界也会持续扣分。"""
    return (raw_actions.abs() - 1.0).clamp_min(0.0).square().sum(dim=-1)


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
