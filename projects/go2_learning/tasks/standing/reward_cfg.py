"""站立任务的奖励权重与容忍范围；在这里调站姿、静止和接触奖励。"""

from dataclasses import dataclass
import math


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

