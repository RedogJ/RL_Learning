"""移动任务的速度跟踪、姿态和接触奖励参数。"""

from dataclasses import dataclass
import math


@dataclass
class RewardCfg:
    # 除 failure_penalty 外，各项为奖励率，计算奖励时乘控制周期 dt。
    # 权重为 0 表示关闭该项；sigma 是 exp 奖励的误差尺度（误差等于 sigma 时约得 36.8%）。

    # 奖励组合规则
    tracking_reward_type: str = "exp"  # exp 为正达标奖励；quadratic 仅用于兼容旧模型回放。
    only_positive_rewards: bool = True  # 任务与常规正则合计下限为 0；独立约束在此之后扣除。
    action_limit_outside_floor: bool = True  # 超界罚分独立扣除，避免被非负截断抵消后失去约束。
    alive: float = 1.0  # 每步基础奖励率，鼓励持续运行；当前公式在终止步也会加入此项。

    # 运动目标：机身线速度与转向角速度
    tracking_linear_weight: float = 300.0  # 机身 XY 速度完全跟踪指令时的最大正奖励率。
    linear_sigma: float = 0.4  # m/s；水平速度误差模长为此值时，收益约为最大值的 36.8%。
    stationary_linear_sigma: float | None = 0.1  # m/s；vx、vy、yaw 指令全零时使用，严格约束慢速漂移；None 沿用 linear_sigma。
    tracking_yaw_weight: float = 200.0  # 机身 Z 角速度完全跟踪转向指令时的最大正奖励率。
    yaw_sigma: float = 0.6  # rad/s；转向角速度误差为此值时，收益约为最大值的 36.8%。

    # 机身姿态与高度
    roll_weight: float = 130.0  # roll=0 时最大正奖励率；exp 模式下不是平方惩罚系数。
    roll_sigma: float = 0.30  # rad；奖励斜率最大处约 12.2°，用于调节横滚容忍度。
    pitch_weight: float = 130.0  # pitch=0 时最大正奖励率；当前俯仰误差较大，优先改善。
    pitch_sigma: float = 0.40  # rad；斜率最大处约 16.2°，用于调节俯仰容忍度。
    height_weight: float = 70.0  # 达到目标高度时最大正奖励率。
    target_height: float = 0.3  # 期望机身根节点距环境原点的 Z 高度，m；当前平地原点位于地面。
    height_sigma: float = 0.10  # m；斜率最大处为高度偏差约 7.1 cm，用于调节高度容忍度。
    roll_velocity_weight: float = 2.0  # 机身坐标系绕 X 轴角速度平方惩罚，rad/s；不严格等于欧拉 roll 导数。
    pitch_velocity_weight: float = 2.0  # 机身坐标系绕 Y 轴角速度平方惩罚，rad/s；近水平时近似 pitch 导数。

    # 关节站姿
    posture_weight: float = 15.0  # 全部关节相对默认站姿的角度偏差平方和惩罚，减少前后过度伸展及不对称姿态。
    posture_stand_still_scale: float = 5.0  # 指令全零且实际水平速度不超过下方阈值时，posture 惩罚的倍数；1 关闭加强。
    posture_velocity_threshold: float = 0.3  # m/s；停车指令下实际速度仍大于此值时，保持基础姿态权重。
    hip_weight: float = 2.0  # 四个髋侧摆角相对 0 rad 的额外平方惩罚；与 posture 中的髋角惩罚叠加。

    # 动作大小、平滑性与限幅
    action_weight: float = 0.0  # 完整机械范围内的归一化动作平方和惩罚；0 对应范围中点，不是电机力矩。
    action_rate_weight: float = 5.0  # 相邻控制步动作变化惩罚；先避免压过姿态修正，稳定后可增大。
    action_limit_weight: float = 10.0  # 独立超界惩罚不再被截断抵消，先用 10 约束原始动作。

    action_activity_weight: float = 15.0  # 同类关节近期 |Δ裁剪动作|/dt 的方差之和惩罚率权重。
    action_activity_tau: float = 0.6 # s；活动量指数平滑时间常数，允许四腿相位不同。

    # 接触与连续离地
    all_feet_airborne_weight: float = 1000.0  # 四脚同时无支撑的惩罚率；按物理帧腾空占比乘 dt，等待期不扣分。
    foot_contact_ratio_weight: float = 40.0  # 所有速度指令下，Σ(1−各足近期接触比例)² 惩罚；无比例阈值或误差截断。
    foot_force_balance_weight: float = 2.0  # 四足平滑竖直接触力的总体方差(N²)惩罚，乘 dt；复用 foot_contact_ratio_tau。
    foot_contact_ratio_tau: float = 0.75  # s；接触比例指数平滑的时间常数。
    foot_air_time_weight: float = 0.0  # 已关闭；保留旧模型配置兼容。
    undesired_contact_weight: float = 40.0  # 每个非足部接触刚体每秒扣分；独立于非负截断；全部接触奖励关闭时才关闭传感器。
    contact_force_threshold: float = 1.0  # N；法向接触合力模长超过此值才计数，抑制微小数值接触。

    # 足部运动质量；权重按当前速度跟踪 300 / Unitree 示例 1.5 的比例起步。
    foot_slide_weight: float = 10.0  # 触地足部世界 XY 速度模长之和惩罚；空中摆腿不扣分。
    foot_time_variance_weight: float = 10.0  # 四足上次腾空、触地时长的样本方差之和惩罚；原始时长不截顶。

    # 回合失败
    failure_penalty: float = 50.0  # 跌倒终止时一次性扣该数值，不乘 dt；正常超时不扣此项。

    def __post_init__(self):
        if not math.isfinite(self.action_activity_tau) or self.action_activity_tau <= 0:
            raise ValueError("action_activity_tau 必须为正数且有限")
        if not math.isfinite(self.foot_contact_ratio_tau) or self.foot_contact_ratio_tau <= 0:
            raise ValueError("foot_contact_ratio_tau 必须为正数且有限")
        if self.tracking_reward_type not in ("exp", "quadratic"):
            raise ValueError("tracking_reward_type 必须为 exp 或 quadratic")
        if self.stationary_linear_sigma is not None:
            if not math.isfinite(self.stationary_linear_sigma) or self.stationary_linear_sigma <= 0:
                raise ValueError("stationary_linear_sigma 必须为正数且有限，或 None")
        for name in ("roll_sigma", "pitch_sigma", "height_sigma", "linear_sigma", "yaw_sigma"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为正数且有限")
        for name in ("undesired_contact_weight", "contact_force_threshold", "tracking_linear_weight", "tracking_yaw_weight",
                     "all_feet_airborne_weight", "foot_air_time_weight", "foot_contact_ratio_weight", "foot_force_balance_weight", "action_activity_weight",
                     "foot_slide_weight", "foot_time_variance_weight",
                     "posture_stand_still_scale", "posture_velocity_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须非负且有限")
