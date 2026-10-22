"""Go2 后腿直立奖励配置；当前基准为 upright_restart_v16。

所有 weight 均填写非负数，正收益/负惩罚的符号由 rewards.py 决定。
除 failure_penalty 外，各项均按奖励率计算，每控制步乘 dt。
hold = exp(-((1 - alignment) / upright_sigma)²)，alignment 为机身 +X 与世界向上的点积。
"""

from dataclasses import dataclass, fields
import math


@dataclass
class RewardCfg:
    # 1. 机身目标：方向与高度分别计分
    upright_hold_weight: float = 500.0  # 正收益：weight * hold，机身前向轴越朝上越高。
    upright_sigma: float = 0.2  # 无量纲；1-alignment 的指数误差尺度，误差等于 sigma 时取最大收益的 e^-1。
    
    target_height: float = 0.4  # m；机身根节点相对环境原点的高度目标。
    height_weight: float = 100.0  # 正收益：目标高度的高斯型收益，全程生效，不乘 hold。
    height_sigma: float = 0.1  # m；高度误差等于 sigma 时取最大收益的 e^-1。

    # 2. 后腿姿态：回正与左右对称是两个目标，均全程生效
    
    hip_rear_weight: float = 200.0  # 正收益：两个后髋接近 0 rad 的指数收益取乘积，最大为此权重。
    hip_rear_sigma: float = 0.5  # rad；后髋侧摆角的指数误差尺度。
    
    rear_symmetry_weight: float = 300.0  # 正收益：左右后腿对称误差的高斯型收益。
    rear_symmetry_sigma: float = 0.3  # rad；RMS(髋角之和、大腿角之差、小腿角之差) 的误差尺度。

    # 3. 脚掌与膝部：支撑、抬脚、膝部离地及触地滑动
    rear_support_weight: float = 5.0  # 正收益：两后脚支撑比例取均值；v16 中膝部同时支撑不取消脚掌收益。
    
    front_lift_weight: float = 5.0  # 正收益：两前脚不支撑的比例取均值，不乘 hold。
    
    rear_knee_clearance_weight: float = 70.0  # 正收益：weight * (1-exp(-max(h,0)/sigma))，h 取较低后小腿 link 原点的世界高度。
    rear_knee_clearance_sigma: float = 0.1  # m；膝部离地收益的高度尺度，不是碰撞表面净空。
    
    foot_slide_weight: float = 3.0  # 负惩罚：触地脚掌的世界 XY 速度模长之和；空中摆腿不扣分。

    # 4. 运动稳定
    balance_weight: float = 30.0  # 负惩罚：-weight * (vx_world² + vy_world²)，不乘 hold；速度单位 m/s。
    angular_velocity_weight: float = 10.0  # 负惩罚：世界 XYZ 角速度平方和乘 hold；包含 Y 轴分量。
    pitch_velocity_weight: float = 4.0  # 负惩罚：机身 Y 轴真实角速度平方，全程生效，与整体角速度项叠加；rad/s。

    # 5. 非足部接触：保留 v16 的豁免与叠加，不按直立程度门控
    contact_force_threshold: float = 1.0  # N；接触看法向合力模长，支撑看世界向上分量，均要求大于阈值。
    undesired_contact_weight: float = 200.0  # 负惩罚：非脚掌接触刚体数，减去本周期有向上支撑的后小腿数。
    rear_knee_contact_weight: float =200.0  # 负惩罚：每条后小腿接触另扣；非支撑接触叠加通用项，脚掌不计入。
    hip_contact_weight: float = 200.0  # 负惩罚：每个髋部接触额外扣；与通用 60 叠加后共扣 120/s。

    # 6. 关节运动与动作超界
    joint_velocity_weight: float = 0.15  # 负惩罚：12 个关节实际角速度平方和，单位 rad/s；不使用观测中的 0.1 缩放。
    action_limit_weight: float = 40.0  # 负惩罚：原始网络动作超出 [-1,1] 部分的平方和，恒定超界也扣分。

    # 7. 存活与结束
    alive: float = 1.0  # 正收益：每秒固定收益，失败步也加入。
    failure_penalty: float = 1000.0  # 一次性负惩罚：跌倒终止扣分，不乘 dt，正常超时不扣。

    # 公式版本：5 仅将版本 1 的水平速度改为平方惩罚，用于区分训练历史。
    reward_schema_version: int = 5  # 1=v16；2=接触/角速度去重；3=速度平方惩罚；4=支撑去重；5=v16+水平速度平方惩罚。
    horizontal_velocity_sigma: float = 0.4  # 仅兼容旧版 1/2 的指数速度收益；当前平方惩罚不使用。

    def __post_init__(self):
        if type(self.reward_schema_version) is not int or self.reward_schema_version not in (1, 2, 3, 4, 5):
            raise ValueError("reward_schema_version 必须为 1、2、3、4 或 5")
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} 必须非负且有限")
        for name in ("upright_sigma", "height_sigma", "horizontal_velocity_sigma",
                     "hip_rear_sigma", "rear_symmetry_sigma", "rear_knee_clearance_sigma", "target_height"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正数")
