"""Go2 后腿直立奖励配置；只列出当前使用的调参项。

连续跟踪误差项为 weight * exp(-误差均方 / sigma²)，weight 表示每秒最高收益。
sigma 为初始/手动值；自动模式在每次开训前校准一次，本次训练中固定。
触地滑动、机体线/角速度、前腿目标、后髋回正、关节速度、输出变化、动作超界、碰撞和失败为负惩罚，不使用自动 sigma。所有 weight 均填写非负数。
持续项每控制步乘 dt；失败罚分一次结算。
hold = exp(-((1 - alignment) / upright_sigma)²)，alignment 为机身 +X 与世界向上的点积。
"""

from dataclasses import dataclass, fields
import math


@dataclass
class RewardCfg:
    # Part 1. 机体线速度与角速度：各轴独立，全程生效，目标均为 0。
    # 线速度使用世界坐标；每步奖励 = -weight * v_axis² * dt；速度单位 m/s。
    linear_velocity_x_weight: float = 1500.0  # vx：世界 X 方向。
    linear_velocity_y_weight: float = 1500.0  # vy：世界 Y 方向。
    linear_velocity_z_weight: float = 1500.0  # vz：世界 Z 方向（上下）；0 关闭。

    # 角速度使用机身坐标；每步奖励 = -weight * omega_axis² * dt；角速度单位 rad/s。
    # 这些是实际角速度分量，不是欧拉角的时间导数，也不乘直立程度 hold。
    angular_velocity_x_weight: float = 2000  # wx：绕机身 X 轴。
    angular_velocity_y_weight: float = 20  # wy：绕机身 Y 轴。
    angular_velocity_z_weight: float = 20  # wz：绕机身 Z 轴。

    # Part 2. 能耗与动作正则：关节运动、驱动力消耗及动作约束统一放在这里。
    # 2.1 关节运动：实际角速度均方惩罚，数值不等于物理能量。
    joint_velocity_weight: float = 30.0  # 负惩罚：-weight * mean(12 个实际关节角速度²) * dt；rad/s，不使用观测的 0.1 缩放。

    # 2.2 输出变化：抑制相邻控制步的网络指令突变。
    action_rate_weight: float = 8000.0  # 负平方惩罚：-weight * mean((本次下发目标角 - 上次下发目标角)²) * dt；rad²，不除 dt，不参与 sigma 校准。
    action_rate_derivative_weight: float = 10.0  # 负平方惩罚：-weight * mean(((本次目标角变化量 - 上次目标角变化量) / dt)²) * dt；0 关闭，不参与 sigma 校准。
    action_filter_alpha: float = 0.0  # 动作低通：上一执行动作的占比；0 关闭，越大越平滑、响应越慢。

    # 2.3 动作超界：约束策略输出；它本身不衡量电机能耗。
    action_limit_weight: float = 200.0  # 负惩罚：原始网络动作超出 [-1,1] 部分的平方和，恒定超界也扣分。

    # Part 3. 目标姿态：机体方向、高度及前后腿关节姿态统一管理。
    # 3.1 机体方向：前向轴朝世界向上。
    upright_hold_weight: float = 1000.0  # 正收益：weight * hold，机身前向轴越朝上越高。
    upright_sigma: float = 0.02  # 无量纲；1-alignment 的指数误差尺度，误差等于 sigma 时取最大收益的 e^-1。

    # 3.2 机体高度：根节点相对环境原点的目标高度。
    target_height: float = 0.4  # m；机身根节点相对环境原点的高度目标。
    height_weight: float = 800.0  # 正收益：目标高度的高斯型收益，全程生效，不乘 hold。
    height_sigma: float = 0.01  # m；高度误差等于 sigma 时取最大收益的 e^-1。

    # 3.3 后腿目标：后髋回正及左右对称。
    hip_rear_weight: float = 3000.0  # 负惩罚：-weight * mean(两个后髋角度²) * dt；目标 0 rad，全程生效。

    rear_symmetry_weight: float = 2000.0  # 正收益上限；误差为髋角之和、大腿角之差、小腿角之差的 RMS。
    rear_symmetry_sigma: float = 0.03736947069471662  # rad。

    # 后腿直立参考：按根节点高 0.4 m、后脚位于机体下方且脚掌球面接地的几何关系定义。
    # 只约束两侧大腿/小腿共四个关节；后髋继续使用上面的回正项。
    rear_posture_weight: float = 20.0  # 正收益上限；误差为四个关节相对参考角的 RMS；0 关闭。
    rear_posture_sigma: float = 0.054258392716485915  # rad。
    rear_posture_thigh: float = 2.6934045  # rad；左右后大腿参考角。
    rear_posture_calf: float = -2.2452164  # rad；左右后小腿参考角。改变目标高度后需重新评估参考角。

    # 3.4 前腿目标：实际关节角的均方误差，全程生效，网络仍控制全部关节。
    front_target_weight: float = 6000.0  # 负惩罚：-weight * mean(六个前腿关节角相对目标的误差²) * dt；rad²，全程生效。

    front_target_hip: float = 0.0  # rad；左右前髋目标。
    front_target_thigh: float = 1.4  # rad；左右前大腿目标。
    front_target_calf: float = -2.4  # rad；左右前小腿目标。

    # Part 4. 支撑与接触：支撑、离地、触地滑动及非足部接触统一管理。
    # 4.1 接触判定：脚掌支撑与非足部接触共用同一个阈值。
    contact_force_threshold: float = 1.0  # N；接触看法向合力模长，支撑看世界向上分量，均要求大于阈值。

    # 4.2 脚掌支撑：后脚支撑、同时腾空与前脚抬起。
    rear_support_weight: float = 700.0  # 正收益：两后脚支撑比例取均值；膝部同时支撑不取消脚掌收益。

    rear_feet_airborne_weight: float = 2000.0  # 负惩罚：两后脚同时无向上支撑的物理帧比例乘 weight；单脚离地不扣。

    front_lift_weight: float = 1.0  # 正收益：两前脚不支撑的比例取均值，不乘 hold。
    # 4.3 膝部离地：衡量后小腿 link 原点高度。
    rear_knee_clearance_weight: float = 0.0  # 正收益：weight * (1-exp(-max(h,0)/sigma))，h 取较低后小腿 link 原点的世界高度。
    rear_knee_clearance_sigma: float = 0.1  # m；膝部离地收益的高度尺度，不是碰撞表面净空。

    # 4.4 触地质量：抑制脚掌触地时滑动。
    foot_slide_weight: float = 600.0  # 负惩罚：-weight * 触地脚掌世界 XY 速度平方的平均值 * dt；无脚触地取 0。

    # 4.5 非足部接触：通用惩罚与部位惩罚可叠加，不按直立程度门控。
    undesired_contact_weight: float = 20.0  # 负惩罚：非脚掌接触刚体数，减去本周期有向上支撑的后小腿数。
    rear_knee_contact_weight: float = 20.0  # 负惩罚：每条后小腿接触另扣；非支撑接触叠加通用项，脚掌不计入。
    hip_contact_weight: float = 20.0  # 负惩罚：每个髋部接触额外扣；与通用非足部接触项叠加。

    # Part 5. 存活与结束
    alive: float = 100.0  # 正收益：每秒固定收益，失败步也加入。
    failure_penalty: float = 40000.0  # 一次性负惩罚：跌倒终止扣分，不乘 dt，正常超时不扣。

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if (field.name in ("front_target_hip", "front_target_thigh", "front_target_calf")
                    or field.name in ("rear_posture_thigh", "rear_posture_calf")):
                if not math.isfinite(value):
                    raise ValueError(f"{field.name} 必须有限")
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} 必须非负且有限")
        for name in (field.name for field in fields(self)
                     if field.name.endswith("_sigma") or field.name == "target_height"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正数")
