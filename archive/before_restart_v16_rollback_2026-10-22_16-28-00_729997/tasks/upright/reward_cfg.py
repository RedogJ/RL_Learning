"""后腿直立的独立奖励配置；除失败项外均为奖励率，计算时乘 dt。"""

from dataclasses import dataclass, fields
import math


@dataclass
class RewardCfg:
    # 公式版本自动随 run.json 保存；版本 4 后脚支撑收益排除同侧膝部同时支撑。
    reward_schema_version: int = 4
    horizontal_velocity_sigma: float = 0.4  # 仅旧版本 1/2 播放的指数速度收益使用；版本 3 及以上不使用。

    # sigma 越小，对误差的容忍范围越窄，必须大于 0。

    # 1. 基础存活
    alive: float = 1.0  # 每秒基础收益；当前公式在失败步也会加入，随后另扣失败罚分。

    # 2. 直立方向：按当前机身方向与竖直目标的误差计算
    upright_hold_weight: float = 1400.0  # 接近完全直立时的指数收益。
    upright_sigma: float = 0.6  # 前向轴与世界向上的点积误差尺度，无量纲；过大会让近水平姿态也拿高分。

    # 3. 机身高度：独立计算
    height_weight: float = 500.0  # 根节点接近目标高度的指数收益，不乘起身进度。
    target_height: float = 0.32  # m；膝离地后由脚掌单独支撑，机身会自然抬高。
    height_sigma: float = 0.5  # m；高度误差等于此值时，指数项约为最大值的 36.8%。

    # 4. 前后脚分工：后脚掌支撑，前脚抬起并收在胸前
    rear_support_weight: float = 20.0  # 逐物理帧、逐后腿计分：脚掌向上支撑且同侧膝部未向上支撑；两腿取均值。
    front_lift_weight: float = 20.0  # 前脚脱离地面的收益，独立于机身起身程度。
    front_foot_position_weight: float = 2000.0  # 两前脚到目标点的距离平方取均值，乘此权重扣分；每脚误差 0.1 m 时扣 10/s。
    # 机身 link 原点坐标系，单位 m：X 朝头部、Y 朝左、Z 朝背部；左右前腿自然屈曲收在胸前。
    front_foot_target_left: tuple[float, float, float] = (0.16, 0.142, -0.15)
    front_foot_target_right: tuple[float, float, float] = (0.16, -0.142, -0.15)

    # 5. 机身平衡：速度目标均为 0，全程按误差平方扣分，不乘直立程度。
    balance_weight: float = 20.0  # 世界水平线速度 vx² + vy² 惩罚；线速度单位 m/s。
    vertical_velocity_weight: float = 20.0  # 世界竖直线速度 vz² 惩罚，抑制上下运动；线速度单位 m/s。
    angular_velocity_weight: float = 20.0  # 机身 X/Z 轴角速度平方和惩罚，不包含 Y 轴。
    pitch_velocity_weight: float = 20.0  # 机身 Y 轴角速度平方惩罚，全程生效；角速度单位 rad/s。

    # 6. 后腿姿态：后髋回正、后腿左右对称
    hip_rear_weight: float = 200.0  # 后髋 RL/RR 接近 0 的指数收益；两腿指数项取乘积，最大为该项权重。
    hip_rear_sigma: float = 1.0  # rad；单腿偏差等于此值时，该腿指数项约为最大值的 36.8%。
    rear_symmetry_weight: float = 30.0  # 后腿左右对称的指数收益；对称时最大，误差达到 sigma 时约 36.8%。
    rear_symmetry_sigma: float = 1.0  # rad；髋和/大腿差/小腿差三项对称误差的 RMS 尺度。

    # 7. 非足部接触：三组互斥，同一个刚体只计入其中一项``
    undesired_contact_weight: float = 300.0  # 其余非足部：机身、头部、大腿、前小腿；排除所有髋部和后小腿。
    rear_knee_contact_weight: float = 300.0  # 每条后小腿接触的总罚分率，支撑/侧向/自碰撞均计入；不包含脚掌。
    hip_contact_weight: float = 300.0  # 每个髋部接触的总罚分率；
    contact_force_threshold: float = 1.0  # N；接触计数比较法向合力模长，足部支撑比较向上分量。

    # 8. 足部接触质量：触地脚掌不应在世界水平方向滑动
    foot_slide_weight: float = 4.0  # 触地足部世界 XY 速度模长之和；空中摆腿不扣分。

    # 9. 动作正则：平滑与网络输出超界是不同目标
    action_rate_weight: float = 30.0  # 相邻控制步裁剪后动作变化的平方和惩罚。
    action_limit_weight: float = 40.0  # 原始网络动作超出 [-1, 1] 部分的平方和惩罚。

    # 10. 回合失败：一次性罚分，不乘 dt
    failure_penalty: float = 100.0  # 跌倒终止时一次扣除，正常超时不扣。

    def __post_init__(self):
        if type(self.reward_schema_version) is not int or self.reward_schema_version not in (1, 2, 3, 4):
            raise ValueError("reward_schema_version 必须为 1、2、3 或 4")
        target_fields = ("front_foot_target_left", "front_foot_target_right")
        for name in target_fields:
            point = getattr(self, name)
            if not isinstance(point, (tuple, list)) or len(point) != 3:
                raise ValueError(f"{name} 必须是三个有限坐标，单位 m")
            try:
                finite = all(math.isfinite(value) for value in point)
            except TypeError:
                finite = False
            if not finite:
                raise ValueError(f"{name} 必须是三个有限坐标，单位 m")
            setattr(self, name, tuple(point))  # 兼容 JSON 恢复的列表，统一为三元组。
        for field in fields(self):
            if field.name in target_fields:
                continue
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} 必须非负且有限")
        for name in ("upright_sigma", "height_sigma", "horizontal_velocity_sigma",
                     "hip_rear_sigma", "rear_symmetry_sigma", "target_height"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正数")
