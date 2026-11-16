from dataclasses import dataclass


@dataclass
class RewardCfg:
    # Part 1：根部世界位姿跟踪。
    root_position_weight: float = 0.5
    root_position_sigma: float = 0.3  # m
    root_orientation_weight: float = 0.5
    root_orientation_sigma: float = 0.4  # rad
    # Part 2：身体空间模仿；位置/方向先按当前根部 xy+yaw 对齐参考。
    body_position_weight: float = 1.0
    body_position_sigma: float = 0.3  # m
    body_orientation_weight: float = 1.0
    body_orientation_sigma: float = 0.4  # rad
    body_linear_velocity_weight: float = 1.0
    body_linear_velocity_sigma: float = 1.0  # m/s
    body_angular_velocity_weight: float = 1.0
    body_angular_velocity_sigma: float = 3.14  # rad/s
    # Part 3：控制正则；负惩罚，最终所有项一起乘控制 dt。
    action_rate_weight: float = 0.1  # sum((当前目标角-上次目标角)^2)，rad²
    joint_limit_weight: float = 2.0  # sum(关节超出软限位的角度)，rad
