"""SuperMimic 表 A.2：Go2 身体跟踪奖励配置。"""
from dataclasses import dataclass


@dataclass
class RewardCfg:
    # 跟踪的身体部位均方误差经过 exp(-误差/sigma²)。
    body_position_weight: float = 1.0
    body_position_sigma: float = 0.3  # m
    body_orientation_weight: float = 1.0
    body_orientation_sigma: float = 0.4  # rad
    body_linear_velocity_weight: float = 1.0
    body_linear_velocity_sigma: float = 1.0  # m/s
    body_angular_velocity_weight: float = 1.0
    body_angular_velocity_sigma: float = 3.141592653589793  # rad/s
    root_position_weight: float = 0.5
    root_position_sigma: float = 0.3  # m
    root_orientation_weight: float = 0.5
    root_orientation_sigma: float = 0.4  # rad
    action_rate_weight: float = 0.1  # 动作差平方和的惩罚
    joint_limit_weight: float = 2.0  # 超出关节范围的绝对偏差之和

    def __post_init__(self):
        for name, value in vars(self).items():
            if name.endswith('_sigma') and value <= 0:
                raise ValueError(f'{name} 必须大于 0')
