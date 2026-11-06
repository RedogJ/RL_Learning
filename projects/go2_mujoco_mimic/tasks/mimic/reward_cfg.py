"""SuperMimic 表 A.2：Go2 身体跟踪奖励配置。"""
from dataclasses import dataclass
import math


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
    joint_position_weight: float = 0.0  # 旧聚合模式的收益上限；设置 joint_position_weights 后不用此项
    joint_position_weights: tuple[float, ...] | None = None  # 12 个独立关节收益上限，按 FL/FR/RL/RR 的 hip/thigh/calf 排列；逐项求和
    joint_position_sigma: float = 1.0  # rad；各关节独立指数项的尺度；joint_sigmas 设置后不再使用
    joint_sigmas: tuple[float, ...] | None = None  # rad；3 项按 hip/thigh/calf 重复四次，或 12 项逐关节设置；None 共用 joint_position_sigma
    joint_position_mode: str = 'per_joint'  # 每个关节单独取指数再平均；group_mse 仅兼容旧模型
    action_rate_weight: float = 0.1  # 动作差平方和的惩罚
    torque_l2_weight: float = 0.0  # 周期平均的 12 个关节 PD 力矩平方和惩罚；0 表示关闭
    torque_mode: str = 'substep_mean'  # 真实仿真逐子步统计；想象用起止状态插值估计；endpoint 兼容旧模型
    joint_acc_l2_weight: float = 0.0  # 12 个关节速度一步差分（加速度）平方和的惩罚；0 表示关闭
    joint_limit_weight: float = 2.0  # 超出关节范围的绝对偏差之和
    failure_penalty: float = 0.0  # 失败时一次性扣分，不乘 dt；正常超时或参考结束不扣

    def __post_init__(self):
        if self.joint_position_mode not in ('per_joint', 'group_mse'):
            raise ValueError('未知 joint_position_mode')
        if self.torque_mode not in ('substep_mean', 'endpoint'):
            raise ValueError('未知 torque_mode')
        for name, value in vars(self).items():
            if name.endswith('_sigma') and value <= 0:
                raise ValueError(f'{name} 必须大于 0')
        if self.joint_position_weights is not None:
            if (len(self.joint_position_weights) != 12
                    or any(not math.isfinite(x) or x < 0 for x in self.joint_position_weights)):
                raise ValueError('joint_position_weights 必须是 12 个有限非负数')
            if self.joint_position_mode != 'per_joint':
                raise ValueError('独立关节权重需要 joint_position_mode=per_joint')
        if self.joint_sigmas is not None:
            if (len(self.joint_sigmas) not in (3, 12)
                    or any(not math.isfinite(x) or x <= 0 for x in self.joint_sigmas)):
                raise ValueError('joint_sigmas 必须是 3 或 12 个有限正数')
            if self.joint_position_mode == 'group_mse' and len(self.joint_sigmas) != 3:
                raise ValueError('group_mse 仅支持 3 项 joint_sigmas')

    @property
    def joint_position_max_reward(self):
        """RSI 使用的关节奖励率上限；独立模式按各关节的实际权重求和。"""
        if self.joint_position_weights is not None:
            return sum(self.joint_position_weights)
        return self.joint_position_weight


def saved_reward_config(raw):
    """按历史语义补缺省字段，不用当前任务的非零权重补旧配置。"""
    defaults = vars(RewardCfg()).copy()
    defaults.update(joint_position_mode='group_mse', torque_mode='endpoint')
    defaults.update(raw)
    for key in ('joint_sigmas', 'joint_position_weights'):
        if defaults[key] is not None:
            defaults[key] = tuple(defaults[key])
    return defaults
