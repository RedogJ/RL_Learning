"""Go2 上的缩小版复现参数；与原论文数值不同之处在 README 标注。"""
from dataclasses import dataclass
import math


@dataclass
class ModelCfg:
    proprio_dim: int = 49  # root v/w/g + q/dq + previous action + root height + 三个积分观测
    intent_dim: int = 6  # running/upright one-hot + commands vx/vy/wz + target height
    action_dim: int = 12
    latent_dim: int = 16
    hidden: int = 256
    history: int = 4
    horizon: int = 16
    embedding: int = 128
    heads: int = 4
    layers: int = 4
    noise_steps: int = 20
    kl_weight: float = 0.001  # 简化意图编码的 KL，避免微弱指令信息被正则项抹去
    learning_rate: float = 0.0001
    emphasis: float = 0.0  # 0 保持旧权重结构；新 diffusion 训练默认设为 6。
    projection_seed: int = 42


@dataclass(frozen=True)
class GuidanceCfg:
    mode: str = 'separate'
    velocity: float = 1.0  # 0.5 * sum_t ||v_horizontal - command||²，m/s；直立目标用 body YZ。
    orientation: float = 1.0  # 重力方向误差平方的时间均值。
    height: float = 4.0  # 高度误差平方的时间均值。
    yaw: float = 1.0  # 角速度误差平方的时间均值。

    def __post_init__(self):
        if self.mode not in ('legacy', 'separate'):
            raise ValueError('引导代价模式必须为 legacy 或 separate')
        for name in ('velocity','orientation','height','yaw'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(f'{name} 引导权重必须有限且非负')
