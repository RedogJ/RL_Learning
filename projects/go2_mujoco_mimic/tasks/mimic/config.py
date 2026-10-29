"""模仿任务参数。"""
from dataclasses import dataclass, field
from .reward_cfg import RewardCfg


@dataclass
class MimicTaskCfg:
    reward: RewardCfg = field(default_factory=RewardCfg)
    max_height_error: float = 0.25  # m，根部相对参考高度误差
    max_tilt_error: float = 0.8  # 根部坐标中重力方向 z 分量之差
    reference_cycles: int = 10  # 每个回合最多连续播放的参考动作周期数。
    random_reference_start: bool = False  # 在一个参考周期内随机初始化阶段。
    history_length: int = 10
    future_length: int = 10
    observation_noise: bool = True
