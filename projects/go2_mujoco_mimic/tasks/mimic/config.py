"""模仿任务参数。"""
from dataclasses import dataclass, field
from .reward_cfg import RewardCfg


@dataclass
class MimicTaskCfg:
    reward: RewardCfg = field(default_factory=RewardCfg)
    max_height_error: float = 0.25  # m，根部相对参考高度误差
    max_tilt_error: float = 0.8  # 根部坐标中重力方向 z 分量之差
    terminate_on_tracking_error: bool = True  # 是否因高度或姿态跟踪误差提前结束。
    reference_loop: bool = True  # False 表示单次动作，轨迹结束不循环。
    reference_cycles: int = 10  # 每个回合最多连续播放的参考动作周期数。
    random_reference_start: bool = False  # 在一个参考周期内随机初始化阶段。
    heading_observation: bool = False  # 额外提供当前机身与参考的航向误差 sin/cos。
    longitudinal_observation: bool = False  # 参考前向上的位置、速度误差。
    history_length: int = 10
    future_length: int = 10
    observation_noise: bool = True
    terminate_on_nonfoot_contact: bool = False  # True 时任一非脚掌几何体接触地面即判失败；默认关闭，旧检查点不受影响。
