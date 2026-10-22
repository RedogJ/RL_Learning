"""站立任务的 CommandCfg 配置。"""

from dataclasses import dataclass


@dataclass
class CommandCfg:
    forward_range: tuple[float, float] = (0.0, 0.0)  # m/s；纯站立不要求前后移动。
    lateral_range: tuple[float, float] = (0.0, 0.0)  # m/s；纯站立不要求侧移。
    yaw_range: tuple[float, float] = (0.0, 0.0)  # rad/s；纯站立不要求转向。
    resample_seconds: float | None = None  # None：只在回合重置时采样，回合内指令不变。
    standing_probability: float = 1.0  # 当前所有环境均练习零速度站立。
    keyboard_forward: float = 0.4
    keyboard_backward: float = 0.3
    keyboard_turn: float = 0.6
