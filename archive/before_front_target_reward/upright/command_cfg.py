"""直立任务不接收移动指令，保留通用环境的三维零指令接口。"""

from dataclasses import dataclass


@dataclass
class CommandCfg:
    forward_range: tuple[float, float] = (0.0, 0.0)
    lateral_range: tuple[float, float] = (0.0, 0.0)
    yaw_range: tuple[float, float] = (0.0, 0.0)
    resample_seconds: float | None = None
    standing_probability: float = 1.0
    keyboard_forward: float = 0.0
    keyboard_backward: float = 0.0
    keyboard_turn: float = 0.0
