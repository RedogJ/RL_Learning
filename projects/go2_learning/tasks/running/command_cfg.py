"""移动任务的 CommandCfg 配置。"""

from dataclasses import dataclass


@dataclass
class CommandCfg:
    forward_fixed_magnitude: float | None = 1.5  # m/s；非 None 时移动回合前后方向各 50%，覆盖 forward_range。
    forward_range: tuple[float, float] = (-0.3, 0.4)  # m/s；机身 X 轴，负值后退、正值前进。
    lateral_range: tuple[float, float] = (0.0, 0.0)  # m/s；机身 Y 轴，当前不要求侧移。
    yaw_range: tuple[float, float] = (0.0, 0.0)  # rad/s；训练只走直线，目标转向角速度为零。
    resample_seconds: float | None = None  # None：只在回合重置时采样，回合内指令不变。
    standing_probability: float = 1.0 / 3.0  # 停车占 1/3，其余前后各半，即 -1.5、0、+1.5 各占 1/3。
    keyboard_forward: float = 1.5
    keyboard_backward: float = 1.5
    keyboard_turn: float = 0.6
