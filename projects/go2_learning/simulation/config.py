"""通用仿真与回合参数的类型；具体数值由每个任务提供。"""

from dataclasses import dataclass
import math


@dataclass
class EnvCfg:
    physics_dt: float
    decimation: int
    episode_seconds: float
    reset_settle_seconds: float
    env_spacing: float

    def __post_init__(self):
        for name in ("physics_dt", "episode_seconds", "env_spacing"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为正数且有限")
        if not isinstance(self.decimation, int) or self.decimation < 1:
            raise ValueError("decimation 必须为正整数")
        if not math.isfinite(self.reset_settle_seconds) or self.reset_settle_seconds < 0:
            raise ValueError("reset_settle_seconds 必须非负且有限")
