"""直立不使用相对四足水平姿态的误差积分；观测仍保留三个零占位。"""

from dataclasses import dataclass
import math


@dataclass
class IntegralCfg:
    enabled: bool = False
    limit: float = 0.5
    height_limit: float = 0.2
    decay_seconds: float = 5.0

    def __post_init__(self):
        for name in ("limit", "height_limit", "decay_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正数且有限")
