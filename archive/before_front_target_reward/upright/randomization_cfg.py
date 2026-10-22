"""直立任务的独立扰动配置；初期训练默认关闭，学会后可显式开启。"""

from dataclasses import dataclass
import math


@dataclass
class RandomizationCfg:
    joint_position_noise: float = 0.05
    push_force_range: tuple[float, float] = (10.0, 30.0)
    push_interval_range: tuple[float, float] = (3.0, 5.0)
    push_duration: float = 0.15
    push_body: str = "base"
    push_point_half_extent: tuple[float, float, float] = (0.1, 0.03, 0.03)

    def __post_init__(self):
        if not math.isfinite(self.joint_position_noise) or self.joint_position_noise < 0:
            raise ValueError("joint_position_noise 必须非负且有限")
        for name in ("push_force_range", "push_interval_range"):
            low, high = getattr(self, name)
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
                raise ValueError(f"{name} 范围无效")
        if not math.isfinite(self.push_duration) or self.push_duration <= 0:
            raise ValueError("push_duration 必须为正数且有限")
        if len(self.push_point_half_extent) != 3 or any(
            not math.isfinite(v) or v < 0 for v in self.push_point_half_extent
        ):
            raise ValueError("push_point_half_extent 必须有三个非负且有限的值")
