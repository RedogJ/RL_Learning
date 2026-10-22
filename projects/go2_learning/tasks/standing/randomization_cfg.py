"""站立任务的 RandomizationCfg 配置。"""

import math
from dataclasses import dataclass


@dataclass
class RandomizationCfg:
    joint_position_noise: float = 0.1  # 默认站姿附近 ±0.1 rad。
    push_force_range: tuple[float, float] = (40.0, 100.0)  # 水平合力大小，N；较原 20～50 N 加倍。
    push_interval_range: tuple[float, float] = (2.0, 4.0)  # 重置/上次推力结束后的等待时间，s。
    push_duration: float = 0.4  # 每次持续施力时间，s；120 Hz 下持续 48 个物理步。
    push_body: str = "base"
    push_point_half_extent: tuple[float, float, float] = (0.15, 0.04, 0.04)  # m；质心周围、沿机身 XYZ 轴的作用点偏移范围。

    def __post_init__(self):
        if not math.isfinite(self.joint_position_noise) or self.joint_position_noise < 0:
            raise ValueError("joint_position_noise 必须非负且有限")
        for name in ("push_force_range", "push_interval_range"):
            low, high = getattr(self, name)
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
                raise ValueError(f"{name} 必须满足 0 <= 最小值 <= 最大值，且为有限值")
        if not math.isfinite(self.push_duration) or self.push_duration <= 0:
            raise ValueError("push_duration 必须为正数且有限")
        if len(self.push_point_half_extent) != 3 or any(
            not math.isfinite(value) or value < 0 for value in self.push_point_half_extent
        ):
            raise ValueError("push_point_half_extent 必须包含三个非负且有限的数值")

