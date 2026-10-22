"""移动任务的 IntegralCfg 配置。"""

import math
from dataclasses import dataclass


@dataclass
class IntegralCfg:
    limit: float = 0.5  # 每轴积分限幅，rad·s；限制历史误差的积累。
    height_limit: float = 0.2  # 高度误差积分限幅，m·s；与姿态积分分别设置。
    decay_seconds: float = 5.0  # 衰减时间常数，s；旧偏差逐渐遗忘，减少反向补偿。

    def __post_init__(self):
        if not math.isfinite(self.limit) or self.limit <= 0:
            raise ValueError("积分 limit 必须为正数且有限")
        if not math.isfinite(self.height_limit) or self.height_limit <= 0:
            raise ValueError("积分 height_limit 必须为正数且有限")
        if not math.isfinite(self.decay_seconds) or self.decay_seconds <= 0:
            raise ValueError("积分 decay_seconds 必须为正数且有限")

