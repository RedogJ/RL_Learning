"""sigma 只在开训前更新一次；采样缓冲不用于 PPO 梯度更新。"""

from dataclasses import dataclass, field
import math

from .tracking_errors import ERROR_SPECS


@dataclass
class SigmaCfg:
    enabled: bool = True
    capacity: int = 16384  # 最多保留多少行近期误差；默认全部 4 项约 0.25 MiB。
    samples_per_step: int = 64  # 每控制步均匀抽取活动环境，在 GPU 写环形缓冲。
    bootstrap_steps: int = 128  # 旧模型无缓冲/目标或采样条件改变时，先采样，不更新网络。
    minimum_samples: int = 128  # 每项有效样本不足时保留 reward_cfg.py 中的初始 sigma。
    quantile: float = 0.75  # 用误差的 75% 分位数，不用加权后的奖励反推误差。
    reward_at_quantile: float = 0.5  # 若不触及限制，使该分位点收益为最高收益的一半。
    change_lower: float = 0.5  # 相容的上阶段 sigma 每次最多收紧 50%。
    change_upper: float = 1.5  # 每次最多放宽 50%；首轮/目标改变不沿用旧尺度。

    # 物理尺度上下界，不跟随缓冲分布移动；不是新的奖励权重。
    # 每项 (最小 sigma, 最大 sigma)，单位见 tracking_errors.py；可自行修改。
    bounds: dict = field(default_factory=lambda: {
        "upright_hold": (0.004, 0.3),
        "height": (0.01, 0.2),
        "rear_symmetry": (0.03, 1.0),
        "rear_posture": (0.05, 1.0),
    })

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled 必须为 bool")
        for name in ("capacity", "samples_per_step", "bootstrap_steps", "minimum_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} 必须为正整数")
        if self.samples_per_step > self.capacity or self.minimum_samples > self.capacity:
            raise ValueError("采样批大小/最低样本数不能大于缓冲容量")
        for name in ("quantile", "reward_at_quantile"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} 必须在 (0,1) 内")
        if not 0 < self.change_lower <= 1 <= self.change_upper < float("inf"):
            raise ValueError("sigma 变化比例要求 0 < 下限 <= 1 <= 上限且有限")
        if set(self.bounds) != set(ERROR_SPECS):
            raise ValueError("sigma bounds 必须覆盖全部连续误差项")
        for name, limits in self.bounds.items():
            if len(limits) != 2 or not all(math.isfinite(x) for x in limits) or not 0 < limits[0] <= limits[1]:
                raise ValueError(f"{name} 的 sigma 上下界无效")
