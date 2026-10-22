"""后腿直立的环境、终止与配置组合。"""

from dataclasses import dataclass, field
import math

from go2_learning.simulation.config import EnvCfg
from .agent_cfg import make_agent_cfg
from .command_cfg import CommandCfg
from .integral_cfg import IntegralCfg
from .randomization_cfg import RandomizationCfg
from .reward_cfg import RewardCfg
from .reset_cfg import ResetCfg


@dataclass
class TerminationCfg:
    minimum_height: float = 0.14
    maximum_tilt_degrees: float = 45.0  # 机身 +X 偏离世界 +Z 超过此角度即失败。

    def __post_init__(self):
        if not math.isfinite(self.minimum_height) or self.minimum_height < 0:
            raise ValueError("minimum_height 必须非负且有限")
        if not math.isfinite(self.maximum_tilt_degrees) or not 0 < self.maximum_tilt_degrees < 180:
            raise ValueError("maximum_tilt_degrees 必须位于 (0, 180)")


@dataclass
class UprightTaskCfg:
    environment: EnvCfg = field(default_factory=lambda: EnvCfg(
        physics_dt=1 / 120, decimation=4, episode_seconds=15.0,
        reset_settle_seconds=0.3, env_spacing=3.0,
    ))
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    commands: CommandCfg = field(default_factory=CommandCfg)
    rewards: RewardCfg = field(default_factory=RewardCfg)
    reset: ResetCfg = field(default_factory=ResetCfg)
    integral: IntegralCfg = field(default_factory=IntegralCfg)
    randomization: RandomizationCfg = field(default_factory=RandomizationCfg)
    agent: dict = field(default_factory=make_agent_cfg)
    train_num_envs: int = 2048
    train_randomization: bool = False
    keyboard_control: bool = False
    lock_front_legs: bool = True  # 将前腿六个关节锁在 reset_cfg 的 front_* 角度。
    front_lock_position_iterations: int = 64  # 提高硬限位/锁定驱动的求解精度。
    front_lock_velocity_iterations: int = 16
