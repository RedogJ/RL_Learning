"""后腿直立的环境、终止与配置组合。"""

from dataclasses import dataclass, field
import math

from go2_learning.simulation.config import EnvCfg
from .agent_cfg import make_agent_cfg
from .command_cfg import CommandCfg
from .integral_cfg import IntegralCfg
from .randomization_cfg import RandomizationCfg
from .reward_cfg import RewardCfg


@dataclass
class TerminationCfg:
    minimum_height: float = 0.14
    maximum_tilt_degrees: float = 120.0  # 机身 +X 偏离世界 +Z 的角度；允许四足水平初始姿态的 90°。

    def __post_init__(self):
        if not math.isfinite(self.minimum_height) or self.minimum_height < 0:
            raise ValueError("minimum_height 必须非负且有限")
        if not math.isfinite(self.maximum_tilt_degrees) or not 90 < self.maximum_tilt_degrees < 180:
            raise ValueError("maximum_tilt_degrees 必须位于 (90, 180)，以允许从水平姿态起身")


@dataclass
class UprightTaskCfg:
    environment: EnvCfg = field(default_factory=lambda: EnvCfg(
        physics_dt=1 / 120, decimation=4, episode_seconds=15.0,
        reset_settle_seconds=0.3, env_spacing=3.0,
    ))
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    commands: CommandCfg = field(default_factory=CommandCfg)
    rewards: RewardCfg = field(default_factory=RewardCfg)
    integral: IntegralCfg = field(default_factory=IntegralCfg)
    randomization: RandomizationCfg = field(default_factory=RandomizationCfg)
    agent: dict = field(default_factory=make_agent_cfg)
    train_num_envs: int = 2048
    train_randomization: bool = False
    keyboard_control: bool = False
