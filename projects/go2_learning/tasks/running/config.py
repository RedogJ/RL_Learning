"""移动任务的组合配置：环境时序、结束条件及各组独立配置。"""

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
    minimum_height: float = 0.18
    maximum_tilt_degrees: float = 60.0

    def __post_init__(self):
        if not math.isfinite(self.minimum_height):
            raise ValueError("minimum_height 必须有限")
        if not math.isfinite(self.maximum_tilt_degrees) or not 0 < self.maximum_tilt_degrees < 180:
            raise ValueError("maximum_tilt_degrees 必须位于 (0, 180)")


@dataclass
class RunningTaskCfg:
    environment: EnvCfg = field(default_factory=lambda: EnvCfg(
        physics_dt=1 / 120, decimation=4, episode_seconds=15.0,
        reset_settle_seconds=0.3, env_spacing=3.0,
    ))
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    commands: CommandCfg = field(default_factory=CommandCfg)
    rewards: RewardCfg = field(default_factory=RewardCfg)
    randomization: RandomizationCfg = field(default_factory=RandomizationCfg)
    integral: IntegralCfg = field(default_factory=IntegralCfg)
    agent: dict = field(default_factory=make_agent_cfg)
    train_num_envs: int = 2048
    train_randomization: bool = True
    keyboard_control: bool = True
