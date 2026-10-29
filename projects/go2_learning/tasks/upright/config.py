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
from .sigma_cfg import SigmaCfg


@dataclass
class TerminationCfg:
    minimum_height: float = 0.14
    maximum_tilt_degrees: float = 120.0  # 机身 +X 偏离世界 +Z 超过此角度即失败。
    reset_on_nonfoot_contact: bool = True  # 任一非脚掌刚体接触超过接触阈值即失败；无膝部支撑豁免。

    def __post_init__(self):
        if type(self.reset_on_nonfoot_contact) is not bool:
            raise ValueError("reset_on_nonfoot_contact 必须为 bool")
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
    sigma_calibration: SigmaCfg = field(default_factory=SigmaCfg)
    reset: ResetCfg = field(default_factory=ResetCfg)
    integral: IntegralCfg = field(default_factory=IntegralCfg)
    randomization: RandomizationCfg = field(default_factory=RandomizationCfg)
    agent: dict = field(default_factory=make_agent_cfg)
    train_num_envs: int = 2048
    train_randomization: bool = False
    finite_horizon: bool = True  # 15 s 为真正任务终点；Critic 观测包含剩余时间。
    keyboard_control: bool = False
    lock_front_legs: bool = False  # 默认由策略控制前腿，通过目标姿态奖励学习。
    front_lock_position_iterations: int = 8  # 保留已校准重置姿态使用的求解精度。
    front_lock_velocity_iterations: int = 4
