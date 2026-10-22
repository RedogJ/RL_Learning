"""移动任务：跟踪机身速度指令，维护独立配置和模型版本。"""

from copy import deepcopy

from go2_learning.tasks.base import Task
from . import TASK_NAME, TASK_VERSION
from .config import RunningTaskCfg, EnvCfg, TerminationCfg
from .command_cfg import CommandCfg
from .integral_cfg import IntegralCfg
from .reward_cfg import RewardCfg
from .rewards import reward_components


class RunningTask(Task):
    name = TASK_NAME
    version = TASK_VERSION

    def __init__(self, cfg=None):
        self.cfg = cfg or RunningTaskCfg()

    @property
    def target_height(self):
        return self.cfg.rewards.target_height

    @property
    def needs_contact_sensor(self):
        return (self.cfg.rewards.undesired_contact_weight != 0
                or self.cfg.rewards.foot_air_time_weight != 0 or self.cfg.rewards.foot_contact_ratio_weight != 0
                or self.cfg.rewards.foot_force_balance_weight != 0 or self.cfg.rewards.all_feet_airborne_weight != 0
                or self.cfg.rewards.foot_slide_weight != 0 or self.cfg.rewards.foot_time_variance_weight != 0)

    @property
    def contact_force_threshold(self):
        return self.cfg.rewards.contact_force_threshold

    def reward_components(self, state, actions, failed, dt):
        return reward_components(state, actions, failed, dt, self.cfg.rewards)

    def failures(self, state):
        cfg = self.cfg.termination
        return (state["height"] < cfg.minimum_height) | self.tilt_failures(state)

    def load_playback_config(self, run_cfg):
        self.cfg.agent = deepcopy(run_cfg["agent_cfg"])
        keyboard_forward = self.cfg.commands.keyboard_forward
        keyboard_backward = self.cfg.commands.keyboard_backward
        self.cfg.commands = CommandCfg(**run_cfg["command_cfg"])
        # 键盘是当前操作者的输入映射，不随旧模型恢复成旧速度。
        self.cfg.commands.keyboard_forward = keyboard_forward
        self.cfg.commands.keyboard_backward = keyboard_backward
        reward_cfg = dict(run_cfg["reward_cfg"])
        # 旧模型保存过的速度门控字段，当前奖励已不再使用。
        reward_cfg.pop("foot_contact_ratio_min", None)
        reward_cfg.pop("standing_linear_threshold", None)
        reward_cfg.pop("standing_yaw_threshold", None)
        reward_cfg.pop("foot_support_weight", None)
        reward_cfg.pop("foot_support_standing_only", None)
        self.cfg.rewards = RewardCfg(**{"stationary_linear_sigma": None, "all_feet_airborne_weight": 0.0, "foot_force_balance_weight": 0.0, "action_activity_weight": 0.0, "foot_air_time_weight": 0.0, "foot_contact_ratio_weight": 0.0,
                                        "foot_slide_weight": 0.0, "foot_time_variance_weight": 0.0,
                                        "posture_stand_still_scale": 1.0, **reward_cfg})
        self.cfg.integral = IntegralCfg(**run_cfg["integral_cfg"])
        self.cfg.environment = EnvCfg(**run_cfg.get("env_cfg", {
            "physics_dt": 1 / 120, "decimation": 4, "episode_seconds": 15.0, "env_spacing": 3.0,
            "reset_settle_seconds": run_cfg.get("reset_settle_seconds", 0.3),
        }))
        self.cfg.termination = TerminationCfg(**run_cfg.get("termination_cfg", {}))
