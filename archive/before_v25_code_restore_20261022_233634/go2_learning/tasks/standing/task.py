"""站立任务：执行站立奖励和失败判断，恢复旧模型保存的任务配置。"""

from copy import deepcopy

from go2_learning.tasks.base import Task
from . import TASK_NAME, TASK_VERSION
from .config import StandingTaskCfg, EnvCfg, TerminationCfg
from .command_cfg import CommandCfg
from .integral_cfg import IntegralCfg
from .reward_cfg import RewardCfg
from .rewards import reward_components


class StandingTask(Task):
    name = TASK_NAME
    version = TASK_VERSION

    def __init__(self, cfg=None):
        self.cfg = cfg or StandingTaskCfg()

    @property
    def target_height(self):
        return self.cfg.rewards.target_height

    @property
    def needs_contact_sensor(self):
        return self.cfg.rewards.undesired_contact_weight != 0 or self.cfg.rewards.foot_support_weight != 0

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
        self.cfg.commands = CommandCfg(**run_cfg["command_cfg"])
        retired = {"roll_integral_weight", "pitch_integral_weight", "height_integral_weight",
                   "tracking_linear_weight", "linear_sigma", "tracking_yaw_weight", "yaw_sigma",
                   "vertical_velocity_weight"}
        saved = {key: value for key, value in run_cfg["reward_cfg"].items() if key not in retired}
        self.cfg.rewards = RewardCfg(**{
            "tracking_reward_type": "quadratic", "only_positive_rewards": False,
            "action_limit_outside_floor": False,
            "roll_velocity_weight": 0., "pitch_velocity_weight": 0.,
            "horizontal_velocity_weight": 0., "yaw_velocity_weight": 0.,
            "undesired_contact_weight": 0., "foot_support_weight": 0.,
            "action_rate_weight": 0., **saved,
        })
        self.cfg.integral = IntegralCfg(**run_cfg["integral_cfg"])
        self.cfg.environment = EnvCfg(**run_cfg.get("env_cfg", {
            "physics_dt": 1 / 120, "decimation": 4, "episode_seconds": 15.0, "env_spacing": 3.0,
            "reset_settle_seconds": run_cfg.get("reset_settle_seconds", 0.3),
        }))
        self.cfg.termination = TerminationCfg(**run_cfg.get("termination_cfg", {}))
