"""从普通四足站姿起身，以两后脚支撑保持机身纵轴朝上。"""

from copy import deepcopy
import math

from go2_learning.tasks.base import Task
from . import TASK_NAME, TASK_VERSION
from .config import UprightTaskCfg, EnvCfg, TerminationCfg
from .command_cfg import CommandCfg
from .integral_cfg import IntegralCfg
from .reward_cfg import RewardCfg
from .rewards import rear_support_fraction, rear_symmetry_error, reward_components, upright_alignment


class UprightTask(Task):
    name = TASK_NAME
    version = TASK_VERSION

    def __init__(self, cfg=None):
        self.cfg = cfg or UprightTaskCfg()

    @property
    def target_height(self):
        return self.cfg.rewards.target_height

    @property
    def needs_contact_sensor(self):
        return True

    @property
    def contact_force_threshold(self):
        return self.cfg.rewards.contact_force_threshold

    @property
    def tilt_description(self):
        return f"机身前向轴偏离世界向上超过 {self.cfg.termination.maximum_tilt_degrees:g}°"

    def tilt_failures(self, state):
        limit = round(math.cos(math.radians(self.cfg.termination.maximum_tilt_degrees)), 12)
        return upright_alignment(state) < limit

    def failures(self, state):
        return (state["height"] < self.cfg.termination.minimum_height) | self.tilt_failures(state)

    def reward_components(self, state, actions, failed, dt):
        return reward_components(state, actions, failed, dt, self.cfg.rewards)

    def metrics(self, state):
        support = state["foot_support_per_foot"]
        front_metrics = {}
        if self.cfg.rewards.front_foot_position_weight != 0:
            distances = state["front_foot_error_b"].norm(dim=-1)
            front_metrics = {
                "Task/front_foot_position_error_m": distances.mean(),
                "Task/front_foot_FL_error_m": distances[:, 0].mean(),
                "Task/front_foot_FR_error_m": distances[:, 1].mean(),
            }
        return {
            "Task/upright_alignment": upright_alignment(state).mean(),
            "Task/rear_support_count": rear_support_fraction(state, self.cfg.rewards).sum(-1).mean(),
            "Task/front_support_count": support[:, :2].sum(-1).mean(),
            "Task/rear_symmetry_error_rad": rear_symmetry_error(state).mean(),
            **front_metrics,
        }

    def load_playback_config(self, run_cfg):
        self.cfg.agent = deepcopy(run_cfg["agent_cfg"])
        self.cfg.commands = CommandCfg(**run_cfg["command_cfg"])
        reward_cfg = dict(run_cfg["reward_cfg"])
        # 已移除的奖励项；旧配置仍可加载，原始历史文件不重写。
        for name in ("upright_progress_weight", "hip_weight", "hip_sigma",
                     "thigh_rear_weight", "thigh_rear_target", "thigh_rear_sigma",
                     "hip_front_weight", "hip_front_sigma",
                     "hip_knee_lift_weight", "hip_knee_lift_sigma",
                     "rear_airborne_weight", "rear_both_airborne_weight",
                     "rear_knee_and_foot_support_weight",
                     "front_pose_weight", "front_pose_target_x", "front_pose_target_y",
                     "front_pose_target_z", "front_pose_sigma",
                     "rear_knee_clearance_weight", "rear_knee_clearance_sigma"):
            reward_cfg.pop(name, None)
        self.cfg.rewards = RewardCfg(**{
            # 无版本字段的旧记录沿用原来的角速度/接触叠加公式。
            "reward_schema_version": 1,
            "hip_contact_weight": {1: 60.0, 2: 120.0}.get(
                reward_cfg.get("reward_schema_version", 1), RewardCfg.hip_contact_weight),
            # 旧模型回放不新增前腿约束；续训由入口使用当前代码配置。
            "pitch_velocity_weight": 0.0, "vertical_velocity_weight": 0.0,
            "rear_knee_contact_weight": 0.0,
            "front_foot_position_weight": 0.0, **reward_cfg,
        })
        self.cfg.integral = IntegralCfg(**run_cfg["integral_cfg"])
        self.cfg.environment = EnvCfg(**run_cfg["env_cfg"])
        self.cfg.termination = TerminationCfg(**run_cfg["termination_cfg"])
