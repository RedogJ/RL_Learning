"""从近直立姿态开始，以两后脚支撑保持机身纵轴朝上。"""

from copy import deepcopy
from dataclasses import asdict
import math

from go2_learning.tasks.base import Task
from . import TASK_NAME, TASK_VERSION
from .config import UprightTaskCfg, EnvCfg, TerminationCfg
from .command_cfg import CommandCfg
from .integral_cfg import IntegralCfg
from .reward_cfg import RewardCfg
from .reset_cfg import ResetCfg
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
        return {
            "Task/upright_alignment": upright_alignment(state).mean(),
            "Task/rear_support_count": rear_support_fraction(state, self.cfg.rewards).sum(-1).mean(),
            "Task/front_support_count": support[:, :2].sum(-1).mean(),
            "Task/rear_symmetry_error_rad": rear_symmetry_error(state).mean(),
        }

    def reset_state(self, root_pose, joint_pos, joint_names):
        cfg = self.cfg.reset
        angle = math.radians(cfg.forward_elevation_deg)
        root_pose[:, 2] = cfg.root_height
        # 绕世界 Y 轴负向旋转，使机身 +X 抬向世界 +Z；Isaac Lab 3 使用 xyzw。
        root_pose[:, 3:] = root_pose.new_tensor((0., -math.sin(angle / 2), 0., math.cos(angle / 2)))
        hold_target = joint_pos.clone()
        for leg in ("FL", "FR", "RL", "RR"):
            prefix = "front" if leg.startswith("F") else "rear"
            offsets = getattr(cfg, f"{prefix}_hold_offset")
            if prefix == "front" and self.cfg.lock_front_legs:
                offsets = (0., 0., 0.)
            for part_id, part in enumerate(("hip", "thigh", "calf")):
                index = joint_names.index(f"{leg}_{part}_joint")
                joint_pos[:, index] = getattr(cfg, f"{prefix}_{part}")
                hold_target[:, index] = joint_pos[:, index] + offsets[part_id]
        return root_pose, joint_pos, hold_target

    def fixed_joint_positions(self):
        if not self.cfg.lock_front_legs:
            return {}
        return {f"{leg}_{part}_joint": getattr(self.cfg.reset, f"front_{part}")
                for leg in ("FL", "FR") for part in ("hip", "thigh", "calf")}

    def configure_scene(self, scene_cfg):
        if self.cfg.lock_front_legs:
            props = scene_cfg.robot.spawn.articulation_props
            props.solver_position_iteration_count = self.cfg.front_lock_position_iterations
            props.solver_velocity_iteration_count = self.cfg.front_lock_velocity_iterations

    def run_config(self, randomization_enabled):
        return {**super().run_config(randomization_enabled), "reset_cfg": asdict(self.cfg.reset),
                "front_legs_fixed": self.cfg.lock_front_legs,
                "front_lock_solver_iterations": [self.cfg.front_lock_position_iterations,
                                                 self.cfg.front_lock_velocity_iterations]}

    def load_playback_config(self, run_cfg):
        # 新模型恢复保存的重置姿态；旧记录没有此字段时使用当前近直立初始姿态。
        self.cfg.reset = ResetCfg(**run_cfg["reset_cfg"]) if run_cfg.get("reset_cfg") else ResetCfg()
        # 旧模型没有锁定记录时也采用当前任务的新行为；新模型恢复保存的开关。
        self.cfg.lock_front_legs = run_cfg.get("front_legs_fixed", True)
        (self.cfg.front_lock_position_iterations, self.cfg.front_lock_velocity_iterations) = run_cfg.get(
            "front_lock_solver_iterations", [64, 16])
        if "front_legs_fixed" not in run_cfg:
            # 旧记录的等待期预载针对 8/4 求解精度；首次启用固定前腿时使用重校准值。
            self.cfg.reset.rear_hold_offset = ResetCfg().rear_hold_offset
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
                     "front_foot_position_weight", "front_foot_target_left", "front_foot_target_right",
                     "vertical_velocity_weight", "action_rate_weight"):
            reward_cfg.pop(name, None)
        self.cfg.rewards = RewardCfg(**{
            # 无版本字段的旧记录沿用原来的角速度/接触叠加公式。
            "reward_schema_version": 1,
            "hip_contact_weight": {1: 60.0, 2: 120.0}.get(
                reward_cfg.get("reward_schema_version", 1), RewardCfg.hip_contact_weight),
            # 更早的模型没有这些奖励时按 0 加载；续训使用当前配置。
            "pitch_velocity_weight": 0.0,
            "rear_knee_contact_weight": 0.0,
            "rear_knee_clearance_weight": 0.0,
            # 旧动作变化权重不能按原数值解释为关节速度权重；旧模型缺少新项时关闭。
            "joint_velocity_weight": 0.0,
            **reward_cfg,
        })
        self.cfg.integral = IntegralCfg(**run_cfg["integral_cfg"])
        self.cfg.environment = EnvCfg(**run_cfg["env_cfg"])
        # 当前任务从近直立重置，播放旧权重也使用当前倾倒阈值，避免旧 120° 覆盖配置。
        self.cfg.termination = TerminationCfg(**{
            **run_cfg["termination_cfg"],
            "maximum_tilt_degrees": self.cfg.termination.maximum_tilt_degrees,
        })
