"""从近直立姿态开始，学习用两后脚支撑直立。"""

from copy import deepcopy
from dataclasses import asdict
import math
import torch

from go2_learning.tasks.base import Task
from . import TASK_NAME, TASK_VERSION
from .config import UprightTaskCfg, EnvCfg, TerminationCfg
from .command_cfg import CommandCfg
from .integral_cfg import IntegralCfg
from .reward_compat import CURRENT_REWARD_SCHEMA, load_reward_config
from .reset_cfg import ResetCfg
from .tracking_errors import continuous_errors
from .rewards import (
    rear_posture_error_squared, rear_support_fraction, rear_symmetry_error,
    reward_components, upright_alignment,
)


class UprightTask(Task):
    name = TASK_NAME
    version = TASK_VERSION

    def __init__(self, cfg=None):
        self.cfg = cfg or UprightTaskCfg()
        self.reward_error_buffer = None  # 只由 train 入口接入；播放不维护误差缓冲。
        self._tracking_errors = None

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
        return self.tilt_failures(state) | self.contact_failures(state)

    def contact_failures(self, state):
        if not self.cfg.termination.reset_on_nonfoot_contact:
            return torch.zeros_like(state["height"], dtype=torch.bool)
        # 使用原始接触数；奖励中的膝部支撑扣除规则不影响终止判定。
        return state["undesired_contact_count"] > 0

    def reward_components(self, state, actions, failed, dt):
        self._tracking_errors = (continuous_errors(state, self.cfg.rewards)
                                 if getattr(self.cfg.rewards, "reward_schema_version", CURRENT_REWARD_SCHEMA) >= 8
                                 else None)
        return reward_components(state, actions, failed, dt, self.cfg.rewards,
                                 self.reward_error_buffer, self._tracking_errors)

    def prepare_training_rewards(self, runner, env, checkpoint_infos):
        from go2_learning.training.reward_calibration import prepare_reward_stage

        return prepare_reward_stage(self, runner, env, checkpoint_infos)

    def critic_observations(self, env):
        remaining = (env.max_episode_length - env.episode_length_buf).clamp_min(0)
        time = (remaining / env.max_episode_length).unsqueeze(-1)
        groups = self.cfg.agent["obs_groups"]["critic"]
        if "critic_time" in groups:
            return {"critic_time": time}
        if "critic_context" in groups:
            # 仅供旧 54 维模型播放时匹配形状；废弃列恒为 0，不维护低通状态。
            return {"critic_context": torch.cat((torch.zeros_like(time), time), dim=-1)}
        return {}

    def metrics(self, state):
        support = state["foot_support_per_foot"]
        raw_metrics = {}
        if self._tracking_errors is not None:
            squared, valid = self._tracking_errors
            for name, value in squared.items():
                error = value.sqrt()
                if name in valid:
                    mask = valid[name]
                    raw_metrics[f"Error/{name}"] = torch.where(mask, error, 0.0).sum() / mask.sum().clamp_min(1)
                    raw_metrics[f"Error/{name}_valid_fraction"] = mask.float().mean()
                else:
                    raw_metrics[f"Error/{name}"] = error.mean()
        return {
            **raw_metrics,  # 固定物理单位的原始误差，不受 sigma/weight 改变影响。
            "Task/upright_alignment": upright_alignment(state).mean(),
            "Task/rear_support_count": rear_support_fraction(state, self.cfg.rewards).sum(-1).mean(),
            "Task/front_support_count": support[:, :2].sum(-1).mean(),
            "Task/rear_symmetry_error_rad": rear_symmetry_error(state).mean(),
            **({"Task/rear_posture_rms_rad": rear_posture_error_squared(state, self.cfg.rewards).sqrt().mean()}
               if self.cfg.rewards.rear_posture_weight != 0.0 else {}),
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

    def make_reset_orientation_sampler(self, env, seed):
        cone = self.cfg.reset.orientation_cone_degrees
        if cone == 0:
            return None
        if abs(self.cfg.reset.forward_elevation_deg - 90) + cone >= self.cfg.termination.maximum_tilt_degrees:
            raise ValueError("初态倾斜分布必须在 upright 失败角阈值以内")
        from go2_learning.control.reset_orientation import ResetOrientationSampler

        return ResetOrientationSampler(env, cone, seed)

    def fixed_joint_positions(self):
        if not self.cfg.lock_front_legs:
            return {}
        return {f"{leg}_{part}_joint": getattr(self.cfg.reset, f"front_{part}")
                for leg in ("FL", "FR") for part in ("hip", "thigh", "calf")}

    def configure_scene(self, scene_cfg):
        props = scene_cfg.robot.spawn.articulation_props
        props.solver_position_iteration_count = self.cfg.front_lock_position_iterations
        props.solver_velocity_iteration_count = self.cfg.front_lock_velocity_iterations

    def run_config(self, randomization_enabled):
        config = super().run_config(randomization_enabled)
        # 公式编号是运行记录元数据，不是训练时需要调节的奖励参数。
        config["reward_cfg"]["reward_schema_version"] = getattr(
            self.cfg.rewards, "reward_schema_version", CURRENT_REWARD_SCHEMA)
        return {**config, "reset_cfg": asdict(self.cfg.reset),
                "sigma_calibration_cfg": asdict(self.cfg.sigma_calibration),
                "finite_horizon": self.cfg.finite_horizon,
                "front_legs_fixed": self.cfg.lock_front_legs,
                "front_lock_solver_iterations": [self.cfg.front_lock_position_iterations,
                                                 self.cfg.front_lock_velocity_iterations]}

    def load_playback_config(self, run_cfg):
        # 播放旧模型时也使用当前任务的初始姿态。
        self.cfg.reset = ResetCfg()
        # 当前播放也交还前腿控制权，不恢复旧模型的机械锁定。
        self.cfg.lock_front_legs = False
        # 当前任务使用常规 8/4 求解精度。
        self.cfg.front_lock_position_iterations = 8
        self.cfg.front_lock_velocity_iterations = 4
        self.cfg.agent = deepcopy(run_cfg["agent_cfg"])
        self.cfg.finite_horizon = run_cfg.get("finite_horizon", False)
        self.cfg.commands = CommandCfg(**run_cfg["command_cfg"])
        self.cfg.rewards = load_reward_config(run_cfg["reward_cfg"])
        # 支持之前保存在顶层的系数；更早没有滤波字段的模型保持关闭。
        self.cfg.rewards.action_filter_alpha = run_cfg["reward_cfg"].get(
            "action_filter_alpha", run_cfg.get("action_filter_alpha", 0.0))
        self.cfg.integral = IntegralCfg(**run_cfg["integral_cfg"])
        self.cfg.environment = EnvCfg(**run_cfg["env_cfg"])
        # 播放旧权重使用当前倾倒阈值和接触终止设置。
        self.cfg.termination = TerminationCfg(**{
            **run_cfg["termination_cfg"],
            "maximum_tilt_degrees": self.cfg.termination.maximum_tilt_degrees,
            "reset_on_nonfoot_contact": self.cfg.termination.reset_on_nonfoot_contact,
        })
