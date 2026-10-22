"""任务接口：定义目标与规则，仿真推进和 RSL-RL 采样由通用环境负责。"""

from dataclasses import asdict
import math


class Task:
    name: str
    version: str

    def reward_components(self, state, actions, failed, dt):
        raise NotImplementedError

    def failures(self, state):
        raise NotImplementedError

    def tilt_failures(self, state):
        """播放可以单独使用倾斜条件，训练仍使用任务的完整失败规则。"""
        limit = round(-math.cos(math.radians(self.cfg.termination.maximum_tilt_degrees)), 12)
        return state["gravity"][:, 2] > limit

    @property
    def tilt_description(self):
        return f"倾斜超过 {self.cfg.termination.maximum_tilt_degrees:g}°"

    def metrics(self, state):
        """任务可追加自己的 GPU 标量日志，通用环境不解释任务目标。"""
        return {}

    def reset_state(self, root_pose, joint_pos, joint_names):
        """返回根位姿、初始关节角、等待期 PD 目标；位置相对环境原点，四元数 xyzw。"""
        return root_pose, joint_pos, joint_pos.clone()

    @property
    def target_height(self):
        raise NotImplementedError

    @property
    def needs_contact_sensor(self):
        raise NotImplementedError

    @property
    def contact_force_threshold(self):
        raise NotImplementedError

    def load_playback_config(self, run_cfg):
        raise NotImplementedError

    def run_config(self, randomization_enabled):
        return {
            "task_name": self.name,
            "task_version": self.version,
            "env_cfg": asdict(self.cfg.environment),
            "termination_cfg": asdict(self.cfg.termination),
            "agent_cfg": self.cfg.agent,
            "command_cfg": asdict(self.cfg.commands),
            "reward_cfg": asdict(self.cfg.rewards),
            "integral_cfg": asdict(self.cfg.integral),
            "randomization_cfg": asdict(self.cfg.randomization) if randomization_enabled else None,
        }
