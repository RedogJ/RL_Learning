import json
from pathlib import Path
import os
import torch
import warp as wp

if os.environ.get('WARP_CACHE_PATH'):
    wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']

from go2_mujoco_mimic.warp_env import EnvCfg, WarpGo2Env, warp_stream_ordered
from .anchor import canonical_goal, world_goal
from .config import reward_config
from .task import AnchorTask

ACTION_MAPPING = 'standing_center_full_range_v1'


def make_config(num_envs=64, seconds=5.5):
    cfg = EnvCfg(num_envs=num_envs, control_hz=50, physics_hz=600,
                 episode_seconds=seconds, clamp_joint_targets=True)
    cfg.task.reward = reward_config()
    cfg.task.reference_loop = False
    cfg.task.reference_cycles = 1
    cfg.task.random_reference_start = True
    cfg.task.heading_observation = False
    cfg.task.longitudinal_observation = False
    cfg.task.max_height_error = .18
    cfg.task.max_tilt_error = .6
    cfg.task.terminate_on_nonfoot_contact = True
    return cfg


class AnchorEnv(WarpGo2Env):
    def __init__(self, cfg, motion):
        self.world_tracking = json.loads(Path(motion).with_name('clip.json').read_text()).get('tracking') == 'world_anchor'
        super().__init__(cfg, motion)
        if abs(self.ref.fps-cfg.control_hz) > 1e-5:
            raise ValueError('Prepare the reference at 50 Hz to match the control clock')
        self.task = AnchorTask(cfg.task, self.world_tracking)
        # Fixed full-range mapping keeps the physical WM action semantics
        # independent of reference phase. No hidden reference-dependent PD offset.
        self.action_scales = torch.maximum(self.joint_limits[:, 1]-self.default_q,
                                           self.default_q-self.joint_limits[:, 0])
        self.reset(torch.ones(cfg.num_envs, device=self.device, dtype=torch.bool))

    def observation(self):
        return policy_input(self, self.qpos, self.history, self.frame, self.frame_limit)


def goal(env, frame, limit, root):
    offsets = torch.arange(1, env.cfg.task.future_length+1, device=env.device)
    indices = torch.minimum(frame[:, None] + offsets, limit[:, None])
    ref = env.ref.sample(indices)
    targets = world_goal(ref, root) if env.world_tracking else canonical_goal(ref)
    return targets.flatten(1)


def policy_input(env, state, history, frame, limit):
    return torch.cat((history.flatten(1), goal(env, frame, limit, state)), -1)
