from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
SLOW_MOTION = PROJECT / 'data/side_run/motion.npy'
# User selected 3x playback as normal speed for this source clip.
DEFAULT_SPEED = 3.
MOTION = PROJECT / 'data/side_run_3x/motion.npy'


@dataclass
class TrainCfg:
    num_envs: int = 64
    cycles: int = 100
    collect_steps: int = 32
    buffer_steps: int = 320
    wm_batch: int = 128
    wm_updates: int = 12
    window: int = 8
    actor_batch: int = 128
    horizon: int = 16
    wm_width: int = 2048
    warmup: int = 20
    critic_warmup: int = 5
    critic_epochs: int = 5
    critic_minibatches: int = 4
    actor_lr: float = 5e-5
    gamma: float = .99
    lambda_: float = .95
    save_interval: int = 50
    seed: int = 42
    actor_hidden: tuple = (512, 256, 128)


def reward_config():
    from go2_mujoco_mimic.tasks.mimic.reward_cfg import RewardCfg
    return RewardCfg(body_position_weight=1., body_position_sigma=.12,
                     body_orientation_weight=.5, body_orientation_sigma=.5,
                     body_linear_velocity_weight=.2, body_linear_velocity_sigma=1.5,
                     body_angular_velocity_weight=.1,
                     root_position_weight=.5, root_position_sigma=.12,
                     root_orientation_weight=.5, root_orientation_sigma=.4,
                     joint_position_weight=2., joint_position_sigma=.4,
                     action_rate_weight=.02, torque_l2_weight=1e-5,
                     joint_acc_l2_weight=1e-7, failure_penalty=1.)
