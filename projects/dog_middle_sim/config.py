"""Middle SuperMimic 配置；网络、仿真、训练分别管理。"""
from dataclasses import dataclass, field
from .tasks.standing.reward_cfg import RewardCfg


@dataclass
class EnvCfg:
    num_envs: int = 64
    physics_hz: int = 500
    control_hz: int = 50
    episode_seconds: float = 10.0
    kp: float = 40.0
    kd: float = 1.0
    action_scale: float = 0.25  # rad；目标角 = 参考默认站姿角 + scale * tanh动作
    reset_joint_noise: float = 0.02  # rad，逐环境独立采样
    reset_tilt_noise: float = 0.04  # rad，初始 roll/pitch
    history: int = 10
    future: int = 10
    height_threshold: float = 0.20
    tilt_threshold: float = 0.8  # 当前/参考重力投影 z 分量之差
    termination_softness: float = 0.05  # 阈值归一化后 sigmoid 的尺度
    cuda_graph: bool = True


@dataclass
class ModelCfg:
    wm_width: int = 2048
    policy_hidden: tuple[int, ...] = (512, 256, 128)
    critic_hidden: tuple[int, ...] = (256, 128)
    latent: int = 32
    root_pos_error: float = 0.01  # WM 损失物理尺度，非奖励 sigma
    root_rot_error: float = 0.02
    joint_pos_error: float = 0.1
    joint_vel_error: float = 3.14


@dataclass
class TrainCfg:
    cycles: int = 1000
    collect_steps: int = 32
    buffer_steps: int = 320
    wm_batch: int = 128
    wm_window: int = 8  # 每条窗口预测的 transition 数
    wm_updates: int = 10
    wm_lr: float = 2e-4
    wm_warmup: int = 10
    critic_warmup: int = 5
    actor_batch: int = 128
    actor_rounds: int = 1  # 每轮重新采起点、重新想象、更新一次 Actor
    horizon: int = 16
    actor_lr: float = 5e-5
    critic_lr: float = 2e-4
    critic_epochs: int = 5
    critic_minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    exploration: float = 0.1
    save_interval: int = 50
    seed: int = 42


@dataclass
class Config:
    env: EnvCfg = field(default_factory=EnvCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)


def from_dict(value):
    return Config(EnvCfg(**value['env']), ModelCfg(**value['model']),
                  TrainCfg(**value['train']), RewardCfg(**value['reward']))
