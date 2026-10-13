# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""交给标准 OnPolicyRunner 的网络、PPO 和采样配置。"""


def make_agent_cfg() -> dict:
    """每次返回独立字典；使用 RSL-RL 5.5.1 的 actor/critic 接口。"""
    return {
        # Runner：每轮采多少步、何时保存、网络读取哪组观测。
        "num_steps_per_env": 24,
        "save_interval": 50,
        "logger": "tensorboard",
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        # Actor 输出动作分布；Critic 输出状态价值。
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "log",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
            "obs_normalization": False,
        },
        # PPO：直接交给库中的标准算法。
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 4,
            # 每个 epoch 使用整批样本。
            "num_mini_batches": 1,
            "learning_rate": 3e-4,
            "schedule": "fixed",
            "desired_kl": None,
            "gamma": 0.99,
            "lam": 0.95,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "value_loss_coef": 0.5,
            "use_clipped_value_loss": False,
            "max_grad_norm": 0.5,
            "normalize_advantage_per_mini_batch": False,
        },
    }
