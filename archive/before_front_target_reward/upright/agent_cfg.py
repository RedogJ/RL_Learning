"""直立任务自己的 RSL-RL 网络、PPO 与采样配置。"""


def make_agent_cfg():
    return {
        "num_steps_per_env": 24,
        "save_interval": 50,
        "logger": "tensorboard",
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel", "hidden_dims": [128, 128], "activation": "tanh",
            "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "log"},
        },
        "critic": {
            "class_name": "MLPModel", "hidden_dims": [128, 128], "activation": "tanh",
            "obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO", "num_learning_epochs": 4, "num_mini_batches": 1,
            "learning_rate": 3e-4, "schedule": "fixed", "desired_kl": None,
            "gamma": 0.999, "lam": 0.95, "clip_param": 0.2, "entropy_coef": 0.01,
            "value_loss_coef": 0.5, "use_clipped_value_loss": False, "max_grad_norm": 0.5,
            "normalize_advantage_per_mini_batch": False,
        },
    }
