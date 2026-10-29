"""直立任务自己的 RSL-RL 网络、PPO 与采样配置。"""


def make_agent_cfg():
    return {
        "num_steps_per_env": 24,
        "check_for_nan": False,  # 稳定训练关闭逐步同步检查；数值排错时设为 True。
        "diagnostics_interval": 24,  # 纯诊断指标按控制步抽样，奖励和失败统计仍逐步计算。
        "save_interval": 50,
        "logger": "tensorboard",
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "critic_time"]},
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
            "class_name": "PPO", "num_learning_epochs": 2, "num_mini_batches": 1,
            "learning_rate": 4e-4, "schedule": "fixed", "desired_kl": None,
            "gamma": 0.9999, "lam": 0.95, "clip_param": 0.4, "entropy_coef": 0.01,
            "value_loss_coef": 0.5, "use_clipped_value_loss": False, "max_grad_norm": 0.5,
            "normalize_advantage_per_mini_batch": False,
        },
    }
