from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg import (
    CartpolePPORunnerCfg,
)


@configclass
class MyCartpolePPORunnerCfg(CartpolePPORunnerCfg):
    experiment_name = "my_cartpole"
    max_iterations = 300

    # 策略网络：根据状态选择动作
    actor = RslRlMLPModelCfg(
        hidden_dims=[32, 32],
        activation="tanh",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.5,
        ),
    )

    # 价值网络：评估当前状态未来能获得多少奖励
    critic = RslRlMLPModelCfg(
        hidden_dims=[32, 32],
        activation="tanh",
        obs_normalization=True,
    )