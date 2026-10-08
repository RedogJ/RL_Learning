import torch


@torch.no_grad()
def collect_rollout(env, policy, critic, observation, num_steps=128):
    storage = {
        "observations": [],
        "actions": [],
        "log_probs": [],
        "values": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "next_values": [],
    }

    for _ in range(num_steps):
        # 根据当前状态采样动作
        distribution = policy.get_distribution(observation)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions).sum(dim=-1)
        values = critic(observation)

        # 执行动作
        next_observation, rewards, terminated, truncated, _ = env.step(
            actions
        )

        # 必须在重置之前计算下一状态的价值
        next_values = critic(next_observation)

        transition = {
            "observations": observation,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "terminated": terminated,
            "truncated": truncated,
            "next_values": next_values,
        }

        for name, value in transition.items():
            storage[name].append(value.clone())

        observation = next_observation

        # 结束的环境重新开始，其他环境继续运行
        done_ids = (terminated | truncated).nonzero(
            as_tuple=False
        ).flatten()

        if done_ids.numel() > 0:
            observation = env.reset(done_ids)

    rollout = {
        name: torch.stack(items, dim=0)
        for name, items in storage.items()
    }

    return rollout, observation
