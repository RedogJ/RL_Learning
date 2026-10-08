# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


@torch.no_grad()
def collect_rollout(env, policy, critic, history, num_steps=128, gamma=0.99):
    storage = {
        "observations": [],
        "actions": [],
        "log_probs": [],
        "values": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "next_values": [],
        "action_masks": [],
        "durations": [],
    }

    for _ in range(num_steps):
        # 根据当前状态采样动作
        distribution = policy.get_distribution(history)
        actions = distribution.sample()
        values = critic(history)

        # 执行动作
        _, rewards, terminated, truncated, info = env.step_sequence(actions, gamma=gamma)
        next_history = env.get_history()
        # 未执行的序列尾部不参与 PPO 的概率比。
        log_probs = (distribution.log_prob(actions).sum(dim=-1) * info["action_masks"]).sum(dim=-1)

        # 必须在重置之前计算下一状态的价值
        next_values = critic(next_history)

        transition = {
            "observations": history,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "terminated": terminated,
            "truncated": truncated,
            "next_values": next_values,
            "action_masks": info["action_masks"],
            "durations": info["durations"],
        }

        for name, value in transition.items():
            storage[name].append(value.clone())

        history = next_history

        # 结束的环境重新开始，其他环境继续运行
        done_ids = (terminated | truncated).nonzero(
            as_tuple=False
        ).flatten()

        if done_ids.numel() > 0:
            env.reset(done_ids)
            history = env.get_history()

    rollout = {
        name: torch.stack(items, dim=0)
        for name, items in storage.items()
    }

    return rollout, history
