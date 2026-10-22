import torch


@torch.no_grad()
def compute_gae(rollout, gamma=0.99, gae_lambda=0.95):
    rewards = rollout["rewards"]
    values = rollout["values"]
    next_values = rollout["next_values"]
    terminated = rollout["terminated"]
    truncated = rollout["truncated"]

    # 形状：[采样步数, 环境数量]
    advantages = torch.zeros_like(rewards)

    # 每个环境分别向前累计
    last_advantage = torch.zeros_like(rewards[0])

    for t in reversed(range(rewards.shape[0])):
        # 真正失败后，不再估计后续收益
        bootstrap_mask = (~terminated[t]).float()

        delta = (
            rewards[t]
            + gamma * bootstrap_mask * next_values[t]
            - values[t]
        )

        # 重置后属于新一轮，优势不能跨轮累计
        continuation_mask = (
            ~(terminated[t] | truncated[t])
        ).float()

        last_advantage = (
            delta
            + gamma * gae_lambda
            * continuation_mask
            * last_advantage
        )

        advantages[t] = last_advantage

    # 价值网络的训练目标
    returns = advantages + values

    return advantages, returns
class PPOTrainer:
    def __init__(self, policy, critic, learning_rate=3e-4):
        self.policy = policy
        self.critic = critic

        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=learning_rate,
        )

        self.num_epochs = 4
        self.clip_ratio = 0.2
        self.entropy_coef = 0.005

    def update(self, rollout, advantages, returns):
        # 合并时间和环境维度：[T, N, ...] → [T*N, ...]
        observations = rollout["observations"].flatten(0, 1)
        actions = rollout["actions"].flatten(0, 1)
        old_log_probs = rollout["log_probs"].reshape(-1).detach()

        advantages = advantages.reshape(-1).detach()
        returns = returns.reshape(-1).detach()

        # 只标准化优势，价值训练目标保持原来的尺度
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        for _ in range(self.num_epochs):
            # 对采样时保存的状态和动作，重新计算 log_prob
            distribution = self.policy.get_distribution(observations)
            new_log_probs = distribution.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO 的裁剪目标
            objective = ratio * advantages
            clipped_objective = torch.clamp(
                ratio,
                1.0 - self.clip_ratio,
                1.0 + self.clip_ratio,
            ) * advantages

            entropy = distribution.entropy().sum(dim=-1).mean()

            policy_loss = (
                -torch.minimum(objective, clipped_objective).mean()
                - self.entropy_coef * entropy
            )

            # 更新策略网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                max_norm=0.5,
            )
            self.policy_optimizer.step()

            # 更新价值网络
            predicted_values = self.critic(observations)
            value_loss = 0.5 * (
                predicted_values - returns
            ).square().mean()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                max_norm=0.5,
            )
            self.critic_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
