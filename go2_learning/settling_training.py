"""给标准 RSL-RL 5.5.1 PPO 添加每环境等待掩码，优化算法仍使用库实现。"""

from copy import copy

import torch


class SettlingPPO:
    """等待期间不推理、不训练；PPO 更新前只打包正式控制的样本。"""

    def __init__(self, algorithm, env):
        if algorithm.actor.is_recurrent or algorithm.critic.is_recurrent:
            raise ValueError("逐回合等待适配当前仅支持工程中的 MLP 策略")
        if algorithm.rnd or algorithm.symmetry:
            raise ValueError("逐回合等待适配尚未接入 RND 或对称增强")
        self.algorithm, self.env = algorithm, env
        st = algorithm.storage
        self.valid = torch.zeros((st.num_transitions_per_env, st.num_envs),
                                 dtype=torch.bool, device=algorithm.device)
        self.distribution_shapes = None

    def __getattr__(self, name):
        return getattr(self.algorithm, name)

    def act(self, obs):
        active = self.env.policy_active.to(self.algorithm.device)
        actions = self.env.standing_actions.to(self.algorithm.device).clone()
        transition = self.algorithm.transition
        if active.any():
            actions[active] = self.algorithm.act(obs[active])
            self.distribution_shapes = [p.shape[1:] for p in transition.distribution_params]
            # 库中的 act 记录的是活动子集，扩展回并行环境维度供固定大小采样缓冲区使用。
            for name in ("values", "actions_log_prob"):
                value = getattr(transition, name)
                full = value.new_zeros((self.env.num_envs, *value.shape[1:]))
                full[active] = value
                setattr(transition, name, full)
            params = []
            for value in transition.distribution_params:
                full = value.new_zeros((self.env.num_envs, *value.shape[1:]))
                full[active] = value
                params.append(full)
            transition.distribution_params = tuple(params)
        else:
            # main 在创建 Runner 前完成首回合等待，所以首次采样一定含活动环境。
            if self.distribution_shapes is None:
                raise RuntimeError("创建 Runner 前必须调用 env.finish_initial_settling()")
            transition.clear()
            transition.values = torch.zeros((self.env.num_envs, 1), device=self.algorithm.device)
            transition.actions_log_prob = torch.zeros(self.env.num_envs, device=self.algorithm.device)
            transition.distribution_params = tuple(
                torch.zeros((self.env.num_envs, *shape), device=self.algorithm.device)
                for shape in self.distribution_shapes
            )
        transition.actions = actions
        transition.observations = obs
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        step = self.algorithm.storage.step
        self.valid[step] = extras["policy_active"].to(self.algorithm.device)
        self.algorithm.process_env_step(obs, rewards, dones, extras)

    @torch.no_grad()
    def compute_returns(self, obs):
        """只沿正式控制样本计算 GAE；终止标志隔断不同回合。"""
        alg, st = self.algorithm, self.algorithm.storage
        active = self.env.policy_active.to(alg.device)
        next_values = torch.zeros((self.env.num_envs, 1), device=alg.device)
        if active.any():
            next_values[active] = alg.critic(obs[active]).detach()
        advantage = torch.zeros_like(next_values)
        for step in reversed(range(st.num_transitions_per_env)):
            valid = self.valid[step, :, None]
            nonterminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + alg.gamma * nonterminal * next_values - st.values[step]
            candidate = delta + alg.gamma * alg.lam * nonterminal * advantage
            advantage = torch.where(valid, candidate, advantage)
            st.returns[step] = torch.where(valid, advantage + st.values[step], 0.0)
            next_values = torch.where(valid, st.values[step], next_values)
        st.advantages = st.returns - st.values
        if self.valid.any() and not alg.normalize_advantage_per_mini_batch:
            values = st.advantages[self.valid]
            st.advantages[self.valid] = (values - values.mean()) / (values.std(unbiased=False) + 1e-8)

    def update(self):
        alg, original = self.algorithm, self.algorithm.storage
        count = int(self.valid.sum())
        if count < 2:
            # 单个样本不足以稳定估计标准化优势，跳过本次优化。
            original.clear()
            self.valid.zero_()
            return {"value": 0.0, "surrogate": 0.0, "entropy": 0.0}
        packed = copy(original)
        packed.num_envs, packed.num_transitions_per_env = 1, count
        packed.step = count
        packed.observations = original.observations[self.valid].unsqueeze(1)
        for name in ("actions", "rewards", "dones", "values", "actions_log_prob", "returns", "advantages"):
            setattr(packed, name, getattr(original, name)[self.valid].unsqueeze(1))
        packed.distribution_params = tuple(p[self.valid].unsqueeze(1) for p in original.distribution_params)
        batches = alg.num_mini_batches
        alg.num_mini_batches = min(batches, max(1, count // 2))
        alg.storage = packed
        try:
            # 原库负责策略损失、价值损失、熵、优化器和观测归一化。
            losses = alg.update()
        finally:
            alg.storage = original
            alg.num_mini_batches = batches
        original.clear()
        self.valid.zero_()
        return losses


class SettlingLogger:
    """保留原 Logger，只让 episode 长度累计正式控制步。"""

    def __init__(self, logger):
        self.logger = logger

    def __getattr__(self, name):
        return getattr(self.logger, name)

    def process_env_step(self, rewards, dones, extras, intrinsic_rewards=None):
        self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
        if self.logger.writer is not None:
            waiting = ~extras["policy_active"].to(self.logger.cur_episode_length.device)
            self.logger.cur_episode_length.sub_(waiting.to(self.logger.cur_episode_length.dtype))
