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
        task_cfg = getattr(getattr(env, "task", None), "cfg", None)
        self.finite_horizon = getattr(task_cfg, "finite_horizon", False)
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
        # 一次生成整数索引，避免每个布尔切片触发动态形状同步。
        ids = active.nonzero(as_tuple=False).flatten()
        cache_selection = getattr(self.env, "cache_policy_selection", None)
        if cache_selection is not None and active.device == torch.device(self.env.device):
            cache_selection(active, ids)
        if ids.numel():
            actions[ids] = self.algorithm.act(obs[ids])
            self.distribution_shapes = [p.shape[1:] for p in transition.distribution_params]
            # 库中的 act 记录的是活动子集，扩展回并行环境维度供固定大小采样缓冲区使用。
            for name in ("values", "actions_log_prob"):
                value = getattr(transition, name)
                full = value.new_zeros((self.env.num_envs, *value.shape[1:]))
                full[ids] = value
                setattr(transition, name, full)
            params = []
            for value in transition.distribution_params:
                full = value.new_zeros((self.env.num_envs, *value.shape[1:]))
                full[ids] = value
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
        if self.finite_horizon:
            # 日志仍保留超时标志；只关闭算法对真正终点的 bootstrap。
            extras = {**extras, "time_outs": torch.zeros_like(dones, dtype=torch.bool)}
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
    """GPU 累计完整回合，按轮批量回传；等待步不计入回合长度。"""

    def __init__(self, logger):
        self.logger = logger
        self._pending_episodes = []
        self._has_intrinsic = None

    def __getattr__(self, name):
        return getattr(self.logger, name)

    def process_env_step(self, rewards, dones, extras, intrinsic_rewards=None):
        logger = self.logger
        if logger.writer is None:
            return
        if "episode" in extras:
            logger.ep_extras.append(extras["episode"])
        elif "log" in extras:
            logger.ep_extras.append(extras["log"])
        self._has_intrinsic = intrinsic_rewards is not None
        if self._has_intrinsic:
            logger.cur_ereward_sum.add_(rewards)
            logger.cur_ireward_sum.add_(intrinsic_rewards)
            logger.cur_reward_sum.add_(rewards + intrinsic_rewards)
        else:
            logger.cur_reward_sum.add_(rewards)
        logger.cur_episode_length.add_(extras["policy_active"].to(logger.cur_episode_length.dtype))
        ended = dones.reshape(-1).bool()
        columns = [logger.cur_reward_sum, logger.cur_episode_length, ended]
        if self._has_intrinsic:
            columns.extend((logger.cur_ereward_sum, logger.cur_ireward_sum))
        # stack 拍下重置前的值；每步只使用固定形状张量，不查询 GPU 上的结束数量。
        self._pending_episodes.append(torch.stack(columns, dim=-1))
        logger.cur_reward_sum.masked_fill_(ended, 0)
        logger.cur_episode_length.masked_fill_(ended, 0)
        if self._has_intrinsic:
            logger.cur_ereward_sum.masked_fill_(ended, 0)
            logger.cur_ireward_sum.masked_fill_(ended, 0)

    def flush_episodes(self):
        """按时间、环境编号顺序更新原库的最近回合队列，每轮仅回传一次。"""
        if not self._pending_episodes:
            return
        rows = torch.stack(self._pending_episodes).flatten(0, 1)
        completed = rows[rows[:, 2].bool()].cpu().tolist()
        self._pending_episodes.clear()
        self.logger.rewbuffer.extend(row[0] for row in completed)
        self.logger.lenbuffer.extend(row[1] for row in completed)
        if self._has_intrinsic:
            self.logger.erewbuffer.extend(row[3] for row in completed)
            self.logger.irewbuffer.extend(row[4] for row in completed)

    def log(self, **kwargs):
        self.flush_episodes()
        return self.logger.log(**kwargs)
