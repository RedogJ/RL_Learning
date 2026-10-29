"""按训练阶段校准正收益的 sigma；本模块不更新策略，也不向 PPO 回放样本。"""

from copy import deepcopy
import json
import math

import torch

from go2_learning.tasks.upright.tracking_errors import ERROR_SPECS


class ErrorReplayBuffer:
    """GPU 环形缓冲；只在校准和保存检查点时拷回 CPU。"""

    def __init__(self, names, cfg, device, binding):
        self.names = tuple(names)
        self.capacity = cfg.capacity
        self.samples_per_step = cfg.samples_per_step
        self.binding = deepcopy(binding)
        self.sigmas = {}
        self.data = torch.empty((self.capacity, len(names)), device=device, dtype=torch.float32)
        self.position = self.size = self.steps = 0

    @torch.no_grad()
    def observe(self, squared, valid):
        if not self.names:
            return
        n = squared[self.names[0]].shape[0]
        if n == 0:
            return
        count = min(n, self.samples_per_step)
        # 等距取样并逐步移动起点，避免总取活动列表开头的环境；不生成 CPU 标量。
        ids = (torch.arange(count, device=self.data.device) * n // count + self.steps) % n
        rows = torch.stack([squared[name].index_select(0, ids) for name in self.names], -1)
        rows = rows.detach().clamp_min(0.0).sqrt().to(dtype=self.data.dtype)
        for column, name in enumerate(self.names):
            if name in valid:
                rows[:, column].masked_fill_(~valid[name].index_select(0, ids), float("nan"))
        first = min(count, self.capacity - self.position)
        self.data[self.position:self.position + first].copy_(rows[:first])
        if first < count:
            self.data[:count - first].copy_(rows[first:])
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.steps += 1

    def clear(self):
        self.position = self.size = self.steps = 0

    def state_dict(self):
        if self.size < self.capacity:
            rows = self.data[:self.size]
        else:
            rows = torch.cat((self.data[self.position:], self.data[:self.position]))
        return {"schema_version": 1, "names": list(self.names), "binding": deepcopy(self.binding),
                "sigmas": dict(self.sigmas), "errors": rows.detach().cpu().clone(),
                "control_steps_observed": self.steps}


def measurement_binding(task, randomization_enabled):
    """目标/接触阈值/采样条件改变时，旧误差不能冒充当前目标下的误差。"""
    cfg = task.cfg.rewards
    run = task.run_config(randomization_enabled)
    binding = {
        "measurement_version": 1,
        "task_name": task.name, "task_version": task.version,
        "targets": {key: getattr(cfg, key) for key in (
            "target_height", "front_target_hip", "front_target_thigh", "front_target_calf",
            "rear_posture_thigh", "rear_posture_calf", "contact_force_threshold")},
        "sampling_context": {key: run.get(key) for key in (
            "env_cfg", "termination_cfg", "reset_cfg", "randomization_cfg", "front_legs_fixed")},
    }
    # tuple/list 的序列化差异不能误判为物理条件改变。
    return json.loads(json.dumps(binding, sort_keys=True))


def compatible_snapshot(snapshot, names, binding):
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        return False, "检查点没有误差缓冲"
    if snapshot.get("binding") != binding or snapshot.get("names") != list(names):
        return False, "误差定义、启用项、目标或采样条件已改变"
    rows = snapshot.get("errors")
    if not isinstance(rows, torch.Tensor) or rows.ndim != 2 or rows.shape[1] != len(names) or not rows.shape[0]:
        return False, "检查点误差缓冲为空或形状不匹配"
    return True, None


def estimate_sigmas(rows, names, reward_cfg, sigma_cfg, previous_sigmas=None):
    """CPU 分位数校准。只返回结果，不在采样/梯度更新期间更改 sigma。"""
    sigmas, statistics = {}, {}
    previous_sigmas = previous_sigmas or {}
    divisor = math.sqrt(-math.log(sigma_cfg.reward_at_quantile))
    for column, name in enumerate(names):
        sigma_field, _, unit = ERROR_SPECS[name]
        initial = getattr(reward_cfg, sigma_field)
        values = rows[:, column]
        values = values[torch.isfinite(values) & (values >= 0)]
        lo, hi = sigma_cfg.bounds[name]
        previous = previous_sigmas.get(sigma_field)
        if previous is not None and (not math.isfinite(previous) or previous <= 0):
            previous = None
        if values.numel() < sigma_cfg.minimum_samples:
            # 没有新证据时不收紧；优先沿用匹配检查点的固定 sigma。
            used = min(hi, max(lo, previous if previous is not None else initial))
            statistics[name] = {"samples": values.numel(), "unit": unit, "sigma": used,
                                "status": "insufficient_samples_keep_scale"}
        else:
            quantile_error = float(torch.quantile(values.float(), sigma_cfg.quantile))
            proposed = quantile_error / divisor
            used = min(hi, max(lo, proposed))
            if previous is not None:
                used = min(previous * sigma_cfg.change_upper, max(previous * sigma_cfg.change_lower, used))
                used = min(hi, max(lo, used))
            statistics[name] = {
                "samples": values.numel(), "unit": unit, "quantile_error": quantile_error,
                "proposed_sigma": proposed, "previous_sigma": previous, "sigma": used,
                "bounds": [lo, hi], "status": "calibrated",
            }
        sigmas[sigma_field] = used
    return sigmas, statistics


@torch.inference_mode()
def collect_initial_errors(runner, env, buffer, control_steps):
    """用当前训练分布采样（含高斯探索），不写 PPO storage，不做反向传播。"""
    actor = runner.alg.actor
    if actor.is_recurrent:
        raise ValueError("sigma 校准当前只支持无递归 Actor")
    was_training = actor.training
    actor.eval()
    try:
        obs = env.get_observations()
        for _ in range(control_steps):
            actions = env.standing_actions.clone()
            ids = env.policy_active.nonzero(as_tuple=False).flatten()
            if ids.numel():
                actions[ids] = actor(obs[ids], stochastic_output=True)
            obs, _, _, _ = env.step(actions)
    finally:
        actor.train(was_training)
        # 校准不留下回合长度、速度或上一动作；正式训练从新的等待期开始。
        env.reset()
        env.finish_initial_settling()


def prepare_reward_stage(task, runner, env, checkpoint_infos):
    """每次 train 调用一次；之后只收集误差，绝不在该次 learn 内改 sigma。"""
    cfg = task.cfg.sigma_calibration
    rewards = task.cfg.rewards
    names = [name for name, (_, weight, _) in ERROR_SPECS.items() if getattr(rewards, weight) != 0]
    initial_sigmas = {ERROR_SPECS[name][0]: getattr(rewards, ERROR_SPECS[name][0]) for name in names}
    report = {"enabled": cfg.enabled, "fixed_during_training": True,
              "initial_sigmas": initial_sigmas, "sigmas": dict(initial_sigmas)}
    if not cfg.enabled or not names:
        task.reward_error_buffer = runner.reward_error_buffer = None
        report["source"] = "manual_fixed" if names else "no_enabled_error_terms"
        print("sigma 自动校准未启用；本次使用 reward_cfg.py 中的固定值")
        return report
    binding = measurement_binding(task, env.randomization is not None)
    buffer = ErrorReplayBuffer(names, cfg, env.device, binding)
    task.reward_error_buffer = runner.reward_error_buffer = buffer
    snapshot = (checkpoint_infos or {}).get("reward_error_buffer")
    matches, reason = compatible_snapshot(snapshot, names, binding)
    if matches:
        rows = snapshot["errors"].detach().cpu()
        previous = snapshot.get("sigmas", {})
        source = "selected_checkpoint_buffer"
        print(f"sigma 校准：读取所选检查点的 {len(rows)} 行近期误差")
    else:
        print(f"sigma 校准：{reason}；先采样 {cfg.bootstrap_steps} 个控制步，期间不更新网络")
        collect_initial_errors(runner, env, buffer, cfg.bootstrap_steps)
        rows = buffer.state_dict()["errors"]
        previous = None
        source = "initial_policy_rollout"
    sigmas, statistics = estimate_sigmas(rows, names, rewards, cfg, previous)
    for name, value in sigmas.items():
        setattr(rewards, name, value)
    # 下一次开训只使用当前阶段近期的数据，不混入更早阶段或校准启动样本。
    buffer.clear()
    buffer.sigmas = dict(sigmas)
    report.update(source=source, fallback_reason=reason, buffer_rows=len(rows), sigmas=sigmas,
                  quantile=cfg.quantile, reward_at_quantile=cfg.reward_at_quantile,
                  statistics=statistics)
    for name, details in statistics.items():
        field = ERROR_SPECS[name][0]
        print(f"  {field}: {initial_sigmas[field]:.5g} → {details['sigma']:.5g} "
              f"{details['unit']}（有效样本 {details['samples']}，{details['status']}）")
    print("sigma 已冻结；本次训练中仅收集误差，下一次 train 才会重新估计")
    return report
