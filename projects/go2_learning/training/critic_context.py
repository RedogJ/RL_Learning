"""续训时将旧 Critic 迁移为 policy 52 维 + 剩余时间 1 维。"""

import torch


def load_with_critic_context(runner, checkpoint, saved_cfg, current_cfg, device):
    old_groups = saved_cfg["obs_groups"]
    new_groups = current_cfg["obs_groups"]
    if old_groups["actor"] != new_groups["actor"]:
        raise ValueError("续训不允许修改 Actor 观测组")
    expected_inputs = {
        ("policy",): 52,
        ("policy", "critic_context"): 54,
        ("policy", "critic_time"): 53,
    }
    if new_groups["critic"] != ["policy", "critic_time"] or tuple(old_groups["critic"]) not in expected_inputs:
        raise ValueError("仅支持旧 52/54 维或当前 53 维 Critic 迁移到 policy + critic_time")
    old_inputs = expected_inputs[tuple(old_groups["critic"])]
    if old_inputs != 53 and (
        saved_cfg["critic"].get("obs_normalization", False)
        or current_cfg["critic"].get("obs_normalization", False)
    ):
        raise ValueError("带观测归一化的 Critic 暂不支持输入维度迁移")
    data = torch.load(checkpoint, weights_only=False, map_location=device)
    migration = None
    if old_inputs != 53:
        key = "mlp.0.weight"
        old = data["critic_state_dict"][key]
        target = runner.alg.critic.state_dict()[key]
        if old.shape[1] != old_inputs or target.shape != (old.shape[0], 53):
            raise ValueError(f"Critic 输入维度不符合 {old_inputs}→53 迁移：{old.shape} → {target.shape}")
        migrated = old.new_zeros(target.shape)
        migrated[:, :52] = old[:, :52]
        if old_inputs == 54:
            # 旧列顺序：[policy(52), cost(1), time(1)]；删除 cost，保留 time。
            migrated[:, 52] = old[:, 53]
            migration = "54_to_53_drop_cost_keep_time"
        else:
            # 更早的 52 维模型没有剩余时间；新增列置零。
            migration = "52_to_53_zero_padded_time"
        data["critic_state_dict"][key] = migrated
    loaded_iteration = runner.alg.load(data, {
        "actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False,
    }, strict=True)
    if loaded_iteration:
        runner.current_learning_iteration = data["iter"]
    return data.get("infos"), migration
