"""仅本实验的奖励覆盖；默认使用所选 upright 专家保存的奖励快照。

例如：REWARD_OVERRIDES = {'action_rate_weight': 8000.0}
修改这里只影响之后新启动的 SuperMimic 实验；续训沿用其检查点保存的奖励。
"""

REWARD_OVERRIDES = {}
