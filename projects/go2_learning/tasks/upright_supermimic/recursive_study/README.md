# 单步递推世界模型对照

这个独立实验使用相同 replay、归一化、8 步自回归损失、优化器和数据抽样，从头训练原 MLP 与 DeepONet。两个模型每个预测步只使用当前模型状态、该步执行动作及 episode context；预测结果递推给下一步，不用未来真实状态。DeepONet 始终查询第 1 步，其他动作槽被遮罩；此设置下时间主干输入恒定，不能体现不同未来时间查询。

每个 cycle 有 10 组，每组抽取两个 2048 窗口的 batch 并分别依次更新一次；默认训练 250 cycles，即每模型 5000 次更新。训练和验证使用不同 seed 采集的 Isaac 数据，32 步测试不跨 episode。输出独立保存在 `logs/upright_supermimic/recursive_study/`，不启动在线 Actor 训练、不修改其他任务或日志。

```bash
bash scripts/run-isaac-project.sh projects/go2_learning \
  go2_learning.tasks.upright_supermimic.recursive_study.train --viz none \
  --checkpoint projects/go2_learning/logs/upright_supermimic/deeponet_study/datasets/20261027_152321_065444/dataset.pt \
  --validation-checkpoint /workspace/go2_learning/logs/upright_supermimic/deeponet_study/datasets/20261027_152429_852598/dataset.pt \
  --cycles 250 --groups-per-cycle 10 --batch 2048
```

`--resume` 可恢复本实验检查点中的模型、优化器和 RNG，`--cycles` 表示追加轮数；训练/验证数据路径及数据采样参数必须保持一致。每 25 轮保存 `latest.pt`，并保留第 100、200 轮及结束轮里程碑。

`evaluate.py` 在不同采集 seed 的全部有效 32 步测试窗口上，使用完全相同的记录动作比较递推 MLP、递推 DeepONet 和已有的单步 MLP、直接查询 DeepONet。窗口连续且允许重叠，不是独立样本。此前其他实验看过同一测试集，所以是探索性比较，不是严格盲测。两种模型参数量不同，输入动作的使用方式亦不同，不能仅归因于网络类型。整个实验是离线预测评估，未证明在线控制表现。
