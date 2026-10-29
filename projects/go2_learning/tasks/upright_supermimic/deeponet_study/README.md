# 时间条件 DeepONet 世界模型对照

这是单独的世界模型实验。可以读取已有 SuperMimic replay 快照，也可以用 `collect.py` 启动 Isaac 并采集独立数据。每次采集必须选择一份策略：`--policy expert` 使用检查点内的原专家，`--policy actor` 使用检查点内的当前训练策略；所有环境都使用同一份。离线训练只读取保存的数据，不启动 Isaac；采集和训练均不修改原任务的 reward、日志、模型、策略或已有 replay。

## 模型

- 分支网络输入：初始状态、最多 8 步已知执行动作、每回合的物理 context。查询第 `k` 步时，第 `k+1` 步之后的动作被遮罩，防止未来信息泄漏。
- 主干网络输入：归一化时间 `k/8`；查询点当前只覆盖离散控制步 `k=1,...,8`。
- 两者输出结合，直接预测第 `k` 步状态；每 8 步用自己的预测状态开始下一段。它是多步时间查询实验，不能宣称未测过的连续时间插值准确。
- 比较模型是从头初始化的 `PaperWorldModel` MLP：两种模型使用同一训练窗口、留出窗口、归一化尺度、学习率和更新次数。MLP 为逐步递推预测；DeepONet 在每个 8 步块内直接查询。两者参数量和结构不同。
- 两个 2048 窗口 batch 一起抽取，然后分别计算和依次执行两次参数更新。第 2 次更新使用第 1 次更新后的权重；不是两个同时执行的优化器步骤。

没有指定独立验证集时，留出窗口取自同一份 replay 的其他环境；指定 `--validation-checkpoint` 可使用相同源策略、不同采集随机种子的独立 Isaac 数据。窗口按 32 步不跨 episode、不含终止筛选；采样有放回，因此 512 个窗口可能重复。原有归一化来自历史检查点，并非只用本实验训练环境重新估计。

## 运行

```bash
bash scripts/run-isaac-project.sh projects/go2_learning \
  go2_learning.tasks.upright_supermimic.deeponet_study.train --viz none \
  --checkpoint projects/go2_learning/logs/upright_supermimic/paper_aligned/paper_aligned_20261027_224833_970025/latest.pt \
  --cycles 10 --groups-per-cycle 10 --batch 2048 --seed 2026
```

每个 cycle 含 10 组，每组两个 batch，即每种模型更新 20 次。输出在 `logs/upright_supermimic/deeponet_study/` 下独立目录，包含配置、每轮训练/留出结果和模型权重。它目前没有接入在线 Actor/Critic：真实交互表现、逐步决策和与论文在线训练的兼容性尚待验证。

采集独立数据集：

```bash
bash scripts/run-isaac-project.sh projects/go2_learning \
  go2_learning.tasks.upright_supermimic.deeponet_study.collect --viz none \
  --checkpoint projects/go2_learning/logs/upright_supermimic/paper_aligned/paper_aligned_20261027_224833_970025/latest.pt \
  --policy expert --num-envs 64 --steps 2000 --seed 2141
```

以另一个 seed 再次采集验证集后，训练命令将 `--checkpoint` 指向训练数据集的 `dataset.pt`，并加 `--validation-checkpoint /workspace/go2_learning/logs/.../dataset.pt`。由于通用启动脚本只自动转换 `--checkpoint` 路径，验证集参数须使用容器路径。训练和验证数据必须来自相同源策略、不同采集种子。当前每隔 `--save-interval` 个 cycle 原子保存权重和优化器状态；命令尚不提供恢复训练的入口。

```bash
bash scripts/run-isaac-project.sh projects/go2_learning \
  go2_learning.tasks.upright_supermimic.deeponet_study.self_test --viz none
```
