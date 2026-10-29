# 十帧历史世界模型对照

两个模型都输入从旧到新排列的状态 `s[t-9:t+1]`（当前帧及前 9 帧）。按 50 Hz 采样，第一帧到最后一帧跨度 0.18 秒。输入为已有 90 维 SuperMimic 适配状态，保留每帧的上一执行动作；不是策略网络的观测历史。

- MLP 每步读取 10 帧、当前执行动作和物理 context，预测下一状态，并把自己的预测加入历史。
- DeepONet 分支读取同一 10 帧、接下来 8 步的执行动作及 context，主干查询第 1–8 步。尚未发生的动作在查询对应时刻之前被遮罩。32 步评估以预测状态更新历史，每 8 步重新开始一个预测块。
- 对固定 replay 的每个有效起点要求：9 个过去状态与后续预测都处于同一 episode、同一 context，且不含重置等待。训练采样长度为 9+8；验证/测试长度为 9+32，测试整段禁止终止。窗口采样有放回。
- 两个模型从头训练，使用相同训练窗口、归一化、优化器和次数。每组先抽两批 2048 窗口，再分别依次更新两次；默认每轮 10 组、100 轮，即每个模型 2000 次参数更新。模型参数量和预测方式不同。
- 验证/测试来自不同随机种子的独立 Isaac 采集，但均使用同一组冻结策略和类似物理随机化。真实执行动作作为模型输入；这不是实时闭环部署测试。

```bash
bash scripts/run-isaac-project.sh projects/go2_learning \
  go2_learning.tasks.upright_supermimic.history_study.train --viz none \
  --checkpoint projects/go2_learning/logs/upright_supermimic/deeponet_study/datasets/20261027_152321_065444/dataset.pt \
  --validation-checkpoint /workspace/go2_learning/logs/upright_supermimic/deeponet_study/datasets/20261027_152429_852598/dataset.pt \
  --cycles 100 --groups-per-cycle 10 --batch 2048
```

输出独立保存在 `logs/upright_supermimic/history_study/`。每 10 轮原子保存权重和优化器状态；目前未实现续训入口。使用 `history_study.evaluate` 可在同一组轨迹上比较本实验两个模型和旧版单帧两个模型；测试集中前 9 帧仅作为历史，不参与预测损失。
