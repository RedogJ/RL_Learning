# 训练配置、日志与检查点

## 当前任务预设

| 参数 | sideflip | backflip |
| --- | --- | --- |
| 启动脚本 | `scripts/train_sideflip.sh` | `scripts/train_backflip.sh` |
| 并行环境 | 2048 | 1024 |
| 每轮真实控制步 | 32/环境 | 32/环境 |
| Replay | 320步/环境 | 320步/环境 |
| WM batch / 窗口 / 更新次数 | 512 / 8 / 30 | 512 / 8 / 30 |
| Actor想象 | 512条×16步，每轮更新1次 | 相同 |
| Actor学习率 | 1e-5 | 4e-6 |
| Critic拟合 | 5 epochs×4 minibatches | 相同 |
| 失败终止 | 高度/倾斜解析终止 | BCE学习式终止 |
| 保存间隔 | 100 cycle | 50 cycle |

`--cycles`表示本次追加轮数；保存编号包含续训前的cycle。每个cycle并不是一个episode。
2048环境预设每轮新增65536条transition；每环境缓存320步，装满后每轮替换10%。
512个训练窗口×8步×30次WM更新是122880个名义训练步，不等于122880条互不重复的新数据。

当前回放示例由分阶段调参和续训产生。后空翻预设使用最终阶段的小学习率；它不是从零训练速度的最优配置保证。

## 到哪里调参数

| 内容 | 位置 |
| --- | --- |
| 侧空翻 / 后空翻权重及sigma | `tasks/mimic/sideflip_reward_cfg.py` / `backflip_reward_cfg.py` |
| 奖励数学形式、dt和一次性失败项 | `rewards.py`、`warp_env.py` |
| 任务失败阈值 | `state_train.py` 的空翻任务配置分支；公共字段在 `config.py` |
| PD、物理/控制频率、动作尺度 | `warp_env.py: EnvCfg` 和 `state_train.py` 任务分支 |
| 网络 | `networks.py`、`state_model.py` |
| 优化次数、采样量、学习率 | 启动脚本或 `train-state --help` |

侧空翻根部相对参考高度误差>0.20m，或局部重力Z分量之差>0.6时失败。0.6不是弧度或欧拉角。
四元数处理支持倒置参考；不会因为机器人在世界中倒置而自动失败。
后空翻保留0.25m/0.8的真实失败阈值。参考最后一帧和正常超时属于截断，不标记成摔倒。

## 续训规则

```bash
bash projects/go2_mujoco_mimic/scripts/train_sideflip.sh \
  --resume /workspace/go2_mujoco_mimic/logs/<原运行目录>/model_1000.pt \
  --cycles 500 --run-name sideflip_continue
```

完整检查点格式为 `go2_core_normalized_v3`，保存Actor、Dynamics、Critic、可选终止头、优化器、归一化、任务配置和RSI统计。
Replay、随机数状态不保存在其中；重启会重新采集数据，因此不是逐比特恢复同一次训练。
`latest.pt`是最后一次保存的编号文件的硬链接或副本，**不是自动挑出的最佳模型**。

改了奖励时追加 `--use-current-reward`，重置Critic与RSI并冻结Actor校准20个cycle。空间跟踪终止规则变化也会重置相关统计。
改任务、参考动作、动作映射和网络输入时存在额外兼容检查，不应跨任务直接完整resume。
只有Actor的演示检查点可回放，但不含WM/优化器，无法完整续训。

从零训练默认先做20个cycle的WM预热，再校准Critic20个cycle；从当前完整检查点正常续训无需重复已完成的预热。

## 如何看日志

| 字段 | 含义 |
| --- | --- |
| `real_reward` | 当前采集块的真实每步平均奖励，含RSI起点混合 |
| `episode_return_mean` / `episode_duration_mean_s` | 这一轮刚结束回合的累计收益/时长；不等待全部环境同步结束 |
| `failures` | 当前采集块累计失败次数，同一环境可多次reset |
| `state_loss` | 归一化状态训练MSE，不是独立验证误差 |
| `imagined_reward_sum` | 折扣和预测存活概率加权的短期奖励 |
| `critic_bootstrap` | 折扣/存活加权的终点估值 |
| `imagined_return` | Actor优化的短期奖励加加权Critic尾项 |
| `eval_*` | 固定种子、关闭噪声、从动作开头进行的独立仿真回放 |

固定起点、无噪声的多个环境不等价于多个独立随机扰动试验。完整率只表示没有触发当前失败条件，姿态精度还要结合位置/朝向误差与实际回放。
比较奖励配置不同时，不能直接用总回报数值判定哪版动作更好。

## 判断是否还值得继续训练

观察完整率、起跳速度、机身朝向跟踪及落地后姿态；不要只看WM loss。
侧空翻历史上在1000轮首次完成动作，之后仍反复；1800轮后固定评估持续完整，约3000～3800轮接近展示效果。该轮次只对应记录的训练分支和配置。
