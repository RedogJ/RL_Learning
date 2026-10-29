# Go2 upright · SuperMimic 实验

本任务使用现有 Isaac Lab + PhysX 和 Go2 模型。独立训练和日志位于本目录与 `projects/go2_learning/logs/upright_supermimic/`；不会修改已有 upright 的源码、配置、权重或日志。

## 正式实验入口

在 RL_Learning 根目录运行：

```bash
bash projects/go2_learning/tasks/upright_supermimic/train.sh \
  --num-envs 64 --cycles 300 --randomization
```

默认从 `upright` 最近完成的 `latest.pt` 初始化专家和 Actor。也可通过 `--checkpoint "projects/go2_learning/logs/upright/某次训练/latest.pt"` 明确指定专家。第一次使用前检查终端打印的专家路径。

每轮交替执行：真实仿真采样并写入逐环境环形缓冲；k 步自回归训练世界模型；从缓冲状态想象 H 步，通过冻结的世界模型反向更新 Actor；Critic 使用想象轨迹的 TD(lambda) 回报。默认 k=H=5，以适应 8 GB 显存，和论文的 k=8、H=32 不同。探索噪声只用于模型采集，动作和低通设置与 Isaac 环境一致。

状态包含策略观测、世界速度、脚掌高度/支撑/接触/速度、非脚掌接触统计、随机推力上下文和后小腿高度。Actor 仍只使用专家的 52 维策略观测。奖励从预测状态按 upright 的实际公式计算，失败通过学习的概率头提供可微生存权重；真实失败标签来自 PhysX。

随机推力开始时的未来抽样不可预测：模型验证使用记录的实际推力上下文，想象中延续已知推力且不猜测未来抽样，并避开想象窗口中即将发生新随机推力的状态。这使多步验证成为**给定实际扰动的条件预测**，不能解释为所有扰动下的自主预测精度。

训练环境分为有探索噪声的采集组、有探索噪声的模型验证组、确定性专家评估组和确定性新策略评估组。评估组使用各自独立采样的扰动，不是严格配对实验。模型多步预测、解析奖励误差、失败预测与真实状态下的奖励公式一致性均通过检查时才允许更新 Actor；若 `actor_updated=false`，表示本轮只继续采样和训练世界模型。

每轮 `metrics.jsonl` 包含真实失败次数、真实奖励、模型多步误差、保持原状态基线、失败预测和专家/策略在线评估。`reward_parity_mae_scaled` 核对解析奖励在真实状态下与环境的差异；`analytical_reward_mae_scaled` 衡量预测状态导致的奖励误差。均使用 `--reward-scale`（默认 1000）缩放。`failure_auc_sampled` 与 Brier 分数仅代表抽样验证，不能作为真实安全保证。

默认每 10 轮在新实验目录原子保存 `latest.pt`；它是最近保存的检查点，不代表表现最佳。每个检查点包含 Actor、Critic、世界模型、优化器、环形缓冲、专家权重和任务配置快照。续训会重新初始化 Isaac 环境并插入缓冲屏障，阻止轨迹跨重启拼接；因此不是逐步完全复现的无缝续训。

```bash
bash projects/go2_learning/tasks/upright_supermimic/train.sh \
  --resume "projects/go2_learning/logs/upright_supermimic/某次训练/latest.pt" \
  --num-envs 64 --cycles 300
```

续训必须使用检查点相同的环境数、缓冲长度、想象步长、随机化、折扣和奖励缩放；学习率默认继承检查点，可用 `--actor-lr` 显式修改。续训读取保存的实验奖励，不重新应用当前覆盖文件。

## 播放和评估

```bash
bash projects/go2_learning/tasks/upright_supermimic/play.sh \
  --checkpoint "projects/go2_learning/logs/upright_supermimic/某次训练/latest.pt"
```

默认 Newton GL，可右键施力；只在 15 秒超时后重置。添加 `--teacher` 播放专家；`--viz none --steps 100` 可做无界面检查。

```bash
bash projects/go2_learning/tasks/upright_supermimic/evaluate.sh \
  --checkpoint "projects/go2_learning/logs/upright_supermimic/某次训练/latest.pt" \
  --pairs 8 --steps 600 --randomization
```

评估输出真实仿真中专家和新策略的失败次数、超时次数、每秒奖励和直立对齐度。随机化时两组扰动独立采样，结果需要多个种子才能可靠比较。

## 调整本实验奖励

编辑本目录 `reward_cfg.py` 的 `REWARD_OVERRIDES`，例如：

```python
REWARD_OVERRIDES = {'action_rate_weight': 8000.0}
```

空字典表示使用所选专家记录的奖励快照。新实验启动时覆盖值仅应用于新任务内存并保存到其检查点；不会修改原 upright 配置。

## 与论文和参考仓库的区别

本实现使用 upright 单任务的解析奖励，不是论文的人形动作轨迹跟踪；训练/验证环境划分、专家约束、模型验证门槛和部分推力处理是为本工程增加的实验选择。参考仓库基于 MuJoCo Warp，本实现不使用 MuJoCo。当前没有论文中的动作库、全身运动模仿、观测历史或完整实验规模；模型误差、失败预测和真实评估需持续观察，不能把短测作为成功证明。

`run.sh` 是早期单步模型实验，`rolling.sh` 是早期简化代理奖励原型；正式实验使用 `train.sh`。

## 论文参数的缩小规模试验

`paper_scale_train.py` 是独立入口，使用三层 2048 宽世界模型、k=8、H=32、每轮 10 次世界模型更新和 2e-4 学习率，并将策略 Adam beta 与梯度裁剪改为论文值。为适应本机，此次试验仍使用 64 个仿真环境、128 个模型窗口及 128 个 Actor 起点，远低于论文中的 4096 个仿真环境、2048 个模型窗口及 4096 个想象起点。保留专家约束、模型验证门槛与本工程奖励；不是论文完整复现。策略与价值网络也并非论文的参考架构，Critic 此处执行 20 次全批更新，并非论文的 5 轮各 4 个不同小批次。

例如使用已复制的 v32 专家：

```bash
bash scripts/run-isaac-project.sh projects/go2_learning \
  go2_learning.tasks.upright_supermimic.paper_scale_train --viz none \
  --checkpoint projects/go2_learning/logs/upright_supermimic/experts/upright_adaptive_v32_2026-10-27_16-21-06_572791/latest.pt \
  --num-envs 64 --cycles 30 --collect-steps 40 --wm-updates 10 --warmup-updates 100 \
  --wm-horizon 8 --wm-batch-size 128 --batch-size 128 --horizon 32 \
  --critic-updates 20 --noise 0.1 --actor-lr 0.0002 --replay-steps 1000 --randomization
```

日志写入 `logs/upright_supermimic/paper_trials/`。检查点可用 `play.sh` 与 `evaluate.sh` 加载策略；如需恢复训练，必须仍使用 `paper_scale_train.py`，并保持环境数、replay 长度、想象长度、世界模型窗口与批量等原参数。普通 `train.sh` 的世界模型尺寸不同，不能加载此检查点。

首次试验 `train_20261027_212901_177163` 的三个真实仿真评估种子中，学生分别失败 181、201、207 次；专家均为 0。每种子是 16 环境、900 控制步，失败后环境会重置并可能再次失败。此配置同时改变多项参数，无法单凭此实验判断退化的单一原因。

## 论文式世界模型入口（context 版）

`paper_aligned.sh` 是独立入口，不修改原 upright 配置及日志；输出位于 `logs/upright_supermimic/paper_aligned/`。它按论文描述实现：

- 每回合采样 friction U[0.3,1.2]、12 维 joint calibration U[-0.01,0.01] rad、机身 COM XYZ 偏移 U[±0.025, ±0.05, ±0.05] m。数值写入 PhysX，context 与真实动作一起存入 replay；Isaac 的地面摩擦设为 1 并使用 multiply 组合。关节标定在本 Go2 适配中解释为执行器位置目标偏差，论文没有给出该项目的具体实现。
- 没有物理随机推力；初始关节/姿态随机化仍用于 Go2。物理步 1/200 s、4 个子步，控制 50 Hz；不改变其他任务。
- 三层 2048 的动力学 MLP，输入状态、实际执行动作和逐回合 context；独立的终止头使用预测下一状态和 context。奖励与无噪声观测使用本任务的解析形式，故不另设学习型奖励或观测头。k=8 多步自回归损失包含每步状态误差及终止 BCE；窗口可以以终止步结束，但不跨回合，终止状态在自动重置前采集。
- 每次循环更新世界模型和 Actor，无旧版的专家动作损失及验证门控。H=32；想象时加入论文的关节位置、关节速度、角速度、重力观测噪声。Critic 接收 context，采用 TD(lambda)，5 个 epoch、各 4 个小批次。优化器、学习率、裁剪、折扣和 25 万轮余弦衰减参考论文。
- 每次训练更新平均采集 2.5 个每环境控制步，将论文的 UTD=0.4 解释为每环境控制步对应 0.4 次策略更新；通过交替采样 2/3 步实现。首轮预填充 k=8，故短测的实际比例会受预热影响。论文原文未明确给出 UTD 的实现代码；这是本适配的解释，不能宣称逐行一致。

显存限制：默认为 64 环境、世界模型批量 2048、想象批量 128；论文分别为 4096、2048 和 4096。世界模型每个 batch 采样 2048 个 8 步窗口，默认每轮执行 10 次优化器更新；论文表 A.5 写的是每次更新 10 个 2048 窗口的 batch，但未明确它们各自更新还是累积梯度后更新，不能据此声称优化器步骤一致。`--wm-batch` 可以调整窗口数，初始归一化校准仍固定采样 128 条，不随 batch 改变。RTX 4060 Laptop 8 GB、64 环境下已对批量 2048 做 3-cycle 短测，实际显存余量会随桌面程序和缓冲区状态变化。其余任务差异：采用 Isaac PhysX 与 Go2、现有 upright 奖励、既有 52 维单帧 Actor 和 v32 初始化；不是论文的人形多动作跟踪、10 帧历史或目标编码器。论文没有公开完整状态定义和每一项归一化实现；我们使用现有 90 维 Go2 状态和初始样本归一化，是可检查的工程适配，不可声称等价于作者未公开代码。

默认训练长度为论文的 **320000 cycle**。先显式做短测，例如：

```bash
bash projects/go2_learning/tasks/upright_supermimic/paper_aligned.sh train \
  --checkpoint projects/go2_learning/logs/upright_supermimic/experts/upright_adaptive_v32_2026-10-27_16-21-06_572791/latest.pt \
  --num-envs 64 --cycles 20 --seed 42
```

续训使用新目录中的 `latest.pt`，需要保持环境数、replay 长度、窗口、想象长度和采集频率一致；恢复网络、优化器、Buffer 和随机状态，但 Isaac 物理场景会重新初始化，并在 Buffer 插入屏障，不能视为逐步无缝续训。

```bash
bash projects/go2_learning/tasks/upright_supermimic/paper_aligned.sh train \
  --resume "projects/go2_learning/logs/upright_supermimic/paper_aligned/目录/latest.pt" \
  --num-envs 64 --cycles 100

bash projects/go2_learning/tasks/upright_supermimic/paper_aligned.sh play \
  --checkpoint "projects/go2_learning/logs/upright_supermimic/paper_aligned/目录/latest.pt"

bash projects/go2_learning/tasks/upright_supermimic/paper_aligned.sh eval \
  --checkpoint "projects/go2_learning/logs/upright_supermimic/paper_aligned/目录/latest.pt" \
  --pairs 8 --steps 1000 --seed 123
```

播放入口默认 Newton GL，50 Hz、随机物理参数，禁用失败重置，仅超时重置；可加 `--teacher` 播放原专家。评估的专家与学生使用独立随机环境，不能当成逐环境配对。

首次 20-cycle 短测 `paper_aligned_20261027_222940_874586` 已验证可运行及续训；在 8 对环境、1000 控制步、种子 123 的真实评估中专家失败 23 次、学生失败 67 次（允许同一环境重复失败）。该短测未训练成功；评价时物理随机化与 50 Hz 均不同于专家原来的 30 Hz 训练条件。
