# Go2 MuJoCo 模仿学习工程

独立工程，使用仓库现有的官方 Go2 MJCF 资源。目标是后续根据参考动作做模仿驱动的世界模型强化学习；参考数据、奖励与训练方式待确定。

目前提供标准 CPU MuJoCo 场景、四足初态、12 关节 PD 控制与状态读取。不是训练入口，也没有接入现有 upright 策略或日志。

```bash
bash projects/go2_mujoco_mimic/run.sh
bash projects/go2_mujoco_mimic/run.sh --view --seconds 30
bash projects/go2_mujoco_mimic/run.sh reference
```

关节顺序为 FL、FR、RL、RR；每条腿依次 hip、thigh、calf。PD 控制为 Kp=25、Kd=0.5，力矩夹在 ±23.7 N·m。模型保留 MJCF 被动参数；具体模型与仿真参数尚需按后续任务校验。

标准 MuJoCo 使用 CPU 仿真；如需 GPU 并行采样，可另行考虑 MuJoCo Warp。目前没有接入该后端。

参考动作来自 [Kine2Go ai4_dog_canter](https://huggingface.co/datasets/MIMUW-Robotics/kine2go/tree/main/data/ai4_dog_canter)，许可与来源见同目录 clip.json。回放是按记录状态直接设置模型姿态，并非动力学仿真或已训练策略。

## GPU 并行训练基础

使用已有 Docker 镜像运行 MuJoCo Warp 的基础检查：

```bash
bash projects/go2_mujoco_mimic/run_gpu.sh --num-envs 64 --steps 100
```

指定已下载的参考动作（路径为容器内路径）：

```bash
bash projects/go2_mujoco_mimic/run_gpu.sh --motion /workspace/go2_mujoco_mimic/data/kine2go/solo8_crawl_fast/motion.npy
```

第一次会编译 Warp 内核，缓存写在本工程 `.cache/warp`；后续运行可复用。上述命令是零残差动作的冒烟检查，不进行策略训练，也不生成模型。

- `warp_env.py`：并行仿真、逐环境重置、参考动作读取、观测和奖励接口；`EnvCfg` 中可调整物理频率、控制频率和控制参数；`tasks/mimic/reward_cfg.py` 单独配置跟踪权重及尺度。
- `tasks/mimic/config.py`、`task.py`、`rewards.py`：分别组织任务配置、结束条件和奖励计算；后续任务可以在 `tasks` 下独立添加。
- 动作是相对默认关节角的 12 维目标偏移，电机在每个物理子步用 PD 计算力矩。
- 自身观测包括关节角偏差、关节速度、机身角速度、重力投影和上次动作，保留 10 帧；目标使用未来 10 帧的关节位置、根部位置与四元数；`tasks/mimic/networks.py` 提供目标编码器（32 维 latent）和共享策略解码器，当前仅验证前向计算，尚未训练。奖励跟踪各身体部位位置、朝向、线速度与角速度，以及根部位置和朝向，并包含动作平滑与关节限位惩罚。参考刚体轨迹由 MuJoCo 正运动学生成，速度采用控制周期位姿差分，属于针对现有数据的近似。
- 每个环境从参考片段第一帧开始、以相邻参考帧的 MuJoCo 广义坐标差分初始化速度，参考片段结束或到达时限时截断；参考高度差或重力投影倾斜误差超过阈值时终止。尚未处理循环衔接、参考速度初始化和更复杂的失败条件。
- 为兼容当前 Warp 对带 margin 的碰撞限制，GPU 模型关闭了 `MULTICCD` 和 `NATIVECCD`。这可能影响碰撞结果，后续需要对照 CPU MuJoCo 验证。

**尚未接入 PPO、世界模型、轨迹缓冲及训练日志。** 当前只是模仿任务的并行环境基础，不能当作已经训练出的策略。

## 世界模型训练首版

```bash
bash projects/go2_mujoco_mimic/run_gpu.sh train --num-envs 64 --cycles 100 --run-name mimic_v1
```

运行时每个 cycle 输出真实仿真平均每步奖励、失败次数、世界模型损失和想象回报，检查点和配置保存在 `logs/<运行名>_<时间>/`。续训可使用容器路径 `--resume /workspace/go2_mujoco_mimic/logs/<运行目录>/latest.pt`，环境参数及动作路径须与原训练一致；重启后 replay 从新数据重新积累。

当前是独立原型：世界模型以观测历史为状态，学习观测增量、奖励和终止，采用多步拟合和通过冻结模型的想象梯度更新策略。它尚未包含论文所需的完整物理状态、sim context、无噪声观测目标、论文同等的网络/损失及验证集评估；这不是经过验证的 SuperMimic 复现。日志中的模型损失是训练损失，不可用于判断泛化能力或稳定性。

性能参数：默认 GPU 环形 replay、世界模型 MLP BF16（`--wm-amp fp32` 可切回）、`--wm-batch 2048`、`--actor-batch 128`、`--wm-updates 5`、`--window 8`、`--horizon 16`，默认每 50 cycle 及结束时保存检查点（`--save-interval` 可调）。这些属于训练配置，变化会影响训练过程；仍需根据显存和收敛情况调节。续训时 replay 重新采集，检查点不包含 replay 或随机数状态。

物理控制的 20 个子步默认使用 Warp CUDA Graph 重放；首次启动会预热并录制图。若要回退到原逐步实现，训练命令追加 `--no-physics-graph`。重放前后使用 CUDA 流依赖确保 PyTorch 与 Warp 数据顺序；`bash projects/go2_mujoco_mimic/run_gpu.sh check-graph` 可对照检查两种实现的短程数值。

MuJoCo Warp 为每环境预留 256 条约束及 128 个接触；更复杂接触仍需监控容量告警。

## 连续状态世界模型试验入口

`train-state` 是独立实验入口：预测 37 维根部与关节状态（输入加动作/context 共 65 维），用 MJCF 驱动的 Torch 可微正向运动学计算 17 个跟踪刚体位姿，身体速度使用控制周期位姿差分，从预测状态解析计算观测和模仿奖励，使用平滑解析终止近似；旧 `train`、旧模型和日志仍保留。从零训练默认先预热世界模型 20 个 cycle，再冻结 Actor 校准 Critic 20 个 cycle，随后更新 Actor。可以通过 `--actor-checkpoint` 只引入旧 Actor，不能将旧观测世界模型权重当成状态模型加载。

该实现参考论文的状态滚动结构，当前尚不等同完整复现：Go2 状态设计、解析平滑终止及有限显存下的 batch/horizon 是适配；sim context 目前固定且未实施论文域随机化，世界模型缺少独立验证集和正式误差评估。暂未验证其长期训练效果。

归一化 37 维检查点格式为 `go2_core_normalized_v3`；258 维世界模型和 Critic 不可续训到新结构。旧 Actor 仍可播放。新训练默认动作尺度为 ±1 rad。日志分别打印想象累计奖励、折扣/存活加权的 Critic 尾项与二者之和。

运动学数值及梯度校验入口：`python -m go2_mujoco_mimic.tasks.mimic.check_kinematics`（在 Docker 环境执行）。

状态归一化：首轮真实采集的当前/下一状态估计逐维均值与标准差，设置物理尺度下限后冻结。Dynamics 输入标准化、输出增量还原物理单位，WM loss 为逐维标准化平方误差；Critic 状态输入使用同一尺度。Actor 观测与物理奖励单位不变。归一化统计写入新运行的 normalization.json 并保存在模型和 Critic checkpoint 中。每轮默认 30 次 WM 更新，打印全部更新 loss 的平均，不能与旧版原始单位 loss 直接比较。未归一化的旧 checkpoint 不支持完整续训到此版本。


当前 train-state 默认：256 个环境，每轮采集 32 步，缓存 320 步/环境（81920 条转移），装满后每轮替换 10%；这不是每个抽样 batch 强制包含 10% 新数据。Actor 每轮一批 128 条×16 步想象，更新一次；WM 每轮默认 30 次更新，预训练 20 cycle。
训练 reset 在参考一个周期内均匀随机选帧，同步恢复根部/关节位置和速度。参考起始阶段与 episode 计时分开，仍运行完整 10 个参考周期。历史重复填充重置时的观测，上次动作置零，不伪造实际经历过的历史。旧 checkpoint 缺少该配置字段时，播放保持首帧初始化。

同起点闭环预测及 Critic 检验（独立启动，不自动加入训练）：
```bash
bash projects/go2_mujoco_mimic/run_gpu.sh evaluate-state --checkpoint /workspace/go2_mujoco_mimic/logs/运行目录/latest.pt --num-envs 32 --horizon 16
```
评估关闭观测/动作噪声，报告真实/想象短期折扣回报、初始 Critic 与完整回合折扣回报、逐步根部位置和动作误差。软/硬终止及噪声条件的差异需要一起解释。

可选自适应 RSI：`--adaptive-rsi --rsi-bin-seconds 0.2`，按分箱混合 50% 均匀、40% 近期低回报、10% 历史困难采样。分箱回报与历史统计存于 checkpoint，续训须继续传 `--adaptive-rsi`。固定评估保持均匀采样。该实现目前对现有周期参考进行测试；单次后空翻数据与时间采样尚未接入，短片段剩余时长的回报偏差需要专门处理。

后空翻实验入口：`train-state --task backflip`。参考通过 `PYTHONPATH=projects python -m go2_mujoco_mimic.prepare_backflip_reference` 从运动学预览以 30 Hz 重采样并保留末态 0.4s；单次动作不循环。默认开启 0.2s 分箱 RSI，20% 从动作开头重置，其余按 50/40/10 混合分箱；短动作各起点根据剩余时长的理论最大奖励归一化。固定评估始终从动作开头开始。姿态误差采用 q/-q 等价处理；行走用 yaw 特征在后空翻中关闭。参考为近似重定向轨迹，开环 PD 未完成动作，未证明可动力学实现；不应将短测结果视为后空翻训练成功。

后空翻关节奖励在 `tasks/mimic/backflip_reward_cfg.py` 中逐关节配置：`joint_position_weights` 与 `joint_sigmas` 各 12 项，顺序为 FL/FR/RL/RR，每条腿 hip/thigh/calf。奖励率是 `sum_i weight_i * exp(-(error_i / sigma_i)^2)`，不对关节取平均；各环境分别计算。当前每关节权重 1，总上限仍为 12。仅把旧 `12 * mean(scores)` 改成 `sum(scores)` 数学上等价，当前 sigma：前腿 `(0.45, 0.65, 0.65)`，后腿 `(0.45, 0.65, 0.50)` rad；后大腿在 4750 基础上从 0.80 收紧到 0.65，根部位置 `root_position_sigma` 从 0.3 收紧到 0.2 m。修改独立权重后，RSI 会按实际权重之和计算奖励上限。

旧 checkpoint 继续使用其保存的奖励形式和参数。用新奖励续训时需要 `--use-current-reward`，现有续训逻辑会重置 Critic/RSI 并先预热 Critic 20 cycle；Actor、WM 和终止头继承原权重。

## 落地后持续站立评估

```bash
bash projects/go2_mujoco_mimic/run_gpu.sh evaluate-hold \
  --checkpoint /workspace/go2_mujoco_mimic/logs/backflip_v22_joint_reward_2026-11-05_10-30-30_673384/model_4750.pt \
  --hold-seconds 2 --num-envs 64
```

从动作第一帧开始，一直跟踪到原动作结束后约 2 秒；`--hold-seconds` 包含参考已有的末态保持时间。原轨迹结束后持续使用最后一帧作为目标，不在中途 reset。评估沿用 checkpoint 的观测噪声，`--no-noise` 可关闭；打印原参考内/延长段的失败数量及关节、机身位姿、速度指标。失败判据仍是 checkpoint 的任务判据。只改变评估环境的终点和时间上限，不改写参考文件、训练配置或模型。4750 的这次 64 环境检查全部完成，延长段平均机身速度约 0.015 m/s，因此暂未延长训练参考。

## 学习式终止（BCE，可选）

`train-state --learned-termination` 启用论文 Eq.1 的学习式终止路径：

- 终止头为 MLP `(256, 128)`，输入 WM 归一化后的预测状态和参考目标，输出 logit；这是本工程选择的头部结构。
- 在真实动作的多步 WM 预测上计算 `BCEWithLogitsLoss`，与状态损失相加，共同反向传播到动力学和终止头。使用原始失败标签，不加类别权重或人为重采样修正概率。
- Replay 分开记录真实失败、回合结束和下一参考帧。失败转移自身参与损失；跨 reset 后的转移被掩码排除。参考结束单独截断想象；正常超时结束采集回合及窗口，不作为失败标签。
- 想象中使用头部的 sigmoid 概率计算存活权重；冻结模型/终止头参数时仍保留对状态、动作的梯度。
- 从解析终止的 checkpoint 切换时保留 Actor 和已训练动力学，重置 Critic：先预热 WM/终止头 20 cycle，再预热 Critic 20 cycle，然后恢复 Actor 更新。网络和预热策略是工程实现选择，不代表完整复现所有论文设置。
- checkpoint 保存 `termination_head`、`opt_d` 和 `termination_mode`。省略选项会继承 checkpoint；旧 checkpoint 默认保持解析 sigmoid；`--no-learned-termination` 显式切回解析方式。

训练打印 BCE、有效失败样本数、失败召回率（阈值 0.5）及 Brier；这些是训练样本上的指标，不能代替独立验证。`evaluate-state` 自动加载终止头，另行报告真实仿真轨迹上的 BCE、Brier、召回及误报率，并区分真实状态输入和一步预测状态输入。

## Critic 历史与剩余时间

`train-state` 的 Critic 输入为：37 维物理状态、190 维未来参考、16 维物理上下文、Actor 实际使用的 10×42 维观测历史、1 维参考终点倒计时，共 664 维；网络仍为 `(256, 128)`、ELU、单个价值输出。倒计时为 `(frame_limit-frame) / 完整参考回合帧数`，RSI 不改变分母。

旧 243 维 Critic 续训时自动保留旧输入权重，新增历史/时间列置零，重新初始化 Critic 优化器；Actor、WM 和终止头按原续训逻辑加载。默认冻结 Actor 校准 Critic 20 cycle（`--critic-input-warmup-cycles 20`），期间仍采集真实数据、更新 WM/Critic。它是预热时长，不保证 Critic 已收敛。新检查点保存 `critic_config`、倒计时定义和预热截止轮次；从新版继续训练不会再次迁移，未完成的预热会接着执行。

想象尾项经过状态和观测历史向 Actor 回传梯度；Critic 回归拟合时对样本与 λ-return 目标 detach。参考结束后的状态不参与拟合。`evaluate-state` 支持新旧两种 Critic，`play` 的 Actor 输入不变。`--actor-critic-tail-weight` 仍须在训练命令中明确传入，省略时默认 1.0。
