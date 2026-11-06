# Go2 SuperMimic：MuJoCo Warp 世界模型强化学习

用真实 GPU 仿真持续采集数据，训练可微世界模型，再通过世界模型中的闭环想象更新策略和 Critic。当前已训练 Go2 **侧空翻与后空翻**，提供参考轨迹、训练入口、回放和诊断工具。

本工程是面向 Go2 的 SuperMimic 思路实现。实现范围与论文差异见[架构说明](docs/architecture.md)，结果与模型来源见[结果说明](docs/results.md)。

## 效果

| 任务 | 最新展示模型 | 视频 |
| --- | --- | --- |
| 侧空翻 | cycle 5800 | [正常速度＋半速](videos/sideflip_latest_cycle5800_20261106.mp4) |
| 后空翻 | cycle 5250 | [正常速度＋半速](videos/backflip_latest_cycle5250_20261106.mp4) |

视频为真实 MuJoCo Warp 闭环策略回放，白色为策略，透明蓝色为参考。固定起点，关闭观测噪声。视频中的模型由多阶段训练产生，不代表从零训练当前配置必定复现相同曲线。

## 快速开始

所有命令从仓库根目录执行。需要 Linux、NVIDIA GPU、Docker 和 NVIDIA Container Toolkit。
已有 `isaaclab-base:local` 镜像可以直接复用；仿真与训练代码本身不依赖 Isaac Lab。

```bash
# 1. 检查 GPU 并行物理环境；首次运行会编译 Warp 内核
bash run.sh supermimic --num-envs 16 --steps 30 \
  --motion /workspace/go2_mujoco_mimic/data/go2_flip_to/sideflip_left_30hz/motion.npy

# 2. 看参考轨迹（运动学播放，不是策略）
bash run.sh supermimic reference \
  --motion /workspace/go2_mujoco_mimic/data/go2_flip_to/sideflip_left_30hz/motion.npy \
  --checkerboard --speed 0.5

# 3. 从零开始侧空翻训练
bash projects/go2_mujoco_mimic/scripts/train_sideflip.sh --cycles 1000

# 4. 回放自己的模型，把 <运行目录> 换成实际目录
bash run.sh supermimic play \
  --checkpoint /workspace/go2_mujoco_mimic/logs/<运行目录>/model_1000.pt
```

新机器的环境安装、独立 MuJoCo 镜像、X11 与路径映射见[环境说明](docs/environment.md)。
拿到组内分享包的同事可直接播放 `checkpoints/` 中的演示策略，见下文。

## 训练、续训与回放

```bash
# 后空翻：启用学习式 BCE 终止，使用独立奖励配置
bash projects/go2_mujoco_mimic/scripts/train_backflip.sh --cycles 1000

# 修改环境数等参数：追加参数覆盖脚本中的同名默认值
bash projects/go2_mujoco_mimic/scripts/train_sideflip.sh --num-envs 256 --cycles 100

# 续训；--cycles 是本次追加轮数
bash projects/go2_mujoco_mimic/scripts/train_sideflip.sh \
  --resume /workspace/go2_mujoco_mimic/logs/<运行目录>/model_1000.pt \
  --cycles 500 --run-name sideflip_continue

# 改了奖励后再续训，追加 --use-current-reward
# 会继承 Actor/世界模型，重置 Critic 与 RSI，并先校准 Critic

# 分享包中的侧空翻演示策略：仅供回放，不是完整续训检查点
bash run.sh supermimic play \
  --checkpoint /workspace/go2_mujoco_mimic/checkpoints/sideflip_policy_5800.pt \
  --no-noise --speed 0.5

# 真实仿真与世界模型短期预测对照（需要完整训练检查点）
bash run.sh supermimic evaluate-state \
  --checkpoint /workspace/go2_mujoco_mimic/logs/<运行目录>/model_1000.pt \
  --num-envs 32 --horizon 16
```

回放窗口空格暂停/继续，关闭窗口或 Ctrl+C 退出。`--no-noise` 关闭观测噪声。
完整参数和日志解释见[训练指南](docs/training.md)。

## 阅读代码

```text
projects/go2_mujoco_mimic/
├── run_gpu.sh                   # Docker 和子命令分发
├── scripts/train_sideflip.sh    # 侧空翻训练参数预设
├── scripts/train_backflip.sh    # 后空翻训练参数预设
├── environment.py              # 官方 Go2 MJCF、CPU 模型与关节顺序
├── warp_env.py                 # GPU 物理、参考动作、观测、reset 和 reward
├── tasks/mimic/
│   ├── state_train.py          # 主训练循环：真实采集→WM→想象→Actor/Critic
│   ├── replay.py               # GPU 时间窗口缓存
│   ├── state_model.py          # 37维状态、归一化、Dynamics、Critic、终止头
│   ├── networks.py             # 目标编码器和策略解码器
│   ├── kinematics.py           # 由预测状态计算身体位姿，保留梯度
│   ├── rewards.py              # 奖励公式，共用于真实仿真与想象
│   ├── reward_cfg.py           # 奖励公共字段与旧配置兼容
│   ├── sideflip_reward_cfg.py  # 侧空翻权重与 sigma
│   ├── backflip_reward_cfg.py  # 后空翻权重与 sigma
│   └── adaptive_rsi.py         # 按参考阶段难度分配 reset 起点
├── play_policy.py              # 策略＋透明参考回放
├── play_reference.py           # 运动学参考回放
├── data/                       # 参考数据，格式和来源见 data/README.md
├── videos/                     # 演示视频与录制元数据
├── docs/                       # 架构、环境、训练与结果
└── tests/                      # 奖励、Critic 梯度及缓存检查
```

推荐顺序：`state_train.py` → `warp_env.py` → `state_model.py` → `networks.py` → `rewards.py`。
`tasks/mimic/train.py`、`world_model.py` 是早期观测世界模型原型，主训练请用 **train-state**。
`diagnostics/` 是研发诊断记录；`docs/history/` 是历史文档，轻量分享包默认不携带它们。

## 检查与分享

```bash
# 在已有运行镜像中执行单元检查和运动学梯度检查
# 具体命令见 docs/environment.md

# 导出用于回放的小模型（已有训练日志的维护者执行）
bash projects/go2_mujoco_mimic/scripts/export_demo_policies.sh

# 生成轻量分享目录及 tar.gz；不携带训练日志、缓存和其他机器人项目
python scripts/export_supermimic.py \
  --output share_exports/go2_supermimic_20261106 --with-demo-policies --archive
```

参考动作与模型来源见 [data/README.md](data/README.md) 和 [第三方来源](THIRD_PARTY_NOTICES.md)。
完整训练检查点和日志由 `.gitignore` 排除，需要另外传给希望续训的同事。
本仓库目前未设置统一的自有代码开源许可证；公开发布时请确定该许可证，并保留第三方材料各自的许可及来源。
