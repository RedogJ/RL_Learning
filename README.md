# RL Learning · Go2 SuperMimic

机器人强化学习实验仓库。当前组内分享重点是 **Go2 的 SuperMimic 风格世界模型训练**：用 MuJoCo Warp 采集真实仿真数据，学习可微动力学，再通过闭环想象更新 Actor/Critic，已完成侧空翻和后空翻参考动作训练。

**从这里开始：[SuperMimic 工程说明](projects/go2_mujoco_mimic/README.md)**

| 内容 | 入口 |
| --- | --- |
| 算法结构、输入输出和论文差异 | [架构说明](projects/go2_mujoco_mimic/docs/architecture.md) |
| Docker、依赖、路径和检查 | [环境说明](projects/go2_mujoco_mimic/docs/environment.md) |
| 训练、续训、参数与日志 | [训练指南](projects/go2_mujoco_mimic/docs/training.md) |
| 演示、模型来源与学习过程 | [实验结果](projects/go2_mujoco_mimic/docs/results.md) |
| 参考格式与数据来源 | [参考轨迹](projects/go2_mujoco_mimic/data/README.md) |

## 演示

- [侧空翻：最新5800轮，正常速度＋半速](projects/go2_mujoco_mimic/videos/sideflip_latest_cycle5800_20261106.mp4)
- [后空翻：最新5250轮，正常速度＋半速](projects/go2_mujoco_mimic/videos/backflip_latest_cycle5250_20261106.mp4)

白色机器人由策略闭环控制，透明蓝色为参考动作；录制使用固定起点、关闭观测噪声。

## 运行 SuperMimic

在仓库根目录执行，已有 `isaaclab-base:local` 镜像可直接复用。
独立MuJoCo镜像与新机器安装见工程环境说明。

```bash
# GPU物理检查
bash run.sh supermimic --num-envs 16 --steps 30 \
  --motion /workspace/go2_mujoco_mimic/data/go2_flip_to/sideflip_left_30hz/motion.npy

# 从当前配置开始新的训练
bash projects/go2_mujoco_mimic/scripts/train_sideflip.sh --cycles 1000
bash projects/go2_mujoco_mimic/scripts/train_backflip.sh --cycles 1000

# 拿到演示权重或分享包后回放
bash run.sh supermimic play \
  --checkpoint /workspace/go2_mujoco_mimic/checkpoints/sideflip_policy_5800.pt --no-noise
```

演示权重由完整训练检查点导出；Git默认不携带日志和权重。
最新训练模型可完整续训，Actor演示权重仅供回放。演示模型经历了多阶段调参，当前从零训练配置不等于历史训练过程的逐步重演。

## 给同事的轻量分享包

```bash
bash projects/go2_mujoco_mimic/scripts/export_demo_policies.sh
python scripts/export_supermimic.py \
  --output share_exports/go2_supermimic_20261106 --with-demo-policies --archive
```

分享包保留运行所需的相对目录：SuperMimic代码、Go2 MJCF/mesh、参考轨迹、最新演示视频与可选Actor权重；排除日志、缓存、历史诊断及其他机器人项目，附带文件校验清单。
第三方来源见[THIRD_PARTY_NOTICES.md](projects/go2_mujoco_mimic/THIRD_PARTY_NOTICES.md)。自有代码的统一开源许可证尚待确定。

## 仓库其他工程

```text
RL_Learning/
├── run.sh                       # 统一入口；bash run.sh --help
├── projects/
│   ├── go2_mujoco_mimic/         # 当前重点：MuJoCo Warp世界模型模仿RL
│   ├── go2_learning/             # Isaac Lab + RSL-RL：standing/running/upright等
│   └── go2_sim2sim/              # Isaac策略的MuJoCo验证
├── examples/                    # PPO及RSL-RL倒立摆学习示例
├── assets/                      # Go2模型和原Isaac地面资源
├── third_party/                 # Isaac Lab、Unitree参考源码
├── beyondmimic/                 # 参考源码
├── scripts/                     # 启动器与分享包导出
├── docs/                        # 原工程目录、Docker与迁移说明
└── archive/                     # 早期示例和维护脚本
```

原Isaac工程仍可用：

```bash
bash run.sh go2 view
bash run.sh go2 train --task standing --max_updates 1000 --run_name stand
bash run.sh go2 play --task running
```

- [Isaac Go2工程](projects/go2_learning/README.md)
- [MuJoCo sim2sim](projects/go2_sim2sim/README.md)
- [最小RSL-RL示例](examples/rsl_rl_version/README.md)
- [手写PPO示例](examples/cartpole_ppo/README.md)
- [原仓库目录说明](docs/structure.md)
- [原Isaac Docker与服务器迁移](docs/docker.md)

## 检查

```bash
python -m unittest discover -s tests -q
```

SuperMimic的奖励、Critic、Replay和运动学梯度检查见工程[环境说明](projects/go2_mujoco_mimic/docs/environment.md#本地检查)。
