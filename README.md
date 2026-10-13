# RL Learning

当前 Go2 学习工程是 **[go2_learning](go2_learning/README.md)**：本地模型、地面、轻量可视化和标准 RSL-RL 接口。
双杆对照工程保存在 **[rsl_rl_version](rsl_rl_version/README.md)**，二者复用现有 Docker 镜像。

## 目录

```text
RL_Learning/
├── go2_learning/            # Go2：场景、站立环境、标准 RSL-RL
├── rsl_rl_version/          # 双杆独立工程：场景、环境、配置、入口
│   └── logs/               # 运行后生成的训练日志和权重，Git 忽略
├── IsaacLab/               # Isaac Lab 框架与 Docker 构建上下文
│   └── scripts/my_cartpole/ # 之前的手写 PPO 学习工程，保留供对照
├── unitree_rl_lab/         # Unitree 参考工程
├── assets/unitree_model/   # 官方 Go2 USD 模型与来源记录
├── docs/                   # 学习笔记
├── archive/                # 早期示例与一次性维护脚本
├── run-cartpole-gui.sh      # 旧工程入口，仍从根目录使用
└── DOCKER_GUIDE.md          # Docker 构建、运行和服务器迁移说明
```

## 当前工程

从本目录执行：

```bash
bash go2_learning/run.sh
```

默认只观察一台 Go2 的 PD 站姿，使用 `newton_gl`。训练入口见工程 README。

## 双杆工程

```bash
cd rsl_rl_version
bash run.sh train --max_updates 2 --num_envs 4
```

正式训练与播放：

```bash
bash run.sh train
bash run.sh play
```

镜像名：`isaaclab-base:local`。移动项目目录无需重建镜像。
构建文件仍是 [IsaacLab/docker/Dockerfile.base](IsaacLab/docker/Dockerfile.base)，
具体命令见 [DOCKER_GUIDE.md](DOCKER_GUIDE.md)。

## 旧实验与整理记录

- 手写 PPO、单杆备份及已有权重保留在 `IsaacLab/scripts/my_cartpole/`。
- 旧实验日志保留在 `IsaacLab/logs/`。
- 学习笔记：[docs/My_Learning.md](docs/My_Learning.md)。其中早期构建命令需在 `IsaacLab/` 下执行。
- 归档位置和用途见 [archive/README.md](archive/README.md)。

2026-10-13 整理时，将独立工程从 `IsaacLab/scripts/my_cartpole/rsl_rl_version/`
整体移到根目录，并清理可重新生成的 Python 字节码缓存。
训练结果被 Git 忽略，需要单独备份和迁移。
