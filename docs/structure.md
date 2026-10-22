# 工程目录与阅读顺序

## 各层放什么

| 目录 | 内容 |
| --- | --- |
| `projects/` | 正在开发的机器人项目，目前是 Go2 |
| `examples/` | 手写 PPO、最小 RSL-RL 双杆及单杆对照 |
| `third_party/` | Isaac Lab 框架和 Unitree 参考源码，保留原目录结构与许可证 |
| `assets/` | 机器人 USD、配置层、地面和贴图，以及下载来源记录 |
| `scripts/` | 启动容器、准备显示与缓存等运行基础设施 |
| `docs/` | 仓库使用与迁移说明 |
| `archive/` | 早期示例、一次性系统维护脚本 |

根目录的 `run.sh` 选择工程，各工程自己的 `run.sh` 选择运行方式。
项目的 Python 包名和容器内路径保持原名，例如 `go2_learning`。

## Go2 阅读顺序

1. [main.py](../projects/go2_learning/main.py)：选任务、建环境、创建 Runner、训练与播放。
2. [standing/config.py](../projects/go2_learning/tasks/standing/config.py)：把各组配置组合成一个站立任务。
3. [simulation/scene.py](../projects/go2_learning/simulation/scene.py)：机器人、地面、PD 和接触传感器。
4. [simulation/env.py](../projects/go2_learning/simulation/env.py)：观测、动作、物理步、重置和任务调用。
5. [standing/task.py](../projects/go2_learning/tasks/standing/task.py) 与 [rewards.py](../projects/go2_learning/tasks/standing/rewards.py)：失败条件与奖励公式。
6. `training/`：模型查找、续训历史、日志和落地等待的 PPO 接入。

## 调参入口

| 参数 | 位置（相对 Go2 工程） |
| --- | --- |
| 奖励权重、sigma、目标高度 | `tasks/standing/reward_cfg.py` |
| PPO、网络、保存频率 | `tasks/standing/agent_cfg.py` |
| 环境数、控制频率、回合长度、失败阈值 | `tasks/standing/config.py` |
| 推力强度、持续时间、作用点 | `tasks/standing/randomization_cfg.py` |
| 速度指令范围 | `tasks/standing/command_cfg.py` |
| 观测积分状态 | `tasks/standing/integral_cfg.py` |
| 站姿关节角、PD、地面 | `simulation/scene.py` |

新任务放在 `tasks/<任务名>/`，拥有自己的配置与奖励，在 `tasks/__init__.py` 注册。
目前注册 `standing` 站立、`running` 速度跟踪与 `upright` 后腿直立。
移动任务的键盘映射与参数见 [running 说明](../projects/go2_learning/tasks/running/README.md)；
直立任务见 [upright 说明](../projects/go2_learning/tasks/upright/README.md)。

## 旧路径对照（2026-10-15）

| 旧位置 | 新位置 |
| --- | --- |
| `go2_learning/` | `projects/go2_learning/` |
| `rsl_rl_version/` | `examples/rsl_rl_version/` |
| `IsaacLab/scripts/my_cartpole/` | `examples/cartpole_ppo/`，含原权重与单杆备份 |
| `IsaacLab/scripts/compare_cartpole_policies.py` | `examples/cartpole_compare.py` |
| `IsaacLab/` | `third_party/IsaacLab/`，原框架日志仍在其 `logs/` |
| `unitree_rl_lab/` | `third_party/unitree_rl_lab/` |
| `DOCKER_GUIDE.md` | `docs/docker.md` |
| `run-cartpole-gui.sh` | `scripts/run-cartpole-official.sh`；手写工程改用 `bash run.sh cartpole-ppo` |

Go2 内部原 `env.py`、`scene.py`、`contacts.py`、`randomization.py` 进入 `simulation/`；
`commands.py`、`attitude_integral.py` 进入 `control/`；模型、历史、日志和等待采样进入
`training/`；鼠标和键盘进入 `ui/`。任务配置由 `tasks/` 下各任务提供，共用环境通过任务接口读取。

## 日志与模型

Go2 原 `logs/<run>/` 和新 `logs/<task>/<run>/` 均随工程整体移动，内容保留。
`latest` 同时查找两种布局。旧记录中的容器路径与旧宿主机来源链按 `logs/` 后的相对路径定位。
无需为了整理重新训练，也无需重建 Docker 镜像。

```bash
bash run.sh go2 play --checkpoint "projects/go2_learning/logs/某次训练/latest.pt"
bash run.sh go2 train --resume "projects/go2_learning/logs/某次训练/latest.pt" --run_name branch
```

日志、权重、缓存继续由 `.gitignore` 排除，上传 GitHub 时不会自动备份这些文件。
任务配置和源码属于 Git 内容。`archive/maintenance/` 的脚本涉及系统存储迁移，
与日常运行无关，具体用途见 [归档说明](../archive/README.md)。
