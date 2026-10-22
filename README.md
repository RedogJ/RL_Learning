# RL Learning

用 Isaac Lab 和 RSL-RL 学习机器人强化学习。当前主工程包含 **Go2 站立抗扰动、速度跟踪与后腿直立**。

## 目录

```text
RL_Learning/
├── run.sh                      # 统一运行入口
├── projects/
│   ├── go2_learning/           # Go2 主工程：standing / running / upright
│   └── go2_sim2sim/            # Go2 的 MuJoCo 独立播放与验证
├── examples/
│   ├── rsl_rl_version/         # 最小 RSL-RL 双杆示例
│   ├── cartpole_ppo/           # 手写 PPO 双杆及单杆备份
│   └── cartpole_compare.py     # 官方单杆奖励对照
├── third_party/
│   ├── IsaacLab/              # 框架源码和 Docker 构建上下文
│   └── unitree_rl_lab/        # Unitree 参考工程
├── assets/                    # 本地 Go2 USD、地面及贴图
├── scripts/                   # Docker 共用启动器与官方例程入口
├── tests/                     # 启动参数与挂载检查，不启动 Docker
├── docs/                      # 使用说明、目录与迁移指南
└── archive/                   # 早期示例与一次性维护脚本
```

## 运行 Go2

在本目录执行，复用已有的 `isaaclab-base:local` 镜像：

```bash
bash run.sh go2 view
bash run.sh go2 train --task standing --max_updates 1000 --run_name stand
bash run.sh go2 play --task standing
```

继续训练最近完成的模型：

```bash
bash run.sh go2 train --task standing --resume latest --max_updates 1000 --run_name continue_stand
```

训练默认无窗口，查看与播放默认使用 `newton_gl`。参数说明见 [Go2 工程](projects/go2_learning/README.md)。
奖励、PPO、扰动及任务配置集中在 [tasks/standing](projects/go2_learning/tasks/standing/)。
移动任务使用 `--task running`，训练与键盘播放见 [running 说明](projects/go2_learning/tasks/running/README.md)。
日志和权重保存在 `projects/go2_learning/logs/`，已被 Git 忽略，需要单独备份。

## 学习示例

```bash
# 标准 RSL-RL 双杆
bash run.sh cartpole-rsl train --max_updates 2 --num_envs 4
bash run.sh cartpole-rsl play

# 手写 PPO 双杆
bash run.sh cartpole-ppo --control train --num_envs 64 --viz none --max_updates 300
bash run.sh cartpole-ppo --control play --num_envs 1

# 查看全部入口
bash run.sh --help
```

## 文档

- [目录职责、阅读顺序与旧路径对照](docs/structure.md)
- [Docker 构建、运行与服务器迁移](docs/docker.md)
- [Go2 任务、参数及模型兼容规则](projects/go2_learning/README.md)
- [Go2 后腿直立任务](projects/go2_learning/tasks/upright/README.md)
- [Go2 MuJoCo sim2sim 验证](projects/go2_sim2sim/README.md)
- [最小 RSL-RL 接入示例](examples/rsl_rl_version/README.md)
- [手写 PPO 对照工程](examples/cartpole_ppo/README.md)
- [归档文件用途](archive/README.md)

## 本地检查

在已有 torch 的 Python 环境执行：

```bash
python -m unittest discover -s tests -q
python -m unittest discover -s projects/go2_learning/tests -t projects -q
python examples/cartpole_ppo/test_double_cartpole.py -q
```

启动器检查使用命令替身；这些检查不验证实际 GPU 仿真。
