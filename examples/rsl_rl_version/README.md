# 独立的 RSL-RL 双杆倒立摆小工程

本文件夹包含场景、环境、训练配置和入口代码，保留最小 RSL-RL 接入形式。
依赖现有的 **Isaac Lab 3 / Isaac Sim 6.1 / RSL-RL 5.5.1** 运行环境；
`run.sh` 使用本机已有的 `isaaclab-base:local` 镜像。

## 核心只有四个文件

| 文件 | 职责 |
|---|---|
| [scene.py](scene.py) | 用 Isaac Lab/USD 创建小车、两根杆、关节和并行场景 |
| [env.py](env.py) | `CartpoleEnv(VecEnv)`：观测、动作、奖励、结束、自动重置 |
| [agent_cfg.py](agent_cfg.py) | Actor、Critic 和 PPO 的配置字典 |
| [main.py](main.py) | 创建环境 → 配置 → 标准 Runner → 训练或播放 |

本工程的 Python 文件不导入上一层的任何项目文件。环境直接实现标准 `VecEnv`，
没有额外的 `vec_env.py` 适配层，也不自行实现网络、采样器、GAE 或 PPO。
这些训练组件由 RSL-RL 根据配置创建。

先阅读 `main.py` 中标号 1～4 的部分：

```python
env = make_env(sim, args.num_envs)
train_cfg = make_agent_cfg()
runner = OnPolicyRunner(env=env, train_cfg=train_cfg, log_dir=..., device=...)
runner.learn(num_learning_iterations=args.max_updates)
```

这对应[官方最小接入示例](https://leggedrobotics.github.io/rsl_rl/guide/overview.html#example-integration)。
`make_env()` 在本文件夹的 `env.py`，`make_agent_cfg()` 在 `agent_cfg.py`。
配置直接用 Python 字典提供，省去 YAML 读取。

## 运行

进入本文件夹，在 Ubuntu 桌面终端运行：

```bash
# 先验证两轮
bash run.sh train --max_updates 2 --num_envs 4

# 正式训练：默认 64 个环境、300 次更新、无窗口
bash run.sh train

# 播放最新完成的训练：默认 1 个环境、newton_gl 可视化
bash run.sh play
```

无窗口播放有限步数：`bash run.sh play --viz none --steps 120`。
训练始终从头开始，`--checkpoint` 用于指定播放权重。

`run.sh` 调用仓库的 `scripts/run-isaac-project.sh`，使用镜像中安装的 Isaac Lab。
不拉取镜像、不联网安装包，无需重新 build。Docker 迁移时一起携带启动器，
见 [Docker 说明](../../docs/docker.md)。在仓库根目录也可使用 `bash run.sh cartpole-rsl train`。
项目在容器内固定挂载为 `/workspace/rsl_rl_version`，以
`python -m rsl_rl_version.main` 启动。

已有原生 Isaac Lab 环境时，也可以在本文件夹的父目录执行：

```bash
uv run --no-project --active python -m rsl_rl_version.main --control train --viz none
```

原生环境须先激活 Isaac Sim 的运行环境；Docker 脚本已处理此步骤。

## 环境和参数

- 世界 → X 轴移动关节 → 小车 → Y 轴转动关节 → 第一根杆 → 第二根杆。
- 每个刚体 1 kg，两根杆各长 1 m；PhysX，步长 `1/120` 秒。
- 观测：`[x, x_dot, theta1, theta1_dot, theta2, theta2_dot]`，杆角为世界绝对倾角。
- 动作：一个小车控制量，限幅到 `[-1, 1]` 后映射到 `[-20, 20] N`。
- 奖励：`(1 - theta1² - theta2² - 0.1*x² - 0.001*action²) * dt`。
- 失败：小车偏移超过 1.8 m，或任意杆倾角超过 30°；每局最多 5 秒。
- 并行环境结束后由 `env.step()` 自动重置，超时另通过 `time_outs` 传给 PPO。

调参位置：

- 环境数、更新轮数：`--num_envs`、`--max_updates`。
- 网络、学习率、采样步数：`agent_cfg.py`。
- 奖励、推力、失败条件：本文件夹的 `env.py`。
- 刚体尺寸、质量、关节：本文件夹的 `scene.py`。

## 日志与权重

每次训练独立保存到本文件夹的 `logs/<时间戳>_<run_name>/`：

- `run.json`：训练时的网络及算法配置。
- `events.out.tfevents.*`：TensorBoard 日志。
- `model_*.pt`：Runner 定期保存的检查点。
- `latest.pt`：本次正常训练完成后保存的权重。

播放读取同目录的 `run.json` 恢复网络配置，但场景和奖励使用当前项目代码。
旧版在 `third_party/IsaacLab/logs/my_cartpole_rsl_rl/` 的记录不会自动搬迁。
如需使用，可以把某次运行的完整文件夹（包括 `run.json` 和权重）复制到这里的 `logs/`。
然后用 `--checkpoint /workspace/rsl_rl_version/logs/<运行目录>/latest.pt` 指定播放。
手写 PPO 的权重格式与 RSL-RL 不同，不能直接加载。

## 测试

`test_rsl_rl.py` 和 `test_support.py` 也完全位于本工程内。
在具有 PyTorch、RSL-RL 5.5.1、TensorDict 和 TensorBoard 的环境中，
从本文件夹的父目录运行：

```bash
uv run --no-project --active python -m rsl_rl_version.test_rsl_rl
```

CPU 测试使用假物理夹具，覆盖观测/奖励、动作限幅、局部重置、超时价值补偿、
双杆锚点、克隆后关节绑定，以及真实 Runner 的两轮训练、保存和加载。
它不验证 PhysX 动力学或 GUI；GPU 验证使用上面的两轮 Docker 命令。
