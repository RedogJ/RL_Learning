# DeepONet Cartpole

独立的倒立摆实验工程：低频读取状态历史并生成控制函数，高频按时间取控制值。
物理场景沿用已经跑通的小车与杆，后端仍是 Isaac Sim PhysX；播放默认使用 `newton_gl` 可视化。

## 开始运行

在 `RL_Learning` 目录执行，使用现有 `isaaclab-base:local` 镜像。

先跑两次 PPO 更新，检查 GPU 仿真流程：

```bash
bash deeponet_cartpole/run.sh train --max_updates 2
```

再正式训练：

```bash
bash deeponet_cartpole/run.sh train --max_updates 300
```

可视化播放：

```bash
bash deeponet_cartpole/run.sh play
```

播放按真实时间节奏执行，按 Ctrl+C 退出。两次更新只是流程检查，通常不足以学会平衡。
训练没有可视化和实时等待，仿真速度由硬件与场景开销决定。

原来的工程仍在 `IsaacLab/scripts/my_cartpole/`，用原来的 `run-cartpole-gui.sh learn ...` 启动。
新工程代码挂载到 `/workspace/deeponet_cartpole`，默认权重保存为本目录下的
`checkpoints/latest_deeponet.pt`，与原来的 `latest.pt` 分开。
运行器复用已有的 Isaac Lab 框架、镜像和缓存；使用 `--pull=never`、`--network none`。
通过原工程已经跑通的 `isaaclab.sh -p` 加载镜像里的 Isaac Sim 环境，
无需指定某个环境初始化脚本；可能出现 `isaaclab.sh` 的弃用提示。
场景由本地几何创建，不需要下载机器人模型。

## 网络如何执行控制

单帧状态是 `[x, x_dot, theta, theta_dot]`，单位分别为 m、m/s、rad、rad/s。

默认配置：

| 项目 | 默认值 |
| --- | --- |
| 并行环境 | 64 |
| 物理步长和指令间隔 | 1/120 秒，即 120 Hz |
| 历史长度 | 5 帧，按物理步保存，从旧到新，最后一帧是当前状态 |
| 一个策略决策的控制段 | 5 步，即 1/24 秒 |
| 正常控制段的状态反馈与 Branch 推理频率 | 24 Hz |
| Branch 输入 | `[环境数, 5, 4]` 展平为 20 维 |
| Branch 输出 | 每个环境 32 维系数 |
| Trunk 输入 | 段内时间 `tau / 控制段时长` |
| 策略输出 | 控制均值函数 `u(tau)`，默认查询 5 个时间点 |

形式为：

```text
过去 5 帧状态 ── Branch ── 32 维系数 b ─┐
                                      ├─ 点积 / sqrt(32) + bias ── u(tau)
段内时间 tau ── Trunk ── 32 维基函数 t ─┘
```

每段开始只编码一次历史，随后按 `0, 1/120, 2/120, 3/120, 4/120` 秒取控制值。
固定时间点在段首批量查询并缓存，逐步施加给仿真。下一段才利用新的历史状态更新控制函数。
提前失败的环境在段末统一重置；失败后的动作不计入训练，历史用新回合初始状态填充。

`networks.py` 还提供任意时间查询接口：

```python
features = policy.encode_history(history)
controls = policy.evaluate_control(features, [0.0, 0.003, 0.02])
# controls: [环境数, 3, 1]，每个查询时间的单位都是秒
```

时间应在当前控制段内。当前实现可以表示段内连续的控制均值，但物理执行仍是离散指令。
相邻控制段之间尚未添加连续性约束。训练时在各执行点独立采样高斯探索噪声，
这些噪声也没有连续性约束；播放使用均值。

Branch/Trunk 点积结构参考 [DeepONet 原论文](https://arxiv.org/abs/1910.03193)。
这里通过 PPO 学习控制函数，没有增加监督学习数据集或系统预测模型。

## 参数在哪改

时间安排、历史长度和并行规模可以从命令行改：

```bash
bash deeponet_cartpole/run.sh train \
  --history_steps 10 --control_steps 10 --num_envs 64 \
  --rollout_steps 64 --max_updates 300 --seed 42
```

该例仍以 120 Hz 执行指令，每 10 步预测一次，正常反馈频率为 12 Hz。
`--control_steps 1` 可用于检查没有控制段间隔时的行为。
播放从权重恢复历史长度、控制段长度和推力上限，环境数量可单独调整。

| 文件 | 内容 |
| --- | --- |
| `main.py` | 入口、物理配置、训练和播放、保存权重 |
| `scene.py` | 小车、杆、关节、环境复制和状态读取 |
| `controller.py` | 保留的 PD 控制器，用于检查物理场景 |
| `env.py` | 奖励、失败条件、历史缓冲、整段动作执行与重置 |
| `networks.py` | DeepONet Actor 和读取历史的 MLP Critic |
| `rollout.py` | 控制段采样，保存执行掩码和持续步数 |
| `ppo.py` | 按持续步数折扣的优势估计与 PPO 更新 |
| `run.sh` | Docker 挂载、缓存、图形授权和启动 |

奖励、20 N 推力上限、30 度失败角度、1.8 m 位置限制和 5 秒回合长度沿用原工程。
Actor 和 Critic 都需要重新训练，旧 MLP 权重无法直接加载。
重复训练会覆盖本工程的默认权重；可另存一次实验：

```bash
bash deeponet_cartpole/run.sh train --checkpoint /workspace/deeponet_cartpole/checkpoints/experiment_01.pt
bash deeponet_cartpole/run.sh play --checkpoint /workspace/deeponet_cartpole/checkpoints/experiment_01.pt
```

传入的路径由容器内 Python 使用，以上路径对应本机的 `deeponet_cartpole/checkpoints/`。

## PPO 的时间单位

一个训练样本对应一段控制。段内奖励按 `gamma**i` 累计；如果实际执行 `d` 步，
下一段价值使用 `gamma**d` 折扣。失败不自举，时间截断使用重置之前的历史做价值自举。
GAE 在决策边界计算，跨段轨迹衰减为 `(gamma * gae_lambda)**d`；
这是按持续时间衰减的决策层优势估计，并不等同于逐物理步计算完整 GAE。
PPO 使用实际执行部分的联合 log probability，未执行尾部被掩码排除。

默认 `gamma=0.99`、`gae_lambda=0.95` 都按物理步解释；
学习率 `3e-4`、4 个优化 epoch、clip `0.2`、entropy 系数 `0.005`。

**比较模型时注意采样预算：** `--rollout_steps` 指控制段数。
默认一轮最多采样 `128 × 64 × 5 = 40960` 个有效物理步；
原工程一轮是 `128 × 64 = 8192` 个物理步。
相同的 300 次更新没有相同的交互预算，应对照日志里的累计有效物理步。
DeepONet 能表达按时间查询的控制函数；是否比 MLP 快、是否改善平衡，需要实测。
降低新状态反馈频率也会影响对段内外界扰动的响应。

## 已有检查

`test_deeponet.py` 使用 CPU 和简化的假场景，检查：

- 任意时间查询和缓存历史接口；
- 历史移位与单环境重置；
- 段内提前失败/截断时的执行掩码、奖励和终止历史；
- 按持续时间折扣、失败与截断的自举边界；
- 重置前的价值估计，以及未执行动作不影响 PPO；
- PPO 更新能改变 Branch 和 Trunk 参数。

可用已有 PyTorch 环境执行 `python deeponet_cartpole/test_deeponet.py`。
这些检查不启动 Isaac Sim；GPU 仿真与训练效果由上面的两轮试跑及后续训练验证。
