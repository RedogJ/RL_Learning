# 双杆倒立摆：手写 PPO 与 RSL-RL 对照

这版直接复用上一层的 `scene.py`、`env.py`，使用真实的 RSL-RL 5.5.1
`OnPolicyRunner` 训练。同样是 PhysX、64 个并行环境、6 维观测、1 维小车推力。
原来的主程序、训练代码和权重保持原样。这里不引入 DirectRLEnv、任务注册或额外的物理后端。

## 先跑两轮，再看代码

在宿主机 `RL_Learning` 根目录运行：

```bash
bash IsaacLab/scripts/my_cartpole/rsl_rl_version/run.sh train --max_updates 2 --num_envs 4
```

这是接口与仿真的短验证，两个更新不足以学会平衡。正式训练：

```bash
bash IsaacLab/scripts/my_cartpole/rsl_rl_version/run.sh train --max_updates 300 --num_envs 64
```

播放最近一次完成训练的本版本权重：

```bash
bash IsaacLab/scripts/my_cartpole/rsl_rl_version/run.sh play
```

播放默认 `--viz newton_gl`、1 个环境，并限制到真实时间。
也可以用 `play --viz kit`，或用 `play --viz none --steps 120` 做有限步的无窗口验证。
训练默认无窗口，不做实时限速；`train --viz newton_gl` 可以观察训练，但会更慢。

每次训练独立保存到：

```text
IsaacLab/logs/my_cartpole_rsl_rl/<时间戳>_baseline/
├── run.json                  # 实际训练配置、环境参数、版本、随机种子
├── events.out.tfevents.*      # TensorBoard 日志
├── model_0.pt                # Runner 定期保存的检查点
├── model_<最后迭代编号>.pt
└── latest.pt                 # 正常完成本次训练后保存
```

两轮训练的迭代编号是 0 和 1。默认 `latest` 只搜索这个 RSL-RL 日志目录，不会加载旧手写权重。
显式选择某次训练可用 `play --checkpoint /workspace/isaaclab/logs/my_cartpole_rsl_rl/<运行目录>/latest.pt`。
容器内路径也可以指向该运行目录下的 `model_*.pt`，它必须和 `run.json` 在同一个目录。
播放会读取保存的网络配置，并检查当前环境参数。`train` 始终从头训练，本练习暂不提供断点续训。

启动脚本复用现有 `isaaclab-base:local`，使用 `--pull=never --network none`。
它激活镜像已有的 Isaac Sim 环境，并通过 `uv run --offline --no-project --active`
调用现有 venv；不会安装依赖或重建镜像。源码只读挂载，日志目录可写。

## 建议在 VS Code 中左右打开这些文件

| 原来手写的部分 | RSL-RL 版本 | 对比重点 |
|---|---|---|
| [main.py](../main.py) | [main.py](main.py) | 场景创建相同；训练循环变为 `runner.learn(...)` |
| [networks.py](../networks.py) | [agent_cfg.py](agent_cfg.py) 的 actor/critic | 6→32→32→1、Tanh、初始 std=0.5 由配置创建 |
| [ppo.py](../ppo.py) 的超参数 | [agent_cfg.py](agent_cfg.py) 的 algorithm | epochs、学习率、gamma、lam 等配置的位置 |
| [rollout.py](../rollout.py) | Runner、PPO、Storage 库内部 | 本工程不再自己存 rollout 或计算 GAE |
| [env.py](../env.py) | [vec_env.py](vec_env.py) | 同一个原环境，适配 RSL-RL 的调用方式 |
| [scene.py](../scene.py) | 直接导入原文件 | 模型、两根杆的关节、克隆修正共用 |

先在新版 `main.py` 中找三个带“对照重点”的注释：

```python
env = CartpoleVecEnv(CartpoleEnv(sim, scene, scene_cfg))
runner = OnPolicyRunner(env, agent_cfg, log_dir=..., device=...)
runner.learn(num_learning_iterations=args.max_updates, init_at_random_ep_len=False)
```

这三句就是入口的核心。前面的代码创建同一个双杆场景；后面的代码保存权重或播放。
这里使用 Python 字典作为配置，不依赖原目录里旧的 `agent_cfg.py`，也不继承官方 Cartpole 任务配置。

## 为什么要有 vec_env.py？

原环境接口：

```python
observation, rewards, terminated, truncated, info = original_env.step(actions)
```

RSL-RL 要求的接口：

```python
observations, rewards, dones, extras = vec_env.step(actions)
```

适配器负责四件事：

1. 把 `[N, 6]` 张量包装成 `TensorDict({"policy": observation})`。
2. 合并 `dones = terminated | truncated`，并单独提供 `extras["time_outs"]`。
3. 自动重置结束的环境，下一次返回这些环境重置后的观测。
4. 提供 `num_envs`、`num_actions`、`episode_length_buf` 等框架需要的属性。

奖励、力的施加、角度计算、失败条件均调用原 `env.py`。
适配器返回独立观测快照，防止 reset 或下一步覆盖已经保存的样本。
原环境会限制实际推力；策略采样的原始动作仍由 RSL-RL 保存，用于计算对应的 log_prob。

## 哪些参数可以直接改？

- 环境数量：命令中的 `--num_envs`。
- 更新轮数：命令中的 `--max_updates`。
- 每轮采样步数：`agent_cfg.py` 中 `num_steps_per_env`。
- 网络宽度：`actor/critic.hidden_dims`。
- 学习率、GAE、PPO 裁剪系数：`algorithm` 字典。
- 奖励与失败条件：上一层共用的 `env.py`。

注意：修改共用的 `scene.py` 或 `env.py` 会同时影响两版；修改本目录的训练配置只影响 RSL-RL 版。
`--seed 42` 固定新版的随机种子，但不能使两种实现的训练轨迹完全相同。

## 相同的任务，不承诺两种 PPO 数值完全相同

已对齐：场景、观测顺序、120 Hz 控制、20 N 推力限制、奖励、失败角度、5 秒超时、
32/32 Tanh 网络、初始动作 std=0.5、无观测标准化、128 步采样、4 epochs、整批更新、
学习率 3e-4、gamma=0.99、lambda=0.95、clip=0.2、entropy=0.005。

仍有库实现差异：

- 原版使用两个独立 Adam；RSL-RL 使用一个优化器管理 Actor 和 Critic。
- RSL-RL 5.5.1 的标准超时补偿使用 `gamma * V(s_t)`；原手写 GAE 使用重置前的
  `gamma * V(s_{t+1})`。本版使用标准库行为，没有自定义 PPO 来隐藏这个差异。
- 初始化、优势标准差的计算、采样顺序等也可能不同。
- 原版打印最后一次更新的损失；RSL-RL 通常打印所有 minibatch 更新的平均损失。
- 两种 checkpoint 结构不同，不能互相直接加载。

因此用这两版学习模块对应关系、观察任务表现；比较损失数值时应考虑上述差异。

## 验证范围

[test_rsl_rl.py](test_rsl_rl.py) 使用真实 RSL-RL 5.5.1 和原工程的 CPU 假物理测试夹具，验证：

- 相同动作对应相同推力、观测和奖励，且不修改采样动作。
- 失败与超时区分、局部重置、历史观测快照。
- 标准 RSL-RL 超时价值补偿。
- 两轮真实 PPO 更新、TensorBoard 输出、保存、重新加载和确定性推理。

这不验证 PhysX 动力学或 GUI。完整验证需在已有 Docker 镜像中运行上面的两轮短命令。
宿主机测试用的临时 Python 环境位于 `/tmp/cartpole-rsl-check`，没有修改 Docker 镜像或原 conda 环境。

在具有 RSL-RL、PyTorch、TensorDict、TensorBoard 的环境中，可从原工程目录运行：

```bash
python rsl_rl_version/test_rsl_rl.py
```

本版本按 5.5.1 固定接口；版本不一致时入口明确报错，避免混用旧 `ActorCritic` 和新 `MLPModel` 接口。
