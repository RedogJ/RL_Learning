# Go2：Isaac Lab → MuJoCo

独立的小工程：加载已有 Actor，在 MuJoCo 的纯平地中推理，不依赖 Isaac Sim、Docker 或 RSL-RL。
模型来自宇树官方 `unitree_mujoco`，固定版本与文件校验记录在资源目录的 `download_manifest.json`。
官方 BSD 3-Clause 许可证随资源保留。

## 运行

在仓库根目录运行，使用已装好 MuJoCo / PyTorch 的 Python 环境（当前机器为 `sim_env`）：

```bash
conda activate sim_env
bash run.sh go2-sim2sim play
```

首次模型资源已下载。迁移到其他电脑时可执行：

```bash
bash run.sh go2-sim2sim fetch-model
```

只下载 Go2 MJCF 及引用的 16 个 OBJ 网格，约 27.1 MiB；校验通过的文件不重复下载。
默认权重固定为：

```text
projects/go2_learning/logs/running/2026-10-17_13-12-58_702879_running_16/model_56500.pt
```

它与同目录 `latest.pt` 的 Actor 张量完全相同。权重和 `run.json` 需要一同保留。
其他权重可以显式指定：

```bash
bash run.sh go2-sim2sim play --checkpoint "projects/go2_learning/logs/running/某次训练/model_XXXXX.pt"
```

不需要 sudo。可用 `GO2_SIM2SIM_PYTHON=/完整路径/bin/python` 指定解释器。

## 键盘与窗口

点击 MuJoCo 窗口后：

| 按键 | 行为 |
| --- | --- |
| 按住 ↑ | 前进，默认 +1.5 m/s |
| 按住 ↓ | 后退，默认 −1.5 m/s |
| 按住 ← / → | 左转 / 右转，默认 ±0.6 rad/s，可与前后组合 |
| 松开方向键 / 窗口失焦 | 指令归零，策略尝试停稳 |
| 空格 | 速度指令归零，停车 |
| R | 从默认站姿重置 |

方向键按住生效，松开停车。通过 X11/XWayland 查询实际键位并检查仿真窗口焦点，
不依赖按键重复事件；GUI 模式以键盘实时输入为准，`--vx/--vy/--yaw` 用于无窗口固定指令测试。
播放保留 15 s 超时与 60° 倾斜重置，不按机身高度重置；阈值取自所选模型配置。
目前只对站立、前进、后退和停车做了闭环验证，转向仍需观察（源任务训练的 yaw 指令为零）。

## 无窗口诊断

```bash
# 站立 / 前进 / 后退；duration 单位为仿真秒，无窗口运行不按墙钟限速
bash run.sh go2-sim2sim test --duration 20
bash run.sh go2-sim2sim test --duration 20 --vx 1.5
bash run.sh go2-sim2sim test --duration 20 --vx -1.5

# 只检查默认站姿的 PD，不执行神经网络
bash run.sh go2-sim2sim test --pd-only --duration 12

# 逐控制步记录 52 维观测、原始动作、目标角、实际关节状态与力矩
bash run.sh go2-sim2sim test --duration 5 --record

# 接入回归测试（需要本地参考权重与机器人资源）
PYTHONPATH=projects python -m unittest go2_sim2sim.tests.test_adapter
```

每次运行保存 `logs/<北京时间>_<run-name>/`：

- `run.json`：模型 SHA256、原始训练配置、迁移参数。
- `metrics.jsonl`：每秒状态。
- `summary.json`：翻倒、超时、平均姿态、速度误差。
- `trace.jsonl`：仅开启 `--record` 时逐控制步记录。

`--no-auto-reset` 可用于连续诊断；`--physics-hz` 必须为策略频率的整数倍。
`logs/` 与原工程一样由 Git 忽略。

## 接口约定

Actor 是无观测归一化的 `52 → 128 → 128 → 12` Tanh MLP，输出使用高斯均值，不采样动作。
只加载 `actor_state_dict` 的 `mlp.*`；Critic、优化器和探索标准差不参与推理。

| 观测索引 | 数据 |
| --- | --- |
| 0:3 | 根刚体质心的机身坐标线速度 |
| 3:6 | 机身坐标角速度 |
| 6:9 | 单位重力方向，机身坐标 |
| 9:21 | 实际关节角减默认关节角 |
| 21:33 | 关节速度 × 0.1 |
| 33:45 | 上一控制步裁剪到 [-1,1] 的动作 |
| 45:48 | vx、vy、yaw-rate 指令 |
| 48:49 | 根 link 高度减目标高度 |
| 49:52 | roll / pitch / height 衰减积分，按各自限幅归一化 |

关节通过名称映射到 `run.json` 顺序，不能使用 MJCF 的电机顺序直接索引。
动作映射为 `center + half_range * clip(action,-1,1)`，沿用保存的机械上下限。
每回合从 0.4 m 出生，前 0.3 s 维持默认站姿，不更新策略或积分；之后每个控制周期结束更新积分。

## 动力学对齐与边界

- 策略 30 Hz；MuJoCo 物理 600 Hz，每个动作保持 20 个子步，子步内重算 PD。
- `Kp=25`、`Kd=0.5`，所有关节力矩上限统一为 ±23.7 N·m。
- 碰撞地面摩擦 0.5，禁用机器人自碰撞。
- 保留官方 MJCF 的关节被动参数：`damping=0.1 N·m·s/rad`、`frictionloss=0.2 N·m`、
  `armature=0.01 kg·m²`；加载器不覆盖这些值，每次运行将实际值保存到 `run.json`。
- 机器人质量、惯量和碰撞几何来自固定版本的官方 MJCF；尚未逐项与二进制 USD 数值比对。
- MuJoCo 使用显式计算的 PD 力矩、`implicitfast` 积分器；Isaac 使用 PhysX 隐式关节驱动，数值行为有差异。
- 高度与线速度来自仿真真值；这份接口用于 sim2sim，尚不是实机状态估计接口。

未修改源训练配置或重新训练策略。测试通过表示这份参考策略已能在这些条件下运行，不能保证任意权重、扰动、地形均能迁移。

## 本机验证记录（2026-10-17）

首次验证使用上述固定权重、MuJoCo 3.7.0、600 Hz 物理 / 30 Hz 控制。
以下记录对应恢复 MJCF 被动参数之前（当时三项均被加载器清零）：

| 测试 | 仿真时长 | 翻倒次数 | 平均水平速度误差 |
| --- | --- | --- | --- |
| 默认站姿 PD | 12 s | 0 | — |
| Actor 零指令站立 | 15 s | 0 | 0.0027 m/s |
| Actor 前进 +1.5 m/s | 20 s | 0 | 0.1592 m/s |
| Actor 后退 −1.5 m/s | 20 s | 0 | 0.1385 m/s |

误差统计包含策略启动阶段；20 s 测试各有一次正常的 15 s 回合超时重置。
额外的 14 s 连续站立→前进→停车→后退→停车测试没有翻倒，两次停车后速度均低于 0.1 m/s。
8 项自动检查覆盖权重矩阵推理、52 维拼接、原工程积分公式、质心坐标变换、关节/电机映射、
动作范围、重置等待、力矩限幅和键盘指令。原生窗口已经完成 3 s 打开、仿真、正常退出检查。

恢复 MJCF 的 `damping=0.1`、`frictionloss=0.2`、`armature=0.01` 后，同日复测：

| 测试 | 仿真时长 | 翻倒次数 | 平均水平速度误差 |
| --- | --- | --- | --- |
| Actor 零指令站立 | 20 s | 0 | 0.0024 m/s |
| Actor 前进 +1.5 m/s | 20 s | 0 | 0.1623 m/s |

两项各包含一次正常超时重置；8 项接入检查和连续指令切换测试也通过。

## 文件

| 文件 | 内容 |
| --- | --- |
| `policy.py` | 固定 Actor 加载、检查点参数与动作映射 |
| `robot.py` | 官方模型、平地、关节/电机映射及物理参数 |
| `observations.py` | 52 维观测和积分 |
| `env.py` | 重置、物理子步、PD、终止判断 |
| `main.py` | 键盘、窗口、无窗口测试与日志 |
| `config.py` | 路径及 PD/频率常量 |
| `fetch_model.py` | 固定版本模型下载与校验 |
