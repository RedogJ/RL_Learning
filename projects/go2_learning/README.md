# Go2 Learning：按任务组织的 Isaac Lab / RSL-RL 工程

目前有 **`standing`：四足站立与抗扰动**、**`running`：速度跟踪与键盘控制**，
以及 **[`upright`](tasks/upright/README.md)：后腿支撑直立**。
各任务在 `tasks/` 下维护独立配置和奖励。
使用现有 `isaaclab-base:local` Docker 镜像、本地 Go2 USD、PhysX 物理和 `newton_gl` 轻量窗口。

## 目录与职责

```text
go2_learning/
├── main.py                  # 选任务，创建环境、Runner，训练或播放
├── paths.py                 # 资源与日志位置
├── simulation/              # 共用仿真
│   ├── env.py               # VecEnv：观测、动作、物理推进与重置
│   ├── config.py            # 环境参数类型；任务给出具体数值
│   ├── scene.py             # Go2、地面、灯光及关节 PD
│   ├── contacts.py          # 足部 / 非足部接触检测
│   └── randomization.py     # 执行初始关节扰动与随机推力
├── control/                 # 共用控制状态
│   ├── commands.py          # 指令采样与键盘指令更新
│   └── attitude_integral.py # 观测中的积分状态
├── tasks/
│   ├── __init__.py          # 任务注册表 TASKS / get_task()
│   ├── base.py              # 任务接口
│   ├── reward_terms.py      # 共用角度 / 动作张量计算
│   ├── standing/
│   │   ├── task.py          # 奖励调用、失败判断、回放配置恢复
│   │   ├── config.py        # 组合配置、环境时序、结束条件与默认开关
│   │   ├── reward_cfg.py    # 奖励权重、sigma、目标高度
│   │   ├── rewards.py       # 奖励公式
│   │   ├── command_cfg.py   # 速度指令参数
│   │   ├── agent_cfg.py     # 网络、PPO、rollout 和保存参数
│   │   ├── randomization_cfg.py # 初始扰动、推力大小 / 时间 / 作用点
│   │   └── integral_cfg.py  # 积分限幅与衰减
│   ├── running/             # 相同配置分组，独立实现速度跟踪
│   └── upright/             # 后腿支撑直立，独立奖励与终止条件
├── training/
│   ├── checkpoints.py      # 按任务查找、核对模型
│   ├── history.py          # 保存和继承训练阶段配置
│   ├── log.py              # 精简终端日志
│   └── settling.py         # 从 PPO / GAE 中排除落地等待样本
├── ui/
│   ├── mouse.py            # 右键施力与箭头
│   ├── height_display.py   # GL 左侧栏机身高度读数
│   └── keyboard.py         # 键盘事件与失焦清零
├── tests/                  # CPU 回归检查
├── run.sh                  # 运行方式；调用仓库共用 Docker 启动器
└── logs/                   # 原有记录；新记录在 <task>/<run>/，Git 忽略
```

**调任务参数进入对应的 `tasks/standing/`、`tasks/running/` 或 `tasks/upright/`**；共用物理、机器人和地面在 `simulation/`。
仓库根目录的 `run.sh` 负责选择工程，`scripts/run-isaac-project.sh` 负责 Docker 环境。
运行时使用镜像中的 Isaac Lab；修改本工程文件后下次启动生效，无需重新 build。

## 任务怎样执行

`train/play/view` 表示运行方式，`--task standing` 表示运行的任务，二者独立。
不写 `--task` 时默认为 `standing`，原命令仍然可用。

```text
main.py 选择任务 → get_task("standing") → StandingTaskCfg
                         ↓
RSL-RL Runner → env.step(actions)
                 ├── 写入关节目标、推进物理、读取状态
                 ├── task.failures(state)
                 ├── task.reward_components(state, actions, failed, dt)
                 └── 处理超时、逐环境重置、返回观测与奖励
```

奖励及失败条件由任务决定。`simulation/env.py` 提供共用 Go2 状态和 52 维观测，负责仿真时序。
两任务共用观测、动作映射与仿真流程；速度目标和奖励分别由任务定义。

## 在哪里调站立参数

| 要修改的内容 | 文件 |
| --- | --- |
| 奖励权重、sigma、参考高度 | [tasks/standing/reward_cfg.py](tasks/standing/reward_cfg.py) |
| 奖励具体公式 | [tasks/standing/rewards.py](tasks/standing/rewards.py) |
| 失败判断的实现 | [tasks/standing/task.py](tasks/standing/task.py) |
| 失败阈值、回合长度、等待时间、频率、默认环境数、默认扰动开关 | [tasks/standing/config.py](tasks/standing/config.py) |
| 目标速度、指令采样方式 | [tasks/standing/command_cfg.py](tasks/standing/command_cfg.py) |
| PPO、网络、每轮采样步数 | [tasks/standing/agent_cfg.py](tasks/standing/agent_cfg.py) |
| 随机推力大小、持续时间、间隔和作用点 | [tasks/standing/randomization_cfg.py](tasks/standing/randomization_cfg.py) |
| 积分限幅和衰减 | [tasks/standing/integral_cfg.py](tasks/standing/integral_cfg.py) |
| 参考关节角、PD、物理关节限幅、地面 | [simulation/scene.py](simulation/scene.py) |

每次 `get_task()` 都创建独立配置；不同任务及不同运行实例可以分别调参。

## 运行站立任务

在 `RL_Learning` 根目录执行，无需重新构建镜像：

```bash
# 查看场景：默认站姿 PD，轻量窗口；无神经网络参与
bash run.sh go2 view --task standing

# 无窗口短检查；这个命令会生成一个短测试模型
bash run.sh go2 train --task standing --num_envs 4 --max_updates 2 --run_name smoke

# 从头训练
bash run.sh go2 train --task standing --max_updates 1000 --run_name stand

# 继续该任务最近完成的训练
bash run.sh go2 train --task standing --resume latest --max_updates 1000 --run_name continue_stand

# 播放该任务最近完成的模型
bash run.sh go2 play --task standing

# 播放时启用随机扰动
bash run.sh go2 play --task standing --randomization

# 关闭训练中的扰动
bash run.sh go2 train --task standing --no-randomization --run_name baseline
```

`standing` 的训练默认启用初始关节随机化和随机推力；view/play 默认关闭。
显式 `--randomization`、`--no-randomization` 优先于任务默认值。
该开关同时控制初始关节角扰动与随机推力。强度以任务配置文件为准。
鼠标右键拖拽在 view/play 窗口可用，点击点施加弹簧阻尼力并显示箭头，上限 150 N。
`newton_gl` 左侧栏顶部显示 `Go2 height | env 0` / `Height: ... m`，每秒刷新 10 次。
数值为环境 0 的机身根节点世界 Z 减去环境原点 Z，与奖励中的 `height` 定义一致；
不是足底距离，也不是机身到崎岖地表的净空。仅 view/play 启用，不增加训练时的数据回传。
随机推力不绘制箭头；其局部作用点随刚体移动，通过施力接口自动产生相对质心的力矩。

键盘检测可使用 `view --keyboard_test`。站立任务尚未开放 `play --keyboard`，
站立速度奖励仍以静止为目标；方向键控制请使用 `running` 任务。

## 运行速度跟踪任务

```bash
# 训练自动采样目标速度，不用键盘
bash run.sh go2 train --task running --max_updates 1000 --run_name running_v1

# 播放默认启用方向键：上下前后，左右转向，松键停车
bash run.sh go2 play --task running
```

速度范围、奖励公式和调参位置见 [running 说明](tasks/running/README.md)。
running 模型与 standing 模型按任务区分，日志放在 `logs/running/`。

## 模型、日志与旧版本兼容

新记录存放在：

```text
logs/standing/<日期时间_运行名称>/
├── run.json                 # 任务、网络、奖励、环境和扰动配置
├── history.json             # 各轮数区间对应的任务与训练配置
├── training_details.log     # RSL-RL 完整日志
├── events.out.tfevents.*     # TensorBoard
├── model_*.pt               # 定期检查点
└── latest.pt                # 本次训练完成时的模型
```

- 播放的 `latest` 只在所选任务目录及旧扁平布局中查找匹配任务的模型。
- 续训的 `latest` 优先使用本任务模型；尚无 running 模型时，可以使用 `logs/running/` 中的 standing 种子。
- 旧站立记录没有 `task_name` 时，按已知的站立版本识别；原有日志随工程整体移动，文件内容不重写。
- `latest` 代表最近完成，**不是表现最好的模型**。短检查也会生成模型；做对照实验应固定具体 `.pt` 来源。
- `play` 使用该模型保存的奖励、网络、环境与结束配置；旧记录缺失的新奖励按原兼容规则关闭。
- `--resume` 恢复 Actor/Critic 权重与轮号，优化器重新初始化，奖励、PPO、任务参数使用当前代码。
- 继续训练继承历史，并把新阶段的任务名、奖励、环境、结束条件和扰动配置一起保存。
- 播放要求任务与检查点一致；续训另支持 standing → running，仍核对网络结构和关节映射。
- 新记录保存来源任务和相对日志路径；旧 `logs/<run>/` 来源链按任务定位到现在的目录。
- 播放的 `--randomization` 使用当前任务的扰动参数，训练时的原配置保存在日志中供核对。

可以指定新旧模型的具体路径：

```bash
bash run.sh go2 play --task standing --checkpoint "projects/go2_learning/logs/原训练目录/latest.pt"
bash run.sh go2 train --task standing --resume "projects/go2_learning/logs/standing/某次训练/latest.pt" --run_name branch
```

## 共用环境的约定

- 站立任务默认物理 120 Hz、控制 30 Hz；每次动作保持 4 个物理步。
- 52 维观测：机身线速度 3、角速度 3、重力方向 3、关节角偏移 12、缩放关节速度 12、上一动作 12、速度指令 3、高度误差 1、积分观测 3。
- 12 维动作映射完整机械关节范围：`q_target = midpoint + half_range * clamp(action, -1, 1)`。
  零动作是机械中点；参考站姿对应的动作通常非零。
- 每回合从默认根位姿、速度和关节状态重置，等待 0.3 秒；等待期间不执行策略、不计奖励或回合时间。
  各环境独立等待，等待样本不进入 PPO / GAE / 观测归一化。
- 站立任务默认低于 0.18 m 或倾斜超过 60° 判失败；15 秒正常超时。
  这些规则现在位于任务配置与 `task.py`，尚未改为摔倒恢复任务。
- 共用场景使用本地棋盘地面，物理平面 z=0，摩擦 0.5、恢复系数 0。

## 奖励与日志

站立任务保留姿态、高度、零速度和四足支撑正奖励，以及姿态角速度、关节站姿、髋角、动作变化等正则。
参考站姿为髋 0、大腿 0.8、小腿 -1.6 rad。姿态惩罚使用实际关节角，髋角也包含在全部关节站姿项内。

任务收益与常规正则可以启用非负截断；原始动作超界、非足部接触和失败罚分独立扣除。
失败项不乘 dt；其他项按奖励率乘控制周期。所有 `Reward/*` 分项之和等于实际环境奖励。

足部支撑按每个物理步的向上接触力判断，对本控制周期的四足支撑比例求平均。
站立指令时四足持续接触收益最高；有移动/转向指令时该项关闭，阈值在任务奖励配置中。
非足部接触按控制周期内法向合力峰值计数；同一小腿多个接触点只算一个刚体。

精简终端保持每轮一行：

- `episode`：近期完成回合的平均正式控制时间，不含等待阶段；长回合未结束前不会进入这个平均值。
- `feet=…/4`：训练样本中物理帧上的平均承重脚掌数，与单台确定性回放不一定相同。
- `speed` / `spin`：世界水平速度大小 / 世界竖直轴角速度绝对值。
- `raw_max` / `clip` / `excess`：原始动作峰值、超界比例、超界平方和。
- `top`：负惩罚项中的最大贡献及占比，不代表失败环境比例或 PPO 梯度占比。

常规训练默认每 24 个控制步采样一次诊断指标（各任务 `agent_cfg.py` 的
`diagnostics_interval`）。奖励、失败次数和回合收益/长度仍完整累计；诊断均值和
各奖励贡献是抽样统计。回合数据在 GPU 上累计，每轮集中回传 CPU。

`--verbose_logs` 恢复每步诊断，并开启各关节角、详细接触指标和完整终端表格。
逐步 NaN 检查默认关闭，数值排错时将 `agent_cfg.py` 中的 `check_for_nan` 改为 `True`。
这些开关不改变物理步长、接触检测、随机推力、奖励或逐回合等待规则，无需重新 build。

## 后续添加任务

1. 新建 `tasks/<任务名>/`，分别定义自己的配置和奖励公式。
2. 实现 `Task` 接口：奖励、失败判断、参考高度、接触传感器需求及回放配置读取。
3. 在 `tasks/__init__.py` 的 `TASKS` 注册类路径。
4. 用 `--task <任务名>` 选择；新日志自动分到该任务目录。

任务可以复用指令采样、扰动、观测和动作执行；任务目标与参数不放进共用 `simulation/env.py`。
如需改变观测、动作或机器人本体，必须同时定义模型兼容规则并验证，不能直接套用旧权重。

## CPU 回归检查

在装有 torch 的 Python 环境执行：

```bash
python -m unittest discover -s projects/go2_learning/tests -t projects -q
```

从仓库根目录执行。检查覆盖配置隔离、任务分派、旧模型读取、按任务查找 latest、历史继承、等待/重置、奖励和随机作用点。
未安装 RSL-RL / TensorDict 时，相应真实 PPO 接入检查会跳过；实际 GPU 仿真仍需在现有 Docker 环境验证。

仓库整体布局及旧路径对照见 [目录说明](../../docs/structure.md)。
