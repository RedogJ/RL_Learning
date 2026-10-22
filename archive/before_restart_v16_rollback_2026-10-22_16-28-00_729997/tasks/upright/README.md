# upright：Go2 后腿直立

从原有四足站姿起身，让 RL、RR 两只后腿的**脚掌着地支撑，膝关节（小腿）离地**，
FL、FR 两只前脚离地，机身前向轴朝世界上方。
使用同一个 Go2 USD、PhysX、RSL-RL 与 `newton_gl`；本任务不改变其他任务的配置。
每回合仍从四足默认姿态开始，先等待 0.3 s 落稳再执行策略，不预先把机器人摆成直立姿势。

## 运行

在仓库根目录执行，无需重新 build：

```bash
# 无窗口短检查；会在本任务目录生成短测试模型
bash run.sh go2 train --task upright --num_envs 16 --max_updates 2 --run_name smoke

# 从零训练；默认 2048 个环境，暂不加入随机推力
bash run.sh go2 train --task upright --max_updates 3000 --run_name upright_v1

# 播放本任务最新完成的模型，支持右键拖拽施力
bash run.sh go2 play --task upright

# 从本任务已保存模型接着训练
bash run.sh go2 train --task upright --resume latest --max_updates 1000 --run_name upright_continue
```

`view --task upright` 只展示默认站姿 PD，没有策略起身行为；需要训练后使用 `play`。
本任务没有键盘移动指令。日志在 `logs/upright/<run>/`，不会自动选到 standing/running 模型。
目前续训只支持 upright → upright，不直接加载四足站立/运动模型。

## 奖励与方向

重力观测 `g_b` 是世界向下方向在机身坐标中的表示。
`alignment = -g_b[:, 0]` 是机身 +X 与世界向上的点积：

- 四足水平：`0`。
- 头朝上、后腿直立：`1`。
- 头朝下、前腿倒立：`-1`。

直接使用该方向，不依赖直立时会发生奇异的欧拉 roll/pitch。
### 配置分块

持续奖励/惩罚均为每秒的量，实际每控制步乘 `dt`；失败罚分一次扣除，不乘 `dt`，超时不扣。
所有权重与目标在 `reward_cfg.py` 中调整，`rewards.py` 负责公式，日志沿用 `Reward/<项名>`。

| 分块 | 配置项 | 作用 |
| --- | --- | --- |
| 存活 | `alive` | 基础收益 |
| 直立 | `upright_hold_weight / upright_sigma` | `weight * exp(-((1-alignment)/sigma)²)` |
| 高度 | `height_weight / target_height / height_sigma` | 根节点接近目标高度的指数收益 |
| 前后脚分工 | `rear_support_weight`、`front_lift_weight`、`front_foot_position_weight` | 后脚支撑、前脚离地并收在胸前 |
| 水平速度 | `balance_weight` | 世界线速度 `-weight * (vx² + vy²)` |
| 竖直速度 | `vertical_velocity_weight` | 世界线速度 `-weight * vz²`，抑制上下运动 |
| 角速度 | `angular_velocity_weight`、`pitch_velocity_weight` | 分别惩罚机身 X/Z 与 Y 轴，互不重复 |
| 后腿姿态 | `hip_rear_weight / hip_rear_sigma`、`rear_symmetry_weight / rear_symmetry_sigma` | 后髋回正与左右关节对称 |
| 非足部接触 | `undesired_contact_weight`、`rear_knee_contact_weight`、`hip_contact_weight` | 按部位互斥计分，见下文 |
| 足部滑动 | `foot_slide_weight` | 触地脚掌世界水平速度模长之和 |
| 动作 | `action_rate_weight`、`action_limit_weight` | 裁剪后动作变化、原始动作超界平方惩罚 |
| 失败 | `failure_penalty` | 跌倒终止的一次性扣分 |

高度、前脚目标、后腿姿态、接触和各项速度约束全程生效。
平衡项全部使用目标为 0 的速度误差平方惩罚，不乘直立程度 `hold`，不进行指数变换或 sigma 归一化。
`balance` 惩罚世界水平线速度，`vertical_velocity` 惩罚世界竖直线速度；
`angular_velocity` 惩罚机身 X/Z 轴角速度，Y 轴只在 `pitch_velocity` 中计一次。
四项每控制步均乘 `dt`。新增竖直速度权重初设为 30，可独立调整；设为 0 可关闭。
`horizontal_velocity_sigma` 移到配置顶部，仅用于旧版本 1/2 播放，不参与版本 3 及以上的训练公式。
这里的轴指机身坐标轴，使用真实角速度，并非欧拉角导数；直立时机身 X 轴朝世界上方。
不使用四足默认站姿的全关节姿态惩罚，也不要求四只脚一直触地。

### 前腿自然屈曲，收在胸前

`front_foot_position_weight` 控制前脚位置平方误差惩罚。目标使用机身 **link 原点**坐标系，
单位 m：X 朝头部、Y 朝左、Z 朝背部；不随世界位置、环境原点或机身姿态变化。
默认左前脚 `front_foot_target_left = (0.16, 0.142, -0.15)`，
右前脚 `front_foot_target_right = (0.16, -0.142, -0.15)`。
该目标接近本地 Go2 MJCF 的前髋 0、大腿 1.4、小腿 -2.4 rad 时的前脚位置，作为胸前屈曲的起点；
没有直接锁定关节角，后腿仍由策略自主控制。实际 USD 播放姿态还需训练后检查。

惩罚为 `-weight * mean(||前脚位置_b - 目标_b||², 两脚) * dt`，
全程生效，不乘起身进度。默认权重 1000，两脚各偏离目标 0.1 m 时每秒扣 10 分，0 可关闭。
目标在环境初始化时传到 GPU，世界点到机身坐标的变换按 xyzw 单位四元数计算。
完整日志记录 `Reward/front_foot_position`、`Task/front_foot_position_error_m`，
以及 `Task/front_foot_FL_error_m` / `Task/front_foot_FR_error_m`（单位 m）。
配置和目标会随训练历史保存；旧模型播放缺少此项时权重恢复为 0，续训采用当前配置。
观测保持 52 维，目标是当前任务内的固定配置，可从已有 upright 检查点继续训练。

### 后腿姿态与脚部目标为何保留多项

`hip_rear_weight` 奖励两个后髋分别接近 0，取两腿指数收益的乘积。
`rear_symmetry_weight` 奖励后腿左右对称，误差定义为
`RMS(hip_RL + hip_RR, thigh_RL - thigh_RR, calf_RL - calf_RR)`，
收益为 `weight * exp(-(error / rear_symmetry_sigma)²)`。
髋外撇在左腿为负、右腿为正；大腿/小腿同轴，对称时左右角相等。
两项目标不同：左右同时大幅外撇也可以对称，因此仍需后髋回正。

`rear_support_weight` 逐物理帧、逐后腿判断：只有脚掌向上支撑、且同侧膝部没有向上支撑时才给分。
脚掌和膝部均以世界向上接触力大于 `contact_force_threshold` 判断支撑。
两只后腿的有效支撑比例取均值，再乘权重和 `dt`；两脚单独支撑时拿满，
两膝共同支撑时该项为 0，只有一侧膝部支撑时仅取消该侧收益。
同一控制周期内脚掌和膝交错支撑，不会把脚掌独立支撑的物理帧一并取消。
膝部侧向接触若没有向上支撑，不取消脚掌收益，但仍触发 `rear_knee_contact`。
后小腿接触惩罚与此项独立计算，不存在膝撑地同时领取该腿支撑收益的情况。
后脚支撑、前脚离地和前脚目标各衡量不同条件。后小腿接触仍由下述接触惩罚约束。

### 非足部接触：按部位只计一次

新公式把非脚掌刚体分成三组：

| 项名 | 部位 | 本次整理后的每个接触刚体罚分率 |
| --- | --- | --- |
| `undesired_contact` | 机身、头部、大腿、前小腿 | 60/s |
| `hip_contact` | 四个髋部 | 120/s |
| `rear_knee_contact` | 两个后小腿，不包含脚掌 | 200/s |

这些数值对应本次整理时的配置，之后以 `reward_cfg.py` 为准。
髋部原来同时扣通用 60 和额外 60，合并为单项 120，保持总强度。
后小腿原来向上支撑扣 200、侧向接触扣 200+60；现在统一扣 200，去掉隐藏叠加。
专用权重设为 0 会完全关闭该部位的接触惩罚，不会重新落入通用项。

任一物理帧的接触法向合力模长超过 `contact_force_threshold` 即计入本控制周期，
同一刚体的多个碰撞几何/接触点只计一个。传感器未按碰撞对象筛选，因此也包含自碰撞。
`Contact/undesired_body_count` 仍描述传感器统计的所有非足部接触数；
奖励中 `undesired_contact` 则扣除了髋部和后小腿。
日志可分别查看 `Reward/undesired_contact`、`Reward/hip_contact`、`Reward/rear_knee_contact`，
以及 `Contact/hip_body_count`、`Contact/rear_knee_contact_count`。

`foot_slide_weight` 惩罚触地脚掌的世界 XY 速度模长之和，空中摆腿不扣，竖直速度不计。
动作平滑项约束相邻控制步的动作变化，超界项约束网络原始输出超出 [-1, 1] 的部分，二者独立保留。

### 旧模型与继续训练

`reward_schema_version` 自动随配置保存：新训练使用版本 4；版本 1/2 播放保留原来的指数水平速度收益与角速度门控，
无版本字段的旧模型按版本 1 加载并保留原接触叠加公式；缺少竖直速度权重时按 0 加载。已有日志与训练历史不重写。已删除的膝部高度收益配置在旧模型加载时忽略。
版本 1–3 播放仍按原脚掌支撑比例计分。
从旧模型继续训练仍使用当前配置和版本 4，并按现有机制继承历史、记录新阶段配置。
观测/动作维数没有改变，无需重新 build 镜像。

## 重置与观测

训练低于 0.14 m 或机身 +X 偏离世界向上超过 120° 时失败，15 s 超时重置。
四足水平对应相对目标 90°，正常直立为 0°，二者都不会因倾角被误重置。
播放沿用共用入口：关闭高度失败，仅保留该任务的方向失败与超时重置。

观测仍为 52 维、动作仍为 12 维完整机械范围映射。
相对四足水平姿态的三个误差积分在本任务中关闭，保留为零占位。
随机化配置独立，初期默认关闭；学会起身后可通过 `--randomization` 开启较轻的推力。

## 日志

- `upright`：机身前向轴与世界向上的点积，接近 1 表示直立。
- `rear=…/2`：有效后脚支撑数，理想为 2；版本 4 排除同侧膝部同时支撑的物理帧。
- `front=…/2`：两只前脚平均支撑数，理想为 0。
- 完整日志另有 `Contact/rear_knee_support_count`（向上支撑的后膝数），以及 `Task/calf_RL/RR_rad`。
- 完整日志中的 `Task/hip_abs_angle_rad` 是四个髋侧摆角的平均绝对值；另有按 `FL/FR/RL/RR`
  分开的有符号 `Task/hip_<leg>_rad`，用于检查左右是否对称。四髋同轴，外撇在左腿为负、右腿为正。
- 共用的 `pitch` 在直立时可能接近 90°，不是姿态错误；应结合 upright、rear、front 判断。

奖励、PPO、终止、扰动与指令参数分别在本目录的对应配置文件中，训练历史和检查点按原有机制保存。
旧模型配置中的 `upright_progress_weight` 在播放加载时忽略，原始训练历史保留；续训使用当前奖励。
