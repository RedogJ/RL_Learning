# Go2：场景与 RSL-RL 入门

独立工程，使用现有 `isaaclab-base:local` 镜像和本地官方 Go2 USD。
物理后端为 PhysX，默认可视化为 `newton_gl`，不打开 Kit 编辑器完整界面。
PhysX 运行仍可能输出 Isaac Sim / Kit 的底层初始化日志。

当前阶段试验**纯站立**：奖励使用姿态、高度、水平零速度、竖直轴零转速、机身 X/Y 角速度、四足支撑及非足部接触约束，不使用误差积分奖励；观测保留姿态与高度积分。所有速度指令为零，训练默认启用随机扰动，view/play 默认关闭，
四条腿采用一致的默认关节角（髋 0、大腿 0.8、小腿 -1.6 rad）。
环境数由 `main.py` 的默认值或 `--num_envs` 指定，每个回合最长 15 秒仿真时间。

## 先看一台 Go2

在 `RL_Learning` 目录执行：

```bash
bash go2_learning/run.sh
```

场景包含 z=0 的官方棋盘地面和一台 Go2。地面使用本地 `GroundPlaneCfg` 资源，
包含静态碰撞平面、100 m × 100 m 可视网格和贴图；静/动摩擦系数均为 0.5，
恢复系数为 0，可在 `scene.py` 调整。机身从 z=0.4 m 放下，
12 个关节由 PD 控制器保持预设站姿。这个模式不加载策略、不训练，也不定时重置。
关闭窗口或按 Ctrl+C 退出。

### 右键施力

在 `view` 或 `play` 的 `newton_gl` 窗口内，**在 Go2 表面按住右键拖动**。
直接调用 Newton Viewer 自带的 `Picking.pick()`，对窗口显示的模型及其状态
进行拾取；用官方 `Picking.update()` 更新同一深度下的鼠标目标，处理显示偏移。
选中对应的机身或腿部刚体，在点击点施加外力。点击空白、地面或 UI 不会给机器人施力。
点击点通过弹簧阻尼力被拉向鼠标目标：`F = k(target - point) - d * point_velocity`。
物体越靠近目标，弹簧力越小；不再按固定的“像素位移→恒定推力”计算。
总力上限 150 N。松开右键、鼠标离开窗口或窗口失去焦点，停止施力。
施力时出现橙红色箭头，从点击点出发并跟随该刚体平移和旋转，方向与外力一致；
50 N 对应 0.5 m 箭头长度。停止施力后箭头消失。

实际外力也作用在这个点，偏离质心时产生力矩 `τ = r × F`。
物理求解仍为 PhysX。官方拾取点通过刚体路径映射到 PhysX，并传入实际施力位置；
箭头使用同一个位置和外力，额外考虑 Viewer 的多环境显示偏移。
`view` 使用 PD 站姿，可能被推倒，不会自动恢复；重新启动可恢复初始状态。
`play` 使用训练策略并保留环境自动重置。无窗口训练不启用鼠标交互。
弹簧刚度、阻尼、力上限在 `interaction.py` 的 `stiffness`、`damping`、`max_force` 修改。
箭头长度比例由同文件的 `arrow_meters_per_newton` 控制，不影响实际力的大小。
箭头刷新上限为 30 Hz，重复使用绘图缓冲区；鼠标事件合并到下一物理步处理。
物理和外力计算仍为 120 Hz。首次拾取可能触发 Warp 内核编译；持续卡顿需进一步测量。

官方实现参考：Newton v1.6.0 的
[Picking](https://github.com/newton-physics/newton/blob/v1.6.0/newton/_src/viewer/picking.py)、
[ViewerGui](https://github.com/newton-physics/newton/blob/v1.6.0/newton/_src/viewer/viewer_gui.py)
和 [力计算内核](https://github.com/newton-physics/newton/blob/v1.6.0/newton/_src/viewer/kernels.py)。
本工程复用已安装库的拾取逻辑，PhysX 侧使用本工程的弹簧增益与 150 N 上限。

无窗口检查（运行 120 个控制步，即 4 秒仿真时间）：

```bash
bash go2_learning/run.sh view --viz none --steps 120
```

## 文件与执行顺序

### 方向键输入检测

```bash
bash go2_learning/run.sh view --keyboard_test
```

点击 `newton_gl` 仿真窗口使其获得键盘焦点，按上下左右键，查看启动终端中的
`[键盘检测]` 输出。按下、松开和失焦清空都会打印；持续按住不重复刷屏。
支持同时按住两个方向键，同轴相反方向相互抵消；全部松开后两轴归零。
切换到终端等其他窗口时也清零，避免漏掉松键事件后一直保持旧输入。
方向信号为 `(上下, 左右)`，上/左为 +1，下/右为 -1。`--keyboard_test` 只显示信号，
不发送速度指令。也可在 `play --keyboard_test` 中检测，此时策略收到零速度目标。
实际键盘控制使用下文的 `play --keyboard`。
无需重新构建镜像。实现采用 pyglet 的
[窗口按键事件](https://pyglet.readthedocs.io/en/latest/programming_guide/keyboard.html)。

### 模块职责

| 文件 | 职责 |
| --- | --- |
| `scene.py` | 本地 USD 路径、地面、Go2 初始站姿、关节 PD |
| `contacts.py` | 按刚体名称区分足部与非足部，读取控制周期内的接触力历史 |
| `env.py` | 实现 RSL-RL `VecEnv`：观测、动作、奖励、结束和重置 |
| `settling_training.py` | 过滤逐回合等待样本，沿用标准 PPO 优化并修正回合时长统计 |
| `training_history.py` | 继承和截断训练历史，随每个检查点保存轮数区间和奖励配置 |
| `interaction.py` | 轻量窗口右键事件和机身外力 |
| `keyboard.py` | 方向键按下/松开状态、失焦清零、输入检测输出 |
| `commands.py` | 目标速度参数、训练时独立随机采样、播放时将按键转成速度指令 |
| `attitude_integral.py` | 每个环境的 roll/pitch/height 积分、衰减、独立限幅、归一化观测及重置 |
| `randomization.py` | 随机化参数、初始关节角采样、每台机器人的定时水平推力 |
| `rewards.py` | 存活、roll/pitch、高度、关节站姿、髋角、动作、动作变化与跌倒奖惩 |
| `training_log.py` | 每轮一行训练摘要，原始日志长表保存到文件，保留 TensorBoard 指标 |
| `agent_cfg.py` | 标准 MLP、PPO、采样参数 |
| `main.py` | 创建环境；view 看场景，train/play 使用 `OnPolicyRunner` |
| `run.sh` | Docker 挂载、缓存、显示和 Python 启动 |

数据流：`OnPolicyRunner → env.step(actions) → 关节位置目标 → PhysX → 观测和奖励`。
这里使用项目自己的环境代码，没有继承宇树现成任务配置。

### 阅读 `env.step()`

`step()` 只串联流程，具体实现仍集中在同一个 `env.py`：

| 方法 | 职责 |
| --- | --- |
| `_set_action()` | 动作限幅，写入关节目标 |
| `_advance_simulation()` | 推进 4 个物理步，每步写入控制、推进仿真、更新状态 |
| `_read_state()` | 读取奖励和结束条件需要的状态 |
| `_compute_termination()` | 判断跌倒与超时，跌倒优先 |
| `_compute_reward()` | 调用独立的速度/姿态奖励公式 |
| `_make_extras()` | 返回超时标志和日志指标 |
| `_reset_finished_envs()` | 重置本步结束的环境 |
| `get_observations()` | 构造 52 维观测，含瞬时高度误差和 3 维归一化姿态/高度积分 |

奖励和结束标志使用重置前的状态；自动重置后返回新一轮初始观测。
`view` 的 `auto_reset=False` 保留本步实际状态。

每个回合重置时，从 `scene.py` 的出生位姿和默认零速度重新开始，
保持默认站姿等待 0.3 秒仿真时间（30 Hz 下为 9 个控制步，共 36 个物理步）。
等待期间不执行策略、不计算奖励/跌倒，不更新误差积分和回合计时，也不施加外力。
每台狗独立计时，其他正在控制的环境正常运行；不缓存或恢复之前落地时的速度和状态。
训练、播放及 view 首次启动先完成第一回合的等待。
`settling_training.py` 在标准 RSL-RL PPO 外增加等待掩码：只对活动环境做策略推理，
更新前过滤等待样本，GAE、优势归一化及观测归一化都只使用正式控制样本。
日志的 episode 长度也只累计正式控制步，不包含等待的 0.3 秒。
参数在 `env.py` 的 `reset_settle_seconds` 中，数值会记录进 `run.json`。
0.3 秒是固定等待时长，不保证此时绝对静止；结束后按实际当前状态接管，不强行清零物理速度。
策略接管后的跌倒仍正常计为失败。

鼠标交互由 `main.py` 接入两个可选回调（这里赋值的是函数，不是调用结果）：

```python
env.before_physics_step = mouse_push.apply
env.on_reset = mouse_push.clear
```

环境负责在每个物理步前调用外力回调、在重置时调用清理回调。
训练时这两个回调默认是 `None`。
随机化是独立的 `DomainRandomizer` 对象，环境仅在重置时调用 `reset()`、
每个物理步调用 `apply()`。随机推力在鼠标回调之后叠加，二者可以同时启用。

### 初始姿态与外力随机化

训练（包括续训）默认开启随机化，view/play 默认关闭。用 `--randomization` 显式开启，
或用 `--no-randomization` 显式关闭，两种参数均优先于默认值。参数集中在 `randomization.py` 的 `RandomizationCfg`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `joint_position_noise` | 0.1 rad | 每次重置，12 个关节各自从默认站姿 ±0.1 rad 均匀采样，并限制在关节软限位内 |
| `push_force_range` | 40～100 N | 水平合力大小均匀采样，世界 XY 平面方向均匀采样；较原 20～50 N 加倍 |
| `push_interval_range` | 2～4 s | 重置后，以及每次推力结束后的随机等待时间 |
| `push_duration` | 0.4 s | 每次保持相同力的持续时间，默认 48 个物理步 |
| `push_body` | `base` | 受力刚体为机身 |
| `push_point_half_extent` | `(0.15, 0.04, 0.04)` m | 围绕机身质心，沿机身前后/左右/上下轴的随机作用点偏移半范围 |

每台机器人独立采样、计时；跌倒重置只清空该环境的推力和计时器。
每次推力开始时独立采样大小、水平世界方向和机身局部作用点；0.4 秒内保持同一力和局部点。
局部点沿机身轴在质心周围前后 ±15 cm、左右 ±4 cm、上下 ±4 cm 的盒内均匀采样，
用于产生不同力臂，不是对 USD 表面的精确拾取。设 `push_point_half_extent=(0, 0, 0)` 可恢复质心施力。
每个物理步用当前机身朝向旋转局部偏移，再加实际质心世界位置，得到世界作用点，
因此作用点随机身运动；力的世界水平方向在该脉冲内保持不变。
通过 `positions` 参数交给施力接口，自动包含相对质心的力矩 `torque = r × F`，不会重复叠加额外力矩。
这会同时扰动平移和转动。例如垂直于 0.1 m 力臂的 50 N 推力产生 5 N·m 力矩。
各环境重置时分别清除该环境的作用点偏移；等待阶段不施力。
作用点配置自动随 `randomization_cfg` 保存到 run.json 和训练历史。
时间均为仿真时间。默认站姿、动作参考位置和奖励参考位置不随初始角扰动改变。
开启随机化时，每回合重新采样初始关节角，并在等待结束后才开始随机推力计时。
质量、摩擦、传感器噪声和动作延迟均未随机化。
随机推力暂不绘制箭头，鼠标交互的箭头继续显示鼠标产生的力。

```bash
# 先检查加入随机化后的采样、重置和训练
bash go2_learning/run.sh train --randomization --num_envs 4 --max_updates 2 --run_name random_smoke
# 从头训练有扰动的策略
bash go2_learning/run.sh train --randomization --max_updates 300 --run_name randomized
# 播放时也开启随机初始角和随机推力
bash go2_learning/run.sh play --randomization
# 不带扰动的基线训练
bash go2_learning/run.sh train --no-randomization --run_name baseline
```

`play --randomization` 使用当前 `RandomizationCfg`；训练时的参数另存于该次
`run.json`，便于对照。开启扰动不会自动改善已有策略的抗扰动能力。

当前推力范围为 40～100 N，持续时间从 0.2 秒提高到 0.4 秒，
单次推力冲量从 8～20 N·s 提高到 16～40 N·s；等待间隔仍为 2～4 秒。
训练默认启用该配置；播放时需显式 `--randomization` 才施加扰动。
该开关也会启用已有的初始关节角 ±0.1 rad 随机化。

```bash
bash go2_learning/run.sh train --resume latest --randomization --max_updates 300 --run_name stronger_push
bash go2_learning/run.sh play --randomization
```

## 站立训练（保留积分观测）

先确认模型加载正常，再检查训练接口：

```bash
bash go2_learning/run.sh train --num_envs 4 --max_updates 2 --run_name stand_height_integral_smoke
```

这个短命令仅验证采样与 PPO 更新。正式训练与播放：

```bash
bash go2_learning/run.sh train --max_updates 300 --run_name stand_height_integral
bash go2_learning/run.sh play
```

- 训练默认无窗口，环境数可用 `--num_envs` 指定；播放默认一个环境和 `newton_gl`。
- 物理频率 120 Hz；每 4 个物理步换一次动作，控制频率 30 Hz。
- 52 维观测：线速度 3、角速度 3、重力方向 3、关节位置偏移 12、关节速度 12、上次动作 12、目标速度 3、瞬时高度误差 1、归一化 roll/pitch/height 积分 3。
- 12 维动作：每个关节的完整机械范围，限位从运行时 `robot.data.joint_pos_limits` 读取，单位 rad。
  `target = (lower + upper) / 2 + action * (upper - lower) / 2`，动作限幅到 `[-1, 1]`；
  -1 对应下限，+1 对应上限，0 对应范围中点。PD 刚度 25、阻尼 0.5。
  默认站姿仍用于初始姿态和关节偏移观测；`view` 自动使用该站姿对应的归一化动作。
  任务版本更新为 `go2_standing_height_integral_full_range_v2`，需从头训练；旧权重及日志保留，
  不可直接用于新映射下的播放或续训。
- 奖励约束 roll/pitch、横滚/俯仰相关机身角速度、机身高度、水平漂移、竖直轴转动、关节站姿、髋角、动作和动作变化，并包含存活与跌倒项。
  积分不参与奖励，仍可用于观测与诊断日志。
- 静止目标采用正收益，零速时最大：
  `horizontal_velocity_weight * exp(-(vx_world**2 + vy_world**2) / horizontal_velocity_sigma**2) * dt`，
  `yaw_velocity_weight * exp(-(omega_world_z / yaw_velocity_sigma)**2) * dt`。
  权重和 sigma 在 `rewards.py` 的 `RewardCfg` 中独立调整，具体数值以当前代码为准；
  sigma 的单位分别为 m/s 和 rad/s，参数仍需训练后检查效果。水平速度使用机身根刚体质心的世界 XY 分量，
  不包含 Z 轴下落速度；转动使用世界 Z 轴角速度，不混用倾斜机身的 Z 轴分量。
  这里的角速度是绕竖直轴的投影，倾斜时不严格等于欧拉 yaw 的导数。
  两项属于任务收益，在非负截断之前计算，每回合 0.3 秒等待期间不计奖励。
  当前目标固定为零，今后移动任务需要改成对非零速度指令的跟踪。
  该目标鼓励停止移动，不会主动回到出生位置或出生朝向；没有加入位置误差或改变网络观测维数。
  权重设 0 可分别关闭；旧模型缺少字段时，播放按权重 0 处理；续训使用当前代码并继承奖励历史。
  修改奖励需要重新训练或续训，已有权重的确定性播放动作不会自动改变。
  详细日志记录 `Reward/horizontal_velocity`、`Reward/yaw_velocity`；
  精简控制台新增 `speed`（世界 XY 速度大小，m/s）和 `spin`（世界 Z 角速度绝对值，rad/s），
  对应 `Task/horizontal_speed_mps` 与 `Task/yaw_velocity_abs_radps`，不会因正负方向相互抵消。
  两个新奖励为正值，既有 `top` 仍只对负惩罚项排序。

  ```bash
  bash go2_learning/run.sh train --resume latest --max_updates 300 --run_name no_drift
  bash go2_learning/run.sh play
  ```

  可将 `latest` 换成具体模型路径来固定续训来源。续训完成后播放最新模型，
  比较 `speed`、`spin` 和站姿是否一起改善。
- 抑制姿态晃动：`-roll_velocity_weight * omega_body_x**2 * dt`
  和 `-pitch_velocity_weight * omega_body_y**2 * dt`。
  两个权重在 `rewards.py` 中调整，设为 0 可单独关闭。
  使用 `root_ang_vel_b` 的 X/Y 分量，单位 rad/s，不包含 Z 轴角速度。
  它们是机身坐标系角速度，近水平时近似 roll/pitch 的欧拉角变化率；不是严格的欧拉角导数。
  两项属于常规正则，在持续奖励的非负截断之前计算；落地等待期间不扣分。
  完整日志/TensorBoard 分别记录 `Reward/roll_velocity`、`Reward/pitch_velocity`
  及 `Task/roll_velocity_abs_radps`、`Task/pitch_velocity_abs_radps`。
  精简终端仍只增加到既有 `top` 排名，不逐项展开。
  权重随 run.json 和训练历史保存；旧模型未记录这两个字段时，播放按 0 处理。
- 四足支撑正奖励：`foot_support_weight * foot_support_fraction * dt`。
  `RewardCfg.foot_support_weight` 初始为 20.0，设为 0 可关闭该奖励。
  每个物理步分别检测四只脚的向上法向合力是否超过 `contact_force_threshold`，
  再对本控制周期的 4 帧、4 只脚求接触比例。四足持续承重比例为 1，三足持续承重为 0.75，
  对应每秒 20 分、15 分；第四只脚只接触一帧时比例为 0.8125，不会得到满分。
  不要求四足受力相等；这是柔性奖励，抬脚不会直接终止回合，每回合等待阶段不计分。
  仅当目标水平速度大小不超过 `standing_linear_threshold=0.05` m/s，且目标转速绝对值
  不超过 `standing_yaw_threshold=0.1` rad/s 时启用。使用目标指令判断，不按实际速度关闭，
  避免通过主动漂移绕开站立奖励；有移动或原地转向指令时，这项奖励为零。
  目前指令全零，因此所有正式控制样本都启用。今后移动任务仍需将现有零速度奖励改为目标速度跟踪。
  奖励、开关阈值自动进入 run.json 和训练历史；旧模型缺少字段时，播放按支撑权重 0 处理。
  详细日志记录 `Reward/foot_support`、`Contact/foot_support_fraction`，
  以及 `Contact/all_feet_support_fraction`（物理帧中四足同时支撑的比例）。
  控制台新增 `feet=…/4`，表示物理帧上平均承重脚掌数；旧的 `Contact/foot_body_count`
  仍是控制周期内接触过的足部数量，二者意义不同。

  ```bash
  bash go2_learning/run.sh train --resume latest --max_updates 300 --run_name four_feet
  bash go2_learning/run.sh play
  ```

  如需比较两版奖励，分别从同一个具体 `.pt` 路径开始续训，使用不同的 run_name；
  不同 run 会独立保存奖励历史。这样不会将第二版建立在第一版的训练结果上。
- 非足部接触惩罚：`-undesired_contact_weight * undesired_contact_count * dt`。
  `rewards.py` 中初始权重为 10.0，即每个非足部接触刚体每秒扣 10 分；
  `contact_force_threshold=1.0` N，只有法向接触合力模长超过阈值才计数。
  四个 `*_foot` 正常触地不扣分；小腿、大腿、髋、机身和头部接触会持续扣分。
  同一小腿的多个碰撞几何或多个接触点只算一个刚体，不按碰撞点数重复扣分。
  例如四条小腿持续接触，每秒合计扣 40 分，30 Hz 下每步约扣 1.333 分。
  接触本身不触发终止，仍使用原来的高度/倾斜失败条件；每回合 0.3 秒等待期间不扣分。
  此项在非负截断之后独立扣除，避免被下限修正抵消。
  场景开启 PhysX 接触报告，传感器每个物理步更新，读取本控制周期的 4 帧历史，
  任一帧超过阈值即视为本周期接触。逐环境重置由 `scene.reset(env_ids)` 清空该环境的传感器历史。
  当前读取各刚体的汇总法向接触力；平地场景没有其他障碍且关闭自碰撞，因此用于检测触地。
  后续加入其他可碰撞物体时，与它们的接触也会计入，若只想约束地面需进一步做对象过滤。
  这不是接触位置坐标或穿透深度的惩罚，也不改变网络输入、动作映射或模型结构。
  非足部惩罚和四足支撑奖励两个权重都为 0 时才关闭传感器；权重和阈值自动保存到 run.json 与训练阶段历史。
  旧模型没有该字段时，播放按权重 0 处理；续训使用当前代码中的配置。
  完整日志/TensorBoard 记录 `Reward/undesired_contact`、`Contact/undesired_body_count`、
  `Contact/undesired_env_fraction`（活动环境中出现非足部接触的比例）和 `Contact/foot_body_count`；
  控制台继续只用已有 `top` 显示最大惩罚项。

  先做小规模接入检查，无需重新 build 镜像：

  ```bash
  bash go2_learning/run.sh train --num_envs 4 --max_updates 2 --run_name contact_smoke
  ```

  检查通过后，继续已有模型并启用新接触奖励：

  ```bash
  bash go2_learning/run.sh train --resume "go2_learning/logs/原训练目录/latest.pt" --max_updates 300 --run_name contact_standing
  ```

  将 `原训练目录` 替换为要继承的那次训练目录。短检查也会生成模型，
  此后 `--resume latest` 会选中短检查的结果，所以接着原策略训练时应指定原模型路径。
- 新训练使用正收益模式（`RewardCfg.tracking_reward_type = "exp"`）：
  `roll_weight * exp(-(roll / roll_sigma)**2)`、
  `pitch_weight * exp(-(pitch / pitch_sigma)**2)`、
  `height_weight * exp(-((height - target_height) / height_sigma)**2)`。
  三项最大每秒收益及容忍宽度在 `rewards.py` 中调节；目标高度 0.30 m。
  roll/pitch 的宽度单位为 rad，高度宽度单位为 m。
  误差等于容忍宽度时得到最大收益的约 36.8%；宽度越小越严格。
  这些权重不再是旧平方惩罚系数，不要直接照搬旧的 30/200 等数值。
  `only_positive_rewards=True` 将任务收益与常规正则的合计截到非负。
  `action_limit_outside_floor=True` 在截断之后独立扣除原始动作超界惩罚，
  防止非负截断抵消约束后，原始动作和探索噪声持续增大。
  再单独扣除 `failure_penalty`（当前 50），正常超时不扣失败罚分。
  因此原始动作严重超界时，即使没有跌倒，净奖励也可以为负。
  `Reward/positive_floor` 记录截断补回的分数，所有分项之和等于实际环境奖励。
  新字段自动写入 run.json 和各阶段历史；旧模型回放缺少模式字段时采用
  `quadratic` 和不截断，恢复旧奖励语义；旧正收益模型缺少
  `action_limit_outside_floor` 时也保留原截断顺序。续训仍使用当前代码配置。
  这次奖励尺度变化较大，建议先不加 `--resume` 从头训练，避免旧 Critic 的负回报预测影响起步：

  ```bash
  bash go2_learning/run.sh train --max_updates 1000 --run_name positive_standing
  ```

  当前调参起点（依据 2026-10-15 最近训练日志，尚未完成新参数的训练验证）：

  | 项目 | sigma | 最大收益率 / 权重 |
  | --- | --- | --- |
  | roll | 0.30 rad | 2.0 / 秒 |
  | pitch | 0.40 rad | 3.0 / 秒 |
  | 高度（目标 0.30 m） | 0.10 m | 4.0 / 秒 |
  | 存活 | — | 1.0 / 秒 |
  | 动作变化惩罚 | — | 0.05 |
  | 原始动作超界惩罚 | — | 10.0，截断后独立扣除 |
  | 失败罚分 | — | 50.0 / 次 |

  三项斜率绝对值最大处分别约为横滚 12.2°、俯仰 16.2°、高度误差 7.1 cm，
  对应日志的典型偏差，避免 sigma=1 时达标要求过宽或过小时收益接近零。
  理想静态站立收益率合计为 10.0；动作修正及超界扣分仍参与总奖励。
  `agent_cfg.py` 的 `entropy_coef` 调到 0.005，减弱旧日志中持续增大的探索噪声。
  旧模型的确定性回放效果与带随机动作的训练指标仍应分别评估。

- 动作平滑项：`-action_rate_weight * sum((action - previous_action)**2) * dt`，
  在 `rewards.py` 中设置，权重以当前代码为准，设为 0 可关闭；每个 30 Hz 控制步计算一次，
  比较限幅后的归一化动作。保持动作不变时该项为零，动作反复跳变时扣分。
  重置时历史动作设为默认站姿对应的动作，因此第一步比较的是策略目标与初始 PD 目标，
  不会带入上一回合的动作。原有 `action_weight` 是另一项，两项可分别调整。
  新训练自动保存新权重；旧检查点未记录此字段时，播放按 0 处理，续训使用当前 `RewardCfg`。
- 髋侧摆角惩罚：`-RewardCfg.hip_weight * sum(q_hip**2) * dt`。
  按名称选择四个 `.*_hip_joint`，目标为 0 rad，减少相对机身的左右侧摆。
  全部关节站姿惩罚：`-posture_weight * sum((joint_pos - default_joint_pos)**2) * dt`，
  参考 `scene.py` 中的默认角：髋 0、大腿 0.8、小腿 -1.6 rad，减少前后过度伸展及不对称姿态。
  站姿修正的调参起点为 `hip_weight=10.0`、`posture_weight=2.0`，具体数值以当前代码为准。
  两项都使用实际关节角，不使用归一化动作；posture 包含髋关节，因此当前髋角总系数为 12。
  这是参考站姿软约束，不会把关节固定，也不直接规定脚掌的世界坐标或间距；
  机身姿态、高度、四足支撑和速度奖励继续共同决定策略。
  当前用于站立阶段，后续移动任务需要重新评估站姿约束的强度或按指令切换。
  TensorBoard 的 `Task/hip_abs_angle_rad` 记录四个髋角绝对值的均值；
  `Task/joint_posture_rms_rad` 记录每台狗全部关节相对参考角的 RMS，再对活动环境取平均。
  `Reward/hip`、`Reward/posture` 记录实际扣分；精简控制台仍用 `top` 排名。
  修改奖励后需重新训练或续训，旧模型回放仍使用原保存的配置。

  ```bash
  bash go2_learning/run.sh train --resume latest --max_updates 300 --run_name tidy_stance
  bash go2_learning/run.sh play
  ```
- 原始动作超界项：`-action_limit_weight * sum(max(abs(raw_action) - 1, 0)**2) * dt`。
  权重在 `rewards.py` 调节。计算使用裁剪前的原始动作，
  范围内不罚，恒定输出 8 也会持续产生超界惩罚值；执行目标仍由裁剪后的动作映射。
  `action_rate_weight` 继续比较裁剪后的相邻动作，两项分别约束超界和指令突变。
  完整日志和 TensorBoard 记录 `Policy/raw_action_abs_max`、`Policy/action_clip_fraction`、
  `Policy/action_limit_violation`；超界比例范围为 0～1，violation 是未乘权重和 dt 的平方和均值。
- 训练/播放时，机身低于 0.18 m 或倾斜过大判失败；15 秒超时，自动重置。
- 当前先验证静态站姿；后续再恢复移动指令和抗推训练。

### 指令从哪里来？

目标为机身坐标下的 `[vx, vy, yaw_rate]`。训练时每台机器人在回合开始时独立采样一次，
整个回合保持不变；跌倒或正常超时重置后重新采样。当前 100% 为全零目标。
`CommandCfg.resample_seconds = None` 表示仅在重置时采样；旧配置中的数值周期仍可读取。
范围在 `commands.py` 的 `CommandCfg`：前后、侧移、转向均设为 `(0.0, 0.0)`。
结束回合的奖励先按旧指令计算，再在重置时采样新指令并返回新回合的观测。

当前纯站立策略请使用普通 `play`；入口会拒绝 `--keyboard`，避免向仅训练过零指令的
策略发送移动命令。键盘检测仍可使用 `view --keyboard_test`。
保留的键盘映射供后续移动阶段使用：

| 按键 | 目标 |
| --- | --- |
| ↑ | 前进 0.4 m/s |
| ↓ | 后退 0.3 m/s |
| ← | 左转 0.6 rad/s |
| → | 右转 0.6 rad/s |
| 全部松开 / 窗口失焦 | 全零速度目标 |

可组合前进与转向；同轴相反方向相互抵消。速度相对狗当前朝向定义。
松键或失焦后给全零速度目标，策略尝试停稳；实际运动取决于策略能力。
普通 `play` 给零速度目标；`view` 仍只有 PD 默认站姿。

### 奖励在哪里改？

`rewards.py` 的 `RewardCfg` 集中列出权重和误差尺度。
roll/pitch 和高度采用瞬时误差平方惩罚，权重读取当前 `RewardCfg`：

```python
roll_reward = -cfg.roll_weight * roll**2
pitch_reward = -cfg.pitch_weight * pitch**2
height_reward = -cfg.height_weight * (height - cfg.target_height)**2
```

roll/pitch 的目标角为 0 rad，分别控制左右侧倾和前后俯仰；它们由机身坐标下的重力方向恢复。
瞬时角度和高度惩罚权重可独立调整；当前任务不指定全局 yaw 朝向角。
连续奖励之和乘 `dt`，跌倒再按 `failure_penalty` 单独扣分。
水平速度跟踪、yaw 角速度跟踪、竖直速度和 roll/pitch 角速度惩罚均已删除。
roll/pitch/height 积分惩罚及其配置字段也已删除；旧检查点的这些历史字段在加载时忽略，
播放时兼容旧配置中的这些字段；续训使用当前代码的 `RewardCfg`。

位置积分模块及其两维观测已经移除，奖励不再依赖累计位置误差。
姿态与高度误差积分目前只作为观测和诊断数据，不参与奖励。
TensorBoard 的 `Task/linear_tracking_error_mps`、`Task/yaw_tracking_error_radps` 记录速度误差，
`Task/roll_abs_rad`、`Task/pitch_abs_rad` 记录绝对倾斜角，单位 rad。
300 轮可作初步尝试，尚不保证形成稳定步态。

### 姿态与高度误差怎样积分？

每个控制步的物理仿真完成后，用该步实际 roll/pitch 和高度误差更新一次：

```python
error = [roll, pitch, height - target_height]
I = clamp(exp(-dt / decay_seconds) * I + error * dt, -limits, limits)
```

姿态积分单位为 rad·s，高度积分单位为 m·s。
高度使用根节点 Z 坐标减去该环境原点 Z，再减目标高度；低于目标时误差为负。
当前 `attitude_integral.py` 的 `IntegralCfg` 默认：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `limit` | 0.5 rad·s | 每轴积分范围为 ±0.5，防止历史误差过度积累 |
| `height_limit` | 0.2 m·s | 高度积分范围为 ±0.2，与角度积分独立限幅 |
| `decay_seconds` | 5.0 s | 旧积分的衰减时间常数；身体摆正后，历史积分逐渐消退 |

网络各轴分别除以自己的积分限幅，范围为 [-1, 1]。
网络还接收瞬时 `height - target_height`，单位 m，从而同时知道当前高度偏差和历史累计偏差。
积分观测仍保留，因此仍是 52 维网络；奖励函数不读取这些积分值。
每个环境的跌倒/超时重置仅清零自己的积分，其他环境保留历史；
本步日志使用重置前的积分，返回新回合观测时再显示归零后的积分。

完整日志和 TensorBoard 记录 `Task/roll_integral_abs_rad_s`、`Task/pitch_integral_abs_rad_s`、
`Task/height_integral_abs_m_s` 和 `Task/height_abs_error_m`。
终端摘要显示平均绝对 roll/pitch 和平均实际高度，便于和之前的基线比较。
这是通过历史状态和奖励引导策略的试验，不直接叠加 PID 控制力矩，改善程度需训练验证。

**需要从头训练当前 52 维输入策略。** 之前的站立权重及其日志保留，
`latest` 只选取标记为 `go2_standing_height_integral_v1` 的训练结果。
播放使用对应训练目录保存的指令和奖励配置。

权重和 TensorBoard 日志保存在 `go2_learning/logs/`，被 Git 忽略。
新训练目录的时间固定使用北京时间（UTC+8），不依赖容器时区；此前使用 UTC 命名的目录应加 8 小时理解。
每次训练保存 `run.json` 记录任务版本、网络配置、指令/奖励/积分配置、随机化配置、种子和环境数；
本版本同时保存 `action_joint_limits`，播放和续训时核对任务版本、关节顺序、参考站姿和机械限位。

### 保存检查点与继续训练

保存的检查点包含 Actor、Critic、Adam 优化器状态和刚完成的训练轮号，
同时在 `infos.training_history` 中携带截至该检查点的完整阶段历史和奖励配置。
`agent_cfg.py` 的 `save_interval` 控制保存间隔（当前为 50），文件名为 `model_N.pt`；
训练完成后还会保存 `latest.pt`。它表示最后保存的模型，不是自动评选的最佳模型。
选择对应文件即可从该模型继续。

```bash
# 从当前任务最近一次完成训练的 latest.pt 继续，再训练 300 轮
bash go2_learning/run.sh train --resume latest --max_updates 300 --run_name resumed
```

也可指定中间检查点；下面的 `<训练目录>` 需要替换为实际目录名：

```bash
bash go2_learning/run.sh train --resume "go2_learning/logs/<训练目录>/model_300.pt" --max_updates 300 --run_name resumed_from_300
```

`--max_updates` 始终表示本次额外执行的更新数。例如已完成 600 次更新后再执行 300 次，
完成后总计 900 次。文件内部 `iter` 从 0 开始，保存的 599 对应已完成 600 次更新；
续训从下一轮开始。终端的 `[当前/总数]` 显示本次续训进度。

`--resume` 只加载 Actor/Critic 网络权重，包含网络内保存的策略标准差和归一化参数。
Adam 优化器重新初始化，PPO 参数、奖励、指令、积分配置以及运行参数均使用当前代码/命令行。
因此修改 `rewards.py` 后继续训练，新奖励会立即用于新进程；启动时会打印实际奖励配置。
轮号仅用于继承历史和连续计数。网络结构、关节名称/顺序、默认参考角和目标角范围仍需兼容，入口会核对。
播放使用所选模型保存的网络及观测配置，方便回看旧模型。

续训写入新训练目录，原权重和日志保留，新 `run.json` 的 `resumed_from` 记录来源。
恢复后仿真从新回合开始，姿态/高度积分清零，随机数从种子重新初始化；
训练使用恢复的网络和新优化器继续采样更新。
`--resume` 用于训练，`--checkpoint` 用于播放；省略 `--resume` 则从头训练。

### 每段奖励和训练轮数怎样记录？

每次定期保存检查点及训练结束时，同时更新新训练目录的 `history.json` 和 `run.json`。
`history.json` 按阶段保存 `start_round`、`end_round`、实际更新数、`reward_cfg`、PPO 配置和来源检查点。
区间采用从 1 开始的闭区间，例如：

```text
第 1～300 轮：posture_weight = 10
第 301～600 轮：posture_weight = 0.5
第 601～800 轮：posture_weight = 0.1
```

每个 `.pt` 自带保存时的历史快照；从第 450 轮的模型分支时，只继承到第 450 轮，
不会混入第 451～600 轮的训练，新的奖励阶段从第 451 轮开始。
文件名 `model_N.pt` 中 N 是零起始轮号，因此 `model_449.pt` 对应已完成 450 轮。
`run.json` 额外记录本段起止轮数、计划结束轮数、已完成更新数和 `resume_mode=weights_only`。
旧模型尚未保存内嵌历史时，沿原有 `run.json` 的 `resumed_from` 链只读补建；
如果祖先文件已缺失，相应历史标为配置未知，不猜测奖励参数。

### 精简训练日志

默认每轮 PPO 更新打印一行摘要，末尾标出当前最大的惩罚项，例如：

```text
[20/300] fallen=2 (total=35) | reward=4.120 | episode=8.32s | roll=2.10deg | pitch=1.40deg | height=0.318m | raw_max=8.00 | clip=70.0% | excess=120.000 | top=action_limit (80.2%)
```

- `fallen`：这一轮采样中所有环境的跌倒次数；`total` 为本次训练调用以来的累计跌倒次数。
- `reward`：最近最多 100 个已完成回合的累计奖励平均值，包含跌倒或超时结束的回合。
- `episode`：同一窗口内已完成回合的平均仿真时长，单位秒。
- `roll` / `pitch`：本轮采样中所有环境、所有控制步的平均绝对倾斜角，单位度，越小越水平。
- `height`：同一范围内的平均实际机身高度，单位 m；目标由该次训练的 `target_height` 决定。
- `raw_max`：本轮正式控制采样中，裁剪前动作绝对值的最大值。
- `clip`：每个正式控制步的动作超界比例（超过 `[-1, 1]` 的关节数 / 全部关节数），再对本轮取均值。
- `excess`：每个正式控制步、每台狗的 `sum(max(abs(raw_action) - 1, 0)**2)` 均值；未乘权重和 dt，权重为 0 时也显示，方便调试。
- `top`：本轮负奖励分项中最大的项目及其占所有负分项的比例。比较时已乘权重和 dt；`failure` 不乘 dt，但同样平摊到全部有效样本。等待样本不参与，活动环境数不同时按样本数加权。正收益和 `positive_floor` 不进入分母；这是下限修正前的负分构成，不代表最终净扣分或 PPO 梯度贡献。没有扣分显示 `none`，没有样本显示 `--`。
- 各分项使用与实际奖励相同的计算结果，并记录至 TensorBoard / 完整日志中的 `Reward/*`。
- 速度、转向误差等详细指标继续保存在 TensorBoard 和完整日志文件中。
- 尚无回合结束时，`reward` 和 `episode` 显示 `--`。

这是每轮 PPO 更新的摘要，不是每只狗每次回合结束都打印。
原始详细长表保存在本次训练目录的 `training_details.log`，TensorBoard 继续保存完整指标。
需要恢复终端长表时，在原训练命令后加 `--verbose_logs`。
启动时的场景信息、警告和异常仍正常显示。已运行的训练需重新启动进程才会使用新显示方式。
适配依据：[RSL-RL v5.5.1 Logger](https://github.com/leggedrobotics/rsl_rl/blob/v5.5.1/rsl_rl/utils/logger.py)。

## 模型和运行环境

模型入口：`../assets/unitree_model/Go2/usd/go2.usd`，其 `configuration/` 子层必须一起保留。
来源与下载校验见 [模型说明](../assets/unitree_model/README.md)。
官方地面及贴图位于 `../assets/isaaclab/ground_plane/`，见 [地面资源说明](../assets/isaaclab/ground_plane/README.md)。
迁移时一起携带 `go2_learning/` 和 `assets/`，保持二者为同级目录。

脚本只读挂载代码和模型、读写挂载日志与缓存，使用 `--pull=never --network none`，
以及 `uv run --offline`。无需重新 build 或安装依赖。

基础场景、站立训练、随机化和方向键输入检测已由用户在本机验证。
姿态/高度积分与环境生命周期已完成 CPU 逻辑检查，实际稳态误差改善仍需 Docker / GPU 训练播放验证。
