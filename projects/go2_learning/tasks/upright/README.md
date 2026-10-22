# upright：Go2 后腿直立

每回合从接近直立的姿态开始，让 RL、RR 两只后腿的**脚掌着地支撑，膝关节（小腿）离地**，
FL、FR 两只前脚离地，机身前向轴朝世界上方。
使用同一个 Go2 USD、PhysX、RSL-RL 与 `newton_gl`；本任务不改变其他任务的配置。
机身前向轴初始朝上（仰角 90°），根节点高 0.45 m，前腿屈曲收起。
先等待 0.3 s，让后脚落稳后执行策略；这段时间不计训练奖励、回合时间或有效策略样本。

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

`view --task upright` 只保持重置用的固定 PD 目标，没有神经网络平衡控制，长时间仍可能倾倒；
观察策略平衡请使用 `play`。
本任务没有键盘移动指令。日志在 `logs/upright/<run>/`，不会自动选到 standing/running 模型。
目前续训只支持 upright → upright，不直接加载四足站立/运动模型。

## 前腿目标姿态

默认取消前腿机械锁定和动作覆盖，12 个网络输出均参与控制（保留通常的动作限幅）。
`reward_cfg.py` 的 `front_target_*` 指定前髋 0、大腿 1.4、小腿 -2.4 rad。
奖励为 `-front_target_weight * mean((实际关节角 - 目标角)**2) * dt`，
只计算两条前腿的六个关节，全程生效；权重 100，可自行调整。
目标角属于奖励配置，重置姿态仍由 `reset_cfg.py` 管理。
播放旧模型也取消前腿锁定；旧记录没有目标奖励时该项按 0 恢复，续训使用当前奖励配置。
训练和播放均恢复 8/4 求解迭代；后腿等待期预载恢复为 `(0, 0.018352, 0.6228905)` rad。

## 奖励：v16 基线及当前修改

曾回滚至 `2026-10-21_21-24-22_524274_upright_restart_v16/run.json`，
对应总训练第 **15001–15500** 轮。下表记录该基线，后续调参数值以 `reward_cfg.py` 为准。
当前默认 `reward_schema_version = 5`：沿用版本 1 的计算方式，仅将水平速度改成平方负惩罚。
配置在 `reward_cfg.py`，公式在 `rewards.py`；持续项每步乘 `dt`，失败罚分一次扣除。

重力观测 `g_b` 是世界向下方向在机身坐标中的表示。
`alignment = -g_b[:, 0]` 为机身 +X 与世界向上的点积：水平为 0、头朝上直立为 1。
`hold = exp(-((1-alignment) / upright_sigma)²)`，不使用直立附近会奇异的欧拉 roll/pitch。

| 分块 | v16 配置 | 计分方式（下列持续项最后均乘 dt） |
| --- | --- | --- |
| 存活 | `alive=1` | 恒定正收益 |
| 直立 | `upright_hold_weight=200`、`upright_sigma=0.2` | `200 * hold` |
| 高度 | `height_weight=100`、`target_height=0.45`、`height_sigma=0.1` | `100 * exp(-((height-0.45)/0.1)²)` |
| 后脚支撑 | `rear_support_weight=5` | 两只后脚掌支撑比例的均值；膝同时支撑不会取消脚掌收益 |
| 后膝离地 | `rear_knee_clearance_weight=70`、`rear_knee_clearance_sigma=0.12` | `70 * (1-exp(-max(h,0)/0.12))`，h 为较低的后小腿 link 原点高度 |
| 前脚抬起 | `front_lift_weight=5` | 前脚不支撑的比例取均值 |
| 水平平衡 | `balance_weight=5`、`horizontal_velocity_sigma=0.4` | `5 * exp(-(vx²+vy²)/0.4²) * hold`，世界坐标 |
| 整体角速度 | `angular_velocity_weight=0.1` | `-0.1 * (wx²+wy²+wz²) * hold`，世界坐标 |
| 机身 Y 轴角速度 | `pitch_velocity_weight=2` | `-2 * wy_body²`，全程计分，不乘 hold |
| 后髋回正 | `hip_rear_weight=200`、`hip_rear_sigma=0.4` | 后髋各自接近 0 的指数收益，左右取乘积 |
| 后腿对称 | `rear_symmetry_weight=100`、`rear_symmetry_sigma=0.8` | 对称误差 RMS 的指数收益 |
| 关节速度（当前替换项） | `joint_velocity_weight=1` | `-weight * sum(qd²)`，12 个实际关节角速度，单位 rad/s |
| 原始动作超界 | `action_limit_weight=40` | 网络动作超出 [-1,1] 部分的平方和惩罚 |
| 通用非足部接触 | `undesired_contact_weight=60` | 非脚掌接触刚体数，减去本周期向上支撑的后小腿数 |
| 后小腿接触 | `rear_knee_contact_weight=150` | 每条后小腿另行扣分，支撑/侧向接触均算，不包含脚掌 |
| 髋部接触 | `hip_contact_weight=60` | 髋部额外扣分，与通用 60 叠加后为每个髋部 120/s |
| 足部滑动 | `foot_slide_weight=4` | 触地脚掌世界 XY 速度模长之和的惩罚 |
| 失败 | `failure_penalty=100` | 跌倒终止一次扣 100，正常超时不扣，不乘 dt |

以上是回滚时数值，后续调参以代码为准。接触阈值为 1 N；后小腿侧向接触仍会同时触发通用项，
向上支撑的后小腿只扣专用 150。接触传感器也可能统计自碰撞。
膝部高度使用 link 原点，未计算碰撞表面到地面的精确距离。

在 v16 基础上，现已将动作变化惩罚替换为实际关节角速度平方和惩罚。
取控制步末的原始 `joint_vel`，不乘观测缩放系数 0.1，不对目标动作做差分；全程生效，每步乘 `dt`。
日志改为 `Reward/joint_velocity`，删除 `Reward/action_rate`。两者量纲不同，数值权重保留 1，实际惩罚强度需重新观察。
旧模型缺少 `joint_velocity_weight` 时回放按 0 加载；续训使用当前代码配置中的新权重。

当前已删除 v16 未使用的前脚位置目标、世界竖直速度奖励及其配置、状态读取与日志；
旧模型配置中相关字段加载时忽略，不再参与回放计分。
版本 1–4 的公式仍用于播放已保存的其他版本模型；当前新训练/续训默认使用版本 5。
训练历史按原机制继承，旧日志和检查点保留。此处根据日志恢复奖励，日志本身不包含全部历史源代码快照。

### 当前水平速度项

```python
balance = -balance_weight * (vx_world**2 + vy_world**2) * dt
```

`balance_weight` 当前为 5，静止时该项为 0；速度越大，扣分越多，不设上限。
本项全程生效，不乘 `hold`，不使用 sigma，不包含世界 Z 速度。
`horizontal_velocity_sigma` 移至配置底部，仅供旧版 1/2 模型恢复原指数收益；当前训练不用调它。
其余奖励的公式由版本 5 保留为版本 1 的形式；历史文件会记录版本号，区分此次改动。

播放 v16 模型：

```bash
bash run.sh go2 play --task upright --checkpoint projects/go2_learning/logs/upright/2026-10-21_21-24-22_524274_upright_restart_v16/latest.pt
```

从 v16 权重继续训练，并使用上述奖励：

```bash
bash run.sh go2 train --task upright --resume projects/go2_learning/logs/upright/2026-10-21_21-24-22_524274_upright_restart_v16/latest.pt --max_updates 500 --run_name upright_v16_rollback
```

## 重置与观测

重置姿态单独放在 [`reset_cfg.py`](reset_cfg.py)：

- `root_height`、`forward_elevation_deg`：机身根节点高度和机身前向轴仰角。
- `front_hip/thigh/calf`、`rear_hip/thigh/calf`：前、后腿的**物理初始关节角**，单位 rad。
- `front_hold_offset`、`rear_hold_offset`：等待期 PD 目标相对初始角的偏置，顺序为 hip/thigh/calf。

实际写入的初始关节角与可动关节 PD 目标分开：`等待期目标 = 初始关节角 + hold_offset`。
固定前腿不叠加偏置，目标直接等于固定角。后腿 Kp=25，若目标与初始角完全相同，
初始没有支撑力矩，0.3 s 内后腿会下沉。
因此等待期用偏置提供抗重力预载；不会把偏置加到实际初始姿态，也不会在策略动作上继续叠加它。
这组数值按当前模型、地面、PD 增益验证，改变增益或初始姿态后应重新检查落稳情况。

每次重置都重新写入姿态，并清零根节点和关节速度，不复用上回合的下落/反弹状态。
并行环境仍各自等待 0.3 s、各自接管；一个环境重置不会重写其他环境。
策略使用的默认关节观测参考不变，因此旧模型仍可加载；旧模型没有 `reset_cfg` 时使用当前近直立重置。
新训练的 `run.json` 与训练历史会记录 `reset_cfg`，播放新模型时恢复保存的重置配置。
续训使用当前代码配置；从新的起始状态保持平衡仍需要策略学习。

训练低于 0.14 m 或机身 +X 偏离世界向上超过 45° 时失败，15 s 超时重置。
四足水平对应相对目标 90°，会判定失败；正常直立为 0°。
倾角用方向向量点积判断，不使用欧拉 roll/pitch。
upright 播放使用当前 `config.py` 的倾倒阈值，旧模型保存的 120° 不会覆盖当前 45°。
播放沿用共用入口：关闭高度失败，仅保留该任务的方向失败与超时重置。

观测仍为 52 维、网络动作仍为 12 维；后腿使用完整机械范围映射，固定前腿的输出被替换。
相对四足水平姿态的三个误差积分在本任务中关闭，保留为零占位。
随机化配置独立，初期默认关闭；学会起身后可通过 `--randomization` 开启较轻的推力。

## 日志

- `upright`：机身前向轴与世界向上的点积，接近 1 表示直立。
- `rear=…/2`：当前版本 1 为两只后脚掌的平均支撑数，膝部同时支撑不会排除。
- `front=…/2`：两只前脚平均支撑数，理想为 0。
- 完整日志另有 `Contact/rear_knee_support_count`（向上支撑的后膝数）、`Task/rear_knee_height`，以及 `Task/calf_RL/RR_rad`。
- 完整日志中的 `Task/hip_abs_angle_rad` 是四个髋侧摆角的平均绝对值；另有按 `FL/FR/RL/RR`
  分开的有符号 `Task/hip_<leg>_rad`，用于检查左右是否对称。四髋同轴，外撇在左腿为负、右腿为正。
- 共用的 `pitch` 在直立时可能接近 90°，不是姿态错误；应结合 upright、rear、front 判断。

奖励、PPO、终止、扰动与指令参数分别在本目录的对应配置文件中，训练历史和检查点按原有机制保存。
旧模型配置中的 `upright_progress_weight` 在播放加载时忽略，原始训练历史保留；续训使用当前奖励。

两后脚同时腾空惩罚：`rear_feet_airborne_weight=100`，逐物理帧判断两后脚
向上接触力均不超过 `contact_force_threshold`，按本控制周期占比乘 `dt` 扣分。
单后脚离地不触发；膝盖接触不豁免；不乘直立程度。日志项为 `Reward/rear_feet_airborne`。

当前代码按 upright_restart_v25 的运行记录恢复：奖励公式版本 5，前腿目标为平方负惩罚；
无单独前腿速度项。latest 恢复最后保存模型的含义，不再自动选择最佳。
运行记录未保存源码快照，因此本次为配置及已知逻辑恢复，不宣称逐字源码复原。
