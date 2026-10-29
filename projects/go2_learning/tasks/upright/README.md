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

## 奖励与 sigma 校准（当前公式版本 8）

- [`reward_cfg.py`](reward_cfg.py)：权重、目标、初始 sigma。
- [`sigma_cfg.py`](sigma_cfg.py)：校准开关、采样容量、分位数、sigma 上下界和阶段变化比例。
- [`tracking_errors.py`](tracking_errors.py)：原始误差定义，奖励与缓冲共用这一份计算。
- [`rewards.py`](rewards.py)：正收益、接触和终止的组合。

### 连续误差用正收益

统一公式：`每步奖励 = weight * exp(-(error / sigma)**2) * dt`。
**weight 现在表示每秒最高收益**，不再是平方负惩罚的系数。
本次保留修改前各项权重数值；最大收益的相对大小仍需按任务效果调节，不会由 sigma 自动平衡。

| 分块 | error 的含义 / 单位 |
| --- | --- |
| 世界 vx、vy、vz | 各轴速度绝对值，m/s，分别设置 weight 和 sigma |
| 机身 wx、wy、wz | 各轴实际角速度绝对值，rad/s；不是欧拉角导数，不乘 hold |
| 关节运动 | 12 个实际关节角速度的 RMS，rad/s，不使用观测中的 0.1 缩放 |
| 直立方向 | `1-alignment`，`alignment=-gravity[:,0]`，机身 +X 与世界向上的点积 |
| 高度 | 根节点相对环境原点的高度与 `target_height` 之差的绝对值，m |
| 后髋回正 | 两后髋相对 0 rad 的 RMS，rad |
| 后腿对称 | 髋角之和、大腿角之差、小腿角之差的 RMS，rad |
| 后腿目标姿态 | 两后大腿/小腿共四个关节相对目标角的 RMS，rad |
| 前腿目标姿态 | 两前腿共六个关节相对目标角的 RMS，rad |
| 触地无滑动 | 触地脚掌世界 XY 速度模长的 RMS，m/s；空中腿不参与，无脚触地时该项收益为 0 |

后腿参考角仍为 thigh=2.6934045、calf=-2.2452164 rad；后髋目标仍为 0。
参考来自实际 USD：后髋纵向位置 -0.1934 m，上下腿各长 0.213 m，脚掌球半径 0.022 m，
假设根节点高 0.4 m、机身竖直、后脚位于根节点下方。这是几何参考；修改高度目标后需重新评估。
前腿目标仍为 hip=0、thigh=1.4、calf=-2.4 rad。
目标姿态只是奖励，不覆盖网络动作，也不改变 PD 目标、机械限位或 reset 姿态。

### 保留原含义的项

- 后脚支撑、前脚抬起：原来的比例正收益。
- 后膝离地：原来的单调离地收益，固定 `rear_knee_clearance_sigma`，默认权重 0。
  link 原点高度不等于碰撞表面净空，不参与自动校准。
- 两后脚同时无支撑、非足部/后小腿/髋部接触：固定负惩罚，不用自动 sigma 放宽。
- 原始动作超出 [-1,1]：超界量平方和的负惩罚，不对已裁剪动作计分。
- alive：固定每秒正收益；失败：固定一次性罚分，正常超时不扣。

接触阈值仍为 1 N。支撑看世界向上分量，接触看合力模长。
通用非足部接触减去有向上支撑的后小腿数，后小腿再由专用项扣分；髋部项可与通用项叠加。
负惩罚仍然存在，所以正收益方案不等于数学上保证策略永不倾向提前结束。
旧的失败吸收代价及低通状态已删除；失败只保留配置中的一次性基础罚分。

### 每次 train 只校准一次

1. 有匹配的检查点误差缓冲：用**所选 model_N.pt 自己携带的缓冲**估计。
2. 旧模型没有缓冲，或目标/启用项/接触阈值/重置等采样条件改变：先执行默认 128 个控制步的采样。
   使用当前 Actor 的高斯探索分布，等待期不入缓冲；不更新网络，不写 PPO storage，不计训练轮数。
   采样后重置所有环境，重新完成落地等待再开始正式训练。
3. 每项取原始 error 的 75% 分位数 q，估计 `sigma = q / sqrt(-log(0.5))`。
   不触及限幅时，分位点收益是该项最高收益的一半。
4. 应用 `sigma_cfg.py` 中固定物理尺度的上下界；与上阶段相容时，最多收紧/放宽 50%。
   首次采样不受旧阶段变化比例限制。样本不足时沿用匹配检查点的尺度，否则用初始值，并遵守上下界。
5. **整个本次 learn（例如 500 轮）固定这些 sigma**。期间只写近期误差，下一次 train 才重新估计。

缓冲最多 16,384 行，每控制步均匀抽取最多 64 个活动环境并轮换采样起点，
全部 4 项约 0.25 MiB；GPU 写入，不在每步拷回 CPU。只在阶段开始校准、保存检查点时拷回 CPU。
它保存原始误差，不保存用于 PPO 回放的训练 transition，不混入更早阶段数据。
滑动项无触地样本以无效值标记，估计时排除，不当成零误差。

`run.json`、`history.json`、模型 infos 会记录实际固定的奖励配置；
`reward_calibration.json` 另存采样来源、样本数、分位数、建议值和实际 sigma。
每个检查点携带当时的误差缓冲；历史模型不重写，播放版本 1–9 保留旧公式，
新模型播放恢复保存的 sigma，不重新校准。
跨阶段比较应看回合长度、失败、速度、高度和姿态原始误差；sigma 改变后总 reward 不再直接可比。
完整日志和 TensorBoard 的 `Error/<项名>` 记录同口径原始 error 均值，不除以 sigma、不乘 weight。
滑动项仅统计触地有效样本，另有 `Error/foot_slide_valid_fraction`，避免把无脚触地误读为零滑动。

```bash
# 用新方案接着训练；首次加载旧模型会先采样校准
bash run.sh go2 train --task upright --resume latest --max_updates 500 --run_name upright_adaptive_v1

# 不自动估计，完全按当前 reward_cfg.py 的 sigma 固定训练
bash run.sh go2 train --task upright --resume latest --max_updates 500 --no-adaptive-sigma --run_name upright_manual_sigma
```

自动模式下，手改 reward_cfg.py 的 sigma 主要影响首次/不足样本时的初始值。
要手动指定下一阶段的尺度，可关闭自动模式；要调整自动策略，可修改分位数、上下界和变化比例。

## 重置与观测

重置姿态单独放在 [`reset_cfg.py`](reset_cfg.py)：

- `root_height`、`forward_elevation_deg`：机身根节点高度和机身前向轴仰角。
- `orientation_cone_degrees`：策略接管时在竖直周围采样的倾斜圆锥半角，默认 10°；0 关闭。
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
默认启用 `config.py` 的 `reset_on_nonfoot_contact=True`：任一非脚掌刚体接触即失败，
训练和播放均生效。只豁免 FL/FR/RL/RR 的四个 `*_foot` 刚体，
机身、头部、髋、大腿、小腿（含膝部）都不豁免，不扣除「膝部向上支撑」接触。
判定沿用 `contact_force_threshold`（默认 1 N），一个控制周期内任一物理帧超过阈值即可触发；
在该控制步末重置发生接触的环境，其余并行环境继续运行。0.3 s 落地等待期仍不判失败。
四足水平对应相对目标 90°，会判定失败；正常直立为 0°。
倾角用方向向量点积判断，不使用欧拉 roll/pitch。
upright 播放使用当前 `config.py` 的倾倒阈值，旧模型保存的 120° 不会覆盖当前 45°。
播放沿用共用入口：关闭高度失败，保留方向失败、非脚掌接触失败与超时重置。

观测仍为 52 维、网络动作仍为 12 维；全部关节使用完整机械范围映射，前腿也由策略控制。
相对四足水平姿态的三个误差积分在本任务中关闭，保留为零占位。
随机化配置独立，初期默认关闭；学会起身后可通过 `--randomization` 开启较轻的推力。

## 日志

- `upright`：机身前向轴与世界向上的点积，接近 1 表示直立。
- `rear=…/2`：两只后脚掌的平均支撑数，膝部同时支撑不会排除。
- `front=…/2`：两只前脚平均支撑数，理想为 0。
- 完整日志另有 `Contact/rear_knee_support_count`（向上支撑的后膝数）、`Task/rear_knee_height`，以及 `Task/calf_RL/RR_rad`。
- 完整日志中的 `Task/hip_abs_angle_rad` 是四个髋侧摆角的平均绝对值；另有按 `FL/FR/RL/RR`
  分开的有符号 `Task/hip_<leg>_rad`，用于检查左右是否对称。四髋同轴，外撇在左腿为负、右腿为正。
- 共用的 `pitch` 在直立时可能接近 90°，不是姿态错误；应结合 upright、rear、front 判断。

奖励、PPO、终止、扰动与指令参数分别在本目录的对应配置文件中，训练历史和检查点按原有机制保存。

## Critic 上下文与旧权重续训

Actor 使用 52 维 policy 观测；Critic 接收同一份状态与一个 `critic_time` 输入：
`剩余控制步数 / 最大控制步数`，总计 **53 维**。剩余时间仅影响价值估计，不传给 Actor。
旧的惩罚率低通、失败后的吸收罚分、参数和日志均已移除。

- 旧 52 维 Critic 续训：追加一列零权重作为时间输入。
- 旧 54 维 Critic 续训：从 `[policy(52), 低通代价(1), 时间(1)]` 删除低通列，保留时间列。
- 当前 53 维 Critic 续训：直接加载。

Actor（含探索标准差）、Critic 其余权重、轮号和历史继承，优化器重新初始化，不改写旧检查点。
旧奖励记录中的吸收权重与低通系数字段读取时忽略，不再启用该机制。
播放旧 54 维模型时仅为匹配旧网络形状提供一列恒零占位，不维护低通状态；Actor 的播放输入不变。

当前 upright 的 `finite_horizon=True`：15 秒为真正任务终点，正常超时关闭价值 bootstrap，
Critic 的时间输入用于区分距离期限不同的状态。其余任务保持原超时处理。
该标志会保存到 run.json 和训练历史。

关节速度项不参与动态 sigma：`-joint_velocity_weight * mean(qdot²) * dt`，
使用 12 个关节的真实角速度（rad/s），当前配置不再需要 `joint_velocity_sigma`。

Part 1 已回滚至 `upright_adaptive_v14_2026-10-26_21-27-53_188637`：
线速度取世界坐标，角速度取机身坐标，各轴使用指数正收益，全程生效、不乘 hold。
`reward_cfg.py` 的默认权重、目标和 sigma 与该记录实际使用的数值一致。
自动 sigma 校准仍按原流程执行；要固定 v14 的 sigma，训练时加 `--no-adaptive-sigma`。

### 命令行更新当前 sigma

在仓库根目录，用安装了 PyTorch 的 Python 执行（不启动仿真）：

```bash
PYTHONPATH=projects /home/redog-j/miniconda3/envs/sim_env/bin/python -m go2_learning.training.update_sigmas --checkpoint latest --recalibrate --write
```

读取所选检查点的近期误差，使用当前 `sigma_cfg.py` 的分位数、±50% 变化范围和
绝对上下界，更新当前启用的自适应 sigma；其他参数保持原值。
省略 `--write` 仅预览，省略 `--recalibrate` 同步检查点当时实际采用的值。
`--checkpoint` 也可指定具体 `model_N.pt` 或训练目录。
原配置自动备份到打印出的临时目录。下次训练加 `--no-adaptive-sigma` 可固定使用写回的值。

### 输出平滑与超界

从奖励版本 11 起，`action_rate_weight` 惩罚实际下发目标角的变化：
`-weight * mean((q_target_t - q_target_previous)²) * dt`，单位使用 rad，未除以 dt。
实现为裁剪后动作差乘各关节机械半行程，与网络观测中的上一执行动作对应。
`action_limit_weight` 继续独立惩罚裁剪前原始输出超界。
两项权重保持当前配置；平滑项不参与 sigma 校准。旧版本 8–10 的回放保留原始输出差公式。

后髋回正从奖励版本 12 起为 `-hip_rear_weight * mean(q_rear_hip²) * dt`，
全程生效，不使用 `hip_rear_sigma` 或自动校准；左右对称奖励保持原公式。
旧版本 8–11 的后髋回放仍按各自保存的指数收益计算。

前腿目标从奖励版本 13 起为六个前腿关节角误差的均方负惩罚，
`-front_target_weight * mean((q_front - q_front_target)²) * dt`。
目标髋/大腿/小腿角保持配置值，全程生效，不使用 `front_target_sigma` 或自动校准。
旧版本 8–12 的回放保留原前腿指数收益。

世界三轴线速度从奖励版本 14 起为各轴独立平方负惩罚：
`-linear_velocity_axis_weight * v_world_axis² * dt`，全程生效，不乘 hold。
对应三个线速度 sigma 已移除。
旧版本 8–13 回放保留当时的线速度公式。

机身三轴角速度从奖励版本 15 起为各轴独立平方负惩罚：
`-angular_velocity_axis_weight * omega_body_axis² * dt`。
全程生效，不乘 hold，使用真实角速度分量而非欧拉角导数；对应三个 sigma 已移除。
旧版本 8–14 按其保存的公式回放，版本 10 保留原有六轴平方惩罚。

触地滑动从奖励版本 16 起为 `-foot_slide_weight * mean(v_world_xy²) * dt`，
只对当前触地脚掌求平均，无脚触地时为 0，不使用 `foot_slide_sigma`。
旧版本 8–15 的播放保留当时的滑动收益公式。
