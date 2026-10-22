# running：Go2 速度跟踪

## 训练与播放

从仓库根目录执行：

```bash
# 从已复制到 logs/running 的站立模型继续训练
bash run.sh go2 train --task running \
  --resume "projects/go2_learning/logs/running/2026-10-15_16-00-18_586737_new1_7/latest.pt" \
  --max_updates 1000 --run_name running_from_new1_7

# 无窗口训练：自动生成速度目标，无需键盘
bash run.sh go2 train --task running --max_updates 1000 --run_name running_v1

# 播放该任务最近完成的模型：默认启用方向键
bash run.sh go2 play --task running
```

播放时点击 `newton_gl` 窗口取得焦点：

| 按键 | 目标 |
| --- | --- |
| ↑ | 前进 1.5 m/s |
| ↓ | 后退 1.5 m/s |
| ← | 左转 0.6 rad/s |
| → | 右转 0.6 rad/s |
| 松开所有键 / 窗口失焦 | 线速度和转向速度目标归零 |

可以同时按住前后键与转向键，例如 ↑+← 表示边前进边左转。
训练时 `yaw_range=(0, 0)`，转向指令恒为零；键盘转向属于训练分布外输入，效果需单独验证。
指令描述的是狗自身坐标系中的速度：转弯后，前进仍沿狗当前朝向。
松键只是给零速度目标，停稳效果取决于训练好的策略。

## 单腿断电回放测试

```bash
# 用已有 running 策略，正常控制 3 秒后切断左前腿驱动；方向键仍可用
bash run.sh go2 play --task running --disable-leg FL --disable-after 3
```

`--checkpoint` 可固定要测试的模型。`--disable-leg` 可选 FL 左前、FR 右前、RL 左后、RR 右后；
当前只支持 running 的单环境播放。`--disable-after 0` 表示落地等待结束后立即断电。
计时按仿真正式控制时间，每次回合重置恢复驱动、重新落稳后再计时。

断电将该腿 hip/thigh/calf 三个关节的隐式 PD 刚度、驱动阻尼、驱动力矩上限设为 0。
保留机器人质量、碰撞、关节限位及被动摩擦/惯量等属性，腿仍可被动运动或触地承重。
参数仅在断电及恢复时写入，正常控制循环不反复写模型属性。
这次测试使用固定策略权重，观察故障应对能力；不执行在线训练。

## 训练中的目标

每个环境在每次回合重置时独立采样，回合内保持不变：

- 前后速度：`+1.5` 或 `-1.5 m/s`（由 `forward_fixed_magnitude` 决定）。
- 侧向速度：`0 m/s`。
- 转向角速度：`0 rad/s`（训练只走直线）。
- 约 1/3 回合设为全零指令，练习停车与站立；其余回合前后方向各半。
- `forward_range` 仅在把 `forward_fixed_magnitude` 设为 `None` 时参与采样。

速度指令本来就在共用环境的 52 维观测中，网络依此选择 12 个关节的目标位置。
训练时使用自动指令，播放时使用键盘；两者由同一个指令模块传入观测和奖励。

## 奖励

速度跟踪替代站立任务的零速度奖励：

```text
linear_reward = tracking_linear_weight * exp(-||v_xy_body - command_xy||² / linear_sigma²)
yaw_reward    = tracking_yaw_weight * exp(-(omega_z_body - command_yaw)² / yaw_sigma²)
```

两项均按控制周期 `dt` 积分为每步奖励。转向使用机身 Z 角速度，接近水平时对应 yaw 角速度。
当前最大线速度收益率为 300、转向收益率为 200；sigma 分别为 0.4 m/s 和 0.6 rad/s。

姿态、高度、站姿正则、动作平滑、超界、非足部接触与失败项由本任务的 `reward_cfg.py` 单独配置。
当前速度跟踪任务没有单独的四脚支撑奖励；接触质量由 `foot_contact_ratio`、`foot_force_balance`、
`all_feet_airborne` 等项约束。
`running` 目前表示速度跟踪任务，尚未指定步态或要求腾空跑步。

### 足部运动质量

在 `reward_cfg.py` 中调整：

- `foot_slide_weight = 10.0`：本控制周期曾触地的足部，按世界 XY 速度模长之和扣分；空中摆腿不扣分。
- `foot_time_variance_weight = 10.0`：四足上次完整腾空时长的样本方差，加上上次完整触地时长的样本方差；原始时长不截顶，不设置目标腾空时长或固定步态。

两项均乘控制周期 `dt`，在常规奖励非负截断之后独立扣除，权重为 0 可关闭。
当前两项权重均为 10；实际强度仍需训练观察。
阶段时长按物理帧累计，仅在触地/离地切换时更新已完成时长；每回合等待期不累计，逐环境重置时清空。
滑动速度按足部刚体质心速度计算，是触地滑动的近似指标。

终端每轮的 `slide`、`time_balance` 为两项实际平均每步奖励（已乘权重和 `dt`，扣分为负）。
完整日志对应 `Reward/foot_slide`、`Reward/foot_time_variance`；配置随 `run.json` 和训练历史保存。
旧模型回放缺少这两项配置时自动关闭；续训使用当前配置，无需重新构建镜像。

### 静止时加强站姿

`posture_stand_still_scale = 5.0`、`posture_velocity_threshold = 0.3`：
当 vx、vy、yaw 三个指令全零，且实际机身水平速度不超过 0.3 m/s 时，
将原有的关节偏差平方和姿态惩罚加强为 5 倍。
基础 `posture_weight = 15.0` 时，静止有效权重为 75，运动时仍为 15；
将倍数改为 1 可关闭这项加强。日志沿用 `Reward/posture`。
新配置保存到训练历史；旧模型回放缺少该倍数时按 1 恢复。

## 调参位置

| 内容 | 文件 |
| --- | --- |
| 训练速度范围、停车比例、键盘速度 | [command_cfg.py](command_cfg.py) |
| 跟踪权重、sigma 及其他奖励权重 | [reward_cfg.py](reward_cfg.py) |
| 奖励公式 | [rewards.py](rewards.py) |
| 默认环境数、回合长度、频率和结束条件 | [config.py](config.py) |
| 网络与 PPO | [agent_cfg.py](agent_cfg.py) |
| 扰动强度 | [randomization_cfg.py](randomization_cfg.py) |

站立配置仍在 `tasks/standing/`。本任务有独立名字和版本，日志保存到
`logs/running/<run>/`，播放只查找真正的 running 模型；先完成 running 训练再播放。
播放恢复该模型保存的配置，续训使用当前配置并继承训练历史。

## 现有策略的崎岖地形测试

```bash
bash run.sh go2 play --task running --checkpoint latest --terrain rough
```

`rough` 为本地生成的 64×64 m 地面，高度在 ±6 cm 范围内采样，再插值形成起伏，最终高度限于 ±8 cm；
固定 `--seed 42` 可复现同一地形。此开关目前仅支持单环境 view/play，默认仍为 `flat`。
它替换原平面碰撞，摩擦系数仍为 0.5；出生高度增加 10 cm，重置等待沿用模型配置。
测试保留旧策略的 52 维观测和原高度基准，没有加入地形扫描或重新训练。
因此这次检验的是平地策略对崎岖地形的适应能力；地面为有限网格，中心距边缘约 32 m。
GL 仅给地面添加高度配色与 1 m 网格纹理：低处偏蓝，高处偏黄。
显示层贴合真实地形并抬高 1 mm 避免闪烁，没有物理碰撞，也不会改变机器狗的渲染。
`play` 关闭高度失败条件，保留模型配置中的倾斜失败与回合超时；训练仍使用完整终止条件。

### 楼梯测试

```bash
bash run.sh go2 play --task running --checkpoint latest --terrain stairs
```

机器人出生于平地区域，沿自身初始朝向前进约 2 m 后进入楼梯。
每级高 5 cm、踏面深 35 cm，上楼 6 级（累计高 30 cm），经过 1.4 m 平台后下楼。
楼梯宽 4 m，周围平地大小为 24×12 m；GL 只对地面显示交替踏面颜色和深色台阶边缘。
几何参数在 `simulation/terrain.py` 顶部的 `STAIR_*` 常量中。

`--resume` 可以指定 `.pt` 文件或包含 `latest.pt` 的运行目录。上面的站立副本仍保留真实
来源配置；训练入口识别 standing → running 迁移，加载网络权重、轮号和历史，优化器重新初始化。
新奖励与速度参数取自当前 `tasks/running/`，本次结果保存到新的 running 运行目录。
从第 9500 轮再训 1000 轮，新阶段为第 9501～10500 轮，原模型目录不会被覆盖。

保存配置同时记录来源任务、来源模型相对 `logs/` 的路径和本次日志相对路径。
整理目录前的 `logs/<run>/` 来源路径会按任务查找，支持现在的 `logs/standing/` 与 `logs/running/` 排布。

```bash
# 优先继续本任务最近完成的训练；尚无 running 模型时使用其目录里的站立种子
bash run.sh go2 train --task running --resume latest --max_updates 1000 --run_name continue_running

# 无窗口播放，只测试零速度指令，运行 120 个控制步
bash run.sh go2 play --task running --no-keyboard --viz none --steps 120
```
