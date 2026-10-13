# Go2：场景与 RSL-RL 入门

独立工程，使用现有 `isaaclab-base:local` 镜像和本地官方 Go2 USD。
物理后端为 PhysX，默认可视化为 `newton_gl`，不打开 Kit 编辑器完整界面。
PhysX 运行仍可能输出 Isaac Sim / Kit 的底层初始化日志。

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

| 文件 | 职责 |
| --- | --- |
| `scene.py` | 本地 USD 路径、地面、Go2 初始站姿、关节 PD |
| `env.py` | 实现 RSL-RL `VecEnv`：观测、动作、奖励、结束和重置 |
| `interaction.py` | 轻量窗口右键事件和机身外力 |
| `agent_cfg.py` | 标准 MLP、PPO、采样参数 |
| `main.py` | 创建环境；view 看场景，train/play 使用 `OnPolicyRunner` |
| `run.sh` | Docker 挂载、缓存、显示和 Python 启动 |

数据流：`OnPolicyRunner → env.step(actions) → 关节位置目标 → PhysX → 观测和奖励`。
这里使用项目自己的环境代码，没有继承宇树现成任务配置。

## 可选：最小站立训练

先确认模型加载正常，再检查训练接口：

```bash
bash go2_learning/run.sh train --num_envs 4 --max_updates 2
```

这个短命令仅验证采样与 PPO 更新，不代表已经学会站立。正式训练与播放：

```bash
bash go2_learning/run.sh train --num_envs 64 --max_updates 300
bash go2_learning/run.sh play
```

- 训练默认无窗口，播放默认一个环境和 `newton_gl`。
- 物理频率 120 Hz；每 4 个物理步换一次动作，控制频率 30 Hz。
- 45 维观测：线速度 3、角速度 3、重力方向 3、关节位置偏移 12、关节速度 12、上次动作 12。
- 12 维动作：默认站姿的关节角偏移，上限 ±0.25 rad；PD 刚度 25、阻尼 0.5。
- 站立奖励鼓励直立、机身高度接近 0.32 m，惩罚移动、偏离站姿、过大动作和跌倒。
- 训练/播放时，机身低于 0.18 m 或倾斜过大判失败；10 秒超时，自动重置。
- 这是站立任务起点，还没有速度指令、行走奖励、地形课程或真实电机模型。

权重和 TensorBoard 日志保存在 `go2_learning/logs/`，被 Git 忽略。
每次训练保存 `run.json` 记录网络配置；修改环境后要自行保持旧权重输入、动作语义兼容。

## 模型和运行环境

模型入口：`../assets/unitree_model/Go2/usd/go2.usd`，其 `configuration/` 子层必须一起保留。
来源与下载校验见 [模型说明](../assets/unitree_model/README.md)。
官方地面及贴图位于 `../assets/isaaclab/ground_plane/`，见 [地面资源说明](../assets/isaaclab/ground_plane/README.md)。
迁移时一起携带 `go2_learning/` 和 `assets/`，保持二者为同级目录。

脚本只读挂载代码和模型、读写挂载日志与缓存，使用 `--pull=never --network none`，
以及 `uv run --offline`。无需重新 build 或安装依赖。

创建时已做 Python/Bash 语法与启动参数检查；由于 Docker 需要终端输入 sudo 密码，
尚未实际验证本工程的 GPU 仿真、模型站姿及 PPO 训练效果。
