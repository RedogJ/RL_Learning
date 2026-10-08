# 小车二阶倒立摆

本目录现在是串联双杆环境：世界通过 X 轴移动关节连接小车，
小车通过 Y 轴转动关节连接第一杆，第一杆上端再连接第二杆。
动作仍是一个水平推力，只施加在小车上。物理后端为 Isaac Sim PhysX。

## 运行

在 `RL_Learning` 目录执行，使用已有的 `isaaclab-base:local` 镜像，无需重新 build。

先查看模型，关闭推力，按 Ctrl+C 退出：

```bash
bash run-cartpole-gui.sh learn --control zero --num_envs 1 --viz newton_gl
```

初始两杆竖直，无推力时可能暂时保持静止。`--control pd` 有初始扰动与双杆角度反馈，
可用于检查关节运动；当前 PD 增益未经双杆调节，不保证平衡。

先做两次 PPO 更新，验证实际 GPU 仿真和训练流程：

```bash
bash run-cartpole-gui.sh learn --control train --num_envs 64 --viz none --max_updates 2
```

正式训练与播放：

```bash
bash run-cartpole-gui.sh learn --control train --num_envs 64 --viz none --max_updates 300
bash run-cartpole-gui.sh learn --control play --num_envs 1 --viz newton_gl
```

默认权重保存到 `checkpoints/latest_double.pt`。旧单杆权重 `checkpoints/latest.pt`
仍保留；输入维度改变后需要重新训练。双杆训练能否在 300 次更新内学会平衡，需要看实际表现。
以上命令中的 `learn` 调用本工程 `main.py`；外层脚本的 `train` 和 `play`
调用 Isaac Lab 官方 Cartpole 工作流。

## 参数和代码

| 文件 | 主要内容 |
| --- | --- |
| `scene.py` | `cart`、第一杆 `ball`、第二杆 `ball2`，两个转动关节和状态读取 |
| `main.py` | 初始化几何、训练/播放入口、默认 64 环境、权重保存 |
| `env.py` | 6 维观测、奖励、失败条件、三刚体重置 |
| `networks.py` | Actor 和 Critic 的输入由 4 维改为 6 维，隐藏层仍为 32/32 |
| `rollout.py` / `ppo.py` | 原有采样、GAE 和 PPO 更新 |

默认小车质量 1 kg，两根杆各长 1 m、质量 1 kg；可修改 `scene.py` 中的
`spawn.size` 和 `mass_props.mass`。创建场景前，`configure_cartpole_scene_cfg()`
从这些尺寸计算初始位置和锚点，修改杆长时无需手算第二杆的高度。

观测按以下顺序排列：

```text
[x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
```

`x` 为小车相对本环境初始中心的位移 [m]；速度为 [m/s]。
两根杆的角度均相对世界竖直方向 [rad]，向 +X 倾斜为正；角速度为世界 Y 轴分量 [rad/s]。
第二个关节的相对角度是 `wrap(theta2 - theta1)`，策略的 `theta2` 使用第二杆绝对倾角。
当前 Isaac Lab 的 `root_link_quat_w.torch` 顺序为 xyzw，由 `quat_apply` 求杆的世界方向。

每条动作执行一个物理步，`dt=1/120` 秒，反馈和推力更新都是 120 Hz。
归一化动作限幅到 [-1, 1] 后乘以 20 N 推力上限。

奖励为：

```text
r = (1 - theta1² - theta2² - 0.1*x² - 0.001*a²) * dt
```

其中 `a` 是限幅后的归一化动作。小车位移超过 1.8 m，或任意一根杆绝对倾角超过 30 度，
即判定失败；达到 5 秒回合上限则截断。重置同时恢复小车与两根杆的位姿和速度。

两个杆关节的相对角度行程仍为 ±90 度；这与 RL 的 30 度失败阈值是不同参数。
连接的两个刚体关闭相互碰撞，避免接头附近的几何碰撞干扰关节运动。
局部锚点和关节碰撞属性参见 [OpenUSD Joint 文档](https://openusd.org/release/api/class_usd_physics_joint.html)。

## 备份和验证

转换前的单杆源文件及权重保存在 `single_pole_backup/`。
DeepONet 实验在独立目录中继续使用它自己的场景和网络。

`test_double_cartpole.py` 的 CPU 检查覆盖尺寸变化时的初始锚点、复制环境的两组关节绑定、
绝对角度及环境原点、第二杆奖励和失败条件、三刚体的局部重置、6 维采样与 PPO 更新。
测试使用实际的 Torch 数学函数和简化的数据夹具，不启动 PhysX。
真实的 GPU 场景、关节运动和收敛效果需通过上面的运行命令验证。
