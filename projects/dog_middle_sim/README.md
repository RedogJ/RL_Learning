# Middle SuperMimic 风格强化学习

独立工程：MuJoCo Warp 真实并行仿真 → Replay → 可微世界模型监督 → 闭环想象 → Actor/Critic。
首个任务为 `standing`，参考来自已经验证能用 PD 维持的静态站姿。
原始 URDF/STL、Go2 工程的配置和日志不修改。

Go2 canter → middle 的独立映射参考已准备，使用 `bash run.sh middle reference --speed 0.5` 查看。
数据格式、映射方法和精度边界见 [参考数据说明](data/README.md)；当前未替换 standing 训练目标。

这是参照官方关节坐标方案搭建的 Middle 实现，不是直接运行官方 G1/mjlab 工程，也不宣称已完成论文复现。
静止站姿本来就能由 PD 维持；短测跑通不能证明学会了新技能或具有抗扰能力。

## 启动

所有命令从 RL_Learning 根目录执行，全部通过 Docker 使用 `isaaclab-base:local`。
宿主机只需 Docker、NVIDIA 驱动及 Container Toolkit；打开窗口还需要 X11 显示授权。
不再调用宿主机 Python，训练、播放、模型生成与参考映射使用同一套依赖。

```bash
# 正式训练由用户启动；每轮打印真实奖励率、回合统计、WM误差和想象目标
bash run.sh middle train --task standing --cycles 500 --num-envs 64 --run-name standing_v1

# 用指定模型续训，cycles 是本次追加轮数；宿主机路径会转换成容器路径
bash run.sh middle train \
  --resume projects/dog_middle_sim/logs/standing/<运行目录>/model_500.pt \
  --cycles 500 --run-name standing_v2

# 播放策略：Docker 内 MuJoCo 窗口，通过 X11 显示到桌面
bash run.sh middle play --checkpoint projects/dog_middle_sim/logs/standing/<运行目录>/model_500.pt

# 在已有 Docker 环境评估 10 秒；真实物理，不通过 WM 生成运动
bash run.sh middle evaluate \
  --checkpoint projects/dog_middle_sim/logs/standing/<运行目录>/model_500.pt --seconds 10

# 原有 PD 演示仍保留
bash run.sh middle
bash run.sh middle --headless --seconds 30

# 多 hinge FK、奖励梯度、Actor/Critic隔离、CUDA Graph及环境独立性检查
bash run.sh middle check
```

`--resume latest` / `play --checkpoint latest` 选择 `logs/standing/*/latest.pt` 中最后修改的一份；
不是最好模型，短测运行也可能被选中，建议使用明确路径。
可通过 `MIDDLE_IMAGE`、`MIDDLE_DOCKER_PYTHON` 替换 Docker 镜像/容器解释器。
所有窗口模式自动传递 DISPLAY/XAUTHORITY；`MIDDLE_PYTHON` 不再使用。
`bash run.sh middle build` 和 `bash run.sh middle check-model` 也在容器中运行。
生成模型只开放 `assets/dog_middle/simulation/` 写入；重新映射开放项目 `data/` 写入。

## 配置与文件

```text
dog_middle_sim/
├── config.py                   # 仿真、网络、训练参数
├── environment.py              # 原生 MuJoCo / Warp 后端、PD、reset、历史观测
├── physics.py                  # Warp PD kernel；CUDA Graph物理子步
├── math_utils.py               # 连续状态、四元数和坐标变换
├── kinematics.py               # 支持同一 body 多 hinge 的可微身体运动学
├── models.py                   # Actor目标编码器、Dynamics、Critic及归一化
├── replay.py                   # 时间×环境的真实transition环形缓存
├── learner.py                  # 多步WM损失、BPTT和λ-return Critic
├── train.py                    # 采集→拟合→想象→更新、日志和检查点
├── play.py                     # 策略真实物理回放/评估
├── tasks/standing/
│   ├── task.py                 # 静态参考、目标、奖励、软/硬终止
│   └── reward_cfg.py           # 分块奖励权重与sigma
├── build_model.py              # 原始URDF→独立MJCF及初始站姿
├── main.py                     # 旧PD演示入口
├── check_model.py              # URDF/MJCF运动学一致性
├── check_training.py           # 训练关键边界检查
├── run.sh / run_gpu.sh         # 模式分发 / 统一Docker入口
└── logs/standing/              # 独立日志，不进入Git
```

统一运行环境的依赖见 `requirements-training.txt`，由镜像提供。
`requirements.txt` 是历史本机查看器依赖，不用于当前 Docker 启动流程。

## 网络与训练过程

- 完整状态37D：根位置3、wxyz四元数4、关节角12、世界线速度3、机身角速度3、关节速度12。
- WM表示34D：高度1、机身重力方向3、机身线速度3、机身角速度3、关节角12、关节速度12。
- WM输入50D：表示34 + 实际PD目标角12 + 固定仿真context4。
- WM输出18D：机身坐标线速度增量3、角速度增量3、关节速度增量12；再积分位姿，
  将速度转换到下一帧机身坐标系。位置积分系数由真实物理子步数计算。
- Dynamics默认2048×3，Linear/LayerNorm/GELU；初始输出层为零。
- Actor：10帧42D自身历史；10帧21D参考经目标编码器得到32D latent，再与历史一起解码12D有界动作。
  静止参考的未来10帧内容相同；目标含关节角12、机身坐标相对位置3、相对朝向6D。
  Actor隐藏层为512/256/128，较官方默认缩小，配置可调整。
- 动作范围[-1,1]；目标角 = 默认站姿角 + 0.25 rad × 动作，再裁到关节限位；每个物理步PD限幅±20 N·m。
- Actor自身观测中的关节速度乘0.1、角速度乘0.2；WM状态和误差日志始终使用真实物理单位。
- 输入归一化由新数据持续更新；WM监督损失按固定物理误差尺度配权。
- Critic包含34D状态表示、210D目标、420D自身历史、4D context、1D剩余时间；隐藏层256/128。
- 前10个cycle只拟合WM，再5个cycle校准Critic，然后更新Actor；续训重新填Replay并执行同样预热。
- 每次Actor更新重新采样起点，闭环执行“Actor→WM→新状态→Actor”；16步回报加折扣Critic尾项，
  软终止累计存活概率；Critic通过同一段想象的λ-return监督，5 epochs×4 minibatches。
- 想象时冻结WM参数但保留输入梯度；Actor尾项冻结Critic参数但保留输入梯度；Critic回归完全detach轨迹。
- 真实失败由参考高度/倾斜误差阈值决定；想象使用同一归一化误差的解析sigmoid，没有BCE终止头。
  超时在真实和想象中均截断，Critic显式看到剩余时间。没有自适应RSI；静止参考没有可区分的运动阶段。
- 真实采集默认附加0.1动作探索噪声，回放采用确定性Actor。reset关节/倾斜噪声逐环境独立采样。
- 新增数据每cycle32步，Replay容量320步，即充满后每cycle替换10%；窗口遇到失败/超时后不监督后续跨reset内容。
- 环境数至少4时，固定保留约20%环境用于WM验证，不进入WM监督batch或想象起点。
- Torch/Warp使用显式stream依赖；固定形状物理子步录入CUDA Graph；不在每个控制步做全设备同步。

## 奖励与指标

奖励集中在 `tasks/standing/reward_cfg.py`，真实与想象共用同一函数：

1. 根部全局位置/朝向跟踪。
2. 身体空间位置/朝向跟踪，先按当前根部水平位置和yaw对齐参考。
3. 身体线速度、角速度跟踪静止目标0。
4. 实际下发目标角变化平方和、关节超软限位惩罚。

全部是每秒奖励率乘控制dt，正收益上限合计5/秒；默认无额外alive或一次性failure罚分。
身体速度以相邻控制帧FK位姿差分计算，真实与想象一致。

- `reward_rate`：当前采集段每环境平均每秒收益，不是回合累计收益。
- 加粗的 `return / episode`：这次采集期间实际结束的回合平均累计收益/存活时间；
  没有结束回合时显示 `--`，不会用短窗口冒充完整回合。
- `WM`：按物理误差尺度加权的多步状态损失。
- `验证关节`：保留环境上自由滚动到窗口末的关节角RMS误差，单位deg。
- `metrics.jsonl` 还记录验证根位置cm、姿态deg、一阶/窗口末误差、各奖励成分、Critic损失等。
- `想象总目标`：折扣的存活加权奖励累计 + Critic尾项，不等同于真实回合收益。

## 保存与续训

每50个cycle和正常训练末尾保存编号模型与硬链接latest，包含Actor、WM、Critic、归一化、优化器、
RNG、参考状态、模型哈希、配置和历史区间；不把Replay和物理环境完整快照塞入检查点。
续训使用当前reward_cfg并继承历史，奖励变化会重置Critic并重新预热。
恢复的是模型和优化状态，不是从中断那一帧逐位重放。模型/关节/站姿不匹配会拒绝加载。
首次正式训练应从零启动，不要把 `_smoke_*` 模型当作已训练的正式版本。

## 转换规则与仿真假设

1. 原始 URDF、STL 不修改。当前使用 `urdf_with_motors/*__with_motors.urdf`，
   保留原有12个关节，以及16个电机相关外壳/安装部件的位姿、网格、碰撞和声明惯性。
   16个外观部件不等于16个驱动自由度，主动关节仍为12个。
2. 源文件四个 `*_UP_virtual` 无几何，填有1 kg、单位惯性占位值。排除这些占位参数，
   将每条腿的两个髋轴放在同一个
   MJCF 大腿 body 中，第一轴在该 body 中偏置、第二轴位于原点。保留两个不共点的
   转轴和运动顺序，不替换成 ball joint。这保持原文件的运动学及其无质量连接假设，
   不表示实机的连接件真的没有质量。
3. 原有九个实体的质量、质心和惯性数值原样保留，包括之前指出的质心疑点；
   额外计入16个电机相关实体各0.01 kg及其声明惯性，总质量约2.846 kg。
   中间/脚端共八个无几何节点的1 kg、单位惯性占位值不计入。
   学习器空间跟踪仍针对九个骨架实体，固定电机附件不重复计奖。
4. 外观和碰撞复用原始 STL。MuJoCo 网格碰撞采用凸包，不是逐三角形凹面碰撞。
   初期只允许机器人与地面接触，关闭机器人自碰撞；碰撞几何为 group 3，默认隐藏。
5. 保留原四个 foot 节点为 site；额外从小腿网格底部选取 `*_sole` site 用于
   站姿初始化/诊断。试验脚掌采用半径2 cm的球形碰撞体，小腿网格仍保留碰撞；
   site 本身不参与碰撞。球体位置与半径需要结合实物校准。
6. 用逆运动学求机身水平的四足站姿，再按全部碰撞网格的最低点放置根高度，避免
   初始穿地。不是从实机导出的站姿。BODY 原点高度不等于机身质心高度。
7. 当前仿真假设：地面摩擦 0.8，关节 damping=0.3、frictionloss=0.02、armature=0.002；
   PD 默认 kp=40、kd=1；电机按原 URDF effort 限幅为 ±20 N·m。
   URDF velocity=1.5 rad/s 不被当成硬速度限幅。均不是经过辨识的电机参数。
8. 这里只能说明模型可加载、运动学一致及当前站姿下 PD 可维持站立；
   不说明可抗任意扰动、不说明已有 RL 策略，也不说明具备 sim2real 精度。

若后续确认中间 link 有真实惯性或修改了关节定义，必须重新评估转换规则；
构建器会对部分不受支持的源结构显式报错，不是通用 URDF 转换工具。
