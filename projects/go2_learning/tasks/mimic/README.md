# Go2 两专家潜变量蒸馏

代码位于 `tasks/mimic`，使用独立生成式学习入口，不注册为 PPO 的 reward task。
现有 standing/running/upright 的奖励及训练代码不改动。

## 专家已固定

`experts.json` 固定源码路径、权重和配置 SHA256。副本位于 `logs/mimic/experts/`：

- upright：`2026-10-22_21-21-35_342854_upright_restart_v25/latest.pt`
- running：`2026-10-17_15-07-50_097333_running_18/latest.pt`

此后的 latest 更新不会改变专家。迁移服务器时一并复制 `logs/mimic/experts`。
`pin` 遇到现有 manifest 只校验，不替换。

## 与 BeyondMimic 的关系和区别

依据论文 v4：https://arxiv.org/html/2508.08241v4 ，Methods 和 S3/S4。
实现条件 VAE、DAgger 数据聚合、学生轨迹采集、状态—latent 联合去噪、历史硬条件和状态目标引导。
这是使用 Go2 任务专家的缩小版实现，尚未证明与原文等效或达到原文效果。

1. 专家是手写奖励训练的 running/upright，不是参考运动跟踪器。
   编码器输入使用技能 one-hot、速度指令和目标高度，共 6 维，代替原文的参考动作/锚点误差。
   解码器输入为 16 维 latent 与 49 维实时本体观测；编码器不直接压缩带尖峰的动作。
2. 扩散状态包含机身速度、重力、关节状态、高度及所有刚体的相对位置/速度。
   采用瞬时完整机身坐标系，避免 upright 时从近竖直机身 X 轴定义 yaw；
   不是原文的混合 character-yaw 坐标。当前只支持机身速度/直立/高度引导，未实现世界航点或避障。
3. 控制周期保持两个专家的 30 Hz；history=4，horizon=16，因此未来约 0.533 秒。
   原文扩散实验 25 Hz、0.64 秒。Go2 使用较小 Transformer，默认 4 层/4 头/128 embedding。
4. 去噪使用 cosine 噪声表与 20 步确定性 DDIM，未实现原文异步 TensorRT/C++ 部署、EMA、
   逐关节对称增强。已加入 Go2 适配的强调投影，见下文。DAgger 使用轨迹聚合再离线更新，不在线回传仿真梯度。
5. 采集执行可加入 OU 动作扰动，监督目标始终是不加噪声的专家输出。
   扩散排除已失败片段，严格保留 reset 边界；截断但未失败的片段也可使用，
   尚未执行原文额外延续到 5 秒的筛选。
6. 只有稳定跑动/直立数据不保证可起身转换。需要分别验证技能保持，再验证切换。
   两专家初始状态、积分、命令和目标高度观测仍按各自保存配置运行。

## 文件职责

| 文件 | 功能 |
|---|---|
| experts.py / experts.json | 专家固定与完整性校验 |
| features.py | 从仿真提取学生输入和 Cartesian 物理状态 |
| simulation.py | 专家播放、学生播放、专家标注与轨迹采集 |
| data.py | 按轨迹划分和技能均衡采样，不跨 reset |
| models.py | 条件 VAE、联合状态—latent 扩散模型 |
| train.py | 离线训练和验证集最优模型保存 |
| main.py | 各阶段命令与 DAgger 调度 |
| config.py | 网络维数、窗口、KL、学习率等 |

## 按顺序运行

全部命令在 RL_Learning 根目录执行。Docker 使用现有 `isaaclab-base:local`，不下载依赖。
数据和模型默认保存在 `logs/mimic`；自定义路径应使用容器路径 `/workspace/go2_learning/logs/...`。

### 1. 播放专家并记录第一批轨迹

```bash
bash run.sh go2 mimic collect --expert running --num_envs 16 --steps 900 --viz none
bash run.sh go2 mimic collect --expert upright --num_envs 16 --steps 900 --viz none
```

保存前置状态、实际动作、专家原始动作标签、本体观测、意图和 reset 边界。
`--controller expert` 为默认值。48 步等短采集仅验证接口，不能作为正式训练数据。

### 2. 首次训练 VAE

```bash
bash run.sh go2 mimic train-vae --updates 2000 --batch_size 256
```

同时要求两专家数据；按完整轨迹划分训练和验证集；按技能均衡抽样。
输出 `logs/mimic/models/vae.pt` 为本次验证 MSE 最低模型。
MSE 下降不等于闭环技能成功，务必分别播放。

```bash
bash run.sh go2 mimic play --expert running --controller vae --num_envs 1 --steps 900 --viz newton_gl
bash run.sh go2 mimic play --expert upright --controller vae --num_envs 1 --steps 900 --viz newton_gl
```

### 3. DAgger：学生执行，专家标注

```bash
bash run.sh go2 mimic dagger --rounds 3 --steps 900 --num_envs 16 --updates 1000
```

每轮依次采集 running/upright 学生访问的状态，由对应冻结专家给动作标签，
聚合全部旧数据再训练 VAE。随轮数减少执行中的专家混合比例。
重置等待阶段使用任务原有 PD 目标，不将这些样本送入训练。

### 4. 固定 VAE，采集潜变量轨迹

先确认 VAE 的两个技能都能保持，再执行：

```bash
bash run.sh go2 mimic collect --expert running --controller vae --num_envs 16 --steps 1200 --noise 0.05 --output /workspace/go2_learning/logs/mimic/latent_data --viz none
bash run.sh go2 mimic collect --expert upright --controller vae --num_envs 16 --steps 1200 --noise 0.05 --output /workspace/go2_learning/logs/mimic/latent_data --viz none
```

这时不要继续更新 VAE；不同 VAE 的潜变量轨迹禁止混用。

### 5. 训练扩散模型

```bash
bash run.sh go2 mimic train-diffusion --data /workspace/go2_learning/logs/mimic/latent_data --emphasis 6 --updates 5000 --batch_size 128
```

默认输出 `logs/mimic/models/diffusion_emphasis.pt`，包含对应 VAE 副本及投影矩阵，避免解码器错配。
原来的 `diffusion.pt` 保留；播放新模型时需显式指定 `--model /workspace/go2_learning/logs/mimic/models/diffusion_emphasis.pt`。
`--emphasis 0` 可训练原结构；不要把旧权重强行加载进投影结构。
状态及 latent 归一化参数存入模型。训练中历史与当前状态硬约束，预测当前 latent 和未来状态/latent。

### 6. 分别验证扩散闭环

```bash
bash run.sh go2 mimic play --expert running --controller diffusion --model /workspace/go2_learning/logs/mimic/models/diffusion_emphasis.pt --num_envs 1 --steps 900 --vx 1.5 --guidance 0.05 --viz newton_gl
bash run.sh go2 mimic play --expert upright --controller diffusion --model /workspace/go2_learning/logs/mimic/models/diffusion_emphasis.pt --num_envs 1 --steps 900 --guidance 0.05 --viz newton_gl
```

`--guidance 0` 关闭引导作为基线；正值使用未来速度、重力方向、高度代价。
`--guidance_mode output` 是原来的输出轨迹修正（默认保留）。
`--guidance_mode input` 是实验性的输入梯度更新：每轮由未来代价经过固定的去噪网络
回传到本轮带噪轨迹，再在 DDIM 更新后作梯度下降。权重不更新，不跨全部去噪轮次反传，
历史与当前观测仍为硬约束。它不是已核验的官方 diffusion 源码还原。
官方公开仓库提供运动跟踪代码。`--cost_mode separate`（默认）使用独立权重，速度代价改为
`0.5 * sum_t ||v_horizontal-command||²`；姿态/高度/转向仍为各自的时间均值。
速度项直接累加各时刻的误差，不除以时间点数，也不使用随时间衰减的权重。
`--cost_mode legacy` 完整保留旧的时间均值混合代价。不同模式下的 guidance 数字不能直接比较。
两种梯度路径的同噪声数值/速度阶跃结果见
`logs/mimic/response/`；input 路径并未解决全部响应问题，暂不替换默认路径。
CUDA 播放默认用 CUDA Graph 提交完整的 20 步去噪，首次推理会先准备图；
采样公式、引导和权重不变，每步仍生成新噪声。`--eager` 可关闭图捕获作对照。
窗口每个控制周期刷新一次，全部物理子步照常执行。
`play` 跳过奖励和训练统计计算，保留相同的动作、积分观测、跌倒/超时与逐回合等待；
`collect` 和 PPO 训练仍计算完整奖励和指标。
播放结束输出 `control_hz` 和 `realtime_factor`（排除前 10 步准备），后者 ≥1 表示达到实时速度。
暂未接入异步实时调度；仿真慢于实时时仍按控制步运行。
不要将冒烟训练生成的模型当作可部署策略。

### 强调投影与独立引导配置

`config.py` 的 `GuidanceCfg` 是默认配置，可通过 CLI 临时覆盖：

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--velocity_guidance_weight` | 1 | 独立的水平速度平方误差时间求和 |
| `--orientation_guidance_weight` | 1 | 重力方向平方误差时间均值 |
| `--height_guidance_weight` | 4 | 高度平方误差时间均值 |
| `--yaw_guidance_weight` | 1 | 键盘 yaw 角速度平方误差时间均值 |

权重设为 0 可关闭对应项，总体 `--guidance` 仍控制更新步长。
直立目标的向上轴为 body X，独立速度项因此使用 body YZ；四足目标使用 body XY。
这是现有完整机身坐标表示的适配，并未切换为论文的混合 yaw 坐标表示。

`projection.py` 对归一化后的物理状态构造 `[AB; I]`，B 在 v/w/gravity/height 维度取 c=6，
其余维度取 1。A 为固定种子的高斯矩阵，行数取原状态维数，并对整个矩阵作共同 RMS 缩放，
保持输入总体尺度，同时保留根特征的相对强调；不做会抵消强调的逐投影通道归一化。
这些维度、A 的行数和共同缩放是 Go2 工程适配，不能称为逐项复现官方表示。
latent 不参与投影。训练/采样均使用同一个保存的矩阵，物理代价通过伪逆回到 m/s、m 等原单位；
历史和当前已知物理观测在逆投影后精确写回。
原始采集轨迹、VAE 与专家权重均可复用，仅需重新训练 diffusion。
训练附带 `.history.jsonl`，记录训练步数、验证损失、物理速度和 latent 重建 RMSE；
这些是去噪验证指标，不代表闭环控制性能。

### 键盘切换技能目标

```bash
bash run.sh go2 mimic play --expert running --controller diffusion --skill_switch --num_envs 1 --steps 18000 --viz newton_gl
```

点击窗口后：Tab 切换 running/upright，1 选 running，2 选 upright。
方向键默认启用：上/下为 ±1.5 m/s，左/右为 ±0.6 rad/s 转向，松键或失焦归零；
upright 下移动指令归零。`--no-keyboard --vx 1.5` 可改为固定速度。
转向使用未来机身 wz 的引导项；现有采集主要覆盖前进/后退/静止，不能保证转向效果。
整个过程使用同一个 diffusion 和 VAE 解码器，不切专家，不在按键时 reset，也不清空扩散历史。
两个任务有效姿态区域的并集用于切换模式的失败判定；超时和失败仍重置到启动技能的初始姿态，
这些 reset 不属于成功过渡。姿态/高度积分按所选技能重新初始化，保持各专家的观测约定。
原来单技能播放的终止规则不变。现有数据没有专门的起立/落地过渡轨迹，因此这是过渡能力测试，
不是已经验证成功的技能衔接策略。

## running 专家验收与数据修复

running 专家按用户确认固定为连续训练线的 running_18，而非日期更晚的 zero_1 从零分支。
固定 -1.5/0/+1.5 m/s、每组 15 秒的物理测试中，去掉起步 2 秒后实测约
-1.691/0.002/1.396 m/s，均无失败。记录在 `logs/mimic/evaluation/expert_audit`。
旧 zero_1 专家、running 数据与联合学生模型位于 `logs/mimic/archive`。
upright 教师未更换，普通监督数据按各自的教师指纹校验，因此可保留旧 upright 标签；
联合 VAE 和扩散模型仍要求两专家整组指纹一致，禁止继续加载旧联合学生。

采集纯 running 专家数据时，会按指令检查两秒起步后的实际响应；
有至少 60 个样本却无法向指令方向达到 0.3 m/s 时拒绝作为移动专家。
这个阈值只用于排除“完全不动/方向反了”的无效专家，不是跟踪精度验收标准。

播放专家或 VAE 可以明确给速度：

```bash
bash run.sh go2 mimic play --expert running --controller expert --vx 1.5 --num_envs 1 --steps 900 --viz newton_gl
bash run.sh go2 mimic play --expert running --controller vae --vx 1.5 --num_envs 1 --steps 900 --viz newton_gl
```

## 学生完整输入与指令辨识（2026-10-23 修复）

running_18 使用的本体信息含三个积分观测。学生现为 49 维：原 46 维后面追加
roll/pitch/height 的归一化积分。新采集直接记录真实积分；旧轨迹按 reset 分段、
按原积分公式逐控制步重建，不修改原始轨迹文件。
离线重建完整 52 维专家输入后，可复算原动作标签；重标注要求最大偏差小于 1e-4。
旧 46 维学生被拒绝加载，需要重新训练。

为避免学生只根据当前速度/上一动作延续现有步态而忽略意图，增加同状态不同指令监督：

```bash
bash run.sh go2 mimic relabel
```

对同一个身体状态，分别向冻结 running 专家查询 -1.5/0/+1.5 的动作答案。
这些样本标记 `supervision_only`，仅训练 VAE，绝不作为实际运动轨迹训练扩散。
同一源轨迹及其所有重标注副本一起划分训练/验证，避免泄漏。
DAgger 每轮自动重新生成该监督集，旧重标注文件归档，不重复叠加相同样本。

KL 系数从 0.01 调至 0.001；训练同时优化随机 latent 与播放实际使用的均值 latent，
各占一半动作 MSE；验证按 running/upright 各占一半，避免样本较多的技能主导模型选择。
这些是适配两个无参考轨迹专家的训练措施，不宣称与原文 VAE 训练完全相同。
