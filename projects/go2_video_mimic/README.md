# Go2 视频动作 · Anchor SuperMimic

独立实验：把 dog2go2 从侧跑视频重定向得到的 Go2 动作，接到现有 SuperMimic
训练链路。目标是学习相对身体动作和跑步节奏，允许机器人在物理世界中自由移动。
参考视频的相机运动和真实世界跑速不在这里恢复。

当前采用用户选定的 **原片3×播放速度作为正常速度**。准备参考、新训练和参考录像
默认统一使用 `data/side_run_3x/motion.npy`：50 Hz、92帧、首尾时间跨度1.82秒。

## 运行

在 `RL_Learning` 根目录执行。另一台机器需要 Linux x86_64、NVIDIA GPU及兼容
CUDA 13.0的驱动、Docker和NVIDIA Container Toolkit，当前本机验证GPU为RTX 4060 Laptop 8GB。
宿主机不需要安装Python环境、Isaac Sim、AniMer或SMAL；Go2资产和3×训练参考均在Git中。

直接复用已验证的 `isaaclab-base:local` 镜像，不需要重新build。
另一台已有同名、同依赖镜像时，直接拉代码运行：

```bash
git clone --depth 1 https://github.com/RedogJ/RL_Learning.git
cd RL_Learning
bash run.sh video-mimic doctor
bash run.sh video-mimic check
bash run.sh video-mimic train --num-envs 64 --cycles 1000
```

另一台没有镜像时，在原机器导出现有镜像，将生成的tar文件复制过去再导入：

```bash
# 原机器（镜像文件较大，存放位置需有足够空间）
docker image save isaaclab-base:local -o /存放位置/isaaclab-base-local.tar

# 另一台机器
docker image load -i /复制位置/isaaclab-base-local.tar
```

这是现成镜像的搬运，不执行构建或重新安装依赖。
已验证的核心版本为PyTorch 2.12.0+cu130、MuJoCo/MuJoCo Warp 3.12.0、
Warp 1.17.0、NumPy 2.5.1、SciPy 1.17.0、Pillow 12.3.0。
首次训练会编译并缓存Warp CUDA内核，这是运行缓存，不是重新构建Docker镜像。
`doctor` 检查GPU分配、核心依赖、Go2资产、默认3×参考及EGL无窗口渲染，
并报告是否有可用的FFmpeg；缺少编码器不影响训练。
训练从已提交的参考直接启动；只有重新导出参考时才需要执行 `prepare`。
显存较少时可降低 `--num-envs`、`--actor-batch`、`--wm-batch` 和 `--wm-width`。

MP4录像另需编码器：将原机器的
`projects/go2_video_mimic/.cache/ffmpeg`（约77MB）复制到另一台仓库的相同位置，
执行 `chmod +x projects/go2_video_mimic/.cache/ffmpeg` 后即可
`bash run.sh video-mimic reference`。只迁移这一个静态文件，不需要第二个镜像。
也可设置 `VIDEO_FFMPEG=/本机静态ffmpeg路径`，或使用镜像已有的FFmpeg。
`SUPERMIMIC_IMAGE`、`SUPERMIMIC_PYTHON` 可指定已有兼容镜像及其中的Python路径。
Git不包含镜像本体、训练权重、日志、缓存和生成录像；从头训练不需要这些文件。
若要续训，另外复制对应日志目录，并使用其中检查点的 `--resume` 参数。

```bash
# 默认准备3×正常速度参考；原片保留慢动作时序，只在这里重定时一次
bash run.sh video-mimic prepare

# 时序、关节速度/加速度、限位与直接PD物理检查
bash run.sh video-mimic timing

# anchor 坐标、速度运输项、奖励梯度、关节顺序与非循环端点检查
bash run.sh video-mimic check

# 新训练默认使用3×参考；20轮世界模型、5轮Critic预热，第26轮起更新Actor
bash run.sh video-mimic train --num-envs 64 --cycles 100 \
  --wm-updates 8 --actor-batch 64 --run-name anchor_side_run_3x

# 从自己项目的检查点续训；cycles 是追加轮数
bash run.sh video-mimic train \
  --resume projects/go2_video_mimic/logs/<运行目录>/model_100.pt --cycles 500

# 无噪声、从参考开头开始，真实物理评估并录像
bash run.sh video-mimic evaluate \
  --checkpoint projects/go2_video_mimic/logs/<运行目录>/model_100.pt \
  --video projects/go2_video_mimic/videos/policy_100.mp4

# 另外测试随机参考阶段；完成率相对于各起点剩余参考时长
bash run.sh video-mimic evaluate \
  --checkpoint projects/go2_video_mimic/logs/<运行目录>/model_100.pt --random-start

# 单独检查3×参考；此视频是运动学播放，输出videos/reference_3x.mp4
bash run.sh video-mimic reference
```

`play` 是 `evaluate` 的同义入口，当前输出文件视频，不打开桌面窗口。
`--video` 录像左边为策略真实物理、右边为按实体 XY/yaw 对齐的蓝色参考，
两边使用跟随实体的观察相机。评估在真实失败处结束该回合，不把失败后的重置算成完成。
无噪声从头评估的多个环境初态相同，结果不代表独立随机扰动下的成功概率。

## 参考数据

`data/source/side_run_go2.pkl` 来自 dog2go2 的
`independent_side_run_9898378.pkl`；原视频、裁剪和来源记录在同目录
`video_source.json`，其中保留 Pexels/Brixiv 的来源链接。原视频本身为慢动作。

`prepare.py` 验证输入，将 FR/FL/RR/RL 转成 FL/FR/RL/RR，xyzw 转 wxyz，
再转换为原 Go2 环境使用的 61 列容器。仅根位姿和关节位置定义参考；
身体位姿及速度由当前 Go2 模型 FK 计算，未用的兼容字段为零。
所有版本均为50 Hz，不闭合端点、不强制循环。采样时查询源动作的 `speed × t`，
速度由重定时后的FK重新计算，实际训练目标也随之改变，并非仅加快录像。
连续轨迹的速度、加速度分别随倍率及其平方缩放；固定50 Hz重采样后的离散峰值
还受采样位置和重建抖动影响，不能直接把下表峰值按倍率相乘。

| 参考 | 帧数 | 首尾时间跨度 | 估计关节周期 | 离散峰值关节速度 | 离散峰值关节加速度 |
| --- | --- | --- | --- | --- | --- |
| 1×慢动作基线 | 274 | 5.46 s | 2.22 s | 12.63 rad/s | 259.7 rad/s² |
| 3×正常速度（当前采用） | 92 | 1.82 s | 0.74 s | 15.34 rad/s | 428.3 rad/s² |
| 估计5× | 55 | 1.08 s | 0.444 s | 20.38 rad/s | 672.5 rad/s² |

用户看过倍率对照后选定3×作为正常速度，元数据记录为 `user_selected_normal_speed`，
录像标为 `NORMAL SPEED 3x - user selected`。原片只给出约24 fps播放信息，
这个选择不推断拍摄帧率。5×保留为历史估计对照。
关节周期由慢动作关节序列自相关估计后按倍率换算。
5×末端裁去源时间轴0.06秒，以保持严格50 Hz，不额外拉伸最后一个控制间隔。
元数据记录倍率、估计状态和裁尾量；三个版本独立保存，旧参考及检查点保持可复查。
新训练默认采用 `side_run_3x`，续训及策略评估沿用检查点原参考。
此前1×/5×检查点仍属于各自历史试验；训练3×应从头启动，不能直接把旧检查点当作3×策略。
仿真仍为600 Hz、控制50 Hz、重力−9.81 m/s²。

[当前3×正常速度参考](videos/reference_3x.mp4)。历史
[1×/3×/5×同时间轴对照](videos/reference_timing_comparison.mp4)保留当时的估计标签；
快版本到终点后标注并冻结末帧，不循环拼接。
以上都是运动学参考，不是训练后的物理表现。
`logs/timing_comparison.json` 还记录了直接关节PD物理探测：1×/3×分别在约0.17/0.19秒
出现非脚端触地；5×到达1.08秒参考末端，但关节RMSE为0.344 rad。
这种短时开环测试只能辅助检查，不能证明已恢复真实速度或已能稳定奔跑。

参考 XY/yaw 选择固定坐标规范；高度、roll/pitch、关节运动保留。
这种处理没有估计相机或物理速度。实体根部仍在真实物理中自由运动。
未来目标与奖励都去除参考平面位移和绝对 yaw，避免从其他通道重新要求跟踪错误轨迹。

## 与原 SuperMimic 的关系

直接导入 `go2_mujoco_mimic` 的：

- Go2模型、`WarpGo2Env`、PD子步和CUDA Graph；
- 37维连续状态 `Dynamics`、`MotionPolicy`、`StateValue`；
- `Replay`、Critic固定目标回归、可微身体运动学与基础奖励函数。

新项目独立实现 anchor 任务、参考准备、训练编排和评估，不修改原任务。
原 Go2/Middle 源码及所有机器人资产在容器内只读；仅本项目的日志、视频和缓存可写。
`prepare` 模式另外开放本项目 `data/`。检查点和日志均不进入 Git。

### Anchor 定义

身体位置减去根的水平平移，并转换到各自 yaw 坐标系；保留世界高度和身体倾斜。
身体朝向去除整体 yaw。相对速度同时扣除根部平移速度和 yaw 转动运输项，
不会继续追踪原视频错误的世界水平速度。根位置项因此只约束高度，根朝向项约束倾斜。
未来10帧目标为关节角12、高度位置3（XY为0）、去yaw四元数4。

保留关节模仿、动作平滑、限位、力矩和加速度项；另有高度门控的足端滑动软惩罚，
这个门控不是接触力传感器。真实非脚端触地、过大高度/倾斜误差终止回合。
想象中的非脚端接触用膝部高度近似软终止，仍需通过真实仿真评估。

### 控制与学习

- 控制50 Hz，物理600 Hz；PD Kp=25/Kd=0.5，力矩限幅23.7 N·m。
  地面geom摩擦0.5，脚掌优先级摩擦沿用Go2模型中的0.4。
- 固定动作映射：`正常站姿角 + 每关节固定尺度 × action`，裁到机械限位。
  尺度覆盖每个关节的完整范围；零动作对应正常站姿。该映射不依赖参考阶段，
  真实环境与世界模型具有一致的动作语义，不引入隐藏的参考PD偏置。
- Actor沿用目标编码器→32维latent→历史解码器；本项目隐藏层512/256/128。
  世界模型默认2048×3，Critic256/128；10帧历史和10帧未来目标。
- 真实仿真→Replay→8步WM监督→16步闭环想象→Actor/Critic，gamma=.99、lambda=.95。
  Critic对固定、detach的lambda目标做5 epochs×4 minibatches更新。
- WM监督和Actor想象起点排除约20%的保留环境；其多步预测误差单独记录。
- reset使用随机参考阶段及参考广义速度；回合到参考终点结束，不拼接不连续的首尾。
- 检查点包含Actor/WM/Critic及优化器、模型/参考哈希、动作映射版本和配置。
  续训重新采集Replay，不承诺逐帧重放；只接受本项目格式，拒绝旧空翻/站立权重静默混用。

## 判断训练结果

`metrics.jsonl` 报告真实奖励率、已结束回合平均收益/时长、失败次数、WM误差、
Actor更新状态和保留环境误差。没有结束回合时使用null，不把采集窗口时长当回合时长。
训练的随机阶段回合长度不同，不能直接把训练平均存活时间当成从头完成率。
最终用 `evaluate` 的真实物理完成率、相对身体误差和实体位移判断。
参考录像和WM loss下降都不代表已经学会稳定跑步。

当前只使用一段短参考；原始重建误差、形态差异和动力学可行性仍会影响结果。

## 慢动作基线100轮试训记录

运行目录：`logs/anchor_side_run_trial_20261117_031236_924140/`。
64个真实环境，2048×3世界模型，每轮8次WM更新，Actor想象64个起点×16步。
这一旧试验使用1×慢动作时序，不代表正常速度跑步训练。7项检查（包括重定时速度和
加速度缩放）、真实GPU采集/Actor/Critic更新、检查点续训均已实际通过。

| 无噪声真实物理评估 | 从头开始 | 随机参考阶段开始 |
| --- | --- | --- |
| 环境数量 | 16，同一确定初态 | 64，不同参考阶段 |
| 到达参考终点比例 | 16/16 | 46/64 |
| 平均存活时长 | 5.46 s | 1.83 s，平均剩余参考2.41 s |
| 相对身体位置RMSE | 0.134 m | 0.138 m |
| 关节位置RMSE | 0.440 rad | 0.425 rad |
| 平均首尾水平位移 | 0.30 m | 0.18 m |

[策略对比视频](videos/policy_100.mp4)：白色实体是真实物理策略，蓝色为anchor对齐参考。
初步策略能在固定初态存活到片段结束，但动作幅度和时序仍有明显跟踪差距，尚未学好奔跑。
训练期间含噪声、随机阶段回合的平均存活时间约0.2→0.8秒，不能与上述从头评估直接混为一个指标。
详细评估JSON、逐帧回放NPZ和检查点都保留在运行目录。

如需专门继续慢动作基线（不会自动切换到3×）：

```bash
bash run.sh video-mimic train \
  --resume projects/go2_video_mimic/logs/anchor_side_run_trial_20261117_031236_924140/model_100.pt \
  --cycles 500 --run-name anchor_side_run_continue
```

日志、权重、缓存与生成视频是本地结果，Git默认排除；迁移现有策略时需另外复制相应运行目录。

## 估计5×的100轮对照试训

运行目录：`logs/anchor_side_run_est5x_trial_20261117_033726_156429/`。
与慢动作基线使用相同64环境、网络规模、8次WM更新及64个Actor想象起点，
从头初始化，没有续用慢动作权重。参考为55帧、首尾时间跨度1.08秒的估计5×版本。

| 无噪声真实物理评估 | 从头开始 | 随机参考阶段开始 |
| --- | --- | --- |
| 环境数量 | 16，同一确定初态 | 64，不同参考阶段 |
| 到达参考终点比例 | 0/16 | 20/64 |
| 平均存活时长 | 0.26 s | 0.243 s，平均剩余参考0.478 s |
| 相对身体位置RMSE | 0.116 m | 0.123 m |
| 关节位置RMSE | 0.416 rad | 0.412 rad |

[5×试训真实物理录像](videos/policy_est5x_100.mp4)仅约0.28秒，保留初始帧和失败终止帧，
未加速编码、未延长存活画面。策略还没有学会完成这段动作。
误差按失败前的有效帧统计，因此不能把较小的身体误差解释为比5.46秒慢动作基线更好。
随机阶段的较高完成比例也包含接近末端的短回合，不能解释为完整片段成功率。
直接PD曾到达参考末端，而此短训策略提前终止：当前结果不足以判断该参考是否可被
充分训练后的策略稳定跟踪，更不能据此确认原片的真实时间倍率。

该5×试验作为历史对照保留，当前默认训练入口已切换到3×。
