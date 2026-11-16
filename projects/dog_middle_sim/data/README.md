# Middle 参考数据

## go2_canter

从之前 Go2 默认行走训练用的 `ai4_dog_canter/motion.npy` 映射而来。
它实际是 canter 慢跑步态，沿用原 `config.json` 的 `[430,459)`：29帧、60 Hz、周期约0.467秒。
原动作来源、作者、许可（CC BY-NC 4.0）及数据 SHA256 保存在 `clip.json` 中；
映射不会改变原参考数据的许可限制。

映射过程：

1. 读取原61列格式和关节分组，用 Go2 模型 FK 计算真实脚端位置。
2. 转换 Go2 前方+X/左方+Y → middle 前方-Y/左方+X，根朝向做坐标基变换。
3. 以源周期平均足端位置作为摆动中心，对齐 middle 自己的站姿；按腿长比例缩放，
   并额外将水平步幅乘0.65、抬脚高度乘0.4，适配 middle 较小的膝关节活动范围。
4. 使用 middle 的实际12转轴、关节限位和脚底碰撞网格，做逐帧IK和整周期平滑优化。
5. 根部高度随脚底约束调整，所有碰撞网格最低点进行地面检查；
   关节/根高度/roll-pitch闭合周期，保留周期根部xy位移与yaw变化。
6. 用相邻周期的姿态通过 MuJoCo 广义位置差分生成 qvel，端点速度不设零。

这不是直接复制 Go2 关节角，也不是迁移 Go2 策略权重。
这是运动学参考，不是训练后策略的真实运动，尚未验证动力学可跟踪性。
特别是原 URDF 声明的1.5 rad/s速度限值并未被满足：实际需求和超限比例记录在clip.json中。
当前 MuJoCo 电机采用力矩限幅，不会自动执行 URDF 的速度限幅；仅放慢回放也不等于解决动力学问题。
现有训练入口仍然是 standing；这份数据先用于映射检查，没有自动替换站立训练目标。

## 查看和重建

在工程根目录：

```bash
# 连续循环，根部累计移动，不回到起点；空格暂停/继续
bash run.sh middle reference --speed 0.5

# 蓝色小球显示IK脚端目标，方便对照几何位置
bash run.sh middle reference --speed 0.5 --show-targets

# 无窗口检查三个周期
bash run.sh middle reference --headless --cycles 3

# 重新映射（不会修改原Go2数据、URDF或MJCF）
OPENBLAS_NUM_THREADS=1 bash run.sh middle retarget
```

## motion.npz 格式

读取使用 `np.load(path, allow_pickle=False)`。

- `qpos`：(T,19)，根位置3、根wxyz四元数4、按 `joint_names` 排列的12关节角。
- `qvel`：(T,18)，世界线速度3、机身角速度3、12关节速度，符合 MuJoCo 自由关节约定。
- `times`：(T,)，秒；`fps`：数据帧率。
- `joint_names`：(12,)，middle 原生关节顺序。
- `target_feet`：(T,4,3)，四足映射目标，世界坐标；z表示目标脚底最低高度。
- `actual_feet`：(T,4,3)，实际 `*_sole` site 位置；不一定就是每个姿态下的最低点。
- `foot_ground_height`：(T,4)，实际小腿碰撞网格的最低高度。

四足顺序 FL、FR、RL、RR。每个周期最后一帧是下一周期起点，
关节角重复第一帧，但根部保留前进位移；`reference.Motion` 用 SE(2) 接续，不能简单取模重置。
加载时验证模型哈希与关节顺序。修改了 MJCF 后需重新映射。
