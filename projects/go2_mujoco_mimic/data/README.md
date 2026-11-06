# 参考轨迹

参考是带时间序列的机器人状态，不是视频，也不是控制指令。训练由Actor输出动作、物理仿真执行；参考回放则直接设置位姿，两者应区分。

| 目录 | 用途与来源 |
| --- | --- |
| `go2_flip_to/sideflip_left_30hz/` | 当前侧空翻训练，77帧、30Hz；由 `quad_sideflip_50hz.txt` 重采样 |
| `go2_flip_to/sideflip_left/`、`sideflip_right/` | 原50Hz转换后的左右侧空翻预览 |
| `solo12_backflip/reference_30hz/` | 当前后空翻训练；Solo12轨迹经Go2运动学适配、重采样并保留末态 |
| `kine2go/` | 早期步态/爬行等参考，不是当前空翻的默认参考 |

## 当前读取格式

`motion.npy` 为有限浮点数组 `(T,61)`；`clip.json` 提供fps等元数据。
当前 `Reference` 实际读取：

- `[:,6:18]`：关节角，文件按同类关节分组，读取时转为FL/FR/RL/RR，每腿hip/thigh/calf。
- `[:,48:51]`：根部世界位置，m。
- `[:,51:55]`：根部四元数，wxyz。

其余列保留数据格式兼容；身体位姿由同一Go2 MJCF正运动学生成，不直接信任文件中的足端列。
参考速度由MuJoCo广义位置差分生成，reset时同时恢复位置与速度。四元数相邻符号连续化，避免q/-q引入插值跳变。

## 重新生成侧空翻参考

在已有NumPy/SciPy的Python环境、仓库根目录执行：

```bash
PYTHONPATH=projects python -m go2_mujoco_mimic.prepare_go2_amp_reference \
  projects/go2_mujoco_mimic/data/go2_flip_to/quad_sideflip_50hz.txt \
  projects/go2_mujoco_mimic/data/go2_flip_to/sideflip_left_30hz --fps 30
```

原始侧空翻文件来自 `yusongmin1/go2_flip_TO` 的 `datasets/go2/mocap_motions_go2/quad_sideflip_50hz.txt`。
原始49列为根部位置3、xyzw四元数4、关节角12、局部足端12、线速度3、角速度3、关节速度12。
项目内转换脚本不改变原始动作的运动任务。

## 后空翻适配

`solo12_backflip/backflip_v4.hdf5` 是Solo12参考，来源为 `upatras-lar/se3_trajopt`。
`play_solo_backflip.py`解释MRP姿态和关节顺序，`retarget_solo_backflip.py`生成Go2适配预览，`prepare_backflip_reference.py`输出30Hz参考。
这是运动学适配；已训练策略完成动作不等于该参考的每一帧都具有精确动力学可行性。

## 来源与许可记录

Kine2Go各片段的来源和许可记录在各自 `clip.json` 中，部分片段是CC BY-NC 4.0，不能统一当作宽松许可的数据。
当前侧空翻和Solo12原始数据的固定来源版本、单独许可文本未完整记录；本整理明确保留此状态，不替它们声明许可证。
第三方来源汇总见[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
