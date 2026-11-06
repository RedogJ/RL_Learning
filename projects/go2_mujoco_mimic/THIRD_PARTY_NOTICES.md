# 第三方方法、模型与数据来源

本文件记录本工程实际使用的来源。许可证以各上游文件及保留的原文为准。

| 内容 | 来源 | 本地记录 |
| --- | --- | --- |
| 世界模型训练思路 | SuperMimic，arXiv:2610.28480 | 结构和适配差异见 docs/architecture.md |
| 身体跟踪奖励参考 | BeyondMimic / `HybridRobotics/whole_body_tracking` | 仓库根 beyondmimic/；轻量分享包不携带该参考仓库 |
| 物理后端 | `google-deepmind/mujoco`、`google-deepmind/mujoco_warp`、NVIDIA Warp | 通过环境依赖安装，保留各包许可 |
| Go2 MJCF和mesh | `unitreerobotics/unitree_mujoco`，revision `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` | 根 assets/unitree_model/Go2/mjcf/ 的 LICENSE（BSD-3-Clause）及download_manifest.json |
| Go2侧空翻参考 | `yusongmin1/go2_flip_TO` | data/go2_flip_to/；固定源版本/单独许可文本待补全 |
| Solo12后空翻参考 | `upatras-lar/se3_trajopt` | data/solo12_backflip/；原始数据固定源版本/单独许可文本待补全 |
| Kine2Go片段 | `MIMUW-Robotics/kine2go`（Hugging Face） | 每段clip.json分别记录来源和许可，其中有CC BY-NC 4.0片段 |

来源链接：

- https://arxiv.org/abs/2610.28480
- https://github.com/HybridRobotics/whole_body_tracking
- https://github.com/unitreerobotics/unitree_mujoco
- https://github.com/yusongmin1/go2_flip_TO
- https://github.com/upatras-lar/se3_trajopt
- https://huggingface.co/datasets/MIMUW-Robotics/kine2go

自有代码尚未设置统一开源许可证。此来源表不对第三方代码、模型或数据重新授权。
