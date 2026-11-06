# 环境、路径与检查

## 已验证的运行环境

当前实验使用 Linux、RTX 4060 Laptop 8GB、Docker 镜像 `isaaclab-base:local`，Python解释器 `/opt/isaaclab-venv/bin/python`。

| 包 | 当前镜像实际版本 |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.12.0+cu130 |
| MuJoCo / MuJoCo Warp | 3.12.0 / 3.12.0 |
| Warp | 1.17.0 |
| NumPy / SciPy | 2.5.1 / 1.17.0 |

CUDA由容器内PyTorch依赖和宿主机驱动提供。驱动、Docker、NVIDIA Container Toolkit需正常工作：

```bash
nvidia-smi
docker run --rm --gpus all --pull=never --entrypoint nvidia-smi isaaclab-base:local
```

启动脚本直接调用 `docker`。用户需要有运行Docker的权限；已有本地运行环境无需重建。

## 两种安装路径

**复用现有镜像（已经验证）。** 默认 `run_gpu.sh` 复用 `isaaclab-base:local`；原始Isaac镜像构建说明在完整仓库的 `docs/docker.md`；组内分享包不携带Isaac源码或镜像。

**独立 MuJoCo 镜像（提供构建配方，尚未独立构建验证）。** 代码本身不导入Isaac Lab，因此提供一个不包含Isaac的配方：

```bash
docker build -f projects/go2_mujoco_mimic/docker/Dockerfile -t go2-supermimic:local .
export SUPERMIMIC_IMAGE=go2-supermimic:local
export SUPERMIMIC_PYTHON=/usr/local/bin/python
bash run.sh supermimic --num-envs 16 --steps 30 \
  --motion /workspace/go2_mujoco_mimic/data/go2_flip_to/sideflip_left_30hz/motion.npy
```

此构建需要联网下载Python、PyTorch和依赖，不能用现有镜像中的测试结果代替独立镜像验收。`requirements.txt`记录当前核心依赖版本，重定向原始后空翻数据还需要h5py。

## 目录映射

| 宿主机 | 容器 |
| --- | --- |
| `projects/go2_mujoco_mimic/` | `/workspace/go2_mujoco_mimic/` |
| `assets/` | `/assets/`，只读 |
| 工程 `.cache/warp/` | `/root/.cache/warp/` |

CLI中的 `--checkpoint`、`--resume`、`--motion` 使用**容器路径**。项目和日志目录需要可写。
默认启动 `--pull=never --network none`；依赖和参考数据应提前准备。首次Warp内核编译比后续启动慢。

## 图形界面

`play`和`reference`将宿主机的DISPLAY、X11 socket与已有XAUTHORITY挂载到容器。
在本地X11桌面运行；SSH服务器优先无窗口训练。`--headless --steps 90` 可以检查策略执行但不打开窗口。

```bash
bash run.sh supermimic play --checkpoint /workspace/go2_mujoco_mimic/checkpoints/sideflip_policy_5800.pt \
  --headless --steps 90 --no-noise
```

## 本地检查

在仓库根目录执行，以下命令在实际运行镜像中检查；若使用独立镜像，替换镜像名和解释器。

```bash
# 奖励、Critic梯度、Replay测试
docker run --rm --pull=never --network none \
  --entrypoint /opt/isaaclab-venv/bin/python -e PYTHONPATH=/workspace \
  -v "$PWD/projects/go2_mujoco_mimic:/workspace/go2_mujoco_mimic" \
  -v "$PWD/assets:/assets:ro" isaaclab-base:local \
  -m unittest discover -s /workspace/go2_mujoco_mimic/tests -q

# MuJoCo与Torch运动学及奖励梯度对照
docker run --rm --pull=never --network none \
  --entrypoint /opt/isaaclab-venv/bin/python -e PYTHONPATH=/workspace \
  -v "$PWD/projects/go2_mujoco_mimic:/workspace/go2_mujoco_mimic" \
  -v "$PWD/assets:/assets:ro" isaaclab-base:local \
  -m go2_mujoco_mimic.tasks.mimic.check_kinematics

# 仓库启动和导出脚本检查，不启动GPU
python -m unittest discover -s tests -q
```

`check-graph` 对比CUDA Graph与普通物理循环。仿真为每环境预留256条约束、128个接触；出现overflow时应排查容量与异常状态。
