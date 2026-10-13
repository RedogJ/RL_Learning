# Docker 构建、运行与服务器迁移说明

核对日期：2026-10-12；目录整理更新：2026-10-13。以下路径以 `RL_Learning/` 为工程根目录。

## 1. 当前要使用哪个 Dockerfile？

当前 Isaac Lab 实验使用的构建文件是：

**[IsaacLab/docker/Dockerfile.base](IsaacLab/docker/Dockerfile.base)**

对应启动脚本使用的镜像名是 **`isaaclab-base:local`**。

| 项目 | 当前文件中的配置 |
|---|---|
| 基础镜像 | `nvcr.io/nvidia/isaac-sim:6.1.0` |
| Isaac Sim 容器内目录 | `/isaac-sim` |
| Isaac Lab 容器内目录 | `/workspace/isaaclab` |
| Python 虚拟环境 | `/opt/isaaclab-venv` |
| Python 依赖解析 | `IsaacLab/pyproject.toml` 和 `IsaacLab/uv.lock` |
| 构建上下文 | 整个 `IsaacLab/` 文件夹 |

Dockerfile 安装 Isaac Lab 及其依赖，包含 RSL-RL；默认还安装其他 RL 库和扩展。
当前小工程只使用其中一部分，但可以直接复用已经构建好的镜像。

工程中其他同名文件的用途：

- [archive/early_gym_demo/docker/Dockerfile](archive/early_gym_demo/docker/Dockerfile)：已归档的普通 Python 环境，基于 `python:3.11-slim`，不能替代 Isaac Lab 镜像。
- [unitree_rl_lab/docker/Dockerfile](unitree_rl_lab/docker/Dockerfile)：Unitree 工程的构建入口，当前双杆小工程不使用它。
- `IsaacLab/docker/Dockerfile.ros2`、`Dockerfile.curobo`、`Dockerfile.kitless`：其他用途的环境，本工程当前启动入口不使用它们。

说明：上述结论来自 Dockerfile、启动脚本和此前构建记录。此次检查未取得 Docker
守护进程的 sudo 访问权限，未重新验证镜像 ID、实际构建参数或镜像内部版本。
当前源码也可能与当初构建时不同；迁移已跑通的环境优先使用现成镜像。

## 2. 如果需要重新构建

在 `RL_Learning/` 根目录执行：

```bash
sudo docker build \
  -f IsaacLab/docker/Dockerfile.base \
  -t isaaclab-base:local \
  IsaacLab
```

最后一个 `IsaacLab` 是构建上下文。Dockerfile 中的 `COPY` 要读取这里的源码、
锁文件和构建脚本，因此不能只复制一个 Dockerfile 去构建。

也可以先进入 `IsaacLab/`，再执行：

```bash
sudo docker build -f docker/Dockerfile.base -t isaaclab-base:local .
```

两种写法的上下文相同。直接执行 `docker build` 不会自动加载 `docker/.env.base`；
这里使用 Dockerfile 自带的默认构建参数。

已有可用镜像时不需要重新构建。重新构建可能下载基础层和依赖，并占用额外磁盘空间。

## 3. 当前独立小工程怎样运行？

入口：
[rsl_rl_version/run.sh](rsl_rl_version/run.sh)

```bash
cd rsl_rl_version

# 两轮短验证，不能用于判断训练收敛
bash run.sh train --max_updates 2 --num_envs 4

# 默认 64 个环境、300 次更新、无窗口训练
bash run.sh train

# 默认打开 newton_gl 窗口，播放最近完成训练的模型
bash run.sh play
```

`run.sh` 自动定位自己的文件夹，使用镜像 `isaaclab-base:local`，以
`python -m rsl_rl_version.main` 启动程序。默认训练不要求桌面；默认播放需要 X11 显示环境。
脚本使用 `--pull=never --network none`，不会自动拉取镜像或联网安装依赖。

当前脚本要求普通用户运行，并通过 sudo 调用 Docker。只有 root 账户、没有 sudo，
或使用作业调度器的服务器，需要相应调整启动方式。

## 4. 镜像、代码和训练结果分别在哪里？

| 内容 | 宿主机位置 | 容器内位置 |
|---|---|---|
| Isaac Sim、Isaac Lab、Python 依赖 | Docker 镜像 | `/isaac-sim`、`/workspace/isaaclab`、`/opt/isaaclab-venv` |
| 独立工程源码 | `rsl_rl_version/` | `/workspace/rsl_rl_version`，只读挂载 |
| 新版日志和权重 | 上述文件夹的 `logs/` | `/workspace/rsl_rl_version/logs`，可写挂载 |
| 用户缓存 | `$HOME/.cache/isaaclab-docker/home` | `/home/ubuntu` |
| Omniverse 缓存 | `$HOME/.cache/ov` | `/home/ubuntu/.cache/ov` |

修改小工程的 Python 代码后，下次启动会读取宿主机的新代码，通常无需重新 build。
挂载进去的最新代码和训练结果不会因导出镜像而自动打包，必须单独携带。

双杆模型由本项目 `scene.py` 创建，没有单独的机器人 USD/URDF 文件需要搬运。
若以后添加外部模型、纹理或数据集，需要一起迁移。

## 5. 迁移到服务器：优先搬现成镜像

需要携带：

1. `isaaclab-base:local` 镜像。
2. 完整的 `rsl_rl_version/` 文件夹；需要播放已有训练时，保留其中的 `logs/`。
3. 需要的外部模型或数据集（当前双杆工程没有这项依赖）。

运行不要求把整个 `RL_Learning/` 都搬过去。为了以后重建环境，可以另行归档
完整的 `IsaacLab/` 构建上下文和本说明。

### 导出与导入示例

下面是之后实际迁移时使用的命令，本次只记录文档，没有执行。
在本机选一个空间充足的目录，导出并压缩镜像：

```bash
(set -o pipefail; sudo docker image save isaaclab-base:local | gzip > isaaclab-base-local.tar.gz)
```

确认命令成功，再把压缩包和工程文件夹传到服务器。服务器导入：

```bash
sudo docker image load -i isaaclab-base-local.tar.gz
sudo docker image inspect isaaclab-base:local --format '{{.Id}} {{.Os}}/{{.Architecture}} {{.Size}}'
```

进入服务器上的 `rsl_rl_version/`，执行两轮短验证后再正式训练。
服务器无桌面时，可以将训练权重和同目录的 `run.json` 传回笔记本播放。

此前镜像列表显示约 **47.6 GB**，这不是此次重新测量的值，也不等于压缩包大小。
若服务器同时保存导出包、镜像层、缓存和输出，先按 **100～150 GB 可用磁盘空间**
做迁移预算；实际需求由存储驱动、压缩率及日志规模决定。

## 6. 服务器条件

- 优先使用与当前宿主机相同的 **Linux x86_64** 环境；最终还需核对镜像架构。
- 安装 Docker、兼容的 NVIDIA 驱动和配置好的 NVIDIA Container Toolkit。
- 宿主机通常不需要重复安装镜像里的 CUDA Toolkit，但驱动必须兼容镜像运行时。
- Isaac Sim 官方要求排除了没有 RT Core 的 A100、H100；不能仅凭能跑 PyTorch
  就认定能原样运行这套 Isaac Sim/PhysX 工程。目标 GPU 需按 Isaac Sim 要求核对。
- 当前脚本的 `--gpus all` 只是暴露 GPU，不会自动启用多卡训练。
  共享服务器上应按管理员分配的 GPU 配置启动参数。
- 当前脚本面向单机 Docker；只提供 Apptainer/Singularity 的集群需要另行适配。

迁移前可收集以下只读信息：

```bash
uname -m
nvidia-smi
df -h
sudo docker info
sudo docker image inspect isaaclab-base:local --format '{{.Id}} {{.Os}}/{{.Architecture}} {{.Size}}'
```

最后一条仅适用于已经有该镜像的机器。

## 7. 搬旧工程时的区别

[run-cartpole-gui.sh](run-cartpole-gui.sh) 是旧工程入口。它还挂载宿主机的
`IsaacLab/source`、`IsaacLab/scripts` 和 `IsaacLab/logs`，这些修改和训练记录需要一起搬。
它无条件检查 `DISPLAY` 和 X11，不能原样用于普通无桌面的 SSH 训练会话。

新版独立小工程使用自己的 `run.sh`，没有这项训练入口限制。

## 参考

- [Docker 镜像导出](https://docs.docker.com/reference/cli/docker/image/save/)
- [Docker 镜像导入](https://docs.docker.com/reference/cli/docker/image/load/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [Isaac Sim 硬件与系统要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
