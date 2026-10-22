# Docker 构建、运行与服务器迁移

目录更新：2026-10-15。命令均从 `RL_Learning/` 根目录执行。

## 当前环境

镜像名为 `isaaclab-base:local`，已有镜像无需因为目录整理而重新构建。
构建文件是 [third_party/IsaacLab/docker/Dockerfile.base](../third_party/IsaacLab/docker/Dockerfile.base)。
构建上下文是完整的 `third_party/IsaacLab/`，不能只复制 Dockerfile。

| 内容 | 容器内位置 |
| --- | --- |
| Isaac Sim | `/isaac-sim`，Isaac Lab 内有运行环境入口 |
| Isaac Lab | `/workspace/isaaclab` |
| Python 虚拟环境 | `/opt/isaaclab-venv` |
| Go2 项目 | `/workspace/go2_learning` |
| RSL-RL 双杆 | `/workspace/rsl_rl_version` |
| 手写 PPO 双杆 | `/workspace/cartpole_ppo` |
| 本地模型与地面 | `/workspace/assets` |

Dockerfile 默认基础镜像为 `nvcr.io/nvidia/isaac-sim:6.1.0`，依赖由
Isaac Lab 的 `pyproject.toml` 和 `uv.lock` 决定，包含 RSL-RL。
本次整理没有执行构建、镜像下载或环境安装。

## 运行

```bash
bash run.sh go2 view
bash run.sh go2 train --task standing --num_envs 4 --max_updates 2 --run_name smoke
bash run.sh go2 play --task standing
bash run.sh cartpole-rsl train --num_envs 4 --max_updates 2
```

各工程的 `run.sh` 设置运行方式和默认参数，再调用
[scripts/run-isaac-project.sh](../scripts/run-isaac-project.sh)。共用启动器：

- 用 `--pull=never --network none` 启动本地镜像。
- 挂载工程源码为只读，日志目录为可写；手写 PPO 另挂载可写 `checkpoints/`。
- 复用 `$HOME/.cache/isaaclab-docker/home` 和 `$HOME/.cache/ov`。
- 需要窗口时提供 X11 授权；无窗口训练不要求 `DISPLAY`。
- 初始化镜像中的 Isaac Sim、虚拟环境与 Isaac Lab Python 包路径。
- 将本工程目录内、宿主机上存在的具体模型路径转换成容器路径。

普通用户启动脚本通过 sudo 调用 Docker；不要先 `sudo bash run.sh`。
宿主机需有可用的 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。
`--gpus all` 暴露 GPU，不代表自动执行多卡训练。

修改 `projects/` 或 `examples/` 中的 Python 代码后，下次启动直接读取新代码。
这些工程使用镜像内的框架；修改宿主机 `third_party/IsaacLab/source` 不会自动覆盖它。
官方单杆入口 `scripts/run-cartpole-official.sh` 则会挂载宿主机的框架源码、脚本和旧日志。

## 重新构建

只有需要更新镜像里的框架或依赖时才构建：

```bash
sudo docker build \
  -f third_party/IsaacLab/docker/Dockerfile.base \
  -t isaaclab-base:local \
  third_party/IsaacLab
```

最后一个参数是构建上下文。命令不会自动读取 `docker/.env.base`。
其他 Dockerfile：Unitree 参考工程的文件在 `third_party/unitree_rl_lab/docker/`；
早期普通 Python 镜像在 `archive/early_gym_demo/docker/`，均不是当前 Go2 的构建入口。

## 迁移到服务器

复制代码时保留仓库布局，并另外携带日志、模型与现成镜像。以只运行 Go2 为例：

```text
RL_Learning/
├── run.sh
├── scripts/run-isaac-project.sh
├── projects/go2_learning/  # 包括需要的 logs/
└── assets/                # 完整 USD、配置层和贴图
```

迁移 RSL-RL 双杆时使用 `examples/rsl_rl_version/`；手写 PPO 使用
`examples/cartpole_ppo/`，保留需要的 `checkpoints/`。启动器共用 assets 挂载，
可保留仓库 assets 目录。第三方源码只在重建镜像或运行官方单杆时需要一起携带。

在空间充足的位置导出现成镜像：

```bash
(set -o pipefail; sudo docker image save isaaclab-base:local | gzip > isaaclab-base-local.tar.gz)
```

服务器导入并检查：

```bash
sudo docker image load -i isaaclab-base-local.tar.gz
sudo docker image inspect isaaclab-base:local --format '{{.Id}} {{.Os}}/{{.Architecture}} {{.Size}}'
nvidia-smi
```

服务器需提供兼容的 GPU、驱动与 Docker GPU 运行环境；无桌面时先做无窗口短训练。
共享服务器按分配情况调整启动器的 GPU 参数。只有 root 账户、没有 sudo 或使用
Apptainer 的机器，需要按目标系统调整启动方式。

镜像导出不会包含宿主机挂载的源码、USD 和训练结果。Git 也忽略训练结果，
这些文件都必须单独迁移。Go2 的容器内项目路径保持一致，旧训练来源链仍可读取。

此前镜像约 47.6 GB；这不是本次实测，也不等于压缩包大小。迁移同时保存镜像、
导出包、缓存和训练结果时，应预留额外空间。
