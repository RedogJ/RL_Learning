#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Run this self-contained project with the existing Isaac Lab image.
set -euo pipefail

mode="${1:-train}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
    train) defaults=(--control train --viz none) ;;
    play) defaults=(--control play --viz newton_gl --num_envs 1) ;;
    *) printf 'Usage: bash run.sh [train|play] [arguments]\n' >&2; exit 2 ;;
esac
[[ "$EUID" -ne 0 ]] || { printf '请以桌面用户运行，脚本会调用 sudo。\n' >&2; exit 1; }
command -v docker >/dev/null || { printf '找不到 docker。\n' >&2; exit 1; }

version_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
container_home="$HOME/.cache/isaaclab-docker/home"
log_dir="$version_dir/logs"
mkdir -p "$log_dir" "$HOME/.cache/ov" \
    "$container_home/.cache" "$container_home/.local/share" \
    "$container_home/.nv" "$container_home/.nvidia-omniverse/logs"

# 无窗口训练不要求 DISPLAY；播放时复用已有 X11 授权方式。
viz=none
all_args=("${defaults[@]}" "$@")
for ((i=0; i<${#all_args[@]}; i++)); do
    case "${all_args[i]}" in
        --viz|--visualizer)
            if ((i+1 >= ${#all_args[@]})); then printf '可视化参数缺少值。\n' >&2; exit 2; fi
            viz="${all_args[i+1]}" ;;
        --viz=*|--visualizer=*) viz="${all_args[i]#*=}" ;;
    esac
done
display_args=()
if [[ "$viz" != none ]]; then
    [[ -n "${DISPLAY:-}" ]] || { printf '可视化需要在 Ubuntu 桌面终端运行。\n' >&2; exit 1; }
    command -v xauth >/dev/null || { printf '找不到 xauth。\n' >&2; exit 1; }
    auth_file="$(mktemp "${TMPDIR:-/tmp}/cartpole-rsl-xauth.XXXXXX")"
    trap 'rm -f -- "$auth_file"' EXIT
    xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$auth_file" nmerge -
    [[ -s "$auth_file" ]] || { printf '没有找到 X11 授权。\n' >&2; exit 1; }
    display_args=(-e DISPLAY="$DISPLAY" -e PYOPENGL_PLATFORM=glx
                  -e XAUTHORITY=/tmp/isaaclab.xauth
                  -v /tmp/.X11-unix:/tmp/.X11-unix:ro
                  -v "$auth_file:/tmp/isaaclab.xauth:ro")
fi
terminal_args=(-i)
if [[ -t 0 && -t 1 ]]; then terminal_args+=(-t); fi
printf 'RSL-RL double Cartpole: %s\n结果目录：%s\n' "$mode" "$log_dir"

sudo docker run --rm "${terminal_args[@]}" --pull=never --network none --gpus all \
    --user "$(id -u):$(id -g)" --entrypoint /bin/bash --shm-size=1g \
    -e ACCEPT_EULA=Y -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e HOME=/home/ubuntu -e VIRTUAL_ENV=/opt/isaaclab-venv \
    "${display_args[@]}" \
    -v "$container_home:/home/ubuntu" \
    -v "$HOME/.cache/ov:/home/ubuntu/.cache/ov" \
    -v "$version_dir:/workspace/rsl_rl_version:ro" \
    -v "$log_dir:/workspace/rsl_rl_version/logs" \
    -w /workspace \
    isaaclab-base:local -c '
        set -e
        sim_root=/workspace/isaaclab/_isaac_sim
        if [[ -f "$sim_root/setup_conda_env.sh" ]]; then
            source "$sim_root/setup_conda_env.sh"
        elif [[ -f "$sim_root/setup_python_env.sh" ]]; then
            export ISAAC_PATH="$sim_root"
            export CARB_APP_PATH="$sim_root/kit"
            export EXP_PATH="$sim_root/apps"
            source "$sim_root/setup_python_env.sh"
        else
            printf "找不到容器内 Isaac Sim 的环境初始化脚本。\n" >&2
            exit 1
        fi
        # Kit 的预装包不能遮盖现有 venv 中的 torch/typing_extensions 等。
        site_dir="$(/opt/isaaclab-venv/bin/python -c "import site; print(site.getsitepackages()[0])")"
        export PYTHONPATH="$site_dir:${PYTHONPATH:-}"
        exec uv run --offline --no-project --active python \
            -m rsl_rl_version.main "$@"
    ' cartpole-rsl "${all_args[@]}"
