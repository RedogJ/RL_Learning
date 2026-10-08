#!/usr/bin/env bash
# Use the existing image; keep this project's code and checkpoints in its own mount.
set -euo pipefail

mode="${1:-train}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "$mode" in
    train|sample) defaults=(--control "$mode" --viz none) ;;
    play|pd) defaults=(--control "$mode" --viz newton_gl --num_envs 1) ;;
    *)
        printf 'Usage: bash deeponet_cartpole/run.sh [train|play|sample|pd] [arguments]\n' >&2
        exit 2
        ;;
esac

if [[ "$EUID" -eq 0 ]]; then
    printf '请以桌面用户运行；脚本会用 sudo 启动 Docker。\n' >&2
    exit 1
fi
command -v docker >/dev/null || { printf '找不到 docker。\n' >&2; exit 1; }

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(dirname -- "$project_dir")"
container_home="$HOME/.cache/isaaclab-docker/home"
mkdir -p "$project_dir/checkpoints" "$HOME/.cache/ov" \
    "$container_home/.cache" "$container_home/.local/share" \
    "$container_home/.nv" "$container_home/.nvidia-omniverse/logs"

# 无窗口训练不需要 DISPLAY；可视化时使用桌面的 X11 授权。
viz=none
for ((index=0; index<${#defaults[@]}; index++)); do
    if [[ "${defaults[index]}" == --viz ]]; then viz="${defaults[index+1]}"; fi
done
extra_args=("$@")
for ((index=0; index<${#extra_args[@]}; index++)); do
    if [[ "${extra_args[index]}" == --viz && $((index+1)) -lt ${#extra_args[@]} ]]; then
        viz="${extra_args[index+1]}"
    elif [[ "${extra_args[index]}" == --viz=* ]]; then
        viz="${extra_args[index]#--viz=}"
    fi
done

display_args=()
if [[ "$viz" != none ]]; then
    [[ -n "${DISPLAY:-}" ]] || { printf '可视化需要在 Ubuntu 桌面终端运行。\n' >&2; exit 1; }
    command -v xauth >/dev/null || { printf '找不到 xauth。\n' >&2; exit 1; }
    gui_auth_file="$(mktemp "${TMPDIR:-/tmp}/deeponet-xauth.XXXXXX")"
    trap 'rm -f -- "$gui_auth_file"' EXIT
    xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$gui_auth_file" nmerge -
    [[ -s "$gui_auth_file" ]] || { printf '没有找到 X11 授权。\n' >&2; exit 1; }
    display_args=(-e DISPLAY="$DISPLAY" -e PYOPENGL_PLATFORM=glx
                  -e XAUTHORITY=/tmp/isaaclab.xauth
                  -v /tmp/.X11-unix:/tmp/.X11-unix:ro
                  -v "$gui_auth_file:/tmp/isaaclab.xauth:ro")
fi

terminal_args=(-i)
if [[ -t 0 && -t 1 ]]; then terminal_args+=(-t); fi
printf 'DeepONet mode: %s\n项目和权重目录: %s\n' "$mode" "$project_dir"

sudo docker run --rm "${terminal_args[@]}" --pull=never --network none --gpus all \
    --user "$(id -u):$(id -g)" --entrypoint /bin/bash --shm-size=1g \
    -e ACCEPT_EULA=Y -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e HOME=/home/ubuntu "${display_args[@]}" \
    -v "$container_home:/home/ubuntu" \
    -v "$HOME/.cache/ov:/home/ubuntu/.cache/ov" \
    -v "$workspace_dir/IsaacLab/source:/workspace/isaaclab/source:ro" \
    -v "$project_dir:/workspace/deeponet_cartpole" \
    -w /workspace/isaaclab \
    isaaclab-base:local -c '
        exec /workspace/isaaclab/isaaclab.sh -p /workspace/deeponet_cartpole/main.py "$@"
    ' deeponet "${defaults[@]}" "$@"
