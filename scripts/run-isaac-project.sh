#!/usr/bin/env bash
# 各独立工程共用：离线 Docker 运行、缓存、X11 授权和 Isaac Sim 环境初始化。
set -euo pipefail
[[ $# -ge 2 ]] || { printf '需要工程目录与 Python 入口。\n' >&2; exit 2; }
project_dir="$(cd -- "$1" && pwd)"
entry="$2"
shift 2
project_name="$(basename -- "$project_dir")"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$EUID" -ne 0 ]] || { printf '请以桌面用户运行，脚本会调用 sudo。\n' >&2; exit 1; }
command -v docker >/dev/null || { printf '找不到 docker。\n' >&2; exit 1; }

container_home="$HOME/.cache/isaaclab-docker/home"
asset_dir="$repo_dir/assets"
log_dir="$project_dir/logs"
if [[ "$project_name" == go2_learning ]]; then
    [[ -f "$asset_dir/unitree_model/Go2/usd/go2.usd" ]] || { printf '缺少 Go2 USD 文件。\n' >&2; exit 1; }
fi
mkdir -p "$log_dir" "$HOME/.cache/ov" \
    "$container_home/.cache" "$container_home/.local/share" \
    "$container_home/.nv" "$container_home/.nvidia-omniverse/logs"

all_args=("$@")
# 宿主机具体权重路径转为容器路径；latest 和已有容器路径原样保留。
for ((i=0; i<${#all_args[@]}; i++)); do
    value_index="$i"
    prefix=''
    case "${all_args[i]}" in
        --checkpoint|--resume)
            ((i+1 < ${#all_args[@]})) || { printf '模型参数缺少路径。\n' >&2; exit 2; }
            value_index="$((i+1))"
            value="${all_args[value_index]}" ;;
        --checkpoint=*|--resume=*)
            prefix="${all_args[i]%%=*}="
            value="${all_args[i]#*=}" ;;
        *) continue ;;
    esac
    if [[ -e "$value" ]]; then
        absolute="$(realpath -- "$value")"
        case "$absolute" in
            "$project_dir"/*)
                all_args[value_index]="${prefix}/workspace/$project_name/${absolute#"$project_dir"/}" ;;
            *) printf '模型需要放在本工程目录内：%s\n' "$project_dir" >&2; exit 2 ;;
        esac
    fi
done

viz=none
for ((i=0; i<${#all_args[@]}; i++)); do
    case "${all_args[i]}" in
        --viz|--visualizer)
            ((i+1 < ${#all_args[@]})) || { printf '可视化参数缺少值。\n' >&2; exit 2; }
            viz="${all_args[i+1]}" ;;
        --viz=*|--visualizer=*) viz="${all_args[i]#*=}" ;;
    esac
done
display_args=()
if [[ "$viz" != none ]]; then
    [[ -n "${DISPLAY:-}" ]] || { printf '可视化需要在 Ubuntu 桌面终端运行。\n' >&2; exit 1; }
    command -v xauth >/dev/null || { printf '找不到 xauth。\n' >&2; exit 1; }
    auth_file="$(mktemp "${TMPDIR:-/tmp}/isaaclab-xauth.XXXXXX")"
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
checkpoint_args=()
if [[ "$project_name" == cartpole_ppo ]]; then
    mkdir -p "$project_dir/checkpoints"
    checkpoint_args=(-v "$project_dir/checkpoints:/workspace/$project_name/checkpoints")
fi
printf '工程：%s\n结果目录：%s\n' "$project_name" "$log_dir"

sudo docker run --rm "${terminal_args[@]}" --pull=never --network none --gpus all \
    --user "$(id -u):$(id -g)" --entrypoint /bin/bash --shm-size=1g \
    -e ACCEPT_EULA=Y -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e HOME=/home/ubuntu -e VIRTUAL_ENV=/opt/isaaclab-venv \
    -e GO2_ASSET_ROOT=/workspace/assets \
    "${display_args[@]}" \
    -v "$container_home:/home/ubuntu" \
    -v "$HOME/.cache/ov:/home/ubuntu/.cache/ov" \
    -v "$project_dir:/workspace/$project_name:ro" \
    -v "$log_dir:/workspace/$project_name/logs" \
    -v "$asset_dir:/workspace/assets:ro" \
    "${checkpoint_args[@]}" \
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
        lab_source=/workspace/isaaclab/source/isaaclab
        [[ -f "$lab_source/isaaclab/__init__.py" ]] || {
            printf "找不到容器内 Isaac Lab Python 包：%s\n" "$lab_source" >&2
            exit 1
        }
        site_dir="$(/opt/isaaclab-venv/bin/python -c "import site; print(site.getsitepackages()[0])")"
        export PYTHONPATH="$lab_source:$site_dir:${PYTHONPATH:-}"
        entry="$1"
        shift
        if [[ "$entry" == *.py ]]; then
            exec uv run --offline --no-project --active python "$entry" "$@"
        fi
        exec uv run --offline --no-project --active python -m "$entry" "$@"
    ' isaac-project "$entry" "${all_args[@]}"
