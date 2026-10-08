#!/usr/bin/env bash
# Run Cartpole PPO training or playback using the existing local Docker image.
set -euo pipefail

mode="${1:-train}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "$mode" in
    learn)
        rl_args=(-p scripts/my_cartpole/main.py)
        ;;
    train)
rl_args=(
    train
    --rl_library rsl_rl
    --task Isaac-Cartpole
    --num_envs 256
    --max_iterations 300
    --viz none
    --logger tensorboard
    --run_name pole_l1
    --seed 42
)
        ;;
    play|play-l2|play-l3)
        checkpoint="latest"
        if [[ "$mode" == "play-l2" ]]; then
            checkpoint="/workspace/isaaclab/logs/rsl_rl/cartpole/2026-10-02_04-24-28_first_gui/model_299.pt"
        elif [[ "$mode" == "play-l3" ]]; then
            checkpoint="/workspace/isaaclab/logs/rsl_rl/cartpole/2026-10-02_05-15-18_pole_l1/model_299.pt"
        fi
        rl_args=(play --rl_library rsl_rl --task Isaac-Cartpole
                 --num_envs 1 --viz newton_gl --checkpoint "$checkpoint" --real-time --seed 42
                 physics=newton_mjwarp env.episode_length_s=60)
        ;;
    compare)
        rl_args=(--checkpoint-l2
                 /workspace/isaaclab/logs/rsl_rl/cartpole/2026-10-02_04-24-28_first_gui/model_299.pt
                 --checkpoint-l3
                 /workspace/isaaclab/logs/rsl_rl/cartpole/2026-10-02_05-15-18_pole_l1/model_299.pt
                 --task Isaac-Cartpole --num_envs 2 --viz newton_gl --real-time --seed 42
                 physics=newton_mjwarp env.episode_length_s=60)
        ;;
    *)
        printf 'Usage: bash run-cartpole-gui.sh [train|play|play-l2|play-l3|compare] [additional Isaac Lab arguments]\n' >&2
        exit 2
        ;;
esac

if [[ "$EUID" -eq 0 ]]; then
    printf 'Run this script as your desktop user; it calls sudo for Docker.\n' >&2
    exit 1
fi
if [[ -z "${DISPLAY:-}" ]]; then
    printf 'DISPLAY is unset. Run this script in a terminal on your Ubuntu desktop.\n' >&2
    exit 1
fi
for tool in docker xauth; do
    if ! command -v "$tool" >/dev/null; then
        printf 'Required command is missing: %s\n' "$tool" >&2
        exit 1
    fi
done

task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
log_dir="$task_dir/IsaacLab/logs"
cache_dir="$HOME/.cache/isaaclab-docker"
container_home="$cache_dir/home"
mkdir -p "$log_dir" "$HOME/.cache/ov" \
    "$container_home/.cache" "$container_home/.local/share" \
    "$container_home/.nv" "$container_home/.nvidia-omniverse/logs"

# A private X11 cookie lets the container display a window on XWayland.
# FamilyWild permits the cookie to be used under the container's hostname.
gui_auth_file="$(mktemp "${TMPDIR:-/tmp}/isaaclab-xauth.XXXXXX")"
trap 'rm -f -- "$gui_auth_file"' EXIT
xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$gui_auth_file" nmerge -
if [[ ! -s "$gui_auth_file" ]]; then
    printf 'No X11 authentication cookie was found for DISPLAY=%s.\n' "$DISPLAY" >&2
    exit 1
fi

printf 'Mode: %s\nTraining results: %s/rsl_rl/cartpole\n' "$mode" "$log_dir"

if [[ "$mode" == "compare" ]]; then
    container_command=(-c 'exec /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/scripts/compare_cartpole_policies.py "$@"'
                       compare "${rl_args[@]}" "$@")
else
    container_command=(-c 'exec /workspace/isaaclab/isaaclab.sh "$@"'
                       isaaclab "${rl_args[@]}" "$@")
fi

sudo docker run --rm -it --pull=never --gpus all \
    --user "$(id -u):$(id -g)" \
    --entrypoint /bin/bash --shm-size=1g \
    -e ACCEPT_EULA=Y -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e HOME=/home/ubuntu -e DISPLAY="$DISPLAY" \
    -e PYOPENGL_PLATFORM=glx \
    -e XAUTHORITY=/tmp/isaaclab.xauth \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v "$gui_auth_file:/tmp/isaaclab.xauth:ro" \
    -v "$container_home:/home/ubuntu" \
    -v "$HOME/.cache/ov:/home/ubuntu/.cache/ov" \
    -v "$task_dir/IsaacLab/source:/workspace/isaaclab/source" \
    -v "$task_dir/IsaacLab/scripts:/workspace/isaaclab/scripts" \
    -v "$log_dir:/workspace/isaaclab/logs" \
    -w /workspace/isaaclab \
    isaaclab-base:local "${container_command[@]}"
