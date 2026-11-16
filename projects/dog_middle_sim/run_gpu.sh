#!/usr/bin/env bash
# 复用已有镜像；项目源码/模型只读，日志和 Warp 缓存单独可写。
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
project="$repo_dir/projects/dog_middle_sim"
mode="${1:-train}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
    train) module=dog_middle_sim.train ;;
    play|evaluate) module=dog_middle_sim.play ;;
    check) module=dog_middle_sim.check_training ;;
    *) printf '用法：bash run_gpu.sh [train|play|evaluate|check] [参数]\n' >&2; exit 2 ;;
esac
[[ -f "$repo_dir/assets/dog_middle/simulation/middle.xml" ]] || {
    printf '请先生成模型：bash run.sh middle --headless --seconds 1 --rebuild\n' >&2
    exit 1
}
mkdir -p "$project/logs" "$project/.cache/warp"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
    case "${args[i]}" in
        --resume|--checkpoint)
            ((i+1<${#args[@]})) || { printf '缺少检查点参数\n' >&2; exit 2; }
            j=$((i+1)); value="${args[j]}"; prefix='' ;;
        --resume=*|--checkpoint=*)
            j=$i; value="${args[i]#*=}"; prefix="${args[i]%%=*}=" ;;
        *) continue ;;
    esac
    if [[ -e "$value" ]]; then
        absolute="$(realpath -- "$value")"
        case "$absolute" in
            "$project"/*) args[j]="${prefix}/workspace/projects/dog_middle_sim/${absolute#"$project"/}" ;;
            *) printf '检查点需位于 %s 内\n' "$project" >&2; exit 2 ;;
        esac
    fi
done
extra=()
if [[ "$mode" == play && -n "${DISPLAY:-}" ]]; then
    extra+=(-e "DISPLAY=$DISPLAY" -e NVIDIA_DRIVER_CAPABILITIES=all
            -e PYGLFW_LIBRARY_VARIANT=x11 -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
    if [[ -n "${XAUTHORITY:-}" && -f "$XAUTHORITY" ]]; then
        extra+=(-e XAUTHORITY=/tmp/xauthority -v "$XAUTHORITY:/tmp/xauthority:ro")
    fi
fi
if [[ "$mode" == evaluate ]]; then args=(--headless "${args[@]}"); fi
exec docker run --rm --pull=never --network none --gpus all \
    --entrypoint "${MIDDLE_DOCKER_PYTHON:-/opt/isaaclab-venv/bin/python}" \
    -e PYTHONPATH=/workspace/projects -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e TZ=Asia/Shanghai \
    "${extra[@]}" \
    -v "$project:/workspace/projects/dog_middle_sim:ro" \
    -v "$project/logs:/workspace/projects/dog_middle_sim/logs" \
    -v "$project/.cache/warp:/root/.cache/warp" \
    -v "$repo_dir/assets:/workspace/assets:ro" \
    "${MIDDLE_IMAGE:-isaaclab-base:local}" -m "$module" "${args[@]}"
