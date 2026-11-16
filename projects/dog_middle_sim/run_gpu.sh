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
    pd) module=dog_middle_sim.main ;;
    reference) module=dog_middle_sim.play_reference ;;
    retarget) module=dog_middle_sim.retarget ;;
    build) module=dog_middle_sim.build_model ;;
    check-model) module=dog_middle_sim.check_model ;;
    *) printf '用法：bash run_gpu.sh [train|play|evaluate|check|pd|reference|retarget|build|check-model] [参数]\n' >&2; exit 2 ;;
esac
mkdir -p "$project/logs" "$project/.cache/warp"
args=("$@")
model_write=false
if [[ "$mode" == build || ( "$mode" == pd && ! -f "$repo_dir/assets/dog_middle/simulation/middle.xml" ) ]]; then
    model_write=true
fi
for arg in "${args[@]}"; do
    if [[ "$mode" == pd && "$arg" == --rebuild ]]; then model_write=true; fi
done
for ((i=0; i<${#args[@]}; i++)); do
    case "${args[i]}" in
        --resume|--checkpoint|--motion|--source|--output)
            ((i+1<${#args[@]})) || { printf '缺少检查点参数\n' >&2; exit 2; }
            j=$((i+1)); value="${args[j]}"; prefix='' ;;
        --resume=*|--checkpoint=*|--motion=*|--source=*|--output=*)
            j=$i; value="${args[i]#*=}"; prefix="${args[i]%%=*}=" ;;
        *) continue ;;
    esac
    if [[ -e "$value" ]]; then
        absolute="$(realpath -- "$value")"
        case "$absolute" in
            "$project"/*) args[j]="${prefix}/workspace/projects/dog_middle_sim/${absolute#"$project"/}" ;;
            "$repo_dir"/*) args[j]="${prefix}/workspace/${absolute#"$repo_dir"/}" ;;
            *) printf '输入/输出路径需位于工程根目录内：%s\n' "$value" >&2; exit 2 ;;
        esac
    fi
done
extra=()
if [[ ( "$mode" == play || "$mode" == pd || "$mode" == reference ) && -n "${DISPLAY:-}" ]]; then
    extra+=(-e "DISPLAY=$DISPLAY" -e NVIDIA_DRIVER_CAPABILITIES=all
            -e PYGLFW_LIBRARY_VARIANT=x11 -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
    if [[ -n "${XAUTHORITY:-}" && -f "$XAUTHORITY" ]]; then
        extra+=(-e XAUTHORITY=/tmp/xauthority -v "$XAUTHORITY:/tmp/xauthority:ro")
    fi
fi
if [[ "$model_write" == true ]]; then
    mkdir -p "$repo_dir/assets/dog_middle/simulation"
    extra+=(-v "$repo_dir/assets/dog_middle/simulation:/workspace/assets/dog_middle/simulation")
fi
if [[ "$mode" == retarget ]]; then
    mkdir -p "$project/data"
    extra+=(-v "$project/data:/workspace/projects/dog_middle_sim/data")
    # 重新映射需要源 Go2 轨迹；普通播放不依赖该目录。
    extra+=(-v "$repo_dir/projects/go2_mujoco_mimic/data:/workspace/projects/go2_mujoco_mimic/data:ro")
fi
if [[ "$mode" == evaluate ]]; then args=(--headless "${args[@]}"); fi
printf 'Middle Docker | mode=%s | image=%s | python=%s\n' \
    "$mode" "${MIDDLE_IMAGE:-isaaclab-base:local}" \
    "${MIDDLE_DOCKER_PYTHON:-/opt/isaaclab-venv/bin/python}"
exec docker run --rm --pull=never --network none --gpus all \
    --workdir /workspace \
    --entrypoint "${MIDDLE_DOCKER_PYTHON:-/opt/isaaclab-venv/bin/python}" \
    -e PYTHONPATH=/workspace/projects -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e TZ=Asia/Shanghai \
    -v "$project:/workspace/projects/dog_middle_sim:ro" \
    -v "$project/logs:/workspace/projects/dog_middle_sim/logs" \
    -v "$project/.cache/warp:/root/.cache/warp" \
    -v "$repo_dir/assets:/workspace/assets:ro" \
    "${extra[@]}" \
    "${MIDDLE_IMAGE:-isaaclab-base:local}" -m "$module" "${args[@]}"
