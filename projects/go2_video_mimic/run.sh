#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
project="$repo_dir/projects/go2_video_mimic"
mode="${1:-train}"
[[ $# == 0 ]] || shift
if [[ "$mode" == --help || "$mode" == -h ]]; then
    echo 'Usage: bash run.sh video-mimic doctor|prepare|timing|check|train|evaluate|play|reference [args]'
    exit 0
fi
image="${SUPERMIMIC_IMAGE:-isaaclab-base:local}"
if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Runtime image unavailable: $image. Import the existing image with docker image load; see the project README." >&2
    exit 1
fi
python_bin="${SUPERMIMIC_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    python_bin=/opt/isaaclab-venv/bin/python
fi
case "$mode" in
    doctor) module=go2_video_mimic.doctor ;;
    train) module=go2_video_mimic.train ;;
    prepare) module=go2_video_mimic.prepare ;;
    timing) module=go2_video_mimic.timing ;;
    evaluate|play|reference) module=go2_video_mimic.evaluate ;;
    check) module=unittest ;;
    *) echo 'Usage: run.sh doctor|prepare|timing|check|train|evaluate|play|reference [args]' >&2; exit 2 ;;
esac
mkdir -p "$project/logs" "$project/videos" "$project/.cache/home" "$project/.cache/warp"
args=("$@")
needs_video=false
[[ "$mode" != reference ]] || needs_video=true
for arg in "${args[@]}"; do [[ "$arg" != --video ]] || needs_video=true; done
encoder="${VIDEO_FFMPEG_CONTAINER:-}"
if [[ -z "$encoder" ]]; then
    if [[ -n "${VIDEO_FFMPEG:-}" ]]; then bash "$project/cache_encoder.sh"; encoder=/cache/ffmpeg
    elif encoder="$(docker run --rm --pull=never --network none --entrypoint /bin/sh "$image" -c 'command -v ffmpeg' 2>/dev/null)"; then
        :
    else
        encoder=/cache/ffmpeg
        if [[ "$needs_video" == true ]]; then bash "$project/cache_encoder.sh"; fi
    fi
fi
if [[ "$mode" == check ]]; then
    args=(discover -s /workspace/projects/go2_video_mimic/tests -v "${args[@]}")
elif [[ "$mode" == reference ]]; then
    args=(--reference-only "${args[@]}")
fi
for ((i=0; i<${#args[@]}; i++)); do
    case "${args[i]}" in
        --resume|--checkpoint|--motion|--source|--output|--video)
            ((i+1<${#args[@]})) || { echo "Missing value for ${args[i]}" >&2; exit 2; }
            j=$((i+1)); value="${args[j]}" ;;
        *) continue ;;
    esac
    if [[ "$value" == /workspace/* ]]; then continue; fi
    absolute="$(realpath -m -- "$value")"
    case "$absolute" in
        "$repo_dir"/*) args[j]="/workspace/${absolute#"$repo_dir"/}" ;;
        *) echo 'Paths must be inside RL_Learning.' >&2; exit 2 ;;
    esac
done
extra=()
if [[ "$mode" == prepare ]]; then
    extra+=(--mount "type=bind,source=$project/data,target=/workspace/projects/go2_video_mimic/data")
fi
exec docker run --rm --init --pull=never --network none --gpus all --shm-size=2g \
    --user "$(id -u):$(id -g)" --workdir /workspace \
    --entrypoint "$python_bin" \
    --env PYTHONPATH=/workspace/projects --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
    --env TZ=Asia/Shanghai --env HOME=/cache/home --env XDG_CACHE_HOME=/cache \
    --env WARP_CACHE_PATH=/cache/warp --env MUJOCO_GL=egl --env PYOPENGL_PLATFORM=egl \
    --env "VIDEO_FFMPEG=$encoder" \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics --env OPENBLAS_NUM_THREADS=1 \
    --mount "type=bind,source=$repo_dir,target=/workspace,readonly" \
    --mount "type=bind,source=$project/logs,target=/workspace/projects/go2_video_mimic/logs" \
    --mount "type=bind,source=$project/videos,target=/workspace/projects/go2_video_mimic/videos" \
    --mount "type=bind,source=$project/.cache,target=/cache" \
    "${extra[@]}" "$image" -m "$module" "${args[@]}"
