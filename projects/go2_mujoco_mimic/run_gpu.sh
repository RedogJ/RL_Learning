#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${SUPERMIMIC_IMAGE:-isaaclab-base:local}"
python_bin="${SUPERMIMIC_PYTHON:-/opt/isaaclab-venv/bin/python}"
module=go2_mujoco_mimic.smoke
if [[ "${1:-}" == train-state ]]; then
  module=go2_mujoco_mimic.tasks.mimic.state_train
  shift
elif [[ "${1:-}" == evaluate-state ]]; then
  module=go2_mujoco_mimic.tasks.mimic.evaluate_state
  shift
elif [[ "${1:-}" == evaluate-hold ]]; then
  module=go2_mujoco_mimic.tasks.mimic.evaluate_hold
  shift
elif [[ "${1:-}" == play-solo-flip ]]; then
  module=go2_mujoco_mimic.play_solo_backflip
  shift
elif [[ "${1:-}" == play ]]; then
  module=go2_mujoco_mimic.play_policy
  shift
elif [[ "${1:-}" == reference ]]; then
  module=go2_mujoco_mimic.play_reference
  shift
elif [[ "${1:-}" == check-graph ]]; then
  module=go2_mujoco_mimic.check_graph
  shift
elif [[ "${1:-}" == profile-collect ]]; then
  module=go2_mujoco_mimic.profile_collect
  shift
elif [[ "${1:-}" == train ]]; then
  module=go2_mujoco_mimic.tasks.mimic.train
  shift
fi
mkdir -p "$root/projects/go2_mujoco_mimic/.cache/warp" "$root/projects/go2_mujoco_mimic/logs"
extra=()
if [[ ( "$module" == go2_mujoco_mimic.play_policy || "$module" == go2_mujoco_mimic.play_solo_backflip || "$module" == go2_mujoco_mimic.play_reference ) && -n "${DISPLAY:-}" ]]; then
  extra+=( -e "DISPLAY=$DISPLAY" -e XAUTHORITY=/tmp/xauthority -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix:ro )
  if [[ -n "${XAUTHORITY:-}" && -f "$XAUTHORITY" ]]; then
    extra+=( -v "$XAUTHORITY:/tmp/xauthority:ro" )
  fi
fi
exec docker run --rm "${extra[@]}" --pull=never --gpus all --network none \
  --entrypoint "$python_bin" \
  -e PYTHONPATH=/workspace \
  -e GO2_PHYSICS_GRAPH="${GO2_PHYSICS_GRAPH:-0}" \
  -v "$root/projects/go2_mujoco_mimic:/workspace/go2_mujoco_mimic" \
  -v "$root/projects/go2_mujoco_mimic/.cache/warp:/root/.cache/warp" \
  -v "$root/assets:/assets:ro" \
  "$image" -m "$module" "$@"
