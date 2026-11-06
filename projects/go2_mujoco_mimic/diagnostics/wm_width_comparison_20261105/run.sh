#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/../.." && pwd)"
workspace_root="$(cd -- "$project_dir/../.." && pwd)"
run_tag="run_$(date +%Y%m%d_%H%M%S)"
output_dir="$script_dir/runs/$run_tag"
container_output="/workspace/go2_mujoco_mimic/diagnostics/wm_width_comparison_20261105/runs/$run_tag"
mkdir -p "$output_dir" "$project_dir/.cache/warp"
printf 'WM 宽度对比：2048×3 与 1024×3，均从零训练\n结果目录：%s\n' "$output_dir"
docker run --rm -i --pull=never --gpus all --network none \
  --entrypoint /opt/isaaclab-venv/bin/python \
  -e PYTHONPATH=/workspace -e GO2_PHYSICS_GRAPH=0 \
  -v "$project_dir:/workspace/go2_mujoco_mimic" \
  -v "$project_dir/.cache/warp:/root/.cache/warp" \
  -v "$workspace_root/assets:/assets:ro" \
  isaaclab-base:local \
  /workspace/go2_mujoco_mimic/diagnostics/wm_width_comparison_20261105/experiment.py \
  --data /workspace/go2_mujoco_mimic/diagnostics/wm_update_isolation_20261105/fixed_stream.pt \
  --source-config /workspace/go2_mujoco_mimic/logs/backflip_v19_bce_2026-11-05_04-25-36_439009/model_3320.pt \
  --output "$container_output" \
  "$@" 2>&1 | tee "$output_dir/console.log"
