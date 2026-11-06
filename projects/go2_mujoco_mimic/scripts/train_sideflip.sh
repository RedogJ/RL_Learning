#!/usr/bin/env bash
# 侧空翻训练预设；追加参数覆盖同名默认值。
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_dir/run_gpu.sh" train-state \
  --task sideflip --num-envs 2048 --cycles 1000 \
  --collect-steps 32 --buffer-steps 320 \
  --wm-batch 512 --wm-updates 30 --window 8 \
  --actor-batch 512 --actor-rounds 1 --actor-lr 1e-5 \
  --horizon 16 --wm-phase-priority \
  --eval-interval 10 --save-interval 100 --run-name sideflip \
  "$@"
