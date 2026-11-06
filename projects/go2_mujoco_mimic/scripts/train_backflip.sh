#!/usr/bin/env bash
# 后空翻训练预设；这是当前配置，不会重演历史调参过程。
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_dir/run_gpu.sh" train-state \
  --task backflip --backflip-joint-mapping --learned-termination \
  --num-envs 1024 --cycles 1000 \
  --collect-steps 32 --buffer-steps 320 \
  --wm-batch 512 --wm-updates 30 --window 8 \
  --actor-batch 512 --actor-rounds 1 --actor-lr 4e-6 \
  --horizon 16 --wm-phase-priority \
  --eval-interval 10 --save-interval 50 --run-name backflip \
  "$@"
