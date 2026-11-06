#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec docker run --rm --pull=never --network none \
  --entrypoint "${SUPERMIMIC_PYTHON:-/opt/isaaclab-venv/bin/python}" \
  -v "$root/projects/go2_mujoco_mimic:/workspace/go2_mujoco_mimic" \
  "${SUPERMIMIC_IMAGE:-isaaclab-base:local}" \
  /workspace/go2_mujoco_mimic/scripts/export_demo_policies.py "$@"
