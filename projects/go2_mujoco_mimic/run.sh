#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$root/projects${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${DISPLAY:-}" ]]; then export PYGLFW_LIBRARY_VARIANT="${PYGLFW_LIBRARY_VARIANT:-x11}"; fi
module=go2_mujoco_mimic.main
if [[ "${1:-}" == reference ]]; then module=go2_mujoco_mimic.play_reference; shift; fi
exec "${GO2_MUJOCO_PYTHON:-python}" -m "$module" "$@"
