#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mode="${1:-pd}"
case "$mode" in
    retarget|reference|train|check|evaluate|play|pd|build|check-model)
        if [[ $# -gt 0 ]]; then shift; fi ;;
    -*) mode=pd ;;
    *) printf '未知运行方式：%s\n' "$mode" >&2; exit 2 ;;
esac
exec bash "$repo_dir/projects/dog_middle_sim/run_gpu.sh" "$mode" "$@"
