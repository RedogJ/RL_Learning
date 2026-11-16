#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$repo_dir/projects${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${DISPLAY:-}" ]]; then export PYGLFW_LIBRARY_VARIANT=x11; fi
mode="${1:-pd}"
case "$mode" in
    retarget|reference)
        shift
        module=dog_middle_sim.retarget
        if [[ "$mode" == reference ]]; then module=dog_middle_sim.play_reference; fi
        exec "${MIDDLE_PYTHON:-python3}" -m "$module" "$@" ;;
    train|check|evaluate)
        shift
        exec bash "$repo_dir/projects/dog_middle_sim/run_gpu.sh" "$mode" "$@" ;;
    play)
        shift
        exec "${MIDDLE_PYTHON:-python3}" -m dog_middle_sim.play "$@" ;;
    pd) if [[ $# -gt 0 ]]; then shift; fi ;;
esac
exec "${MIDDLE_PYTHON:-python3}" -m dog_middle_sim.main "$@"
