#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$project_dir/../.." && pwd)"
mode=play
if [[ $# -gt 0 && "$1" != --* ]]; then mode="$1"; shift; fi
python_bin="${GO2_SIM2SIM_PYTHON:-python}"
export PYTHONPATH="$repo_dir/projects${PYTHONPATH:+:$PYTHONPATH}"
# GLFW 的 Python wheel 同时提供两套库；选 X11 库后再导入，不能对 Wayland 库强设 X11。
if [[ -n "${DISPLAY:-}" ]]; then
    export PYGLFW_LIBRARY_VARIANT="${PYGLFW_LIBRARY_VARIANT:-x11}"
fi
case "$mode" in
    fetch-model) exec "$python_bin" -m go2_sim2sim.fetch_model "$@" ;;
    play|test)
        if ! "$python_bin" -c 'import torch, mujoco' >/dev/null 2>&1; then
            printf '当前 Python 缺少 torch 或 mujoco。请先 conda activate sim_env，或设置 GO2_SIM2SIM_PYTHON。\n' >&2
            exit 1
        fi
        defaults=()
        if [[ "$mode" == test ]]; then defaults=(--headless --duration 30); fi
        exec "$python_bin" -m go2_sim2sim.main "${defaults[@]}" "$@" ;;
    *) printf '用法：bash run.sh [play|test|fetch-model] [参数]\n' >&2; exit 2 ;;
esac
