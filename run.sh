#!/usr/bin/env bash
# 从仓库根目录选择一个工程；参数原样交给各工程入口。
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="${1:---help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$project" in
    go2) exec bash "$repo_dir/projects/go2_learning/run.sh" "$@" ;;
    go2-sim2sim) exec bash "$repo_dir/projects/go2_sim2sim/run.sh" "$@" ;;
    cartpole-rsl) exec bash "$repo_dir/examples/rsl_rl_version/run.sh" "$@" ;;
    cartpole-ppo) exec bash "$repo_dir/examples/cartpole_ppo/run.sh" "$@" ;;
    cartpole-official) exec bash "$repo_dir/scripts/run-cartpole-official.sh" "$@" ;;
    cartpole-compare) exec bash "$repo_dir/scripts/run-cartpole-official.sh" compare "$@" ;;
    -h|--help)
        printf '%s\n' \
            '用法：bash run.sh <工程> [运行方式] [参数]' \
            '  go2                Go2 主工程：view（默认）、train、play' \
            '  go2-sim2sim        独立 MuJoCo：play（默认）、test、fetch-model' \
            '  cartpole-rsl       RSL-RL 双杆示例：train（默认）、play' \
            '  cartpole-ppo       手写 PPO 双杆示例：参数见 examples/cartpole_ppo/README.md' \
            '  cartpole-official  Isaac Lab 官方单杆：train（默认）、play' \
            '  cartpole-compare   官方单杆 L2 / L3 权重对照'
        ;;
    *) printf '未知工程：%s；使用 bash run.sh --help 查看入口。\n' "$project" >&2; exit 2 ;;
esac
