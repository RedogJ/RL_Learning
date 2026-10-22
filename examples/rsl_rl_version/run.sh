#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 选择运行方式，Docker 环境初始化由共用启动器完成。
set -euo pipefail

mode="${1:-train}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
    train) defaults=(--control train --viz none) ;;
    play) defaults=(--control play --viz newton_gl --num_envs 1) ;;
    *) printf 'Usage: bash run.sh [train|play] [arguments]\n' >&2; exit 2 ;;
esac
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$project_dir/../.." && pwd)"
exec bash "$repo_dir/scripts/run-isaac-project.sh" "$project_dir" rsl_rl_version.main "${defaults[@]}" "$@"
