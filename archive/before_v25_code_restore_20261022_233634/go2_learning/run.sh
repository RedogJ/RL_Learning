#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 选择运行方式，Docker 环境初始化由共用启动器完成。
set -euo pipefail

mode=view
if [[ $# -gt 0 && "$1" != --* ]]; then mode="$1"; shift; fi
case "$mode" in
    view) defaults=(--control view --viz newton_gl --num_envs 1) ;;
    train) defaults=(--control train --viz none) ;;
    play) defaults=(--control play --viz newton_gl --num_envs 1) ;;
    *) printf 'Usage: bash run.sh [view|train|play] [arguments]\n' >&2; exit 2 ;;
esac
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$project_dir/../.." && pwd)"
exec bash "$repo_dir/scripts/run-isaac-project.sh" "$project_dir" go2_learning.main "${defaults[@]}" "$@"
