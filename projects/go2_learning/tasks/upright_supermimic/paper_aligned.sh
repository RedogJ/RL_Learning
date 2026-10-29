#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_dir="$(cd -- "$project_dir/../.." && pwd)"
mode="${1:-train}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  train) entry=paper_aligned_train; viz=none ;;
  eval) entry=paper_aligned_evaluate; viz=none ;;
  play) entry=paper_aligned_play; viz=newton_gl ;;
  *) printf '用法：%s {train|eval|play} [参数...]\n' "$0" >&2; exit 2 ;;
esac
exec bash "$repo_dir/scripts/run-isaac-project.sh" "$project_dir" "go2_learning.tasks.upright_supermimic.$entry" --viz "$viz" "$@"
