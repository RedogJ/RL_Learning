#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_dir="$(cd -- "$project_dir/../.." && pwd)"
exec bash "$repo_dir/scripts/run-isaac-project.sh" "$project_dir" go2_learning.tasks.upright_supermimic.rolling --viz none "$@"
