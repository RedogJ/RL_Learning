"""宿主机与容器共用的资源、日志位置；不依赖 Isaac Sim。"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_ROOT = PROJECT_ROOT / "logs"
# Docker 启动器显式传入 /workspace/assets；宿主机采用仓库中的 assets/。
ASSET_ROOT = Path(os.environ.get("GO2_ASSET_ROOT", PROJECT_ROOT.parents[1] / "assets"))
USD_PATH = ASSET_ROOT / "unitree_model/Go2/usd/go2.usd"
GROUND_PATH = ASSET_ROOT / "isaaclab/ground_plane/default_ground_plane.usda"
