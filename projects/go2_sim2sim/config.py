"""迁移参数与路径；策略相关参数以检查点旁的 run.json 为准。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[1]
MODEL_ROOT = REPO_ROOT / "assets/unitree_model/Go2/mjcf"
DEFAULT_CHECKPOINT = REPO_ROOT / (
    "projects/go2_learning/logs/running/"
    "2026-10-17_13-12-58_702879_running_16/model_56500.pt"
)

# 与目前 Isaac scene.py 的驱动配置对应，后端数值行为仍需独立验证。
KP = 25.0
KD = 0.5
TORQUE_LIMIT = 23.7
FRICTION = 0.5
PHYSICS_HZ = 600  # 策略仍为 30 Hz；每个目标保持 20 个物理步。
INITIAL_HEIGHT = 0.4
