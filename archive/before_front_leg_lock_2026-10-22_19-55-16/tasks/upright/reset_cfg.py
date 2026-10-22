"""直立任务的重置姿态；角度为 rad，机身前向轴仰角单独使用 deg。"""

from dataclasses import dataclass, fields
import math


@dataclass
class ResetCfg:
    root_height: float = 0.45  # m；机身根节点相对环境原点的高度。
    forward_elevation_deg: float = 90.0  # 0 为水平、90 为头朝上直立。
    front_hip: float = 0.0
    front_thigh: float = 1.4
    front_calf: float = -2.4  # 两条前腿屈曲收在胸前。
    rear_hip: float = 0.0
    rear_thigh: float = 2.5595642
    rear_calf: float = -1.9862825  # 后脚支撑；90° / 0.45 m 下脚掌球心初始高约 0.0223 m。
    # hip、thigh、calf 的 PD 目标偏置，单位 rad；物理初始角本身不加这些数值。
    # 当前 Kp=25 时用于抵抗重力，避免 0.3 s 等待期腿软下沉；策略接管后不再加偏置。
    front_hold_offset: tuple[float, float, float] = (0.0, -0.0110096, -0.0067346)
    rear_hold_offset: tuple[float, float, float] = (0.0, 0.0183520, 0.6228905)

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith("_hold_offset"):
                if len(value) != 3 or not all(math.isfinite(x) for x in value):
                    raise ValueError(f"{field.name} 必须是三个有限角度，顺序为 hip/thigh/calf")
                setattr(self, field.name, tuple(value))  # JSON 读取后的列表也统一为 tuple。
            elif not math.isfinite(value):
                raise ValueError(f"{field.name} 必须为有限数值")
        if self.root_height <= 0:
            raise ValueError("root_height 必须为正数")
        if not 0 < self.forward_elevation_deg < 180:
            raise ValueError("forward_elevation_deg 必须位于 (0, 180)")
