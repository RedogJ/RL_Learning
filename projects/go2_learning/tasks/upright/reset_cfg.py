"""直立任务的重置姿态；角度为 rad，机身前向轴仰角单独使用 deg。"""

from dataclasses import dataclass, fields
import math


@dataclass
class ResetCfg:
    root_height: float = 0.45  # m；机身根节点相对环境原点的高度。
    forward_elevation_deg: float = 90.0  # 0 为水平、90 为头朝上直立。
    orientation_cone_degrees: float = 10.0  # 策略接管时，机身前向轴相对竖直最多倾斜此角度。
    front_hip: float = 0.0
    front_thigh: float = 1.4
    front_calf: float = -2.4  # 前腿屈曲收在胸前。
    rear_hip: float = 0.0
    rear_thigh: float = 2.5595642
    rear_calf: float = -1.9862825  # 后脚支撑。
    # hip、thigh、calf 的 PD 目标偏置，单位 rad；物理初始角本身不加这些数值。
    # 近直立等待期的 PD 目标偏置，用于抵抗重力。
    front_hold_offset: tuple[float, float, float] = (0.0, -0.0110096, -0.0067346)
    rear_hold_offset: tuple[float, float, float] = (0.0, 0.018352, 0.6228905)

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith("_hold_offset"):
                if len(value) != 3 or not all(math.isfinite(x) for x in value):
                    raise ValueError(f"{field.name} 必须是三个有限角度，顺序为 hip/thigh/calf")
                setattr(self, field.name, tuple(value))  # JSON 读取后的列表也统一为 tuple。
            elif not math.isfinite(value):
                raise ValueError(f"{field.name} 必须为有限数值")
        if not 0 <= self.orientation_cone_degrees < 90:
            raise ValueError("orientation_cone_degrees 必须位于 [0,90)")
        if self.root_height <= 0:
            raise ValueError("root_height 必须为正数")
        if not 0 <= self.forward_elevation_deg < 180:
            raise ValueError("forward_elevation_deg 必须位于 [0, 180)")
