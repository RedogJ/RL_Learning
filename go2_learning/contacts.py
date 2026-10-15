"""接触检测：按刚体名称区分足部与非足部，不依赖 USD 的排列顺序。"""

LEGS = ("FL", "FR", "RL", "RR")
FOOT_NAMES = tuple(f"{leg}_foot" for leg in LEGS)
REQUIRED_BODY_NAMES = ("base", "Head_upper", "Head_lower") + tuple(
    f"{leg}_{part}" for leg in LEGS for part in ("hip", "thigh", "calf", "foot")
)


def select_contact_bodies(body_names):
    """足部不扣分，其余刚体扣分；模型合并/遗漏刚体时明确报错。"""
    names = list(body_names)
    missing = sorted(set(REQUIRED_BODY_NAMES) - set(names))
    if missing or len(set(names)) != len(names):
        raise ValueError(f"Go2 接触传感器刚体不完整或重名，缺少 {missing}；实际为 {names}")
    feet = [i for i, name in enumerate(names) if name in FOOT_NAMES]
    undesired = [i for i, name in enumerate(names) if name not in FOOT_NAMES]
    return feet, undesired


def contact_counts(force_history, foot_ids, undesired_ids, threshold):
    """输入 (环境, 物理历史步, 刚体, XYZ)，输出每台狗触地的刚体数量。

    某刚体在本控制周期任一物理步的法向合力模长 > threshold 即算接触。
    同一小腿有多个碰撞几何、多个接触点，仍只计一个刚体。
    """
    touching = force_history.norm(dim=-1).amax(dim=1) > threshold
    # 平地支撑要求向上的法向力；逐物理帧统计，不能用窗口峰值代替持续支撑。
    foot_support = force_history[:, :, foot_ids, 2] > threshold
    return {
        "foot_contact_count": touching[:, foot_ids].float().sum(dim=-1),
        "undesired_contact_count": touching[:, undesired_ids].float().sum(dim=-1),
        "foot_support_fraction": foot_support.float().mean(dim=(1, 2)),
        "all_feet_support_fraction": foot_support.all(dim=-1).float().mean(dim=1),
    }


class Go2Contacts:
    """只负责读取传感器；场景管理其更新与逐环境重置，奖励由 rewards.py 计算。"""

    def __init__(self, sensor, threshold, control_substeps):
        self.sensor, self.threshold = sensor, threshold
        self.control_substeps = control_substeps
        if sensor.cfg.history_length < control_substeps:
            raise ValueError("接触传感器 history_length 必须覆盖一个控制周期的全部物理步")
        self.foot_ids, self.undesired_ids = select_contact_bodies(sensor.body_names)
        self.undesired_names = [sensor.body_names[i] for i in self.undesired_ids]

    def read(self):
        # 当前 Isaac Lab 使用 ProxyArray；.torch 取得 GPU 上的张量，无需复制到 CPU。
        history = self.sensor.data.net_normal_forces_w_history.torch
        # 只看本次控制周期；更长历史不应把上次碰撞重复计入。
        return contact_counts(history[:, :self.control_substeps], self.foot_ids,
                              self.undesired_ids, self.threshold)
