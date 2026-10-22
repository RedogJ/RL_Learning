"""接触检测：按刚体名称区分足部与非足部，不依赖 USD 的排列顺序。"""

import math
import torch

LEGS = ("FL", "FR", "RL", "RR")
FOOT_NAMES = tuple(f"{leg}_foot" for leg in LEGS)
HIP_NAMES = tuple(f"{leg}_hip" for leg in LEGS)
# 后腿膝关节区域由小腿刚体承担；接触与向上支撑分别统计，是否扣分由任务决定。
REAR_KNEE_NAMES = tuple(f"{leg}_calf" for leg in LEGS[2:])
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


def contact_counts(force_history, foot_ids, undesired_ids, threshold, hip_ids=None, rear_ids=None,
                   knee_ids=None):
    """输入 (环境, 物理历史步, 刚体, XYZ)，输出每台狗触地的刚体数量。

    某刚体在本控制周期任一物理步的法向合力模长 > threshold 即算接触。
    同一小腿有多个碰撞几何、多个接触点，仍只计一个刚体。
    hip_ids 单独统计四个髋部刚体；rear_ids 是足部顺序中两只后脚的位置，
    用于统计两后脚在同一物理帧都失去向上支撑的比例。knee_ids 是后腿小腿（膝关节）
    刚体，用于统计「膝+脚掌同时支撑」。未提供时对应量保持为 0。
    """
    touching = force_history.norm(dim=-1).amax(dim=1) > threshold
    # 平地支撑要求向上的法向力；逐物理帧统计，不能用窗口峰值代替持续支撑。
    foot_support = force_history[:, :, foot_ids, 2] > threshold
    hip_contact_count = (touching[:, hip_ids].float().sum(dim=-1) if hip_ids is not None
                         else touching.new_zeros(touching.shape[0]))
    zero = touching.new_zeros(touching.shape[0])
    rear_airborne = ((~foot_support[:, :, rear_ids].any(dim=-1)).float().mean(dim=-1)
                     if rear_ids is not None else zero)
    if knee_ids is not None:
        knee_support = force_history[:, :, knee_ids, 2] > threshold
        rear_knee_contact_count = touching[:, knee_ids].float().sum(dim=-1)
        # 本周期有向上支撑的膝；侧向/自碰撞也可能计入上面的接触数，但不一定向上支撑。
        rear_knee_support_count = knee_support.any(dim=1).float().sum(dim=-1)
        rear_knee_support_fraction = knee_support.float().mean(dim=(1, 2))
        if rear_ids is not None:
            # 每条后腿只有膝和脚掌在同一物理帧都支撑，才算这条腿被支撑。
            rear_leg_supported = foot_support[:, :, rear_ids] & knee_support
            rear_leg_supported_fraction = rear_leg_supported.float().mean(dim=1)
            rear_knee_and_foot_fraction = rear_leg_supported.float().mean(dim=(1, 2))
            rear_both_unsupported_fraction = (~rear_leg_supported.any(dim=-1)).float().mean(dim=-1)
        else:
            rear_leg_supported_fraction = touching.new_zeros((touching.shape[0], 2))
            rear_knee_and_foot_fraction = zero
            rear_both_unsupported_fraction = zero
    else:
        rear_knee_contact_count = zero
        rear_knee_support_count = zero
        rear_knee_support_fraction = zero
        rear_leg_supported_fraction = touching.new_zeros((touching.shape[0], 2))
        rear_knee_and_foot_fraction = zero
        rear_both_unsupported_fraction = zero
    return {
        "foot_contact": touching[:, foot_ids],
        "foot_contact_count": touching[:, foot_ids].float().sum(dim=-1),
        "undesired_contact_count": touching[:, undesired_ids].float().sum(dim=-1),
        "hip_contact_count": hip_contact_count,
        "rear_feet_airborne_fraction": rear_airborne,
        "rear_knee_contact_count": rear_knee_contact_count,
        "rear_knee_support_count": rear_knee_support_count,
        "rear_knee_support_fraction": rear_knee_support_fraction,
        "rear_leg_supported_fraction": rear_leg_supported_fraction,
        "rear_knee_and_foot_support_fraction": rear_knee_and_foot_fraction,
        "rear_both_unsupported_fraction": rear_both_unsupported_fraction,
        "foot_support_fraction": foot_support.float().mean(dim=(1, 2)),
        "foot_support_per_foot": foot_support.float().mean(dim=1),
        "all_feet_airborne_fraction": (~foot_support.any(dim=-1)).float().mean(dim=1),
        "all_feet_support_fraction": foot_support.all(dim=-1).float().mean(dim=1),
    }


class FootPhaseTimes:
    """逐物理步记录完整的足部阶段；等待期与逐环境重置均清空历史。"""

    def __init__(self, template):
        self.air = torch.zeros_like(template)
        self.contact = torch.zeros_like(template)
        self.last_air = torch.zeros_like(template)
        self.last_contact = torch.zeros_like(template)

    def update(self, touching, dt, active):
        active = active[:, None]
        # 沿用 Isaac 接触计时规则：阶段结束时包含状态切换所在的物理周期。
        self.last_air.copy_(torch.where(
            active, torch.where(touching & (self.air > 0), self.air + dt, self.last_air), 0.0))
        self.last_contact.copy_(torch.where(
            active, torch.where(~touching & (self.contact > 0), self.contact + dt, self.last_contact), 0.0))
        self.air.copy_(torch.where(active & ~touching, self.air + dt, 0.0))
        self.contact.copy_(torch.where(active & touching, self.contact + dt, 0.0))

    def reset(self, env_ids):
        for value in (self.air, self.contact, self.last_air, self.last_contact):
            value[env_ids] = 0.0


class Go2Contacts:
    """只负责读取传感器；场景管理其更新与逐环境重置，奖励由 rewards.py 计算。"""

    def __init__(self, sensor, threshold, control_substeps, track_phase_times=False):
        self.sensor, self.threshold = sensor, threshold
        self.control_substeps = control_substeps
        if sensor.cfg.history_length < control_substeps:
            raise ValueError("接触传感器 history_length 必须覆盖一个控制周期的全部物理步")
        self.foot_ids, self.undesired_ids = select_contact_bodies(sensor.body_names)
        self.hip_ids = [sensor.body_names.index(name) for name in HIP_NAMES]
        self.knee_ids = [sensor.body_names.index(name) for name in REAR_KNEE_NAMES]
        self.leg_order = [self.foot_ids.index(sensor.body_names.index(name)) for name in FOOT_NAMES]
        self.rear_order = self.leg_order[2:]  # FL、FR、RL、RR 中后两只脚的位置。
        self.air_time = None
        self.contact_ratio = None
        self.mean_force_z = None
        self.track_phase_times = track_phase_times
        self.phase_times = None
        self.ratio_initial = 1.0  # 重置视为充分支撑，避免尚无历史时产生启动罚分。
        self.undesired_names = [sensor.body_names[i] for i in self.undesired_ids]

    def update_air_time(self, physics_dt, active, ratio_tau=0.5, ratio_initial=1.0):
        """每控制步只调用一次；历史第 0 帧最新，按旧到新处理短暂落地。"""
        history = self.sensor.data.net_normal_forces_w_history.torch
        force_z = history[:, :self.control_substeps, self.foot_ids, 2]
        support = force_z > self.threshold
        if self.air_time is None:
            self.air_time = history.new_zeros((history.shape[0], len(self.foot_ids)))
        self.ratio_initial = ratio_initial
        if self.contact_ratio is None:
            self.contact_ratio = torch.full_like(self.air_time, ratio_initial)
        if self.mean_force_z is None:
            self.mean_force_z = torch.zeros_like(self.air_time)
        if self.track_phase_times and self.phase_times is None:
            self.phase_times = FootPhaseTimes(self.air_time)
        touching = (history[:, :self.control_substeps, self.foot_ids].norm(dim=-1) > self.threshold
                    if self.track_phase_times else None)
        alpha = -math.expm1(-physics_dt / ratio_tau)
        for frame in range(self.control_substeps - 1, -1, -1):
            if self.phase_times is not None:
                self.phase_times.update(touching[:, frame], physics_dt, active)
            force_updated = self.mean_force_z + alpha * (force_z[:, frame] - self.mean_force_z)
            self.mean_force_z.copy_(torch.where(active[:, None], force_updated, 0.0))
            updated = self.contact_ratio + alpha * (support[:, frame].float() - self.contact_ratio)
            self.contact_ratio.copy_(torch.where(active[:, None], updated, ratio_initial))
            self.air_time.copy_(torch.where(
                active[:, None] & ~support[:, frame], self.air_time + physics_dt, 0.0,
            ))

    def reset(self, env_ids):
        if self.air_time is not None:
            self.air_time[env_ids] = 0.0
        if self.contact_ratio is not None:
            self.contact_ratio[env_ids] = self.ratio_initial
        if self.mean_force_z is not None:
            self.mean_force_z[env_ids] = 0.0
        if self.phase_times is not None:
            self.phase_times.reset(env_ids)

    def diagnostic_line(self, waiting=False, env_id=0):
        """显示当前帧接触与累计离地时间；只在低频播放诊断中传回 CPU。"""
        history = self.sensor.data.net_normal_forces_w_history.torch
        force_z = history[env_id, 0, self.foot_ids, 2]
        air = self.air_time[env_id] if self.air_time is not None else torch.zeros_like(force_z)
        ratio = self.contact_ratio[env_id] if self.contact_ratio is not None else torch.full_like(force_z, self.ratio_initial)
        rows = torch.stack((force_z, air, ratio), dim=-1).cpu().tolist()
        by_name = {self.sensor.body_names[index]: row for index, row in zip(self.foot_ids, rows)}
        parts = []
        for leg in LEGS:
            force, seconds, recent = by_name[f"{leg}_foot"]
            contact = "触地" if force > self.threshold else "离地"
            parts.append(f"{leg} {contact} {seconds:.2f}s Fz={force:.1f}N 近期接触={recent:.0%}")
        status = "重置等待，计时暂停" if waiting else "控制中"
        return f"[足部 env={env_id} {status}] " + " | ".join(parts)

    def read(self):
        # 当前 Isaac Lab 使用 ProxyArray；.torch 取得 GPU 上的张量，无需复制到 CPU。
        history = self.sensor.data.net_normal_forces_w_history.torch
        # 只看本次控制周期；更长历史不应把上次碰撞重复计入。
        result = contact_counts(history[:, :self.control_substeps], self.foot_ids,
                                self.undesired_ids, self.threshold, self.hip_ids, self.rear_order,
                                self.knee_ids)
        # 对外逐脚支撑数据固定为 FL、FR、RL、RR，不依赖传感器排列。
        result["foot_support_per_foot"] = result["foot_support_per_foot"][:, self.leg_order]
        result["foot_air_time"] = (self.air_time if self.air_time is not None else
                                   history.new_zeros((history.shape[0], len(self.foot_ids))))
        result["foot_contact_ratio"] = (self.contact_ratio if self.contact_ratio is not None else
                                        history.new_full((history.shape[0], len(self.foot_ids)), self.ratio_initial))
        result["foot_mean_force_z"] = (self.mean_force_z if self.mean_force_z is not None else
                                       history.new_zeros((history.shape[0], len(self.foot_ids))))
        if self.track_phase_times:
            zero = history.new_zeros((history.shape[0], len(self.foot_ids)))
            result["foot_last_air_time"] = self.phase_times.last_air if self.phase_times is not None else zero
            result["foot_last_contact_time"] = self.phase_times.last_contact if self.phase_times is not None else zero
        return result
