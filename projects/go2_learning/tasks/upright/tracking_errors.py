"""连续误差的唯一口径：奖励计算和 sigma 采样共用，不含权重、dt 或 sigma。"""

import torch


# 项名 -> (sigma 字段, weight 字段, 原始误差单位)。缓冲区保存 sqrt(误差均方)。
ERROR_SPECS = {
    "upright_hold": ("upright_sigma", "upright_hold_weight", "1-alignment"),
    "height": ("height_sigma", "height_weight", "m"),
    "rear_symmetry": ("rear_symmetry_sigma", "rear_symmetry_weight", "rad"),
    "rear_posture": ("rear_posture_sigma", "rear_posture_weight", "rad"),
}


def upright_alignment(state):
    """机身 +X 与世界 +Z 的点积；不使用欧拉角。"""
    return (-state["gravity"][:, 0]).clamp(-1.0, 1.0)


def rear_symmetry_error_squared(state):
    """两后髋相加、两后大腿相减、两后小腿相减后取均方，单位 rad²。"""
    hip = state["hip_joint_pos"][:, 2:]
    thigh = state["thigh_joint_pos"][:, 2:]
    calf = state["calf_joint_pos"][:, 2:]
    return (hip.sum(-1).square() + (thigh[:, 0] - thigh[:, 1]).square()
            + (calf[:, 0] - calf[:, 1]).square()) / 3.0


def rear_symmetry_error(state):
    return rear_symmetry_error_squared(state).sqrt()


def rear_posture_error_squared(state, cfg):
    """四个后大腿/小腿关节对直立参考角的均方误差，单位 rad²。"""
    return sum(
        (state[f"{part}_joint_pos"][:, 2:] - getattr(cfg, f"rear_posture_{part}")).square().sum(-1)
        for part in ("thigh", "calf")
    ) / 4.0


def continuous_errors(state, cfg):
    """返回逐环境误差均方和有效掩码；只读取已启用项需要的状态。"""
    squared, valid = {}, {}
    for axis_id, axis in enumerate("xyz"):
        for prefix, state_key in (("linear_velocity", "linear_velocity_w"),
                                  ("angular_velocity", "angular_velocity")):
            name = f"{prefix}_{axis}"
            if getattr(cfg, f"{name}_weight") != 0:
                squared[name] = state[state_key][:, axis_id].square()
    if cfg.joint_velocity_weight != 0:
        squared["joint_velocity"] = state["joint_velocity"].square().mean(-1)
    if cfg.upright_hold_weight != 0:
        squared["upright_hold"] = (1.0 - upright_alignment(state)).square()
    if cfg.height_weight != 0:
        squared["height"] = (state["height"] - cfg.target_height).square()
    if cfg.hip_rear_weight != 0:
        squared["hip_rear"] = state["hip_joint_pos"][:, 2:].square().mean(-1)
    if cfg.rear_symmetry_weight != 0:
        squared["rear_symmetry"] = rear_symmetry_error_squared(state)
    if cfg.rear_posture_weight != 0:
        squared["rear_posture"] = rear_posture_error_squared(state, cfg)
    if cfg.front_target_weight != 0:
        squared["front_target"] = sum(
            (state[f"{part}_joint_pos"][:, :2] - getattr(cfg, f"front_target_{part}")).square().sum(-1)
            for part in ("hip", "thigh", "calf")
        ) / 6.0
    if cfg.foot_slide_weight != 0:
        touching = state["foot_contact"].float()
        count = touching.sum(-1)
        # 空中腿不进入均方；无脚触地时误差为 0，旧版回放不领滑动收益。
        squared["foot_slide"] = (
            state["foot_velocity_w"][..., :2].square().sum(-1) * touching
        ).sum(-1) / count.clamp_min(1.0)
        valid["foot_slide"] = count > 0
    return squared, valid


def positive_error_rewards(squared, valid, dt, cfg, like):
    result = {}
    for name, (sigma_field, weight_field, _) in ERROR_SPECS.items():
        if name not in squared:
            # 保留关闭项的日志键，方便比较不同阶段。
            result[name] = torch.zeros_like(like)
            continue
        value = getattr(cfg, weight_field) * torch.exp(-squared[name] / getattr(cfg, sigma_field)**2) * dt
        result[name] = value * valid[name] if name in valid else value
    return result
