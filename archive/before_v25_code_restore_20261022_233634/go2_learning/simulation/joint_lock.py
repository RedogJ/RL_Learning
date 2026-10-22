"""用物理限位锁住指定关节；保留原有 DOF 与观测顺序以兼容检查点。"""

import math

import torch


def lock_joint_positions(robot, positions):
    """一次性设置零宽限位和锁定驱动，返回按名称匹配的关节索引。"""
    names = list(robot.joint_names)
    ids = [names.index(name) for name in positions]
    if not ids:
        return []
    if not all(math.isfinite(value) for value in positions.values()):
        raise ValueError("固定关节角必须为有限数值")
    original_limits = robot.data.joint_pos_limits.torch[:, ids]
    target = original_limits.new_tensor(list(positions.values())).expand(original_limits.shape[0], -1)
    if not ((target >= original_limits[..., 0]) & (target <= original_limits[..., 1])).all():
        raise ValueError("固定关节角超出原模型机械限位")
    robot.write_joint_position_limit_to_sim_index(
        limits=torch.stack((target, target), dim=-1), joint_ids=ids, warn_limit_violation=False,
    )
    # 高刚度驱动补强强冲击下的限位精度，模拟机械制动；不是正常电机的 PD/力矩参数。
    # 目标由环境固定，策略不能驱动这些关节；无需逐物理步改写关节状态。
    robot.write_joint_stiffness_to_sim_index(stiffness=1e6, joint_ids=ids)
    robot.write_joint_damping_to_sim_index(damping=1e3, joint_ids=ids)
    robot.write_joint_effort_limit_to_sim_index(limits=1e6, joint_ids=ids)
    return ids
