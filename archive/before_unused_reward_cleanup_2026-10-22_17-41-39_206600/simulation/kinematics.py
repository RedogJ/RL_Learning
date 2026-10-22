"""仅依赖 torch 的批量坐标变换；四元数顺序与当前 Isaac Lab 一致，为 xyzw。"""

import torch


def points_in_body_frame(points_w, body_pos_w, body_quat_w):
    """世界点 (N, K, 3) 转到机身 link 原点坐标系；输入四元数应为单位四元数。"""
    offset = points_w - body_pos_w[:, None, :]
    xyz = body_quat_w[:, None, :3].expand_as(offset)
    cross = 2.0 * torch.cross(xyz, offset, dim=-1)
    return offset - body_quat_w[:, None, 3:] * cross + torch.cross(xyz, cross, dim=-1)
