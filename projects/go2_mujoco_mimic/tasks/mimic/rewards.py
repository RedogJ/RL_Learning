"""身体位置、朝向、速度跟踪；输出各项奖励率。"""
import torch
import torch.nn.functional as F


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw*bw - (av*bv).sum(-1, keepdim=True),
                      aw*bv + bw*av + torch.cross(av, bv, dim=-1)), dim=-1)


def rotation_vector(q_to, q_from):
    q = F.normalize(quat_mul(F.normalize(q_to, dim=-1), quat_conjugate(F.normalize(q_from, dim=-1))), dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    length = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, q[..., :1].clamp_min(0))
    return v * torch.where(length > 1e-7, angle / length.clamp_min(1e-7), 2.0)


def gravity_local(q):
    g = torch.zeros_like(q)
    g[..., 3] = -1
    return quat_mul(quat_mul(quat_conjugate(q), g), q)[..., 1:]


def compute_reward(cfg, actual, ref, action, previous_action, joint_pos, joint_limits, dt):
    pos = (actual['pos'] - ref['pos']).square().sum(-1).mean(-1)
    ori = rotation_vector(actual['quat'], ref['quat']).square().sum(-1).mean(-1)
    vel = (actual['linear_vel'] - ref['linear_vel']).square().sum(-1).mean(-1)
    ang = (actual['angular_vel'] - ref['angular_vel']).square().sum(-1).mean(-1)
    root_pos = (actual['pos'][:, 0] - ref['pos'][:, 0]).square().sum(-1)
    root_ori = rotation_vector(actual['quat'][:, 0], ref['quat'][:, 0]).square().sum(-1)
    limits = (joint_limits[:, 0] - joint_pos).clamp_min(0) + (joint_pos - joint_limits[:, 1]).clamp_min(0)
    terms = {
        'body_position': cfg.body_position_weight * torch.exp(-pos / cfg.body_position_sigma**2),
        'body_orientation': cfg.body_orientation_weight * torch.exp(-ori / cfg.body_orientation_sigma**2),
        'body_linear_velocity': cfg.body_linear_velocity_weight * torch.exp(-vel / cfg.body_linear_velocity_sigma**2),
        'body_angular_velocity': cfg.body_angular_velocity_weight * torch.exp(-ang / cfg.body_angular_velocity_sigma**2),
        'root_position': cfg.root_position_weight * torch.exp(-root_pos / cfg.root_position_sigma**2),
        'root_orientation': cfg.root_orientation_weight * torch.exp(-root_ori / cfg.root_orientation_sigma**2),
        'action_rate': -cfg.action_rate_weight * (action - previous_action).square().sum(-1),
        'joint_limit': -cfg.joint_limit_weight * limits.sum(-1),
    }
    return sum(terms.values()) * dt, terms
