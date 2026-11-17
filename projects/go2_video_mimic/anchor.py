"""Reference and reward coordinates, independent of planar camera motion.

Positions/orientations are expressed in each actor's yaw frame, retaining world
height and roll/pitch. Velocities subtract anchor translation and yaw transport.
This does not estimate the video's physical speed or its camera trajectory.
"""
import torch
import torch.nn.functional as F
from go2_mujoco_mimic.tasks.mimic.kinematics import rotate
from go2_mujoco_mimic.tasks.mimic.rewards import quat_mul, quat_conjugate


def yaw_quat(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)) / 2
    zero = torch.zeros_like(yaw)
    return torch.stack((yaw.cos(), zero, zero, yaw.sin()), -1)


def remove_yaw(q):
    out = quat_mul(quat_conjugate(yaw_quat(q)), q)
    return torch.where(out[..., :1] < 0, -out, out)


def canonical_goal(reference):
    p = reference['pos']
    position = torch.cat((torch.zeros_like(p[..., :2]), p[..., 2:3]), -1)
    return torch.cat((reference['joints'], position, remove_yaw(reference['quat'])), -1)


def features(pose):
    p, q = pose['pos'], F.normalize(pose['quat'], dim=-1)
    anchor, rootq = p[:, :1], q[:, 0]
    inv_yaw = quat_conjugate(yaw_quat(rootq))[:, None].expand_as(q)
    relative = p - anchor
    local_p = rotate(inv_yaw, relative)
    # World height is retained, unlike planar translation.
    local_p = local_p + torch.cat((torch.zeros_like(anchor[..., :2]), anchor[..., 2:3]), -1)
    local_q = quat_mul(inv_yaw, q)
    v, omega = pose['linear_vel'], pose['angular_vel']
    forward = rotate(rootq, rootq.new_tensor([1., 0., 0.]))
    df = torch.cross(omega[:, 0], forward, dim=-1)
    yaw_rate = (forward[:, 0]*df[:, 1] - forward[:, 1]*df[:, 0]) / forward[:, :2].square().sum(-1).clamp_min(1e-5)
    yaw_omega = torch.stack((torch.zeros_like(yaw_rate), torch.zeros_like(yaw_rate), yaw_rate), -1)[:, None]
    local_v = rotate(inv_yaw, v - v[:, :1] - torch.cross(yaw_omega.expand_as(relative), relative, dim=-1))
    local_v = local_v + torch.cat((torch.zeros_like(v[:, :1, :2]), v[:, :1, 2:3]), -1)
    local_w = rotate(inv_yaw, omega - yaw_omega)
    out = dict(pose)
    out.update(pos=local_p, quat=local_q, linear_vel=local_v, angular_vel=local_w)
    return out
