"""Torch 四元数均为 wxyz；所有运算支持批维和反向传播。"""
import torch
import torch.nn.functional as F


def unit(q):
    return F.normalize(q, dim=-1, eps=1e-8)


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def multiply(a, b):
    a, b = torch.broadcast_tensors(a, b)
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.cross(av, bv, dim=-1)), -1)


def rotate(q, v):
    shape = torch.broadcast_shapes(q.shape[:-1], v.shape[:-1])
    q, v = q.expand(*shape, 4), v.expand(*shape, 3)
    t = 2*torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1]*t + torch.cross(q[..., 1:], t, dim=-1)


def inverse_rotate(q, v):
    return rotate(conjugate(q), v)


def exp_quat(rotvec):
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle/2), 0.5*torch.sinc(angle/(2*torch.pi))*rotvec), -1)


def rotvec(a, b):
    """从 b 到 a 的世界坐标旋转向量；q/-q 等价。"""
    q = unit(multiply(unit(a), conjugate(unit(b))))
    q = torch.where(q[..., :1] < 0, -q, q)
    sine = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2*torch.atan2(sine, q[..., :1].clamp_min(1e-8))
    factor = torch.where(sine > 1e-6, angle/sine.clamp_min(1e-8), 2.0)
    return q[..., 1:]*factor


def gravity(q):
    return inverse_rotate(q, q.new_tensor([0., 0., -1.]))


def yaw_quat(q):
    w,x,y,z = unit(q).unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))/2
    return torch.stack((yaw.cos(), yaw*0, yaw*0, yaw.sin()), -1)


def rotation6d(q):
    x = rotate(q, q.new_tensor([1., 0., 0.]))
    y = rotate(q, q.new_tensor([0., 1., 0.]))
    return torch.cat((x, y), -1)


def represent(state):
    q = unit(state[..., 3:7])
    # 34D: z, gravity_b, velocity_b, angular_velocity_b, joints, joint_velocities.
    return torch.cat((state[..., 2:3], gravity(q), inverse_rotate(q, state[..., 19:22]),
                      state[..., 22:25], state[..., 7:19], state[..., 25:37]), -1)


def proprio(state, last_action, default):
    return torch.cat((state[..., 7:19]-default, state[..., 25:37]*0.1,
                      state[..., 22:25]*0.2, gravity(unit(state[..., 3:7])), last_action), -1)
