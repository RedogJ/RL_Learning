"""从编译后的 MJCF 读取结构，使用 Torch 可微计算跟踪刚体位姿。"""
import torch
from torch import nn
import torch.nn.functional as F
from .rewards import quat_mul, rotation_vector


def rotate(q, v):
    v = v.expand(*q.shape[:-1], 3)
    u = q[..., 1:]
    t = 2 * torch.cross(u, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(u, t, dim=-1)


class BodyKinematics(nn.Module):
    def __init__(self, model, body_ids, joint_ids):
        super().__init__()
        import mujoco
        ids = list(map(int, body_ids))
        mapping = {bid: i for i, bid in enumerate(ids)}
        joint_map = {int(j): i for i, j in enumerate(joint_ids)}
        self.parents, self.joints = [], []
        axes, anchors, refs = [], [], []
        for i, bid in enumerate(ids):
            if i == 0:
                if model.body_jntnum[bid] != 1 or model.jnt_type[model.body_jntadr[bid]] != mujoco.mjtJoint.mjJNT_FREE:
                    raise ValueError('首个跟踪刚体必须是浮动根部')
                self.parents.append(-1); self.joints.append(-1)
                axes.append([0., 0., 0.]); anchors.append([0., 0., 0.]); refs.append(0.)
                continue
            parent = mapping[int(model.body_parentid[bid])]
            if parent >= i: raise ValueError('刚体必须按父节点在前排序')
            self.parents.append(parent)
            count = int(model.body_jntnum[bid])
            if count == 0:
                self.joints.append(-1); axes.append([0., 0., 0.]); anchors.append([0., 0., 0.]); refs.append(0.)
            elif count == 1:
                j = int(model.body_jntadr[bid])
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError('目前仅支持 Go2 的转动关节')
                self.joints.append(joint_map[j]); axes.append(model.jnt_axis[j].copy())
                anchors.append(model.jnt_pos[j].copy()); refs.append(model.qpos0[model.jnt_qposadr[j]])
            else:
                raise ValueError('每个非根部刚体最多支持一个关节')
        import numpy as np
        for name, value in [('offset', model.body_pos[ids]), ('rotation', model.body_quat[ids]),
                            ('axis', axes), ('anchor', anchors), ('reference', refs)]:
            self.register_buffer(name, torch.tensor(np.asarray(value), dtype=torch.float32))

    def forward(self, state):
        positions = [state[..., :3]]
        rotations = [F.normalize(state[..., 3:7], dim=-1)]
        for i in range(1, len(self.parents)):
            parent = self.parents[i]
            pq = rotations[parent]
            q = quat_mul(pq, self.rotation[i].expand_as(pq))
            p = positions[parent] + rotate(pq, self.offset[i])
            j = self.joints[i]
            if j >= 0:
                half = (state[..., 7+j] - self.reference[i]) / 2
                dq = torch.cat((half.cos()[..., None], half.sin()[..., None]*self.axis[i]), -1)
                anchor_world = p + rotate(q, self.anchor[i])
                q = quat_mul(q, dq)
                p = anchor_world - rotate(q, self.anchor[i])
            positions.append(p); rotations.append(q)
        return torch.stack(positions, -2), torch.stack(rotations, -2)

    def transition(self, previous, current, dt):
        """与真实环境一致：身体速度使用一个控制周期的位姿差分。"""
        old_pos, old_quat = self(previous)
        pos, quat = self(current)
        return {'pos': pos, 'quat': quat, 'linear_vel': (pos-old_pos)/dt,
                'angular_vel': rotation_vector(quat, old_quat)/dt}
