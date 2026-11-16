"""通用浮动基座 + 顺序多 hinge 的可微 FK，支持 middle 的双髋轴 body。"""
import numpy as np
import torch
from torch import nn
import mujoco
from .math_utils import rotate, multiply, exp_quat, unit
from .build_model import JOINTS


class Kinematics(nn.Module):
    def __init__(self, model):
        super().__init__()
        # 电机外壳是父实体的固定附件；跟踪9个骨架实体，避免重复计奖和冗余FK。
        self.body_ids = [i for i in range(1,model.nbody) if '_motor_' not in model.body(i).name]
        self.body_names = [model.body(i).name for i in self.body_ids]
        mapping = {bid:i for i,bid in enumerate(self.body_ids)}
        joint_map = {model.joint(name).id:i for i,name in enumerate(JOINTS)}
        self.parents, self.chains = [], []
        for bid in self.body_ids:
            self.parents.append(mapping.get(int(model.body_parentid[bid]), -1))
            chain = []
            for j in range(model.body_jntadr[bid], model.body_jntadr[bid]+model.body_jntnum[bid]):
                if j in joint_map:
                    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                        raise ValueError('当前 FK 仅支持 hinge')
                    chain.append((j, joint_map[j]))
            self.chains.append(chain)
        for name, values in [('offset',model.body_pos),('rotation',model.body_quat),
                             ('axis',model.jnt_axis),('anchor',model.jnt_pos),('q0',model.qpos0)]:
            self.register_buffer(name, torch.tensor(np.array(values), dtype=torch.float32))
        self.qadr = [int(x) for x in model.jnt_qposadr]

    def forward(self, state):
        ps, qs = [state[..., :3]], [unit(state[..., 3:7])]
        for i, bid in enumerate(self.body_ids[1:], 1):
            parent = self.parents[i]
            p = ps[parent] + rotate(qs[parent], self.offset[bid])
            q = multiply(qs[parent], self.rotation[bid].expand_as(qs[parent]))
            for jid, index in self.chains[i]:
                anchor = p + rotate(q, self.anchor[jid])
                dq = exp_quat((state[..., 7+index]-self.q0[self.qadr[jid]])[..., None]*self.axis[jid])
                q = multiply(q, dq)
                p = anchor - rotate(q, self.anchor[jid])
            ps.append(p); qs.append(q)
        return torch.stack(ps, -2), torch.stack(qs, -2)
