"""真实仿真与想象共用参考、观测目标、奖励和终止规则。"""
import torch
from ...math_utils import (unit, rotate, inverse_rotate, multiply, conjugate,
                           rotvec, yaw_quat, rotation6d, gravity)


class StandingTask:
    def __init__(self, env_cfg, reward_cfg, kinematics, reference, limits, context):
        self.cfg, self.reward_cfg, self.fk = env_cfg, reward_cfg, kinematics
        self.reference = reference
        self.default = reference[7:19]
        self.limits, self.context = limits, context
        self.dt = 1/env_cfg.control_hz
        self.reference_pose = self.fk(reference.unsqueeze(0))
        self.max_step_reward = self.dt * sum(getattr(reward_cfg,k) for k in vars(reward_cfg)
                                           if k.endswith('_weight') and k not in ('action_rate_weight','joint_limit_weight'))
        for name, value in vars(reward_cfg).items():
            if name.endswith('_sigma') and value <= 0:
                raise ValueError(f'{name} 必须为正')

    def targets(self, action):
        q = self.default + self.cfg.action_scale*action.clamp(-1, 1)
        return q.clamp(self.limits[:,0],self.limits[:,1])

    def goal(self, state):
        q = unit(state[...,3:7]); refq = self.reference[3:7].expand_as(q)
        target = torch.cat((self.default.expand(*state.shape[:-1],12),
                            inverse_rotate(q,self.reference[:3]-state[...,:3]),
                            rotation6d(multiply(conjugate(q),refq))), -1)
        return target.unsqueeze(-2).expand(*state.shape[:-1],self.cfg.future,21).flatten(-2)

    def termination(self, state, soft=False):
        height = (state[...,2]-self.reference[2]).abs()/self.cfg.height_threshold
        tilt = (gravity(unit(state[...,3:7]))[...,2]-gravity(self.reference[3:7])[2]).abs()/self.cfg.tilt_threshold
        error = torch.maximum(height,tilt)
        return torch.sigmoid((error-1)/self.cfg.termination_softness) if soft else error > 1

    def reward(self, previous, state, action, last_action):
        pos, q = self.fk(state)
        old_pos, old_q = self.fk(previous)
        refpos, refq = self.reference_pose
        rootq = unit(state[...,3:7])
        delta = yaw_quat(multiply(rootq,conjugate(self.reference[3:7].expand_as(rootq))))
        anchor = torch.cat((state[...,:2],self.reference[2:3].expand(*state.shape[:-1],1)),-1)
        local_targets = refpos - self.reference[:3]
        body_target = anchor.unsqueeze(-2)+rotate(delta.unsqueeze(-2),local_targets)
        orient_target = multiply(delta.unsqueeze(-2).expand_as(q),refq.expand_as(q))
        errors = {
            'root_position': (state[...,:3]-self.reference[:3]).square().sum(-1),
            'root_orientation': rotvec(rootq,self.reference[3:7].expand_as(rootq)).square().sum(-1),
            'body_position': (pos-body_target).square().sum(-1).mean(-1),
            'body_orientation': rotvec(q,orient_target).square().sum(-1).mean(-1),
            'body_linear_velocity': ((pos-old_pos)/self.dt).square().sum(-1).mean(-1),
            'body_angular_velocity': (rotvec(q,old_q)/self.dt).square().sum(-1).mean(-1),
        }
        cfg = self.reward_cfg
        terms = {k:getattr(cfg,k+'_weight')*torch.exp(-v/getattr(cfg,k+'_sigma')**2)
                 for k,v in errors.items()}
        terms['action_rate'] = -cfg.action_rate_weight*(self.targets(action)-self.targets(last_action)).square().sum(-1)
        # 软限位为实际角度范围的 95%，并非动作限幅。
        center = self.limits.mean(-1); half = (self.limits[:,1]-self.limits[:,0])*0.475
        excess = (center-half-state[...,7:19]).clamp_min(0)+(state[...,7:19]-center-half).clamp_min(0)
        terms['joint_limit'] = -cfg.joint_limit_weight*excess.sum(-1)
        return {k:v*self.dt for k,v in terms.items()}
