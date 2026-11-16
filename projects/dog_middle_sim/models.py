"""目标编码策略、速度增量世界模型、特权 Critic。"""
import torch
from torch import nn
from .math_utils import (represent, rotate, inverse_rotate, unit, multiply,
                         exp_quat, rotvec)


def mlp(inputs, hidden, outputs):
    layers=[]
    for width in hidden:
        layers.extend((nn.Linear(inputs,width),nn.LayerNorm(width),nn.GELU()))
        inputs=width
    layers.append(nn.Linear(inputs,outputs))
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    def __init__(self, floor):
        super().__init__()
        self.register_buffer('mean',torch.zeros(len(floor)))
        self.register_buffer('variance',torch.ones(len(floor)))
        self.register_buffer('count',torch.tensor(0.0))
        self.register_buffer('floor',torch.tensor(floor))

    @torch.no_grad()
    def update(self, x):
        x=x.reshape(-1,len(self.mean)).float()
        n=x.shape[0]; total=self.count+n
        delta=x.mean(0)-self.mean
        var=(self.variance*self.count+x.var(0,unbiased=False)*n
             +delta.square()*self.count*n/total)/total
        self.mean.add_(delta*n/total)
        self.variance.copy_(var)
        self.count.copy_(total)

    def forward(self,x):
        return (x-self.mean)/self.variance.sqrt().clamp_min(self.floor)


class Actor(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.proprio_normalizer=Normalizer([.1]*42)
        self.goal_normalizer=Normalizer([.1]*(cfg.env.future*21))
        self.encoder=mlp(cfg.env.future*21,cfg.model.policy_hidden,cfg.model.latent)
        self.decoder=mlp(cfg.env.history*42+cfg.model.latent,cfg.model.policy_hidden,12)
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)

    def forward(self,history,goal):
        z=torch.tanh(self.encoder(self.goal_normalizer(goal)))
        return torch.tanh(self.decoder(torch.cat((self.proprio_normalizer(history).flatten(-2),z),-1)))


class Dynamics(nn.Module):
    def __init__(self,cfg,default,limits):
        super().__init__()
        self.dt=1/cfg.env.control_hz
        substeps=cfg.env.physics_hz//cfg.env.control_hz
        self.coeff=(substeps+1)/(2*substeps)
        self.action_scale=cfg.env.action_scale
        self.register_buffer('default',default.clone())
        self.register_buffer('limits',limits.clone())
        self.normalizer=Normalizer([.1]+[.1]*3+[.2]*3+[.5]*3+[.1]*12+[1.]*12)
        self.action_normalizer=Normalizer([.1]*12)
        w=cfg.model.wm_width
        self.net=mlp(34+12+4,(w,w,w),18)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def processed(self,action):
        return (self.default+self.action_scale*action.clamp(-1,1)).clamp(self.limits[:,0],self.limits[:,1])

    @torch.no_grad()
    def update_normalization(self,states,actions):
        self.normalizer.update(represent(states))
        self.action_normalizer.update(self.processed(actions))

    def forward(self,state,action,context):
        rep=represent(state)
        x=torch.cat((self.normalizer(rep),self.action_normalizer(self.processed(action)),context),-1)
        delta=self.net(x)
        dv,dw,dj=delta[...,:3],delta[...,3:6],delta[...,6:]
        q=unit(state[...,3:7]); v_b=rep[...,4:7];w_b=state[...,22:25]
        next_pos=state[...,:3]+rotate(q,v_b+self.coeff*dv)*self.dt
        next_q=unit(multiply(q,exp_quat((w_b+self.coeff*dw)*self.dt)))
        next_j=state[...,7:19]+(state[...,25:37]+self.coeff*dj)*self.dt
        next_v=rotate(q,v_b+dv)
        next_w=inverse_rotate(next_q,rotate(q,w_b+dw))
        return torch.cat((next_pos,next_q,next_j,next_v,next_w,state[...,25:37]+dj),-1)


class Critic(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.net=mlp(34+cfg.env.future*21+cfg.env.history*42+4+1,cfg.model.critic_hidden,1)

    def forward(self,obs):
        return self.net(obs).squeeze(-1)


def critic_observation(model,task,state,history,step,episode_steps):
    remaining=((episode_steps-step)/episode_steps).clamp(0,1).unsqueeze(-1)
    return torch.cat((model.normalizer(represent(state)),task.goal(state),history.flatten(-2),
                      task.context.expand(*state.shape[:-1],4),remaining),-1)


def state_loss(predicted,target,cfg):
    # 固定物理尺度配权；不以统计方差代替监督目标权重。
    return ((predicted[...,:3]-target[...,:3]).square().sum(-1)/cfg.root_pos_error**2
            +rotvec(predicted[...,3:7],target[...,3:7]).square().sum(-1)/cfg.root_rot_error**2
            +(predicted[...,7:19]-target[...,7:19]).square().sum(-1)/cfg.joint_pos_error**2
            +(predicted[...,25:37]-target[...,25:37]).square().sum(-1)/cfg.joint_vel_error**2)/30
