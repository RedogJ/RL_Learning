"""论文式上下文条件世界模型；奖励和观测使用 Go2 的解析接口。"""
import torch
from torch import nn
from go2_learning.tasks.upright_supermimic.paper_context import CONTEXT_DIM


def enforce_executed_action(state,executed_action):
    """状态中的上一动作精确等于滤波后的实际执行动作。"""
    return torch.cat((state[:,:33],executed_action,state[:,45:]),-1)


class PaperWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim=state_dim
        self.action_dim=action_dim
        dim=state_dim+action_dim+CONTEXT_DIM
        self.net=nn.Sequential(nn.Linear(dim,2048),nn.SiLU(),nn.Linear(2048,2048),nn.SiLU(),
                               nn.Linear(2048,2048),nn.SiLU(),nn.Linear(2048,state_dim))
        # 论文式 d(s_{t+1},g)：本单任务目标固定，终止头从预测下一状态与 context 输出 logit。
        self.termination=nn.Sequential(nn.Linear(state_dim+CONTEXT_DIM,256),nn.SiLU(),nn.Linear(256,1))
        with torch.no_grad(): self.termination[-1].bias[-1]=-4.0
        self.register_buffer('input_mean',torch.zeros(dim))
        self.register_buffer('input_std',torch.ones(dim))
        self.register_buffer('delta_mean',torch.zeros(state_dim))
        self.register_buffer('delta_std',torch.ones(state_dim))

    @staticmethod
    def pack(state,action,context):
        return torch.cat((state,action,context),-1)

    @torch.no_grad()
    def calibrate(self,state,action,context,next_state):
        x=self.pack(state,action,context);delta=next_state-state
        self.input_mean.copy_(x.mean(0));self.input_std.copy_(x.std(0,unbiased=False).clamp_min(1e-3))
        self.delta_mean.copy_(delta.mean(0));self.delta_std.copy_(delta.std(0,unbiased=False).clamp_min(1e-3))
        # 稀少的脚部接触和速度不能因初期数据恒定而放大 1000 倍。
        self.input_std[55:80].clamp_(min=.05);self.delta_std[55:80].clamp_(min=.05)

    def forward(self,state,action,context):
        inputs=(self.pack(state,action,context)-self.input_mean)/self.input_std
        z=self.net(inputs.clamp(-10,10))
        nxt=state+z*self.delta_std+self.delta_mean
        # 实验不使用随机外力；历史接口中的外力槽不由网络预测。
        nxt=torch.cat((nxt[...,:80],torch.zeros_like(nxt[...,80:88]),nxt[...,88:]),-1)
        nxt=enforce_executed_action(nxt,action)
        ctx_start=self.state_dim+self.action_dim
        xn=(nxt-self.input_mean[:self.state_dim])/self.input_std[:self.state_dim]
        cn=(context-self.input_mean[ctx_start:])/self.input_std[ctx_start:]
        logit=self.termination(torch.cat((xn,cn),-1).clamp(-10,10)).squeeze(-1)
        return nxt,logit


class PaperCritic(nn.Module):
    def __init__(self,state_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(state_dim+CONTEXT_DIM+1,256),nn.SiLU(),
                               nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1))

    def forward(self,state,context,remaining):
        return self.net(torch.cat((state,context,remaining.unsqueeze(-1)),-1)).squeeze(-1)
