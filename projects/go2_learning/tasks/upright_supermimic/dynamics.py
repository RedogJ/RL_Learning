"""世界模型：预测下一观测、非终止奖励和失败概率。"""
import torch
from torch import nn

class Dynamics(nn.Module):
    def __init__(self, obs_dim, action_dim, width=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = nn.Sequential(nn.Linear(obs_dim + action_dim, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, obs_dim + 2))
        # 失败是稀疏事件；初始化为较小概率，仍由数据学习校准。
        with torch.no_grad():
            self.net[-1].bias[-1] = -4.0
        self.register_buffer('input_mean', torch.zeros(obs_dim + action_dim))
        self.register_buffer('input_std', torch.ones(obs_dim + action_dim))
        self.register_buffer('delta_mean', torch.zeros(obs_dim))
        self.register_buffer('delta_std', torch.ones(obs_dim))

    @torch.no_grad()
    def calibrate(self, obs, action, next_obs):
        x = torch.cat((obs, action), -1); d = next_obs - obs
        self.input_mean.copy_(x.mean(0)); self.input_std.copy_(x.std(0, unbiased=False).clamp_min(1e-3))
        self.delta_mean.copy_(d.mean(0)); self.delta_std.copy_(d.std(0, unbiased=False).clamp_min(1e-3))
        # 启动采样可能还没有推力或接触切换；按物理单位给稀有状态设下限，防止新事件被除以 0.001。
        if self.obs_dim >= 88:
            self.input_std[55:80].clamp_(min=.05)
            self.delta_std[55:80].clamp_(min=.05)
            self.input_std[80:88].copy_(torch.maximum(self.input_std[80:88], x.new_tensor([.5,.1,10.,10.,10.,.05,.05,.05])))
            self.delta_std[80:88].copy_(torch.maximum(self.delta_std[80:88], x.new_tensor([.03,.03,5.,5.,5.,.01,.01,.01])))

    def forward(self, obs, action):
        z = self.net(((torch.cat((obs, action), -1)-self.input_mean)/self.input_std).clamp(-10.,10.))
        next_obs = obs + z[..., :self.obs_dim] * self.delta_std + self.delta_mean
        return next_obs, z[..., self.obs_dim], z[..., self.obs_dim + 1]


class Critic(nn.Module):
    def __init__(self, obs_dim, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim+1, width), nn.SiLU(), nn.Linear(width,width), nn.SiLU(), nn.Linear(width,1))

    def forward(self, obs, remaining):
        return self.net(torch.cat((obs, remaining.unsqueeze(-1)), -1)).squeeze(-1)
