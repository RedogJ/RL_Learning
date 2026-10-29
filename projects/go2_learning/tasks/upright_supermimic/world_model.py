"""简化世界模型：根据当前观测和原始策略动作预测下一观测的变化。"""
import torch
from torch import nn


class WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + action_dim, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, obs_dim))
        self.register_buffer('input_mean', torch.zeros(obs_dim + action_dim))
        self.register_buffer('input_std', torch.ones(obs_dim + action_dim))
        self.register_buffer('delta_mean', torch.zeros(obs_dim))
        self.register_buffer('delta_std', torch.ones(obs_dim))

    @torch.no_grad()
    def calibrate(self, obs, action, next_obs):
        x = torch.cat((obs, action), dim=-1)
        delta = next_obs - obs
        self.input_mean.copy_(x.mean(0))
        self.input_std.copy_(x.std(0, unbiased=False).clamp_min(1e-3))
        self.delta_mean.copy_(delta.mean(0))
        self.delta_std.copy_(delta.std(0, unbiased=False).clamp_min(1e-3))

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=-1)
        delta = self.net((x - self.input_mean) / self.input_std) * self.delta_std + self.delta_mean
        return obs + delta

    def loss(self, obs, action, next_obs):
        return (((self(obs, action) - next_obs) / self.delta_std).square()).mean()
