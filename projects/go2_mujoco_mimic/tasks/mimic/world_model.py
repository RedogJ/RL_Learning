"""基于观测历史的可微世界模型；首版不含特权物理参数。"""
import torch
from torch import nn
from .networks import mlp


class WorldModel(nn.Module):
    def __init__(self, obs_dim=610, action_dim=12, hidden=(512, 512, 512)):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, hidden, 43)

    def forward(self, obs, action):
        result = self.net(torch.cat((obs, action), -1))
        # 增量预测可测的前 30 维；上次动作为已执行动作，目标序列外部推进。
        predicted = obs[:, 378:408] + result[:, :30]
        return torch.cat((predicted, action), -1), result[:, 30:42], result[:, 42]


class Value(nn.Module):
    def __init__(self, obs_dim=610):
        super().__init__()
        self.net = mlp(obs_dim, (256, 128), 1)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)
