import torch.nn as nn

import math

import torch
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, observation_dim=6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(observation_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.log_std = nn.Parameter(torch.full((1,), math.log(0.5)))

    def forward(self, observation):
        return self.net(observation)
    
    def get_distribution(self, observation):
        mean = self(observation)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)


class ValueNetwork(nn.Module):
    def __init__(self, observation_dim=6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(observation_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, observation):
        return self.net(observation).squeeze(-1)
