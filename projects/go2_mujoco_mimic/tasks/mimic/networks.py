"""SuperMimic 风格的参考动作编码器和策略解码器。"""
import torch
from torch import nn


def mlp(input_dim, hidden, output_dim):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class MotionPolicy(nn.Module):
    """10 帧目标 -> 32 维 latent；latent 与 10 帧自身观测 -> 12 维动作。"""
    def __init__(self, history_length=10, future_length=10, proprio_dim=42,
                 goal_dim=19, latent_dim=32, hidden=(2048, 1024, 512, 256), heading_dim=0):
        super().__init__()
        self.heading_dim = heading_dim
        self.history_length = history_length
        self.future_length = future_length
        self.proprio_dim = proprio_dim
        self.goal_dim = goal_dim
        self.encoder = mlp(future_length * goal_dim, hidden, latent_dim)
        self.decoder = mlp(history_length * proprio_dim + latent_dim + heading_dim, hidden, 12)

    def forward(self, observation):
        expected = self.history_length*self.proprio_dim + self.future_length*self.goal_dim + self.heading_dim
        if observation.shape[-1] != expected:
            raise ValueError(f'观测末维应为 {expected}，实际为 {observation.shape[-1]}')
        history = observation[..., :self.history_length*self.proprio_dim]
        goal_end = self.history_length*self.proprio_dim + self.future_length*self.goal_dim
        goals = observation[..., self.history_length*self.proprio_dim:goal_end]
        heading = observation[..., goal_end:]
        z = torch.tanh(self.encoder(goals))
        return torch.tanh(self.decoder(torch.cat((history, z, heading), dim=-1)))
