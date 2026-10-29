"""过去 10 帧状态作为输入的 MLP 和时间条件 DeepONet。"""
import torch
from torch import nn

from go2_learning.tasks.upright_supermimic.paper_context import CONTEXT_DIM
from go2_learning.tasks.upright_supermimic.paper_model import enforce_executed_action

HISTORY = 10
HORIZON = 8


class HistoryBase(nn.Module):
    def __init__(self, state_dim=90, action_dim=12):
        super().__init__()
        self.state_dim, self.action_dim = state_dim, action_dim
        dim = state_dim + action_dim + CONTEXT_DIM
        self.register_buffer('input_mean', torch.zeros(dim))
        self.register_buffer('input_std', torch.ones(dim))
        self.register_buffer('delta_mean', torch.zeros(state_dim))
        self.register_buffer('delta_std', torch.ones(state_dim))
        self.termination = nn.Sequential(nn.Linear(state_dim + CONTEXT_DIM, 256), nn.SiLU(), nn.Linear(256, 1))
        with torch.no_grad():
            self.termination[-1].bias[-1] = -4.0

    def norm_history(self, history):
        if history.ndim != 3 or history.shape[1:] != (HISTORY, self.state_dim):
            raise ValueError('需要按时间从旧到新排列的 10 帧状态')
        return ((history - self.input_mean[:self.state_dim]) / self.input_std[:self.state_dim]).clamp(-10, 10).flatten(1)

    def norm_context(self, context):
        d, a = self.state_dim, self.action_dim
        return ((context - self.input_mean[d+a:]) / self.input_std[d+a:]).clamp(-10, 10)

    def finish(self, current, z, action, context, step=1):
        nxt = current + step * (z * self.delta_std + self.delta_mean)
        nxt = torch.cat((nxt[:, :80], torch.zeros_like(nxt[:, 80:88]), nxt[:, 88:]), -1)
        nxt = enforce_executed_action(nxt, action)
        xn = (nxt - self.input_mean[:self.state_dim]) / self.input_std[:self.state_dim]
        logit = self.termination(torch.cat((xn, self.norm_context(context)), -1).clamp(-10, 10)).squeeze(-1)
        return nxt, logit


class HistoryMLP(HistoryBase):
    def __init__(self, state_dim=90, action_dim=12):
        super().__init__(state_dim, action_dim)
        dim = HISTORY * state_dim + action_dim + CONTEXT_DIM
        self.net = nn.Sequential(nn.Linear(dim, 2048), nn.SiLU(), nn.Linear(2048, 2048), nn.SiLU(),
                                 nn.Linear(2048, 2048), nn.SiLU(), nn.Linear(2048, state_dim))

    def forward(self, history, action, context):
        d, a = self.state_dim, self.action_dim
        an = ((action - self.input_mean[d:d+a]) / self.input_std[d:d+a]).clamp(-10, 10)
        z = self.net(torch.cat((self.norm_history(history), an, self.norm_context(context)), -1))
        return self.finish(history[:, -1], z, action, context)


class HistoryDeepONet(HistoryBase):
    def __init__(self, state_dim=90, action_dim=12, rank=16, width=1536):
        super().__init__(state_dim, action_dim)
        self.rank = rank
        dim = HISTORY * state_dim + HORIZON * action_dim + CONTEXT_DIM
        self.branch = nn.Sequential(nn.Linear(dim, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU(), nn.Linear(width, state_dim * rank))
        self.trunk = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, rank))
        self.bias = nn.Parameter(torch.zeros(state_dim))

    def forward(self, history, actions, context, step):
        if not 1 <= step <= HORIZON or actions.shape[1:] != (HORIZON, self.action_dim):
            raise ValueError('需要 8 个动作及 1..8 的预测步数')
        d, a = self.state_dim, self.action_dim
        an = (actions - self.input_mean[d:d+a]) / self.input_std[d:d+a]
        mask = (torch.arange(HORIZON, device=history.device) < step).view(1, -1, 1)
        an = torch.where(mask, an.clamp(-10, 10), torch.zeros_like(an))
        features = torch.cat((self.norm_history(history), an.flatten(1), self.norm_context(context)), -1)
        coeff = self.branch(features).reshape(-1, d, self.rank)
        basis = self.trunk(history.new_full((1, 1), step / HORIZON)).reshape(1, 1, self.rank)
        z = (coeff * basis).sum(-1) + self.bias
        return self.finish(history[:, -1], z, actions[:, step - 1], context, step)
