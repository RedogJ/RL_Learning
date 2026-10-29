"""以初态、动作序列和未来时间查询状态的 DeepONet。"""
import torch
from torch import nn

from go2_learning.tasks.upright_supermimic.paper_context import CONTEXT_DIM
from go2_learning.tasks.upright_supermimic.paper_model import enforce_executed_action


class TimeDeepONet(nn.Module):
    """动作序列仅输入到查询时刻；连续时间插值并未经过验证。"""

    def __init__(self, state_dim=90, action_dim=12, horizon=8, rank=16, width=1536):
        super().__init__()
        self.state_dim, self.action_dim, self.horizon, self.rank = state_dim, action_dim, horizon, rank
        dim = state_dim + action_dim * horizon + CONTEXT_DIM
        self.branch = nn.Sequential(nn.Linear(dim, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU(), nn.Linear(width, state_dim * rank))
        self.trunk = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, rank))
        self.bias = nn.Parameter(torch.zeros(state_dim))
        self.termination = nn.Sequential(nn.Linear(state_dim + CONTEXT_DIM, 256), nn.SiLU(), nn.Linear(256, 1))
        with torch.no_grad():
            self.termination[-1].bias[-1] = -4.0
        self.register_buffer('input_mean', torch.zeros(state_dim + action_dim + CONTEXT_DIM))
        self.register_buffer('input_std', torch.ones(state_dim + action_dim + CONTEXT_DIM))
        self.register_buffer('delta_mean', torch.zeros(state_dim))
        self.register_buffer('delta_std', torch.ones(state_dim))

    def forward(self, state, actions, context, step):
        """查询 step（1..horizon）步后的状态。actions 包含固定长度的真实动作序列。"""
        if not 1 <= step <= self.horizon:
            raise ValueError('step 超出训练窗口')
        d, a = self.state_dim, self.action_dim
        sn = (state - self.input_mean[:d]) / self.input_std[:d]
        an = (actions - self.input_mean[d:d+a]) / self.input_std[d:d+a]
        cn = (context - self.input_mean[d+a:]) / self.input_std[d+a:]
        # 未来动作不允许泄漏：在归一化后将尚不可用的位置清零。
        mask = (torch.arange(self.horizon, device=state.device) < step).view(1, -1, 1)
        an = torch.where(mask, an.clamp(-10, 10), torch.zeros_like(an))
        branch_in = torch.cat((sn.clamp(-10, 10), an.flatten(1), cn.clamp(-10, 10)), dim=-1)
        coeff = self.branch(branch_in).reshape(-1, d, self.rank)
        time = state.new_full((1, 1), step / self.horizon)
        basis = self.trunk(time).reshape(1, 1, self.rank)
        z = (coeff * basis).sum(-1) + self.bias
        nxt = state + step * (z * self.delta_std + self.delta_mean)
        nxt = torch.cat((nxt[:, :80], torch.zeros_like(nxt[:, 80:88]), nxt[:, 88:]), -1)
        nxt = enforce_executed_action(nxt, actions[:, step - 1])
        xn = (nxt - self.input_mean[:d]) / self.input_std[:d]
        logit = self.termination(torch.cat((xn, cn), -1).clamp(-10, 10)).squeeze(-1)
        return nxt, logit
