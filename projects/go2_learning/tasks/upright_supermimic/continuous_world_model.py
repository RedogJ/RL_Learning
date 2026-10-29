"""独立实验：连续状态、LayerNorm 和在线统计的世界模型组件。

尚未接入在线训练；调用方须提供没有接触布尔值的连续状态和配套的真实样本。
"""
import math

import torch
from torch import nn


def disable_contact_rewards(cfg):
    """只作用于传入的实验奖励配置；不改变真实接触终止条件。"""
    names = ("rear_support_weight", "rear_feet_airborne_weight", "front_lift_weight",
             "rear_knee_clearance_weight", "foot_slide_weight", "undesired_contact_weight",
             "rear_knee_contact_weight", "hip_contact_weight")
    for name in names:
        if hasattr(cfg, name):
            setattr(cfg, name, 0.0)
    return cfg

class RunningNorm(nn.Module):
    """按批合并总体均值与方差；达到样本数上限后冻结。"""

    def __init__(self, dim, eps=1e-2, max_count=200_000_000):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.eps = eps
        self.max_count = max_count

    @torch.no_grad()
    def update(self, x):
        x = x.detach().reshape(-1, self.mean.numel()).float()
        if not len(x) or self.count.item() >= self.max_count:
            return
        n = x.shape[0]
        mean = x.mean(0)
        var = x.var(0, unbiased=False)
        total = self.count + n
        delta = mean - self.mean
        self.mean.add_(delta * (n / total))
        self.var.copy_((self.var * self.count + var * n + delta.square() * self.count * n / total) / total)
        self.count.copy_(total)

    def std(self):
        return (self.var + self.eps**2).sqrt()

    def forward(self, x):
        return (x - self.mean) / self.std()


class ContinuousWorldModel(nn.Module):
    """保持 context 维度；预测连续状态的增量，不包含二分类终止头。"""

    def __init__(self, state_dim, action_dim, hidden=2048, context_dim=16):
        super().__init__()
        self.state_dim = state_dim
        self.amp_dtype = None  # 仅世界模型 MLP 运算可选混合精度；状态和归一化始终 FP32。
        self.in_norm = RunningNorm(state_dim + action_dim + context_dim)
        self.out_norm = RunningNorm(state_dim, eps=1e-4)
        layers = []
        size = state_dim + action_dim + context_dim
        for _ in range(3):
            layers.extend((nn.Linear(size, hidden), nn.LayerNorm(hidden), nn.SiLU()))
            size = hidden
        layers.append(nn.Linear(size, state_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @torch.no_grad()
    def update_statistics(self, state, action, context, next_state):
        self.in_norm.update(torch.cat((state, action, context), -1))
        self.out_norm.update(next_state - state)

    def forward(self, state, action, context):
        x = torch.cat((state, action, context), -1)
        x = self.in_norm(x)
        if self.amp_dtype is None:
            z = self.net(x)
        else:
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                z = self.net(x)
            z = z.float()
        delta = z * self.out_norm.std() + self.out_norm.mean
        return state + delta


class ContinuousDeepONet(ContinuousWorldModel):
    """单步自回归 DeepONet；每次以固定时间 t+1 查询下一状态。"""

    def __init__(self, state_dim, action_dim, hidden=1536, context_dim=16, rank=16):
        nn.Module.__init__(self)
        self.state_dim, self.rank = state_dim, rank
        self.in_norm = RunningNorm(state_dim + action_dim + context_dim)
        self.out_norm = RunningNorm(state_dim, eps=1e-4)
        layers = []
        size = state_dim + action_dim + context_dim
        for _ in range(3):
            layers.extend((nn.Linear(size, hidden), nn.LayerNorm(hidden), nn.SiLU()))
            size = hidden
        layers.append(nn.Linear(size, state_dim * rank))
        self.branch = nn.Sequential(*layers)
        self.trunk = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 128),
                                   nn.SiLU(), nn.Linear(128, rank))
        self.bias = nn.Parameter(torch.zeros(state_dim))
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self, state, action, context):
        x = self.in_norm(torch.cat((state, action, context), -1))
        coeff = self.branch(x).reshape(*state.shape[:-1], self.state_dim, self.rank)
        basis = self.trunk(state.new_ones((1, 1)))
        delta = ((coeff * basis).sum(-1) + self.bias) * self.out_norm.std() + self.out_norm.mean
        return state + delta


class DirectTimeDeepONet(ContinuousWorldModel):
    """从初态和动作序列直接查询第 1..horizon 步。"""

    def __init__(self, state_dim=55, action_dim=12, horizon=16, rank=16, width=1536, context_dim=16):
        nn.Module.__init__(self)
        self.state_dim, self.action_dim, self.horizon, self.rank = state_dim, action_dim, horizon, rank
        self.in_norm = RunningNorm(state_dim + action_dim + context_dim)
        self.out_norm = RunningNorm(state_dim, eps=1e-4)
        size = state_dim + action_dim * horizon + context_dim
        layers = []
        for _ in range(3):
            layers.extend((nn.Linear(size, width), nn.LayerNorm(width), nn.SiLU()))
            size = width
        layers.append(nn.Linear(size, state_dim * rank))
        self.branch = nn.Sequential(*layers)
        self.trunk = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 128),
                                   nn.SiLU(), nn.Linear(128, rank))
        self.bias = nn.Parameter(torch.zeros(state_dim))
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self, state, actions, context, step):
        if not 1 <= step <= self.horizon or actions.shape[-2:] != (self.horizon, self.action_dim):
            raise ValueError('查询步数或动作序列形状不正确')
        d, a = self.state_dim, self.action_dim
        mean, std = self.in_norm.mean, self.in_norm.std()
        sn = (state - mean[:d]) / std[:d]
        an = (actions - mean[d:d+a]) / std[d:d+a]
        cn = (context - mean[d+a:]) / std[d+a:]
        mask = torch.arange(self.horizon, device=state.device) < step
        an = torch.where(mask.view(1, -1, 1), an, torch.zeros_like(an))
        coeff = self.branch(torch.cat((sn, an.flatten(1), cn), -1)).reshape(-1, d, self.rank)
        basis = self.trunk(state.new_full((1, 1), step / self.horizon))
        delta = ((coeff * basis).sum(-1) + self.bias) * self.out_norm.std() + self.out_norm.mean
        result = state + step * delta
        return torch.cat((result[:, :33], actions[:, step-1], result[:, 45:]), -1)


def soft_upright_tilt_failure(gravity_body, max_tilt_degrees, softness=0.02):
    """连续近似机身前向轴倾斜终止；不包含接触终止。"""
    if softness <= 0:
        raise ValueError("softness 必须为正")
    alignment = -gravity_body[..., 0]
    limit = math.cos(math.radians(max_tilt_degrees))
    return torch.sigmoid((limit - alignment) / softness)
