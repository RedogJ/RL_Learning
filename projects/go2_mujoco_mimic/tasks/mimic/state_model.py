"""Go2 连续状态动力学：状态维度与字段定义见 StateLayout。"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from .networks import mlp
from .rewards import gravity_local


@dataclass(frozen=True)
class StateLayout:
    # root pos 3 + quat 4 + joint pos 12 + root velocity 6 + joint velocity 12
    core: int = 37
    @property
    def size(self): return self.core


LAYOUT = StateLayout()


def build_state(env):
    return torch.cat((env.qpos[:, :7], env.qpos[:, env.qids],
                      env.qvel[:, :6], env.qvel[:, env.vids]), -1)


def obs_from_state(state, previous_action, default_q, noise=False):
    quat = F.normalize(state[:, 3:7], dim=-1, eps=1e-8)
    values = [state[:, 7:19]-default_q, state[:, 25:37], state[:, 22:25], gravity_local(quat), previous_action]
    if noise:
        values = [x + (torch.rand_like(x)*2-1)*n for x,n in zip(values,(.01,.5,.2,.05,0.0))]
    return torch.cat(values, -1)


def normalize_state(state):
    return torch.cat((state[..., :3], F.normalize(state[..., 3:7], dim=-1, eps=1e-8),
                      state[..., 7:]), -1)


class StateNormalizer(nn.Module):
    """首批真实状态校准后冻结；尺度下限防止静止维度放大噪声。"""
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(LAYOUT.size))
        self.register_buffer('scale', torch.ones(LAYOUT.size))
        self.register_buffer('ready', torch.tensor(False))

    @torch.no_grad()
    def calibrate(self, states):
        flat = states.reshape(-1, LAYOUT.size).float()
        floor = flat.new_tensor([.1]*3 + [.1]*4 + [.1]*12 + [.2]*3 + [.5]*3 + [1.]*12)
        self.mean.copy_(flat.mean(0))
        self.scale.copy_(flat.std(0, unbiased=False).clamp_min(floor))
        self.ready.fill_(True)

    def forward(self, state):
        return (state-self.mean)/self.scale


class Dynamics(nn.Module):
    """论文宽度 2048×3，预测连续状态增量；context 为明确的物理参数。"""
    def __init__(self, context_dim=16, width=2048):
        super().__init__()
        self.normalizer = StateNormalizer()
        self.net = mlp(LAYOUT.size+12+context_dim, (width, width, width), LAYOUT.size)
        # 让初始模型近似恒等映射，防止随机初始化破坏物理状态。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, state, action, context):
        delta = self.net(torch.cat((self.normalizer(state), action, context), -1))
        return normalize_state(state + delta.float()*self.normalizer.scale)


class StateValue(nn.Module):
    def __init__(self, context_dim=16):
        super().__init__()
        self.normalizer = StateNormalizer()
        self.net = mlp(LAYOUT.size+190+context_dim, (256,128), 1)
    def forward(self, state, goal, context):
        return self.net(torch.cat((self.normalizer(state), goal, context), -1)).squeeze(-1)
