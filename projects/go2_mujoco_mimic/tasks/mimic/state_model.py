"""Go2 连续状态动力学：状态维度与字段定义见 StateLayout。"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from .networks import mlp
from .rewards import gravity_local
from .kinematics import rotate


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
    def __init__(self, context_dim=16, width=2048, kinematics=None):
        super().__init__()
        self.normalizer = StateNormalizer()
        self.kinematics = kinematics
        self.net = mlp(LAYOUT.size+12+context_dim+(8 if kinematics is not None else 0), (width, width, width), LAYOUT.size)
        # 让初始模型近似恒等映射，防止随机初始化破坏物理状态。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def foot_features(self, state):
        """球面距平地高度与竖直速度；从状态计算且保留梯度。"""
        pos, quat = self.kinematics(state)
        calfs = [3, 7, 11, 15]
        offset = state.new_tensor([-.002, 0., -.213])
        feet = pos[:, calfs] + rotate(quat[:, calfs], offset)
        h = feet[..., 2] - .022
        # MuJoCo free joint 的角速度在机身坐标系，线速度在世界坐标系。
        omega = rotate(F.normalize(state[:, 3:7], dim=-1, eps=1e-8), state[:, 22:25])
        velocity = state[:, None, 19:22] + torch.cross(
            omega[:, None, :].expand_as(feet), feet - state[:, None, :3], dim=-1)
        bodies = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
        qjoint = quat[:, bodies].reshape(-1, 4, 3, 4)
        pjoint = pos[:, bodies].reshape(-1, 4, 3, 3)
        axes = self.kinematics.axis[bodies].reshape(1, 4, 3, 3).expand_as(pjoint)
        anchors = self.kinematics.anchor[bodies].reshape(1, 4, 3, 3).expand_as(pjoint)
        axis_world = rotate(qjoint, axes)
        anchor_world = pjoint + rotate(qjoint, anchors)
        joint_velocity = state[:, 25:37].reshape(-1, 4, 3, 1)
        velocity = velocity + (torch.cross(axis_world, feet[:, :, None, :] - anchor_world, dim=-1)
                               * joint_velocity).sum(dim=2)
        return torch.cat((h / .5, velocity[..., 2] / 5.0), -1)

    def forward(self, state, action, context):
        features = [self.normalizer(state), action, context]
        if self.kinematics is not None:
            features.append(self.foot_features(state))
        delta = self.net(torch.cat(features, -1))
        return normalize_state(state + delta.float()*self.normalizer.scale)


class StateValue(nn.Module):
    """有限回合价值；history_length=0 仅供读取旧版 Critic/历史诊断。"""
    def __init__(self, context_dim=16, history_length=0, proprio_dim=42, goal_dim=190):
        super().__init__()
        self.config = dict(context_dim=context_dim, history_length=history_length,
                           proprio_dim=proprio_dim, goal_dim=goal_dim)
        self.history_length = history_length
        self.proprio_dim = proprio_dim
        self.base_input_dim = LAYOUT.size + goal_dim + context_dim
        self.input_dim = self.base_input_dim + (history_length * proprio_dim + 1 if history_length else 0)
        self.normalizer = StateNormalizer()
        self.net = mlp(self.input_dim, (256,128), 1)

    def forward(self, state, goal, context, history=None, remaining_time=None):
        parts = [self.normalizer(state), goal, context]
        if self.history_length:
            if history is None or remaining_time is None:
                raise ValueError('新版 Critic 必须提供策略历史和归一化剩余时间')
            if history.shape != (*state.shape[:-1], self.history_length, self.proprio_dim):
                raise ValueError('Critic 历史形状与配置不符')
            if remaining_time.shape != (*state.shape[:-1], 1):
                raise ValueError('Critic 剩余时间须为每个样本一维')
            # Actor 尾项需要经过历史回传梯度；仅在 Critic 回归训练处 detach。
            parts.extend((history.flatten(-2), remaining_time))
        return self.net(torch.cat(parts, -1)).squeeze(-1)

    def load_for_resume(self, weights):
        """旧输入列完整保留，新历史/时间列补零；返回是否需要重置优化器并预热。"""
        old = weights['net.0.weight']
        target = self.net[0].weight
        migrated = old.shape != target.shape
        if migrated:
            if not self.history_length or old.shape != (target.shape[0], self.base_input_dim):
                raise ValueError(f'不支持 Critic 输入迁移 {tuple(old.shape)} → {tuple(target.shape)}')
            weights = weights.copy()
            expanded = old.new_zeros(target.shape)
            expanded[:, :self.base_input_dim] = old
            weights['net.0.weight'] = expanded
        self.load_state_dict(weights)
        return migrated


def critic_remaining_time(env, frame, frame_limit):
    """参考终点倒计时，以完整参考回合长度归一化；RSI 不改变分母。"""
    full_span = max(1, (len(env.ref.frames)-1) * env.cfg.task.reference_cycles)
    return ((frame_limit-frame).float().clamp_min(0) / full_span).unsqueeze(-1)


class TerminationHead(nn.Module):
    """d_omega(s_hat, g)：输出 logits，用真实失败标签做 BCE；输入状态由 WM normalizer 归一化。"""
    def __init__(self, goal_dim=190):
        super().__init__()
        self.net = mlp(LAYOUT.size + goal_dim, (256, 128), 1)
        # 新增头预热期间以低失败率起步，避免随机输出立即截断全部想象。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -4.6)

    def forward(self, normalized_state, goal):
        return self.net(torch.cat((normalized_state, goal), -1)).squeeze(-1)
