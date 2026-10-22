"""Go2 状态的强调投影；不投影 latent，不改变磁盘上的原始轨迹格式。"""
import torch
from torch import nn


class EmphasisProjection(nn.Module):
    def __init__(self, state_dim, emphasis, seed):
        super().__init__()
        if state_dim < 34:
            raise ValueError('Go2 强调投影需要包含机身速度、重力方向和高度的状态')
        generator=torch.Generator().manual_seed(seed)
        # 本项目根特征：v(3)、w(3)、gravity(3)、height(1)。
        # 当前表示没有论文的根部相对 xy/完整旋转，保留已采集数据的坐标约定。
        weights=torch.ones(state_dim,dtype=torch.float64)
        weights[:9]=emphasis;weights[33]=emphasis
        a=torch.randn(state_dim,state_dim,generator=generator,dtype=torch.float64)
        matrix=torch.cat((a*weights[None],torch.eye(state_dim,dtype=torch.float64)),dim=0)
        # 整体缩放保持单位方差输入的平均投影方差为 1；不逐通道再归一化，保留相对强调。
        # A 的行数取 state_dim、整体缩放及根特征集合均为本 Go2 原型的适配。
        matrix=matrix/matrix.square().sum(1).mean().sqrt()
        self.register_buffer('matrix',matrix.float())
        self.register_buffer('inverse',torch.linalg.pinv(matrix).float())

    def forward(self, state):
        return state @ self.matrix.T

    def decode(self, state):
        return state @ self.inverse.T
