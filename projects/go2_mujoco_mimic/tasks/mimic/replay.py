"""真实仿真轨迹的 GPU 环形缓存，供当前和历史训练入口共用。"""
import torch


class Replay:
    """预分配 GPU 环形缓存，窗口索引按真实时间顺序处理环绕。"""
    def __init__(self, capacity, n, device):
        self.n, self.capacity, self.device = n, capacity, device
        self.count = 0
        self.storage = None

    def add(self, row):
        if self.storage is None:
            self.storage = [torch.empty((self.capacity, *x.shape), device=self.device, dtype=x.dtype) for x in row]
        slot = self.count % self.capacity
        for dst, src in zip(self.storage, row):
            dst[slot].copy_(src.detach())
        self.count += 1

    def sample(self, batch, length):
        size = min(self.count, self.capacity)
        if size < length:
            raise ValueError('连续轨迹不足以采样')
        start = torch.randint(self.count-size, self.count-length+1, (batch,), device=self.device)
        envs = torch.randint(self.n, (batch,), device=self.device)
        slots = (start[:, None] + torch.arange(length, device=self.device)[None, :]) % self.capacity
        ids = envs[:, None].expand_as(slots)
        return tuple(col[slots, ids] for col in self.storage)
