"""对固定 replay 缓存有效连续窗口，避免跨回合与跨物理 context。"""
import torch


class WindowSampler:
    def __init__(self, buf, horizon, env_ids, *, allow_last_done=True):
        self.buf, self.horizon = buf, horizon
        self.env_ids = env_ids.to(device=buf.device, dtype=torch.long)
        start = max(0, buf.count - buf.capacity)
        end = buf.count - horizon
        if horizon < 1 or end < start:
            raise ValueError('replay 不足以形成窗口')
        self.starts = torch.arange(start, end + 1, device=buf.device)
        flags = torch.ones((len(self.starts), len(self.env_ids)), dtype=torch.bool, device=buf.device)
        ref = buf.context[self.starts % buf.capacity][:, self.env_ids]
        for k in range(horizon):
            slots = (self.starts + k) % buf.capacity
            flags &= (buf.step_id[slots] == self.starts + k)[:, None]
            flags &= buf.active[slots][:, self.env_ids]
            flags &= (buf.context[slots][:, self.env_ids] == ref).all(-1)
            if k < horizon - 1 or not allow_last_done:
                flags &= ~buf.done[slots][:, self.env_ids]
        self.valid = flags.nonzero()
        if not len(self.valid):
            raise ValueError('没有不跨回合和物理参数的有效窗口')

    def sample(self, count):
        if count < 1:
            raise ValueError('采样数量必须为正')
        b = self.buf
        picks = self.valid[torch.randint(len(self.valid), (count,), device=b.device)]
        t = (self.starts[picks[:, 0], None] + torch.arange(self.horizon, device=b.device)) % b.capacity
        e = self.env_ids[picks[:, 1], None].expand_as(t)
        return tuple(getattr(b, name)[t, e] for name in ('state', 'action', 'context', 'next_state', 'fail', 'done', 'remaining'))
