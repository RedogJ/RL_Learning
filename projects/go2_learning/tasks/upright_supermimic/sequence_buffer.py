"""时间序列环形缓冲；窗口不能跨等待或回合终止。"""
import torch

class SequenceBuffer:
    def __init__(self, capacity, num_envs, obs_dim, action_dim, device):
        self.capacity, self.num_envs, self.device = capacity, num_envs, device
        shape = (capacity, num_envs)
        self.obs = torch.zeros(*shape, obs_dim, device=device)
        self.action = torch.zeros(*shape, action_dim, device=device)
        self.next_obs = torch.zeros_like(self.obs)
        self.reward = torch.zeros(*shape, device=device)
        self.fail = torch.zeros(*shape, dtype=torch.bool, device=device)
        self.done = torch.zeros_like(self.fail)
        self.active = torch.zeros_like(self.fail)
        self.remaining = torch.zeros(*shape, device=device)
        self.next_remaining = torch.zeros(*shape, device=device)
        self.step_id = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.count = 0

    def add(self, obs, action, next_obs, reward, fail, done, active, remaining, next_remaining):
        i = self.count % self.capacity
        for key, value in locals().copy().items():
            if key in ('obs','action','next_obs','reward','fail','done','active','remaining','next_remaining'):
                getattr(self, key)[i].copy_(value)
        self.step_id[i] = self.count
        self.count += 1

    def windows(self, n, horizon, env_ids):
        end = self.count - horizon
        start = max(0, self.count - self.capacity)
        if end < start:
            raise ValueError('缓冲区没有足够的连续时间步')
        # 随机候选中剔除跨回合/等待以及未填满的窗口。
        starts = torch.arange(start, end+1, device=self.device)
        offsets = torch.arange(horizon, device=self.device)
        idx = (starts[:,None] + offsets[None,:]) % self.capacity
        chronological = (self.step_id[idx] == starts[:,None] + offsets).all(1)
        flags = (self.active[idx][:,:,env_ids] & ~self.done[idx][:,:,env_ids]).all(1)
        candidates = (flags & chronological[:,None]).nonzero()
        if len(candidates) == 0:
            raise ValueError('没有跨回合前的有效连续窗口')
        chosen = candidates[torch.randint(len(candidates), (n,), device=self.device)]
        slots = idx[chosen[:,0]]; env = env_ids[chosen[:,1]][:,None].expand_as(slots)
        return self.obs[slots,env], self.action[slots,env], self.next_obs[slots,env]

    def transitions(self, n, env_ids):
        length = min(self.count,self.capacity)
        idx = torch.arange(length,device=self.device)
        valid = self.active[idx][:,env_ids].nonzero()
        if len(valid) == 0: raise ValueError('缓冲区没有有效转移')
        chosen = valid[torch.randint(len(valid),(n,),device=self.device)]
        t = idx[chosen[:,0]]; e = env_ids[chosen[:,1]]
        return tuple(getattr(self,k)[t,e] for k in ('obs','action','next_obs','reward','fail','done','remaining','next_remaining'))
