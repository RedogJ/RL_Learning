"""逐回合的速度切换；切换指令不会重置物理状态或切断采集轨迹。"""
import torch


class TransitionCommands:
    def __init__(self, num_envs, device, dt, seed):
        self.dt=dt
        self.generator=torch.Generator(device=device).manual_seed(seed)
        self.durations=torch.tensor([1.5,2.5,2.,2.5,2.,2.,2.,2.],device=device)
        self.pattern=torch.tensor([0.,1.5,0.,-1.5,0.,1.5,-1.5,0.],device=device)
        self.ends=torch.empty(num_envs,len(self.pattern),device=device)
        self.sign=torch.empty(num_envs,device=device)
        self.reset(torch.arange(num_envs,device=device))

    def reset(self, ids):
        jitter=.8+.4*torch.rand(len(ids),len(self.pattern),device=self.ends.device,generator=self.generator)
        self.ends[ids]=(self.durations*jitter).cumsum(-1)
        self.sign[ids]=torch.where(torch.rand(len(ids),device=self.ends.device,generator=self.generator)<.5,-1.,1.)

    def apply(self, env):
        # episode_length_buf 只累计正式控制时间，逐回合 .3s 等待不推进指令阶段。
        age=env.episode_length_buf*self.dt
        phase=(age[:,None]>=self.ends).sum(-1).clamp_max(len(self.pattern)-1)
        env.commands.values.zero_()
        env.commands.values[:,0]=self.pattern[phase]*self.sign

