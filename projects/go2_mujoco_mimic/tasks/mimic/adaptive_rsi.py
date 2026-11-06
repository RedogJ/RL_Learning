"""按参考片段起点回报估计的 RSI 采样；统计留在 GPU。"""
import math
import torch


class AdaptiveRSI:
    def __init__(self, period_frames, fps, device, bin_seconds=.2, gamma=.99,
                 max_reward_rate=5., max_episode_steps=140, control_dt=1/30, ema=.1, temperature=25., history_decay=.99):
        if period_frames < 1 or fps <= 0 or bin_seconds <= 0 or not 0 < gamma < 1 or max_reward_rate <= 0 or max_episode_steps < 1 or control_dt <= 0 or not 0 < ema <= 1 or not 0 <= history_decay < 1:
            raise ValueError('RSI 参数无效')
        self.period_frames = int(period_frames)
        self.fps = float(fps)
        self.frames_per_bin = max(1, round(bin_seconds*fps))
        self.n_bins = math.ceil(period_frames/self.frames_per_bin)
        self.ema = ema
        self.temperature = temperature
        self.history_decay = history_decay
        # 每个环境的回合目标可不同；这里先以该任务最大回合长度的上界归一化。
        self.gmax = (max_reward_rate * control_dt) * (1 - gamma**max_episode_steps) / (1-gamma)
        self.mean_return = torch.zeros(self.n_bins, device=device)
        self.seen = torch.zeros(self.n_bins, device=device, dtype=torch.bool)
        self.historical = torch.full((self.n_bins,), 1/self.n_bins, device=device)
        self.counts = torch.zeros(self.n_bins, device=device, dtype=torch.long)

    def probabilities(self):
        # 尚无统计的片段采用已知片段的平均回报，不把未采样当作失败。
        fallback = (self.mean_return * self.seen).sum() / self.seen.sum().clamp_min(1)
        estimates = torch.where(self.seen, self.mean_return, fallback)
        hard = torch.softmax(-self.temperature * estimates / self.gmax, dim=0)
        historical = self.historical / self.historical.sum().clamp_min(1e-12)
        return .5 / self.n_bins + .4 * hard + .1 * historical

    @torch.no_grad()
    def update(self, starts, discounted_returns, done):
        bins=(starts.remainder(self.period_frames)//self.frames_per_bin).clamp(max=self.n_bins-1)
        counts=torch.zeros_like(self.mean_return)
        sums=torch.zeros_like(self.mean_return)
        counts.scatter_add_(0,bins,done.float())
        sums.scatter_add_(0,bins,torch.where(done,discounted_returns,0.))
        present=counts>0
        batch_mean=sums/counts.clamp_min(1)
        new_values=torch.where(self.seen, (1-self.ema)*self.mean_return+self.ema*batch_mean, batch_mean)
        self.mean_return=torch.where(present,new_values,self.mean_return)
        self.seen |= present
        self.counts += counts.long()
        # 存储过去困难分布的指数移动平均。
        fallback=(self.mean_return*self.seen).sum()/self.seen.sum().clamp_min(1)
        hard=torch.softmax(-self.temperature*torch.where(self.seen,self.mean_return,fallback)/self.gmax,dim=0)
        updated_history=self.history_decay*self.historical+(1-self.history_decay)*hard
        self.historical=torch.where(present.any(),updated_history,self.historical)

    def state_dict(self):
        return {'period_frames':self.period_frames,'fps':self.fps,'frames_per_bin':self.frames_per_bin,
                'n_bins':self.n_bins,'mean_return':self.mean_return,'seen':self.seen,
                'historical':self.historical,'counts':self.counts}

    def load_state_dict(self, data):
        if (data['period_frames'],data['fps'],data['frames_per_bin'],data['n_bins']) != (self.period_frames,self.fps,self.frames_per_bin,self.n_bins):
            raise ValueError('RSI 参考片段或分箱参数与 checkpoint 不一致')
        for name in ('mean_return','seen','historical','counts'):
            getattr(self,name).copy_(data[name])
