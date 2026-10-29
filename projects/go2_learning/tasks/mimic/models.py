"""条件 VAE 与联合状态—latent 的去噪 Transformer；不依赖仿真器。"""
import math
import torch
from torch import nn
from .config import GuidanceCfg


def mlp(input_dim, hidden, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden),
                         nn.ELU(), nn.Linear(hidden, output_dim))


class ConditionalVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = mlp(cfg.intent_dim, cfg.hidden, 2 * cfg.latent_dim)
        self.decoder = mlp(cfg.latent_dim + cfg.proprio_dim, cfg.hidden, cfg.action_dim)
        self.register_buffer('proprio_mean', torch.zeros(cfg.proprio_dim))
        self.register_buffer('proprio_std', torch.ones(cfg.proprio_dim))

    def encode(self, intent):
        mean, logvar = self.encoder(intent).chunk(2, -1)
        return mean, logvar.clamp(-10, 5)

    def decode(self, latent, proprio):
        return self.decoder(torch.cat((latent, (proprio-self.proprio_mean)/self.proprio_std), -1))

    def sample_latent(self, intent, sample=True):
        """返回实际抽取的 z 及分布参数；采集必须记录 z，不能误记成 mean。"""
        mean, logvar = self.encode(intent)
        latent = mean + torch.randn_like(mean) * (.5 * logvar).exp() if sample else mean
        return latent, mean, logvar

    def forward(self, intent, proprio, sample=True):
        latent, mean, logvar = self.sample_latent(intent, sample)
        return self.decode(latent, proprio), mean, logvar


class TrajectoryDiffusion(nn.Module):
    def __init__(self, state_dim, cfg):
        super().__init__()
        self.cfg, self.state_dim = cfg, state_dim
        if not 0 <= cfg.future_keyframe_prob <= 1:
            raise ValueError('future_keyframe_prob 必须在 [0,1]')
        self.raw_dim = state_dim + cfg.latent_dim
        self.projection = None
        if cfg.emphasis > 0:
            from .projection import EmphasisProjection
            self.projection=EmphasisProjection(state_dim,cfg.emphasis,cfg.projection_seed)
        self.model_state_dim = state_dim * (2 if self.projection is not None else 1)
        self.dim = self.model_state_dim + cfg.latent_dim
        self.length = cfg.history + 1 + cfg.horizon
        self.input = nn.Linear(self.dim, cfg.embedding)
        self.position = nn.Parameter(torch.randn(1, self.length, cfg.embedding) * .01)
        if cfg.command_dim not in (0, 6):
            raise ValueError('command_dim 必须为 0 或 6')
        self.command_encoder = mlp(6, cfg.embedding, cfg.embedding) if cfg.command_dim else None
        # 指令以物理单位输入，固定尺度不依赖验证集或未来指令。
        if cfg.command_dim:
            self.register_buffer('command_scale', torch.tensor([1., 1., 1.5, 1.5, 1., .4]))
        self.noise = nn.Sequential(nn.Linear(2, cfg.embedding), nn.SiLU(), nn.Linear(cfg.embedding, cfg.embedding))
        layer = nn.TransformerEncoderLayer(cfg.embedding, cfg.heads, cfg.embedding * 4,
                                          dropout=0., activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.output = nn.Linear(cfg.embedding, self.dim)
        self.register_buffer('mean', torch.zeros(self.raw_dim))
        self.register_buffer('std', torch.ones(self.raw_dim))
        times = torch.linspace(0, 1, cfg.noise_steps+1)
        schedule = torch.cos((times+.008)/1.008 * math.pi/2).square()
        alpha_bar = (schedule/schedule[0])[1:].clamp_min(1e-5)
        self.register_buffer('alpha_bar', alpha_bar)

    def forward(self, x, levels, command=None):
        tokens = self.input(x) + self.position + self.noise(levels.float() / self.cfg.noise_steps)
        if self.command_encoder is not None:
            if command is None or command.shape != (x.shape[0], 6):
                raise ValueError('条件扩散需要每个环境的六维当前指令')
            tokens = tokens + self.command_encoder(command / self.command_scale)[:, None, :]
        return self.output(self.transformer(tokens))

    def command_from_target(self, target):
        # 与 features.extract 的 intent 顺序相同；target 中重力方向区分技能。
        upright = (target[:, 3].abs() > .5).to(target.dtype)
        yaw = target[:, 7:8] if target.shape[-1] > 7 else torch.zeros_like(target[:, :1])
        return torch.cat(((1-upright)[:, None], upright[:, None], target[:, :2], yaw, target[:, 6:7]), -1)

    def encode_state(self, state):
        return self.projection(state) if self.projection is not None else state

    def encode(self, raw):
        normalized=(raw-self.mean)/self.std
        return torch.cat((self.encode_state(normalized[...,:self.state_dim]),
                          normalized[...,self.state_dim:]),-1)

    def decode(self, projected):
        state=projected[...,:self.model_state_dim]
        if self.projection is not None:state=self.projection.decode(state)
        return torch.cat((state,projected[...,self.model_state_dim:]),-1)*self.std+self.mean

    def loss(self, clean, return_metrics=False, command=None):
        raw=clean
        clean = self.encode(clean)
        n, t, _ = clean.shape
        # 状态与 latent 各自、各时间步独立噪声等级；0 为已知观测。
        levels = torch.randint(1, self.cfg.noise_steps+1, (n, t, 2), device=clean.device)
        alpha = self.alpha_bar[(levels-1).clamp_min(0)]
        alpha = torch.cat((alpha[:, :, :1].expand(-1, -1, self.model_state_dim),
                           alpha[:, :, 1:].expand(-1, -1, self.cfg.latent_dim)), -1)
        x = alpha.sqrt()*clean + (1-alpha).sqrt()*torch.randn_like(clean)
        # 训练真实条件：历史状态/latent、当前状态可见，当前 latent 需要预测。
        known = torch.zeros_like(x, dtype=torch.bool)
        known[:, :self.cfg.history] = True
        known[:, self.cfg.history, :self.model_state_dim] = True
        if self.cfg.future_keyframe_prob:
            anchors = torch.rand(n, self.cfg.horizon, device=clean.device) < self.cfg.future_keyframe_prob
            known[:, self.cfg.history+1:, :self.model_state_dim] = anchors[:, :, None]
            levels[:, self.cfg.history+1:, 0].masked_fill_(anchors, 0)
        x = torch.where(known, clean, x)
        levels[:, :self.cfg.history] = 0
        levels[:, self.cfg.history, 0] = 0
        prediction = self(x, levels, command)
        loss=(prediction-clean).square()[~known].mean()
        if not return_metrics:return loss
        physical=self.decode(prediction)
        future=slice(self.cfg.history+1,None)
        metrics={
            'future_velocity_rmse':float((physical[:,future,:2]-raw[:,future,:2]).square().mean().sqrt()),
            'current_latent_rmse':float((physical[:,self.cfg.history,self.state_dim:]-raw[:,self.cfg.history,self.state_dim:]).square().mean().sqrt()),
        }
        return loss,metrics

    def guidance_terms(self, prediction, target, cfg):
        physical = self.decode(prediction)
        if cfg.mode=='separate':
            horizon=physical[:,self.cfg.history:,:self.state_dim]
            error=horizon[:,:,:3]-target[:,None,:3]
            # 每个预测姿态有自己的机身坐标；用该帧重力方向定义世界水平面。
            # 不能根据目标技能提前假设 body X 已朝上，否则起身前会漏罚后退、误罚上升。
            gravity=horizon[:,:,6:9]
            vertical=gravity/gravity.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            planar_error=error-(error*vertical).sum(-1,keepdim=True)*vertical
            per_time=planar_error.square().sum(-1)
            planar=.5*per_time.sum()
            orientation=(horizon[:,:,6:9]-target[:,None,3:6]).square().sum(-1).mean(1).sum()
            height=(horizon[:,:,33]-target[:,None,6]).square().mean(1).sum()
            yaw=((horizon[:,:,5]-target[:,None,7]).square().mean(1).sum()
                 if target.shape[-1]>7 else planar.new_zeros(()))
            return {'velocity':cfg.velocity*planar,'orientation':cfg.orientation*orientation,
                    'height':cfg.height*height,'yaw':cfg.yaw*yaw}
        future = physical[:, self.cfg.history+1:, :self.state_dim]
        velocity = (future[:, :, :3]-target[:, None, :3]).square().sum(-1)
        orientation = (future[:, :, 6:9]-target[:, None, 3:6]).square().sum(-1)
        height = (future[:, :, 33]-target[:, None, 6]).square()
        cost = (velocity+orientation+height*4).mean(1).sum()
        if target.shape[-1] > 7:
            cost = cost + (future[:, :, 5]-target[:, None, 7]).square().mean(1).sum()
        return {'legacy':cost}

    def guidance_cost(self, prediction, target, cfg=None):
        return sum(self.guidance_terms(prediction,target,cfg or GuidanceCfg(mode='legacy')).values())

    def sample(self, history, current_state, target, strength=.05, noise=None,
               guidance_mode='output', guidance_cfg=None, start_step=None, keyframes=None):
        if guidance_mode not in ('output', 'input'):
            raise ValueError(f'未知引导路径：{guidance_mode}')
        # 暖启动诊断可从较低噪声层开始；noise 此时须由调用方构造为该层的带噪轨迹。
        # 默认保持原来的完整去噪过程。
        start_step=self.cfg.noise_steps-1 if start_step is None else start_step
        if not isinstance(start_step,int) or not 0<=start_step<self.cfg.noise_steps:
            raise ValueError('start_step 必须是有效噪声层索引')
        # DDIM eta=0，20 步采样；与原文随机 DDPM 的实现差异在 README 记录。
        device = current_state.device
        n = current_state.shape[0]
        x = (torch.randn(n, self.length, self.dim, device=device)
             if noise is None else noise.clone())
        known_value = torch.zeros_like(x)
        known_value[:, :self.cfg.history] = self.encode(history)
        known_value[:, self.cfg.history, :self.model_state_dim] = self.encode_state(
            (current_state-self.mean[:self.state_dim])/self.std[:self.state_dim])
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, :self.cfg.history] = True
        mask[:, self.cfg.history, :self.model_state_dim] = True
        if keyframes is not None:
            values, frame_mask = keyframes
            if values.shape != (n, self.cfg.horizon, self.state_dim) or frame_mask.shape != (n, self.cfg.horizon):
                raise ValueError('关键帧需要 [N,H,state_dim] 状态与 [N,H] 布尔掩码')
            if frame_mask.dtype != torch.bool or values.device != device or frame_mask.device != device:
                raise ValueError('关键帧状态/掩码须与观测同设备，掩码必须为bool')
            encoded=self.encode_state((values-self.mean[:self.state_dim])/self.std[:self.state_dim])
            known_value[:, self.cfg.history+1:, :self.model_state_dim] = encoded
            mask[:, self.cfg.history+1:, :self.model_state_dim] = frame_mask[:, :, None]
        command = self.command_from_target(target) if self.command_encoder is not None else None
        separate=guidance_cfg is not None and guidance_cfg.mode=='separate'
        for step in reversed(range(start_step+1)):
            x = torch.where(mask, known_value, x)
            levels = torch.full((n, self.length, 2), step+1, device=device)
            levels[:, :self.cfg.history] = 0
            levels[:, self.cfg.history, 0] = 0
            if keyframes is not None:
                levels[:, self.cfg.history+1:, 0].masked_fill_(frame_mask, 0)
            input_grad = None
            if strength and guidance_mode == 'input':
                # 仅对本轮去噪输入回传：利用网络的状态/latent 关系，不跨 20 轮反传。
                with torch.enable_grad():
                    noisy = x.detach().requires_grad_(True)
                    prediction = self(noisy, levels, command)
                    cost_prediction=torch.where(mask,known_value,prediction) if separate else prediction
                    input_grad = torch.autograd.grad(self.guidance_cost(cost_prediction, target,guidance_cfg), noisy)[0]
                prediction = prediction.detach()
                input_grad = input_grad.detach().masked_fill(mask, 0).clamp(-10, 10)
            else:
                with torch.no_grad():
                    prediction = self(x, levels, command)
            if strength and guidance_mode == 'output':
                with torch.enable_grad():
                    clean = prediction.detach().requires_grad_(True)
                    cost_prediction=torch.where(mask,known_value,clean) if separate else clean
                    cost = self.guidance_cost(cost_prediction, target,guidance_cfg)
                    grad = torch.autograd.grad(cost, clean)[0]
                    prediction = (clean-strength*grad.clamp(-10, 10)).detach()
            alpha = self.alpha_bar[step]
            previous = self.alpha_bar[step-1] if step else x.new_ones(())
            epsilon = (x-alpha.sqrt()*prediction)/(1-alpha).sqrt()
            x = previous.sqrt()*prediction + (1-previous).sqrt()*epsilon
            if input_grad is not None:
                # 实验性 posterior-sampling 更新：DDIM 后沿输入梯度下降。
                # 保留旧 output 更新，便于在同一权重/噪声下对照，不宣称二者数值等价。
                x = x - strength*input_grad
        x = torch.where(mask, known_value, x)
        result=self.decode(x)
        # 逆投影存在浮点舍入；已知物理观测最后精确写回。
        result[:,:self.cfg.history]=history
        result[:,self.cfg.history,:self.state_dim]=current_state
        if keyframes is not None:
            result[:,self.cfg.history+1:,:self.state_dim]=torch.where(
                frame_mask[:,:,None],values,result[:,self.cfg.history+1:,:self.state_dim])
        return result
