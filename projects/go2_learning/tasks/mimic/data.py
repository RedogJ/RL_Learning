"""轨迹分界、技能均衡抽样和防止跨 reset 的窗口。"""
from pathlib import Path
import torch


def load_trajectories(folder, latent_only=False):
    trajectories = []
    manifests = set()
    latent_modes = set()
    signature = None
    for path in sorted(Path(folder).glob('*.pt')):
        data = torch.load(path, map_location='cpu', weights_only=False)
        if data.get('format') != 'go2_mimic_v1':
            continue
        from .experts import load_experts
        pinned = load_experts()
        # 普通 DAgger 标签只依赖对应教师；另一专家更换不使其失效。
        # 潜变量轨迹则来自联合 VAE，必须整组来源一致。
        old = data.get('experts', {})
        compatible = (old == pinned if latent_only else
                      old.get(data['expert']) == pinned.get(data['expert']))
        if not compatible:
            raise ValueError(f'{path} 来自不同专家，不能混用')
        if data['expert'] == 'running' and data.get('controller') == 'expert':
            validate_running_response(data['trajectories'], data['control_dt'])
        current = (tuple(data['joint_names']), tuple(data['body_names']), data['control_dt'])
        if signature is not None and current != signature:
            raise ValueError('轨迹关节顺序、刚体顺序或控制周期不一致')
        signature = current
        if latent_only:
            if data.get('teacher_mix', 0.) != 0.:
                continue
            if not data.get('vae_sha256'):
                continue
            manifests.add(data['vae_sha256'])
        if latent_only and data.get('supervision_only',False):
            continue
        if latent_only:
            mode=data.get('latent_mode','mean')
            if mode not in ('mean','posterior_sample'):
                raise ValueError(f'{path} 的 latent_mode 无效：{mode}')
            latent_modes.add(mode)
        for trajectory_index, trajectory in enumerate(data['trajectories']):
            if latent_only and trajectory['failed']:
                continue
            trajectory = dict(trajectory, expert=data['expert'],
                              latent_mode=data.get('latent_mode','mean'),
                              command_profile=data.get('command_profile','episode'),
                              trajectory_id=trajectory.get('trajectory_id',f'{path.name}#{trajectory_index}'))
            if not latent_only:
                from .supervision import complete_proprio
                trajectory = complete_proprio(trajectory,data['expert'],data['control_dt'])
            trajectories.append(trajectory)
    if latent_only and len(manifests) > 1:
        raise ValueError('不能混用不同 VAE 的潜变量轨迹，请为每个 VAE 使用独立数据目录')
    if latent_only and len(latent_modes)>1:
        raise ValueError('不能混用均值和分布采样的潜变量轨迹，请使用独立数据目录')
    if not trajectories:
        raise ValueError(f'{folder} 没有可用轨迹')
    return trajectories, manifests


def balanced_indices(labels, batch_size, generator=None):
    groups = torch.unique(labels)
    if len(groups) < 2:
        raise ValueError('学生训练必须同时包含 running 和 upright 数据')
    result = []
    for index, group in enumerate(groups):
        ids = (labels == group).nonzero(as_tuple=False).flatten()
        count = batch_size // len(groups) + (index < batch_size % len(groups))
        result.append(ids[torch.randint(len(ids), (count,), generator=generator, device=ids.device)])
    return torch.cat(result)


def priority_indices(labels, priority, batch_size, fraction):
    if fraction==0:return balanced_indices(labels,batch_size)
    result=[];groups=torch.unique(labels)
    if len(groups)<2:raise ValueError('训练必须同时包含两种技能')
    for index,group in enumerate(groups):
        count=batch_size//len(groups)+(index<batch_size%len(groups))
        all_ids=(labels==group).nonzero(as_tuple=False).flatten()
        selected=all_ids[priority[all_ids]];other=all_ids[~priority[all_ids]]
        if not len(selected) or not len(other):
            result.append(all_ids[torch.randint(len(all_ids),(count,),device=all_ids.device)]);continue
        n=round(count*fraction)
        result.extend((selected[torch.randint(len(selected),(n,),device=selected.device)],
                       other[torch.randint(len(other),(count-n,),device=other.device)]))
    return torch.cat(result)


class Windows:
    def __init__(self, trajectories, cfg):
        self.trajectories = trajectories
        self.length = cfg.history + 1 + cfg.horizon
        self.history = cfg.history
        self.command_conditioned = bool(cfg.command_dim)
        self.excluded_command_changes = 0
        self.indices, labels, transitions = [], [], []
        for i, trajectory in enumerate(trajectories):
            intent=trajectory.get('intent')
            changed=torch.zeros(len(trajectory['state']),dtype=torch.long)
            if intent is not None:changed[1:]=((intent[1:,2:5]-intent[:-1,2:5]).abs().amax(-1)>.1).long()
            prefix=torch.cat((changed.new_zeros(1),changed.cumsum(0))).tolist()
            for start in range(len(trajectory['state'])-self.length+1):
                if self.command_conditioned:
                    if intent is None or intent.shape[-1] != 6:
                        raise ValueError('条件扩散数据必须保存六维 intent')
                    current = start + cfg.history
                    # 只检查当前之后的指令；保留当前帧刚收到停车/反向命令的样本。
                    if not torch.allclose(intent[current: start+self.length],
                                          intent[current].expand(self.length-cfg.history, -1), atol=1e-6, rtol=0):
                        self.excluded_command_changes += 1
                        continue
                self.indices.append((i, start))
                labels.append(0 if trajectory['expert'] == 'running' else 1)
                transitions.append(prefix[start+self.length]>prefix[start+1])
        self.labels = torch.tensor(labels)
        self.transitions = torch.tensor(transitions,dtype=torch.bool)
        if not self.indices:
            raise ValueError('完整轨迹太短，无法构建历史/未来窗口')

    def batch(self, ids):
        return torch.stack([torch.cat((self.trajectories[i]['state'][start:start+self.length],
                                      self.trajectories[i]['latent'][start:start+self.length]), -1)
                            for i, start in (self.indices[j] for j in ids.tolist())])

    def commands(self, ids):
        return torch.stack([self.trajectories[i]['intent'][start+self.history]
                            for i, start in (self.indices[j] for j in ids.tolist())])

    def sample_ids(self, batch_size, transition_fraction=0.):
        return priority_indices(self.labels,self.transitions,batch_size,transition_fraction)


def running_response(trajectories, dt):
    """按指令分别统计两秒起步后的实际速度，不能用正负速度总均值验收。"""
    rows = []
    skip = round(2 / dt)
    for trajectory in trajectories:
        if len(trajectory['state']) > skip:
            rows.append((trajectory['intent'][skip:, 2], trajectory['state'][skip:, 0]))
    result = {}
    for name, sign in (('backward', -1), ('stationary', 0), ('forward', 1)):
        commands, velocities = [], []
        for c, v in rows:
            mask = c < -.5 if sign == -1 else c > .5 if sign == 1 else c.abs() < .05
            if mask.any():
                commands.append(c[mask]); velocities.append(v[mask])
        if commands:
            c, v = torch.cat(commands), torch.cat(velocities)
            result[name] = {'samples': len(c), 'command_vx': float(c.mean()),
                            'actual_vx': float(v.mean()), 'mean_abs_error': float((v-c).abs().mean())}
    return result


def validate_running_response(trajectories, dt):
    result = running_response(trajectories, dt)
    for name, sign in (('backward', -1), ('forward', 1)):
        row = result.get(name)
        if row and row['samples'] >= 60 and sign * row['actual_vx'] < .3:
            raise ValueError(f'running 专家速度响应未通过：{name} 指令 {row["command_vx"]:.2f}，'
                             f'实际 {row["actual_vx"]:.3f} m/s；禁止采集/训练为移动专家')
    return result
