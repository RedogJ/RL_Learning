"""真实轨迹关键帧库与按物理控制步前移的稀疏未来状态约束。"""
from pathlib import Path
import torch
from .experts import load_experts


class KeyframeTimeline:
    def __init__(self,path,model,num_envs,device,dt,initial='running',entry=.5,interval=.2):
        saved=torch.load(path,map_location=device,weights_only=False)
        if saved.get('format')!='go2_mimic_keyframes_v1':
            raise ValueError('未知关键帧库格式')
        if saved['experts']!=load_experts():raise ValueError('关键帧库专家来源不一致')
        if abs(saved['control_dt']-dt)>1e-8:raise ValueError('关键帧库控制周期不一致')
        # 关键帧不含latent，但状态/关节顺序必须来自同一模型数据来源。
        if saved['vae_sha256']!=getattr(model,'vae_sha256',saved['vae_sha256']):
            raise ValueError('关键帧库与扩散数据的VAE来源不一致')
        self.reference=saved['upright_states'].to(device=device,dtype=model.mean.dtype)
        if self.reference.ndim!=2 or self.reference.shape[1]!=model.state_dim or len(self.reference)<2:
            raise ValueError('关键帧状态维数或长度不正确')
        if not torch.isfinite(self.reference).all():raise ValueError('关键帧状态必须有限')
        if model.cfg.future_keyframe_prob<=0:
            raise ValueError('此模型没有未来关键帧条件训练，请使用独立keyframe模型')
        self.horizon=model.cfg.horizon
        self.entry=max(1,round(entry/dt));self.interval=max(1,round(interval/dt))
        self.elapsed=torch.zeros(num_envs,device=device,dtype=torch.long)
        self.offsets=torch.arange(1,self.horizon+1,device=device)
        self.skill=initial

    def select(self,skill):
        if skill!=self.skill:
            self.elapsed.zero_();self.skill=skill

    def future(self):
        absolute=self.elapsed[:,None]+self.offsets[None]
        age=absolute-self.entry
        mask=(age>=0)&(age<len(self.reference))&(age.remainder(self.interval)==0)
        if self.skill!='upright':mask=torch.zeros_like(mask)
        # 真实片段按控制步推进，结束后撤掉约束，避免把非零速度末帧无限重复。
        index=age.clamp(0,len(self.reference)-1)
        return self.reference[index],mask

    def advance(self,active):
        self.elapsed.add_(active.long())

    def reset(self,ids):
        self.elapsed[ids]=0


def build_library(data_folder,output,vae_sha256):
    """从成功的真实upright片段提取2秒目标；不制造中间起身状态。"""
    from .experts import sha256
    candidates=[]
    for path in sorted(Path(data_folder).glob('*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('expert')!='upright' or saved.get('vae_sha256')!=vae_sha256:continue
        for i,t in enumerate(saved['trajectories']):
            if not t['failed'] and len(t['state'])>=90:
                candidates.append((path,saved,i,t))
    if not candidates:raise ValueError('找不到至少3秒的成功upright真实片段')
    path,saved,index,t=candidates[0]
    # 跳过片段开头1秒，提取接下来2秒，长度按真实控制周期确定。
    dt=saved['control_dt'];start=round(1/dt);length=round(2/dt)
    payload=dict(format='go2_mimic_keyframes_v1',experts=saved['experts'],vae_sha256=vae_sha256,
        control_dt=dt,joint_names=saved['joint_names'],body_names=saved['body_names'],
        upright_states=t['state'][start:start+length].clone(),
        source=dict(file=path.name,sha256=sha256(path),trajectory_index=index,start=start,length=length),
        note='真实upright保持片段；不含已验证的四足起身过程')
    output=Path(output)
    if output.exists():raise FileExistsError(f'拒绝覆盖关键帧库：{output}')
    output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,output)
    return payload['source']
