"""保留专家完整本体信息，并在同一状态上查询不同速度意图的监督动作。"""
import math
from pathlib import Path
import json
import torch
from torch import nn
from .experts import expert_path


def complete_proprio(trajectory, expert, dt):
    """旧 46 维数据按 episode 内状态重建积分；原始文件不重写。

    当前观测的积分由上一物理控制步末（即当前记录状态）的误差更新。
    每段记录均从 reset 等待结束开始，首帧积分为零。
    """
    proprio=trajectory['proprio']
    if proprio.shape[-1]==49:return trajectory
    if proprio.shape[-1]!=46:raise ValueError('未知本体观测维数')
    cfg=json.loads(expert_path(expert).with_name('run.json').read_text())['integral_cfg']
    integral=torch.zeros((len(proprio),3),dtype=proprio.dtype,device=proprio.device)
    if cfg.get('enabled',True):
        from go2_learning.tasks.reward_terms import roll_pitch_from_gravity
        roll,pitch=roll_pitch_from_gravity(trajectory['state'][:,6:9])
        errors=torch.stack((roll,pitch,proprio[:,45]-trajectory['intent'][:,5]),-1)
        limits=proprio.new_tensor((cfg['limit'],cfg['limit'],cfg['height_limit']))
        decay=math.exp(-dt/cfg['decay_seconds'])
        for i in range(1,len(integral)):
            integral[i]=(integral[i-1]*decay+errors[i]*dt).clamp(-limits,limits)
        integral=integral/limits
    return dict(trajectory,proprio=torch.cat((proprio,integral),-1))


def teacher_observation(proprio,intent):
    if proprio.shape[-1]!=49:raise ValueError('专家监督需要包含积分的 49 维本体信息')
    return torch.cat((proprio[:,:45],intent[:,2:5],proprio[:,45:46]-intent[:,5:6],proprio[:,46:49]),-1)


def mean_teacher(expert,device='cpu'):
    path=expert_path(expert)
    cfg=json.loads(path.with_name('run.json').read_text())['agent_cfg']['actor']
    if cfg.get('obs_normalization') or cfg['class_name']!='MLPModel':
        raise ValueError('离线指令重标注当前仅支持无观测归一化的 MLP 专家')
    activation={'tanh':nn.Tanh,'elu':nn.ELU,'relu':nn.ReLU}[cfg['activation']]
    dims=[52]+cfg['hidden_dims']+[12];layers=[]
    for i,(a,b) in enumerate(zip(dims,dims[1:])):
        layers.append(nn.Linear(a,b))
        if i<len(dims)-2:layers.append(activation())
    model=nn.Sequential(*layers).to(device)
    saved=torch.load(path,map_location=device,weights_only=False)['actor_state_dict']
    model.load_state_dict({k.removeprefix('mlp.'):v for k,v in saved.items() if k.startswith('mlp.')})
    return model.eval()


def split_trajectories(trajectories,cfg):
    training,validation=[],[]
    for expert in ('running','upright'):
        groups={}
        for trajectory in trajectories:
            if trajectory['expert']==expert:
                key=trajectory.get('source_trajectory_id',trajectory['trajectory_id'])
                groups.setdefault(key,[]).append(trajectory)
        if len(groups)<2:raise ValueError(f'{expert} 至少需要两条独立轨迹')
        for index,key in enumerate(sorted(groups)):
            (validation if index%5==0 else training).extend(groups[key])
    return training,validation


def relabel_running(args):
    from datetime import datetime
    from .data import load_trajectories
    from .experts import load_experts
    from .train import save_atomic
    trajectories,_=load_trajectories(args.data)
    trajectories=[t for t in trajectories if t['expert']=='running' and not t.get('supervision_only',False)]
    teacher=mean_teacher('running',args.device)
    augmented=[]
    for t in trajectories:
        # 原始标签复算必须一致，避免遗漏积分或搞错观测布局后仍继续重标注。
        with torch.no_grad():
            obs=teacher_observation(t['proprio'],t['intent']).to(args.device)
            original=teacher(obs).cpu()
        if (original-t['teacher_action']).abs().max()>1e-4:
            raise ValueError('重建的专家输入与原始标签不一致，停止重标注')
        for command in (-1.5,0.,1.5):
            intent=t['intent'].clone();intent[:,2]=command
            with torch.no_grad():
                action=teacher(teacher_observation(t['proprio'],intent).to(args.device)).cpu()
            augmented.append(dict(t,intent=intent,teacher_action=action,
                                  source_trajectory_id=t['trajectory_id'],supervision_only=True))
    if not augmented:raise ValueError('没有可重标注的 running 数据')
    path=Path(args.data)/'relabel_running.pt'
    # 只保存监督相关字段；不冒充实际发生过的运动轨迹。
    for t in augmented:
        for key in ('latent','executed_action'):t.pop(key,None)
    example=next(p for p in sorted(Path(args.data).glob('running_*.pt')))
    source=torch.load(example,weights_only=False,map_location='cpu')
    # 旧重标注文件保留到归档，避免重复聚合导致样本权重不断膨胀。
    import shutil
    from .experts import ROOT
    legacy=list(Path(args.data).glob('relabel_running_*.pt'))
    if path.exists():legacy.append(path)
    if legacy:
        archive=ROOT/'archive'/('relabel_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        archive.mkdir(parents=True,exist_ok=True)
        for old_path in legacy:shutil.copy2(old_path,archive/old_path.name)
    save_atomic(path,{'format':'go2_mimic_v1','expert':'running','controller':'counterfactual_supervision',
        'experts':load_experts(),'joint_names':source['joint_names'],'body_names':source['body_names'],
        'control_dt':source['control_dt'],'supervision_only':True,'trajectories':augmented})
    for old_path in legacy:
        if old_path!=path:old_path.unlink()
    print(f'同状态不同指令监督已保存：{path}，{len(augmented)} 组',flush=True)
