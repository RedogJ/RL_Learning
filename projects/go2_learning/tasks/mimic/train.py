"""离线训练；不导入 Isaac Sim，不运行 PPO。"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import torch
from .config import ModelCfg
from .models import ConditionalVAE, TrajectoryDiffusion
from .data import load_trajectories, balanced_indices, priority_indices, Windows
from .experts import sha256, load_experts


def load_vae(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get('experts') != load_experts():
        raise ValueError('VAE 的专家来源与当前固定专家不一致')
    if saved['config']['proprio_dim'] != ModelCfg().proprio_dim:
        raise ValueError('旧学生缺少积分观测，请使用49维新学生，不能续训旧46维模型')
    model = ConditionalVAE(ModelCfg(**saved['config'])).to(device)
    model.load_state_dict(saved['model'])
    return model, saved


def save_atomic(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); torch.save(payload, temp); temp.replace(path)


def data_provenance(args):
    return {'files':[{'name':p.name,'sha256':sha256(p)} for p in sorted(Path(args.data).glob('*.pt'))],
            'seed':args.seed,'transition_fraction':getattr(args,'transition_fraction',0.),
            'updates':args.updates,'batch_size':args.batch_size}


def train_vae(args):
    torch.manual_seed(args.seed)
    trajectories, _ = load_trajectories(args.data)
    provenance=data_provenance(args)
    # 同状态的不同指令标签必须跟随源轨迹一起划分，避免训练/验证泄漏。
    from .supervision import split_trajectories
    train, valid = split_trajectories(trajectories,ModelCfg())
    def tensors(rows):
        return {key: torch.cat([t[key] for t in rows]).to(args.device)
                for key in ('proprio', 'intent', 'teacher_action')}
    train_data, valid_data = tensors(train), tensors(valid)
    valid_labels=torch.cat([torch.full((len(t['proprio']),), int(t['expert']=='upright')) for t in valid]).to(args.device)
    valid_priority=torch.cat([torch.full((len(t['proprio']),),t.get('command_profile')=='transitions',dtype=torch.bool)
                              for t in valid]).to(args.device)
    labels = torch.cat([torch.full((len(t['proprio']),), int(t['expert']=='upright')) for t in train]).to(args.device)
    priority=torch.cat([torch.full((len(t['proprio']),),t.get('command_profile')=='transitions',dtype=torch.bool)
                        for t in train]).to(args.device)
    cfg = ModelCfg()
    if args.resume:
        model, _ = load_vae(args.resume, args.device);cfg = model.cfg
    else:
        model = ConditionalVAE(cfg).to(args.device)
        with torch.no_grad():
            model.proprio_mean.copy_(train_data['proprio'].mean(0))
            model.proprio_std.copy_(train_data['proprio'].std(0).clamp_min(.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    best = float('inf')
    for step in range(args.updates):
        model.train()
        ids = priority_indices(labels,priority,args.batch_size,getattr(args,'transition_fraction',0.))
        prediction, mean, logvar = model(train_data['intent'][ids],train_data['proprio'][ids])
        reconstruction = (prediction-train_data['teacher_action'][ids]).square().mean()
        kl = -.5 * (1+logvar-mean.square()-logvar.exp()).sum(-1).mean()
        # 保留均值路径的辅助重建约束；带噪 latent 路径也参与训练，播放/采集使用实际采样。
        mean_action=model.decode(mean,train_data['proprio'][ids])
        mean_reconstruction=(mean_action-train_data['teacher_action'][ids]).square().mean()
        loss = .5*(reconstruction+mean_reconstruction) + cfg.kl_weight*kl
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.);optimizer.step()
        if step % 50 == 0 or step == args.updates-1:
            model.eval()
            with torch.no_grad():
                per_skill=[];validation_groups={}
                for skill in (0,1):
                    groups=[]
                    for dynamic in (False,True):
                        available=((valid_labels==skill)&(valid_priority==dynamic)).nonzero(as_tuple=False).flatten()
                        if not len(available):continue
                        vi=available[torch.linspace(0,len(available)-1,min(2048,len(available)),device=args.device).long()]
                        mse=(model(valid_data['intent'][vi],valid_data['proprio'][vi],sample=False)[0]-valid_data['teacher_action'][vi]).square().mean()
                        groups.append(mse);validation_groups[f'{skill}_{"transitions" if dynamic else "episode"}']=float(mse)
                    per_skill.append(torch.stack(groups).mean())
                value = float(torch.stack(per_skill).mean())
            print(f'vae {step+1}/{args.updates} train_mse={float(reconstruction.detach()):.5f} validation_mse={value:.5f}',flush=True)
            if value < best:
                best = value
                save_atomic(args.output, {'model': model.state_dict(), 'config': asdict(cfg),
                     'validation_mse': value,'validation_groups':validation_groups,
                     'training_step':step+1,'requested_updates':args.updates,
                     'data_provenance':provenance,
                     'stage': 'conditional_vae', 'experts': load_experts()})


def train_diffusion(args):
    torch.manual_seed(args.seed)
    trajectories, fingerprints = load_trajectories(args.data, latent_only=True)
    provenance=data_provenance(args)
    vae, _ = load_vae(args.vae, args.device)
    if fingerprints != {sha256(args.vae)}:
        raise ValueError('轨迹与指定的 VAE 不匹配')
    cfg = replace(vae.cfg,emphasis=getattr(args,'emphasis',6.),
                  command_dim=6 if getattr(args,'command_conditioned',False) else 0,
                  future_keyframe_prob=getattr(args,'future_keyframe_prob',0.))
    # 按完整片段划分验证集。
    training, validation = [], []
    for expert in ('running', 'upright'):
        group = [t for t in trajectories if t['expert']==expert and len(t['state'])>=cfg.history+1+cfg.horizon]
        if len(group)<2:raise ValueError(f'{expert} 至少需要两条成功且足够长的轨迹')
        validation.extend(group[::5]);training.extend(t for i,t in enumerate(group) if i%5)
    windows, valid = Windows(training,cfg), Windows(validation,cfg)
    model = TrajectoryDiffusion(training[0]['state'].shape[-1],cfg).to(args.device)
    all_data = torch.cat([torch.cat((t['state'],t['latent']),-1) for t in training]).to(args.device)
    with torch.no_grad():
        model.mean.copy_(all_data.mean(0));model.std.copy_(all_data.std(0).clamp_min(.05))
    del all_data
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=.001)
    best = float('inf')
    history_path=Path(args.output).with_suffix('.history.jsonl')
    history_path.parent.mkdir(parents=True,exist_ok=True)
    history_path.write_text('')
    print(f'diffusion emphasis={cfg.emphasis:g}, physical_state_dim={model.state_dim}, '
          f'model_state_dim={model.model_state_dim}; command_dim={cfg.command_dim}; output={args.output}',flush=True)
    print(f'windows train={len(windows.indices)} valid={len(valid.indices)}; '
          f'excluded future-command changes={windows.excluded_command_changes+valid.excluded_command_changes}',flush=True)
    for step in range(args.updates):
        ids = windows.sample_ids(args.batch_size,getattr(args,'transition_fraction',0.))
        batch = windows.batch(ids).to(args.device)
        command=windows.commands(ids).to(args.device) if cfg.command_dim else None
        model.train();loss = model.loss(batch,command=command)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%50==0 or step==args.updates-1:
            model.eval()
            # 固定验证噪声，不影响训练随机数流。
            with torch.random.fork_rng(devices=[torch.device(args.device).index or 0] if str(args.device).startswith('cuda') else []),torch.no_grad():
                torch.manual_seed(100)
                ids=balanced_indices(valid.labels,min(args.batch_size,128))
                validation_loss,metrics=model.loss(valid.batch(ids).to(args.device),return_metrics=True,
                    command=valid.commands(ids).to(args.device) if cfg.command_dim else None)
                value=float(validation_loss)
            row={'step':step+1,'loss':float(loss.detach()),'validation_loss':value,**metrics}
            with history_path.open('a') as log:log.write(json.dumps(row)+'\n')
            print(f'diffusion {step+1}/{args.updates} loss={row["loss"]:.5f} validation={value:.5f} '
                  f'vxy_rmse={metrics["future_velocity_rmse"]:.3f} latent_rmse={metrics["current_latent_rmse"]:.3f}',flush=True)
            if value<best:
                best=value
                vae_saved=torch.load(args.vae,map_location='cpu',weights_only=False)
                save_atomic(args.output,{'model':model.state_dict(),'config':asdict(cfg),
                    'state_dim':model.state_dim,'vae':vae_saved,'vae_sha256':sha256(args.vae),
                    'validation_loss':value,'validation_metrics':metrics,'training_step':step+1,
                    'requested_updates':args.updates,'data_provenance':provenance,
                    'latent_mode':trajectories[0]['latent_mode'],
                    'stage':'state_latent_diffusion','experts':load_experts()})
