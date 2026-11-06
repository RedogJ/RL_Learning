"""Fixed-trajectory WM fitting diagnostic. Does not update production checkpoints."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import mujoco_warp as mjw
from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env, warp_stream_ordered
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.state_model import Dynamics, TerminationHead, build_state
from go2_mujoco_mimic.tasks.mimic.state_train import goal
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=600)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    if (args.output / 'metrics.jsonl').exists():
        raise ValueError('Choose an empty output directory; existing results will not be overwritten')
    started = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = load_config(ck)
    cfg.num_envs = 384
    cfg.task.observation_noise = True
    cfg.task.random_reference_start = False
    env = WarpGo2Env(cfg, Path(ck['motion']))
    env.enable_cuda_graph()
    actor = MotionPolicy(history_length=cfg.task.history_length,
                         future_length=cfg.task.future_length,
                         heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval()
    actor.load_state_dict(ck['actor'])
    actor.requires_grad_(False)
    actor_before = {k: v.clone() for k,v in actor.state_dict().items()}
    model = Dynamics(kinematics=env.kinematics if cfg.wm_foot_features else None).cuda()
    model.load_state_dict(ck['model'])
    normalizer_before = {k: v.clone() for k,v in model.normalizer.state_dict().items()}
    head = None
    if ck.get('termination_head') is not None:
        head = TerminationHead(cfg.task.future_length*19).cuda()
        head.load_state_dict(ck['termination_head'])
    opt_m = torch.optim.Adam(model.parameters(), lr=2e-4)
    opt_m.load_state_dict(ck['opt_m'])
    opt_d = torch.optim.Adam(head.parameters(), lr=2e-4) if head is not None else None
    if opt_d is not None: opt_d.load_state_dict(ck['opt_d'])
    parameters = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    ctx = torch.zeros(cfg.num_envs,16,device='cuda');ctx[:,0] = .5
    first_frame = round(.6*env.ref.fps)
    steps = round(.8/env.dt)
    assert steps >= 16

    @warp_stream_ordered
    def reset_segment(env):
        env.reset(torch.ones(cfg.num_envs,device='cuda',dtype=torch.bool))
        env.frame.fill_(first_frame);env.start_frame.fill_(first_frame);env.steps.zero_()
        env.previous_action.zero_()
        env.qpos[:,:3] = env.ref.position[first_frame]
        env.qpos[:,3:7] = env.ref.quaternion[first_frame]
        env.qpos[:,env.qids] = env.ref.joints[first_frame]
        env.qvel[:] = env.ref.velocity[first_frame]
        env.previous_joint_vel.copy_(env.qvel[:,env.vids])
        mjw.forward(env.model,env.data)
        env.history.copy_(env.proprioception()[:,None,:].expand_as(env.history))

    print(f'采集固定数据：checkpoint={ck["cycle"]}；384条独立噪声轨迹；参考{first_frame/env.ref.fps:.2f}–{first_frame/env.ref.fps+steps*env.dt:.2f}s', flush=True)
    torch.manual_seed(1401)
    reset_segment(env)
    states = [build_state(env).clone()];actions=[];dones=[];labels=[];frames=[env.frame.clone()];masks=[]
    alive = torch.ones(cfg.num_envs,device='cuda',dtype=torch.bool)
    with torch.no_grad():
        for _ in range(steps):
            action = (actor(env.observation()) + torch.randn(cfg.num_envs,12,device='cuda')*.1).clamp(-1,1)
            masks.append(alive.clone());actions.append(action.clone())
            _,_,terminated,truncated,_ = env.step(action)
            done=terminated|truncated;dones.append(done);labels.append(terminated.float())
            states.append(build_state(env).clone());frames.append(env.frame.clone())
            alive &= ~done
    data={'states':torch.stack(states),'actions':torch.stack(actions),'dones':torch.stack(dones),
          'labels':torch.stack(labels),'frames':torch.stack(frames),'masks':torch.stack(masks),
          'context':ctx,'frame_limit':env.frame_limit.clone()}
    assert all(torch.isfinite(v).all() for v in data.values())
    dataset_path=args.output/'fixed_trajectories.pt'
    torch.save({**{k:v.cpu() for k,v in data.items()},'env':asdict(cfg),'motion':ck['motion'],
                'train_envs':list(range(256)),'validation_envs':list(range(256,384))},dataset_path)
    del states,actions,dones,labels,frames,masks
    st=data['states'];ac=data['actions'];dn=data['dones'];lb=data['labels'];fr=data['frames'];mk=data['masks']

    def windows(length,lo,hi):
        starts=torch.arange(steps-length+1,device='cuda')[:,None].expand(-1,hi-lo).reshape(-1)
        ids=torch.arange(lo,hi,device='cuda')[None].expand(steps-length+1,-1).reshape(-1)
        valid=mk[starts,ids]
        return starts[valid],ids[valid]

    train8=windows(8,0,256);val8=windows(8,256,384)
    g=torch.Generator(device='cuda').manual_seed(7301)
    def fixed_subset(pair):
        idx=torch.randperm(len(pair[0]),device='cuda',generator=g)[:512]
        return pair[0][idx],pair[1][idx]
    eval8={'train':fixed_subset(train8),'validation':fixed_subset(val8)}
    eval16={'train':fixed_subset(windows(16,0,256)),
            'validation':fixed_subset(windows(16,256,384))}
    assert min(len(train8[0]),len(val8[0]),len(eval16['validation'][0])) > 0
    config={'checkpoint':str(args.checkpoint),'cycle':ck['cycle'],'env':asdict(cfg),'motion':ck['motion'],
            'segment_seconds':[first_frame/env.ref.fps,first_frame/env.ref.fps+steps*env.dt],
            'seed_collect':1401,'seed_sample':7301,'train_envs':256,'validation_envs':128,
            'valid_train_transitions':int(mk[:,:256].sum()),'valid_validation_transitions':int(mk[:,256:].sum()),
            'train_windows8':len(train8[0]),'validation_windows8':len(val8[0]),
            'batch':512,'window':8,'updates':args.updates,'optimizer':'restored Adam states',
            'wm_learning_rates':[x['lr'] for x in opt_m.param_groups],
            'loss':'original normalized 37D recursive 8-step MSE + original BCE weight 1; original quaternion sign alignment; masks include failure step',
            'sampling':'uniform windows from fixed training trajectories; no phase priority in this diagnostic',
            'eval':'fixed windows; no weight updates from validation; physical metrics against recorded physics, same actions',
            'actor_frozen':True,'critic_untouched':True,'normalization_frozen':True}
    (args.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    print('数据固定：'+json.dumps({k:config[k] for k in ('valid_train_transitions','valid_validation_transitions','train_windows8','validation_windows8')},ensure_ascii=False),flush=True)

    def predict_loss(starts,ids,length,with_bce):
        pred=st[starts,ids];live=torch.ones(len(ids),device='cuda');loss=pred.new_zeros(());bce=loss.clone()
        for t in range(length):
            with torch.autocast('cuda',dtype=torch.bfloat16): pred=model(pred,ac[starts+t,ids],ctx[ids])
            target=st[starts+t+1,ids]
            tq=target[:,3:7]
            tq=torch.where((pred[:,3:7]*tq).sum(-1,keepdim=True)<0,-tq,tq)
            target=torch.cat((target[:,:3],tq,target[:,7:]),-1)
            err=((pred-target)/model.normalizer.scale).square().mean(-1)
            loss+=(err*live).sum()/live.sum().clamp_min(1)
            if with_bce and head is not None:
                logits=head(model.normalizer(pred),goal(env,fr[starts+t+1,ids],data['frame_limit'][ids]))
                bce+=(F.binary_cross_entropy_with_logits(logits,lb[starts+t,ids],reduction='none')*live).sum()/live.sum().clamp_min(1)
            live=live*(~dn[starts+t,ids])
        return loss/length,bce/length

    @torch.no_grad()
    def evaluate(update):
        model.eval()
        if head is not None:head.eval()
        row={'update':update,'elapsed_seconds':time.monotonic()-started}
        for split in ('train','validation'):
            sids,eids=eval8[split]
            loss,bce=predict_loss(sids,eids,8,True)
            result={'loss8':loss.item(),'bce8':bce.item(),'physical':{}}
            sids,eids=eval16[split];pred=st[sids,eids];live=torch.ones(len(eids),device='cuda',dtype=torch.bool)
            for t in range(16):
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(pred,ac[sids+t,eids],ctx[eids])
                if t+1 in (1,8,16):
                    err=(pred-st[sids+t+1,eids])[live]
                    ang=rotation_vector(pred[live,3:7],st[sids+t+1,eids][live,3:7]).norm(dim=-1)
                    rm=lambda a,b:err[:,a:b].square().mean().sqrt().item()
                    result['physical'][t+1]={'samples':len(err),
                        'root_position_rms_cm':err[:,:3].square().sum(-1).mean().sqrt().item()*100,
                        'root_rotation_rms_deg':torch.rad2deg(ang.square().mean().sqrt()).item(),
                        'root_rotation_p90_deg':torch.rad2deg(ang.quantile(.9)).item(),
                        'joint_position_rmse_deg':math.degrees(rm(7,19)),
                        'linear_velocity_rmse_m_s':rm(19,22),'angular_velocity_rmse_rad_s':rm(22,25),
                        'joint_velocity_rmse_rad_s':rm(25,37)}
                live &= ~dn[sids+t,eids]
            row[split]=result
        with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print('EVAL '+json.dumps(row,ensure_ascii=False),flush=True)
        model.train()
        if head is not None:head.train()
        return row

    rows=[evaluate(0)]
    torch.manual_seed(9301)
    checkpoints={60,180,args.updates}
    start_train=time.monotonic();interval=[]
    for update in range(1,args.updates+1):
        idx=torch.randint(len(train8[0]),(512,),device='cuda')
        loss,bce=predict_loss(train8[0][idx],train8[1][idx],8,True)
        opt_m.zero_grad(set_to_none=True)
        if opt_d is not None:opt_d.zero_grad(set_to_none=True)
        (loss+bce).backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        opt_m.step()
        if opt_d is not None:opt_d.step()
        interval.append(torch.stack((loss.detach(),bce.detach())))
        if update%20==0:
            av=torch.stack(interval).mean(0).tolist();interval=[]
            print(f'WM固定数据 [{update}/{args.updates}] state={av[0]:.6f} BCE={av[1]:.6f} 训练用时={time.monotonic()-start_train:.1f}s',flush=True)
        if update in checkpoints: rows.append(evaluate(update))
    assert all(torch.equal(v,actor_before[k]) for k,v in actor.state_dict().items())
    assert all(torch.equal(v,normalizer_before[k]) for k,v in model.normalizer.state_dict().items())
    torch.save({'model':model.state_dict(),'termination_head':head.state_dict() if head is not None else None,
                'source_checkpoint':str(args.checkpoint),'additional_updates':args.updates},args.output/'wm_diagnostic_final.pt')
    summary={'config':config,'evaluations':rows,'checks':{'actor_unchanged':True,'normalizer_unchanged':True,
             'production_checkpoints_untouched':True},'elapsed_seconds':time.monotonic()-started}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('DONE '+str(args.output/'summary.json'),flush=True)

if __name__=='__main__':main()
