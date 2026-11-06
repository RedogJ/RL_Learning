"""Controlled WM update attribution, with fixed Actor/data/minibatches and held-out probes."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env
from go2_mujoco_mimic.tasks.mimic.train import Replay
from go2_mujoco_mimic.tasks.mimic.adaptive_rsi import AdaptiveRSI
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.state_model import Dynamics, TerminationHead, build_state
from go2_mujoco_mimic.tasks.mimic.state_train import goal
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--old-data',type=Path,required=True)
    p.add_argument('--recent-data',type=Path,required=True)
    p.add_argument('--probe-indices',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cycles',type=int,default=10)
    p.add_argument('--num-envs',type=int,default=2048)
    p.add_argument('--fixed-stream',type=Path)
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    if (args.output/'events.jsonl').exists():raise ValueError('Choose a new output directory')
    start=time.monotonic();torch.backends.cuda.matmul.allow_tf32=True
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=load_config(ck);cfg.num_envs=args.num_envs
    env=WarpGo2Env(cfg,Path(ck['motion']));env.enable_cuda_graph()
    actor=MotionPolicy(history_length=cfg.task.history_length,future_length=cfg.task.future_length,
                       heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval()
    actor.load_state_dict(ck['actor']);actor.requires_grad_(False)
    actor_reference={k:v.cpu().clone() for k,v in actor.state_dict().items()}
    context=torch.zeros(args.num_envs,16,device='cuda');context[:,0]=.5
    capacity=320;collect_steps=32;updates=60;batch=512;window=8
    total_steps=capacity+args.cycles*collect_steps
    r=cfg.task.reward
    max_rate=sum(getattr(r,k) for k in ('body_position_weight','body_orientation_weight','body_linear_velocity_weight',
        'body_angular_velocity_weight','root_position_weight','root_orientation_weight','joint_position_weight'))
    rsi=AdaptiveRSI(len(env.ref.frames)-1,env.ref.fps,env.device,bin_seconds=.2,max_reward_rate=max_rate,
                    max_episode_steps=min(math.ceil(cfg.episode_seconds/env.dt),len(env.ref.frames)-1),control_dt=env.dt)
    if not cfg.task.reference_loop:rsi.gmax=1.
    if ck.get('adaptive_rsi') is not None:rsi.load_state_dict(ck['adaptive_rsi'])
    stream=Replay(total_steps,args.num_envs,env.device)
    if args.fixed_stream is None:
        torch.manual_seed(4420)
        env.reset(torch.ones(args.num_envs,device='cuda',dtype=torch.bool),rsi.probabilities(),rsi.frames_per_bin,start_probability=.2)
        state=build_state(env);discounted=torch.zeros(args.num_envs,device='cuda');discount=torch.ones_like(discounted)
        print(f'固定Actor={ck["cycle"]}；先采集{total_steps}步×{args.num_envs}环境，前320步填满缓冲，之后每32步做一轮对照。',flush=True)
        with torch.no_grad():
            for t in range(total_steps):
                frame=env.frame.clone()
                action=(actor(env.observation())+torch.randn(args.num_envs,12,device='cuda')*.1).clamp(-1,1)
                _,reward,failed,truncated,_=env.step(action)
                target=build_state(env);done=failed|truncated
                stream.add((state,action,target,done.float(),frame,failed.float(),env.frame.clone(),env.frame_limit.clone()))
                discounted+=discount*reward;discount*=.99
                remaining=torch.ceil((env.frame_limit-env.start_frame).float()/env.ref.fps/env.dt).clamp_min(1)
                possible=(max_rate*env.dt)*(1-torch.pow(.99,remaining))/.01
                rsi.update(env.start_frame,discounted/possible.clamp_min(1e-6),done)
                discounted=torch.where(done,0.,discounted);discount=torch.where(done,1.,discount)
                env.reset(done,rsi.probabilities(),rsi.frames_per_bin,start_probability=.2)
                state=torch.where(done[:,None],build_state(env),target)
                if (t+1)%32==0:print(f'采集 [{t+1}/{total_steps}] 用时={time.monotonic()-start:.1f}s',flush=True)
        field_names=('state','action','next_state','done','frame','failure','next_frame','frame_limit')
        data={name:value for name,value in zip(field_names,stream.storage)}
        assert stream.count==total_steps
        torch.save({'fields':{k:v.cpu() for k,v in data.items()},'context':context.cpu(),
                    'env':asdict(cfg),'motion':ck['motion'],'source_actor':str(args.checkpoint),
                    'seed_collect':4420,'capacity_simulated':capacity,'collect_steps':collect_steps},args.output/'fixed_stream.pt')
        print('固定数据流已保存；所有对照组只读取同一份数据。',flush=True)
    else:
        saved_stream=torch.load(args.fixed_stream,map_location='cpu',weights_only=False)
        assert saved_stream['source_actor']==str(args.checkpoint)
        data={k:v.cuda() for k,v in saved_stream['fields'].items()}
        context=saved_stream['context'].cuda()
        assert data['state'].shape[:2]==(total_steps,args.num_envs)
        del saved_stream
        print('复用固定数据流：'+str(args.fixed_stream),flush=True)

    probe_names=('states','actions','dones','labels','frames','masks','context','frame_limit')
    probe_data={}
    for name,path in [('old_policy_3270',args.old_data),('recent_policy_3330',args.recent_data)]:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        probe_data[name]={k:saved[k].cuda() for k in probe_names}
    indices=torch.load(args.probe_indices,map_location='cuda',weights_only=False)
    configs=[('joint_lr2e-4',False,2e-4),('detached_bce_lr2e-4',True,2e-4),('joint_lr1e-4',False,1e-4)]
    branches={}
    for name,detach,lr in configs:
        model=Dynamics(kinematics=env.kinematics if cfg.wm_foot_features else None).cuda()
        model.load_state_dict(ck['model'])
        head=TerminationHead(cfg.task.future_length*19).cuda();head.load_state_dict(ck['termination_head'])
        opt_m=torch.optim.Adam(model.parameters(),lr=lr);opt_m.load_state_dict(copy.deepcopy(ck['opt_m']))
        for group in opt_m.param_groups:group['lr']=lr
        opt_d=torch.optim.Adam(head.parameters(),lr=2e-4);opt_d.load_state_dict(copy.deepcopy(ck['opt_d']))
        branches[name]={'model':model,'head':head,'opt_m':opt_m,'opt_d':opt_d,'detach':detach,
                        'parameters':list(model.parameters())+list(head.parameters()),
                        'normalizer':{k:v.clone() for k,v in model.normalizer.state_dict().items()}}
    priority=ck['wm_priority'];bin_frames=priority['bin_frames'];error_ema=priority['error_ema'].cuda().clone()
    seen=priority['seen'].cuda().clone();n_bins=len(error_ema)
    sampler=torch.Generator(device='cuda').manual_seed(88120)
    sampling=[];events=[];gradients=[];cycle_stats=[]
    config={'source_checkpoint':str(args.checkpoint),'cycles':args.cycles,'num_envs':args.num_envs,
            'buffer_steps':capacity,'prefill_steps':capacity,'collect_steps':collect_steps,'wm_updates':updates,
            'wm_batch':batch,'window':window,'seed_collect':4420,'seed_sampler':88120,
            'branches':[{'name':n,'wm_lr':lr,'detach_bce_at_wm_output':d,'termination_lr':2e-4} for n,d,lr in configs],
            'data':'fixed actor 3320 with original RSI, observation noise and .1 action noise; real physics; generated once',
            'sampling':'current 320-step subrange only; baseline phase-priority EMA; identical sampled windows for all branches',
            'probes':'independent saved old/new policy validation windows, never used for training',
            'limitation':'Actor/Critic frozen. Fresh replay prefilling and fresh RNG; not a replay of historical 3320..3330 training.',
            'fixed_stream_source':str(args.fixed_stream) if args.fixed_stream else str(args.output/'fixed_stream.pt'),
            'optimizer_states':'deepcopied independently per branch including CPU step counters',
            'production_untouched':True}
    (args.output/'config.json').write_text(json.dumps(config,indent=2))

    def sample_batch(available):
        ti=torch.randint(available-capacity,available-window+1,(batch*4,),device='cuda',generator=sampler)
        ei=torch.randint(args.num_envs,(batch*4,),device='cuda',generator=sampler)
        bins=(data['frame'][ti,ei].remainder(len(env.ref.frames)-1)//bin_frames).clamp(max=n_bins-1)
        fallback=(error_ema*seen).sum()/seen.sum().clamp_min(1)
        estimates=torch.where(seen,error_ema,fallback).clamp_min(1e-6)
        weights=(estimates/estimates.mean().clamp_min(1e-6)).clamp(.25,4.)
        choose=torch.cat((torch.arange(batch//2,device='cuda'),torch.multinomial(weights[bins],batch-batch//2,replacement=True,generator=sampler)))
        ti=ti[choose];ei=ei[choose];bins=bins[choose]
        slots=ti[:,None]+torch.arange(window,device='cuda')[None]
        ids=ei[:,None].expand_as(slots)
        row={k:v[slots,ids] for k,v in data.items()}
        row['context']=context[ei]
        row['goals']=goal(env,row['next_frame'].reshape(-1),row['frame_limit'].reshape(-1)).reshape(batch,window,-1)
        return row,ti,ei,bins

    def loss_on(branch,row):
        model=branch['model'];head=branch['head'];pred=row['state'][:,0]
        live=torch.ones(len(pred),device='cuda');loss=pred.new_zeros(());bce=loss.clone()
        per=live.clone().zero_();counts=per.clone();positive=loss.clone()
        for t in range(window):
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(pred,row['action'][:,t],row['context'])
            target=row['next_state'][:,t];quat=target[:,3:7]
            quat=torch.where((pred[:,3:7]*quat).sum(-1,keepdim=True)<0,-quat,quat)
            truth=torch.cat((target[:,:3],quat,target[:,7:]),-1)
            err=((pred-truth)/model.normalizer.scale).square().mean(-1)
            loss+=(err*live).sum()/live.sum().clamp_min(1)
            logits=head(model.normalizer(pred.detach() if branch['detach'] else pred),row['goals'][:,t])
            term=F.binary_cross_entropy_with_logits(logits,row['failure'][:,t],reduction='none')
            bce+=(term*live).sum()/live.sum().clamp_min(1)
            per+=err.detach()*live;counts+=live;positive+=(live*row['failure'][:,t]).sum()
            live=live*(1-row['done'][:,t])
        return loss/window,bce/window,per/counts.clamp_min(1),counts.sum(),positive

    @torch.no_grad()
    def evaluate(branch):
        model=branch['model'];head=branch['head'];model.eval();head.eval();result={}
        for name,d in probe_data.items():
            ti,ei=indices[name]['8'];pred=d['states'][ti,ei];live=torch.ones(len(ei),device='cuda');loss=live.new_zeros(());bce=loss.clone()
            for t in range(8):
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(pred,d['actions'][ti+t,ei],d['context'][ei])
                truth=d['states'][ti+t+1,ei].clone()
                truth[:,3:7]=torch.where((pred[:,3:7]*truth[:,3:7]).sum(-1,keepdim=True)<0,-truth[:,3:7],truth[:,3:7])
                err=((pred-truth)/model.normalizer.scale).square().mean(-1)
                loss+=(err*live).sum()/live.sum().clamp_min(1)
                logits=head(model.normalizer(pred),goal(env,d['frames'][ti+t+1,ei],d['frame_limit'][ei]))
                term=F.binary_cross_entropy_with_logits(logits,d['labels'][ti+t,ei],reduction='none')
                bce+=(term*live).sum()/live.sum().clamp_min(1);live*=~d['dones'][ti+t,ei]
            q={'loss8':(loss/8).item(),'bce8':(bce/8).item(),'physical':{}}
            ti,ei=indices[name]['16'];pred=d['states'][ti,ei];live=torch.ones(len(ei),device='cuda',dtype=torch.bool)
            for t in range(16):
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(pred,d['actions'][ti+t,ei],d['context'][ei])
                if t+1 in (1,8,16):
                    err=(pred-d['states'][ti+t+1,ei])[live]
                    angle=rotation_vector(pred[live,3:7],d['states'][ti+t+1,ei][live,3:7]).norm(dim=-1)
                    rms=lambda lo,hi:err[:,lo:hi].square().mean().sqrt().item()
                    q['physical'][t+1]={'samples':len(err),'root_position_rms_cm':err[:,:3].square().sum(-1).mean().sqrt().item()*100,
                        'root_rotation_rms_deg':torch.rad2deg(angle.square().mean().sqrt()).item(),
                        'joint_position_rmse_deg':math.degrees(rms(7,19)),'linear_velocity_rmse_m_s':rms(19,22),
                        'angular_velocity_rmse_rad_s':rms(22,25),'joint_velocity_rmse_rad_s':rms(25,37)}
                live &= ~d['dones'][ti+t,ei]
            result[name]=q
        model.train();head.train()
        return result

    def record(cycle,update,phase,names):
        row={'cycle':cycle,'update_in_cycle':update,'phase':phase,'global_updates':cycle*updates+update,
             'elapsed_seconds':time.monotonic()-start,'branches':{name:evaluate(branches[name]) for name in names}}
        events.append(row)
        with (args.output/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        compact={name:{probe:round(v['physical'][16]['root_rotation_rms_deg'],2) for probe,v in values.items()} for name,values in row['branches'].items()}
        print('PROBE',cycle,update,phase,json.dumps(compact),flush=True)
        return row

    record(0,0,'initial',list(branches))
    for cycle in range(args.cycles):
        available=capacity+(cycle+1)*collect_steps
        phase_counts=torch.zeros(n_bins,device='cuda',dtype=torch.long)
        losses={name:[] for name in branches}
        record(cycle,0,'before_wm',['joint_lr2e-4'])
        for update in range(1,updates+1):
            row,ti,ei,bins=sample_batch(available)
            sampling.append(torch.stack((ti,ei),-1).cpu())
            phase_counts+=torch.bincount(bins,minlength=n_bins)
            for name,branch in branches.items():
                lm,ld,per,valid,positive=loss_on(branch,row)
                diagnostic=(name=='joint_lr2e-4' and update in (1,20,60))
                wm_params=tuple(branch['model'].parameters())
                if diagnostic:
                    gm=torch.autograd.grad(lm,wm_params,retain_graph=True)
                    gd=torch.autograd.grad(ld,wm_params,retain_graph=True)
                    a2=sum(x.square().sum() for x in gm);b2=sum(x.square().sum() for x in gd);dot=sum((x*y).sum() for x,y in zip(gm,gd))
                    grow={'cycle':cycle,'update':update,'state_gradient_norm':a2.sqrt().item(),
                          'bce_gradient_norm':b2.sqrt().item(),'cosine':(dot/(a2*b2).sqrt().clamp_min(1e-20)).item(),
                          'bce_to_state_norm_ratio':(b2/a2.clamp_min(1e-20)).sqrt().item(),
                          'state_loss_before':lm.item(),'bce_before':ld.item(),'valid_steps':valid.item(),'positive_steps':positive.item()}
                    before=[p.detach().clone() for p in wm_params]
                branch['opt_m'].zero_grad(set_to_none=True);branch['opt_d'].zero_grad(set_to_none=True)
                (lm+ld).backward()
                norm=torch.nn.utils.clip_grad_norm_(branch['parameters'],1.,error_if_nonfinite=True)
                branch['opt_m'].step();branch['opt_d'].step()
                losses[name].append(torch.stack((lm.detach(),ld.detach(),norm.detach())))
                if diagnostic:
                    with torch.no_grad():
                        grow['preclip_joint_gradient_norm']=norm.item()
                        grow['adam_step_dot_state_gradient']=sum(((p-old)*g).sum() for p,old,g in zip(wm_params,before,gm)).item()
                        grow['adam_step_dot_bce_gradient']=sum(((p-old)*g).sum() for p,old,g in zip(wm_params,before,gd)).item()
                        nl,nd,_,_,_=loss_on(branch,row)
                        grow.update(state_loss_after=nl.item(),bce_after=nd.item())
                    gradients.append(grow)
                    with (args.output/'gradients.jsonl').open('a') as f:f.write(json.dumps(grow)+'\n')
                    del gm,gd,before
                if name=='joint_lr2e-4':
                    totals=torch.zeros_like(error_ema).scatter_add_(0,bins,per)
                    counts=torch.bincount(bins,minlength=n_bins);present=counts>0
                    mean=totals/counts.clamp_min(1)
                    error_ema=torch.where(present,torch.where(seen,.9*error_ema+.1*mean,mean),error_ema);seen|=present
            if update in (1,10,20,40):record(cycle,update,'within_wm',['joint_lr2e-4'])
        metrics={'cycle':cycle,'available_steps':available,'buffer_lower_step':available-capacity,
                 'phase_sample_counts':phase_counts.tolist(),'wm_error_ema':error_ema.tolist(),
                 'branch_training_means':{k:torch.stack(v).mean(0).tolist() for k,v in losses.items()}}
        cycle_stats.append(metrics)
        with (args.output/'cycle_stats.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        record(cycle,updates,'after_wm',list(branches))
    torch.save(torch.stack(sampling),args.output/'sample_schedule.pt')
    for name,branch in branches.items():
        assert all(torch.equal(v,branch['normalizer'][k]) for k,v in branch['model'].normalizer.state_dict().items())
        torch.save({'model':branch['model'].state_dict(),'termination_head':branch['head'].state_dict(),
                    'source_checkpoint':str(args.checkpoint),'branch':name,'updates':args.cycles*updates},args.output/f'{name}_diagnostic.pt')
    assert all(torch.equal(v.cpu(),actor_reference[k]) for k,v in actor.state_dict().items())
    summary={'config':config,'events':events,'gradient_checks':gradients,'cycle_stats':cycle_stats,
             'checks':{'actor_unchanged':True,'normalizers_unchanged':True,'all_branches_same_data_and_minibatches':True,
                       'only_current_320_steps_sampled':True,'production_untouched':True},'elapsed_seconds':time.monotonic()-start}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('DONE',args.output/'summary.json',flush=True)

if __name__=='__main__':main()
