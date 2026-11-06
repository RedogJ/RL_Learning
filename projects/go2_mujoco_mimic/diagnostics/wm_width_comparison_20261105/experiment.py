"""Paired from-scratch WM width comparison on a fixed full-motion dataset."""
import argparse
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
from go2_mujoco_mimic.tasks.mimic.state_model import Dynamics, TerminationHead
from go2_mujoco_mimic.tasks.mimic.state_train import goal
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--source-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=4800)
    p.add_argument('--seeds',default='42,117')
    p.add_argument('--wm-lr',type=float,default=1e-4)
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    if (args.output/'metrics.jsonl').exists():raise ValueError('Choose an unused output directory')
    torch.backends.cuda.matmul.allow_tf32=True
    started=time.monotonic()
    ck=torch.load(args.source_config,map_location='cpu',weights_only=False)
    cfg=load_config(ck);cfg.num_envs=1
    env=WarpGo2Env(cfg,Path(ck['motion']))
    source=torch.load(args.data,map_location='cpu',weights_only=False)
    data={k:v.cuda() for k,v in source['fields'].items()}
    context=source['context'].cuda();del source
    steps,n=data['state'].shape[:2]
    assert n==2048 and steps>=32
    train_end=1536;batch=512;window=8
    normalizer={k:v.clone() for k,v in ck['model'].items() if k.startswith('normalizer.')}
    # Only fixed input/output scales are reused. Both networks and both optimizers start from scratch.
    def make_pairs(length,lo,hi,count,g):
        # Unique sampled windows; disjoint environment sets separate whole trajectories.
        ix=torch.randperm((steps-length+1)*(hi-lo),device='cuda',generator=g)[:count]
        return ix.div(hi-lo,rounding_mode='floor'),ix.remainder(hi-lo)+lo
    probe_generator=torch.Generator(device='cuda').manual_seed(552701)
    probes={name:{str(length):make_pairs(length,lo,hi,512,probe_generator) for length in (8,16)}
            for name,lo,hi in [('train',0,train_end),('validation',train_end,n)]}
    assert not torch.isin(probes['train']['16'][1],probes['validation']['16'][1]).any()
    torch.save({name:{k:tuple(x.cpu() for x in v) for k,v in values.items()} for name,values in probes.items()},args.output/'probe_indices.pt')
    config={'dataset':str(args.data),'dataset_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
            'source_config_only':str(args.source_config),'normalizer':'same frozen source scales, no model/head/optimizer weights reused',
            'steps':steps,'envs':n,'train_envs':[0,train_end-1],'validation_envs':[train_end,n-1],
            'train_transitions':steps*train_end,'validation_transitions':steps*(n-train_end),
            'widths':[2048,1024],'hidden_layers':3,'seeds':[int(x) for x in args.seeds.split(',')],
            'updates':args.updates,'batch':batch,'window':window,'wm_lr':args.wm_lr,'head_lr':2e-4,
            'loss':'original 8-step recursive normalized MSE + BCE weight 1, BCE backpropagates into WM, shared norm clip 1',
            'sampling':'uniform fixed data windows, same minibatch sequence for both widths and initialization seeds',
            'amp':'BF16 matmul, TF32 allowed','timing':'CUDA events around forward/backward/clip/Adam only; shared sampling, evaluation and saving excluded',
            'env':asdict(cfg),'scope':'offline fixed-data capacity diagnostic; no Actor/Critic updates or policy evaluation',
            'production_untouched':True}
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    print('数据划分：'+json.dumps({k:config[k] for k in ('steps','envs','train_transitions','validation_transitions','wm_lr','seeds')}) ,flush=True)
    del ck

    def build_batch(ti,ei):
        slots=ti[:,None]+torch.arange(window,device='cuda')[None]
        ids=ei[:,None].expand_as(slots)
        row={k:v[slots,ids] for k,v in data.items()}
        row['context']=context[ei]
        row['goals']=goal(env,row['next_frame'].reshape(-1),row['frame_limit'].reshape(-1)).reshape(len(ti),window,-1)
        return row

    def losses(branch,row):
        model=branch['model'];head=branch['head'];pred=row['state'][:,0]
        live=torch.ones(len(pred),device='cuda');lm=pred.new_zeros(());ld=lm.clone()
        for t in range(window):
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(pred,row['action'][:,t],row['context'])
            target=row['next_state'][:,t]
            quat=target[:,3:7]
            quat=torch.where((pred[:,3:7]*quat).sum(-1,keepdim=True)<0,-quat,quat)
            truth=torch.cat((target[:,:3],quat,target[:,7:]),-1)
            err=((pred-truth)/model.normalizer.scale).square().mean(-1)
            lm+=(err*live).sum()/live.sum().clamp_min(1)
            logits=head(model.normalizer(pred),row['goals'][:,t])
            bce=F.binary_cross_entropy_with_logits(logits,row['failure'][:,t],reduction='none')
            ld+=(bce*live).sum()/live.sum().clamp_min(1)
            live=live*(1-row['done'][:,t])
        return lm/window,ld/window

    @torch.no_grad()
    def evaluate(branch):
        model=branch['model'];model.eval();branch['head'].eval();out={}
        for name,pairs in probes.items():
            row=build_batch(*pairs['8']);lm,ld=losses(branch,row)
            result={'loss8':lm.item(),'bce8':ld.item(),'physical':{}}
            ti,ei=pairs['16'];pred=data['state'][ti,ei];live=torch.ones(len(ti),device='cuda',dtype=torch.bool)
            for t in range(16):
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(pred,data['action'][ti+t,ei],context[ei])
                if t+1 in (1,8,16):
                    truth=data['next_state'][ti+t,ei]
                    err=(pred-truth)[live]
                    ang=rotation_vector(pred[live,3:7],truth[live,3:7]).norm(dim=-1)
                    rms=lambda lo,hi:err[:,lo:hi].square().mean().sqrt().item()
                    result['physical'][t+1]={'samples':len(err),
                        'root_position_rms_cm':err[:,:3].square().sum(-1).mean().sqrt().item()*100,
                        'root_rotation_rms_deg':torch.rad2deg(ang.square().mean().sqrt()).item(),
                        'joint_position_rmse_deg':math.degrees(rms(7,19)),
                        'linear_velocity_rmse_m_s':rms(19,22),'angular_velocity_rmse_rad_s':rms(22,25),
                        'joint_velocity_rmse_rad_s':rms(25,37)}
                live &= ~data['done'][ti+t,ei].bool()
            out[name]=result
        model.train();branch['head'].train()
        return out

    all_rows=[];timing_rows=[]
    milestones={0,100,300,600,1200,2400,3600,args.updates}
    for seed in config['seeds']:
        branches={}
        # Equal seed tags, zero output layers, identical initial head tensors.
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed+7000)
            initial_head=TerminationHead(cfg.task.future_length*19).cuda().state_dict()
        for width in config['widths']:
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed)
                model=Dynamics(width=width,kinematics=env.kinematics if cfg.wm_foot_features else None).cuda()
                head=TerminationHead(cfg.task.future_length*19).cuda()
            model.normalizer.load_state_dict({k.removeprefix('normalizer.'):v for k,v in normalizer.items()})
            head.load_state_dict(initial_head)
            om=torch.optim.Adam(model.parameters(),lr=args.wm_lr);od=torch.optim.Adam(head.parameters(),lr=2e-4)
            branches[width]={'model':model,'head':head,'om':om,'od':od,'params':list(model.parameters())+list(head.parameters()),
                'parameter_count':sum(p.numel() for p in model.parameters()),'train_ms':0.,'events':[],
                'time_targets':{5.:False,15.:False,30.:False,60.:False},'losses':[]}
        sampler=torch.Generator(device='cuda').manual_seed(93117)
        print(f'开始 seed={seed}；WM参数='+json.dumps({w:b['parameter_count'] for w,b in branches.items()}),flush=True)
        def record(update,width,reason):
            branch=branches[width]
            row={'seed':seed,'width':width,'update':update,'reason':reason,'parameter_count':branch['parameter_count'],
                 'train_seconds':branch['train_ms']/1000,'elapsed_seconds':time.monotonic()-started,
                 'metrics':evaluate(branch)}
            all_rows.append(row)
            with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            val=row['metrics']['validation']['physical'][16]
            print('WIDTH',seed,width,update,reason,'time',round(row['train_seconds'],2),
                  'val16',json.dumps({k:round(val[k],4) for k in ('root_position_rms_cm','root_rotation_rms_deg','joint_position_rmse_deg')}),flush=True)
        for width in branches:record(0,width,'initial')
        assert all_rows[-1]['metrics']==all_rows[-2]['metrics']
        for update in range(1,args.updates+1):
            ti=torch.randint(steps-window+1,(batch,),device='cuda',generator=sampler)
            ei=torch.randint(train_end,(batch,),device='cuda',generator=sampler)
            row=build_batch(ti,ei)
            order=[2048,1024] if update%2 else [1024,2048]
            for width in order:
                branch=branches[width]
                before=torch.cuda.Event(enable_timing=True);after=torch.cuda.Event(enable_timing=True)
                before.record()
                lm,ld=losses(branch,row)
                branch['om'].zero_grad(set_to_none=True);branch['od'].zero_grad(set_to_none=True)
                (lm+ld).backward()
                torch.nn.utils.clip_grad_norm_(branch['params'],1.,error_if_nonfinite=True)
                branch['om'].step();branch['od'].step()
                after.record();branch['events'].append((before,after))
                branch['losses'].append(torch.stack((lm.detach(),ld.detach())))
            if update%100==0 or update in milestones:
                torch.cuda.synchronize()
                for width,branch in branches.items():
                    ms=sum(a.elapsed_time(b) for a,b in branch['events']);branch['events']=[]
                    branch['train_ms']+=ms
                    avg=torch.stack(branch['losses']).mean(0).tolist();count=len(branch['losses']);branch['losses']=[]
                    timing={'seed':seed,'width':width,'update':update,'interval_updates':count,'interval_seconds':ms/1000,
                            'cumulative_train_seconds':branch['train_ms']/1000,'state_loss':avg[0],'bce':avg[1]}
                    timing_rows.append(timing)
                    with (args.output/'timings.jsonl').open('a') as f:f.write(json.dumps(timing)+'\n')
                    crossed=[budget for budget,seen in branch['time_targets'].items() if not seen and branch['train_ms']>=budget*1000]
                    for budget in crossed:branch['time_targets'][budget]=True
                    if update in milestones or crossed:
                        record(update,width,'fixed_update' if update in milestones else 'time_budget_'+','.join(map(str,crossed)))
                    elif update%600==0:
                        print(f'进度 seed={seed} width={width} [{update}/{args.updates}] state={avg[0]:.5f} BCE={avg[1]:.5f} train={branch["train_ms"]/1000:.1f}s',flush=True)
        for width,branch in branches.items():
            assert all(torch.equal(branch['model'].state_dict()[k],v.cuda()) for k,v in normalizer.items())
            torch.save({'format':'diagnostic_wm_width_only','width':width,'seed':seed,'updates':args.updates,
                        'model':branch['model'].state_dict(),'termination_head':branch['head'].state_dict(),
                        'opt_m':branch['om'].state_dict(),'opt_d':branch['od'].state_dict(),
                        'sampler_state':sampler.get_state(),'config':config},args.output/f'wm_width{width}_seed{seed}_final.pt')
        del branches,initial_head
        torch.cuda.empty_cache()
    result={'config':config,'evaluations':all_rows,'timings':timing_rows,'elapsed_seconds':time.monotonic()-started,
            'checks':{'both_networks_and_optimizers_initialized_from_scratch':True,'normalizers_unchanged':True,
                      'train_validation_environment_ids_disjoint':True,'identical_initial_predictions':True,
                      'same_training_minibatch_sequence_across_widths_and_seeds':True,'production_untouched':True}}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    print('DONE',args.output/'summary.json',flush=True)

if __name__=='__main__':main()
