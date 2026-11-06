"""Measure saved historical WMs on fixed old/new policy trajectories; no updates."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import mujoco_warp as mjw
import torch
from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env, warp_stream_ordered
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.state_model import Dynamics, build_state
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--recent-data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    if (a.output/'history.json').exists():raise ValueError('Choose a new output directory')
    torch.backends.cuda.matmul.allow_tf32=True
    start=time.monotonic()
    ck=torch.load(a.run/'model_3270.pt',map_location='cpu',weights_only=False)
    cfg=load_config(ck);cfg.num_envs=384;cfg.task.observation_noise=True;cfg.task.random_reference_start=False
    env=WarpGo2Env(cfg,Path(ck['motion']));env.enable_cuda_graph()
    actor=MotionPolicy(history_length=cfg.task.history_length,future_length=cfg.task.future_length,
                       heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval()
    actor.load_state_dict(ck['actor']);actor.requires_grad_(False)
    first_frame=round(.6*env.ref.fps);steps=round(.8/env.dt)

    @warp_stream_ordered
    def reset(env):
        env.reset(torch.ones(384,device='cuda',dtype=torch.bool))
        env.start_frame.fill_(first_frame);env.frame.fill_(first_frame);env.steps.zero_();env.previous_action.zero_()
        env.qpos[:,:3]=env.ref.position[first_frame];env.qpos[:,3:7]=env.ref.quaternion[first_frame]
        env.qpos[:,env.qids]=env.ref.joints[first_frame];env.qvel[:]=env.ref.velocity[first_frame]
        env.previous_joint_vel.copy_(env.qvel[:,env.vids]);mjw.forward(env.model,env.data)
        env.history.copy_(env.proprioception()[:,None,:].expand_as(env.history))

    print('采集旧策略3270的固定探针轨迹：0.6–1.4s；随后交叉评估各历史WM，无参数更新。',flush=True)
    torch.manual_seed(1401);reset(env)
    states=[build_state(env).clone()];actions=[];dones=[];labels=[];frames=[env.frame.clone()];masks=[]
    alive=torch.ones(384,device='cuda',dtype=torch.bool)
    for _ in range(steps):
        action=(actor(env.observation())+torch.randn(384,12,device='cuda')*.1).clamp(-1,1)
        actions.append(action.clone());masks.append(alive.clone())
        _,_,failed,truncated,_=env.step(action)
        dones.append((failed|truncated).clone());labels.append(failed.float())
        states.append(build_state(env).clone());frames.append(env.frame.clone());alive &= ~(failed|truncated)
    ctx=torch.zeros(384,16,device='cuda');ctx[:,0]=.5
    old={'states':torch.stack(states),'actions':torch.stack(actions),'dones':torch.stack(dones),
         'labels':torch.stack(labels),'frames':torch.stack(frames),'masks':torch.stack(masks),
         'context':ctx,'frame_limit':env.frame_limit.clone()}
    torch.save({**{k:v.cpu() for k,v in old.items()},'env':asdict(cfg),'motion':ck['motion'],
                'train_envs':list(range(256)),'validation_envs':list(range(256,384)),
                'actor_checkpoint':str(a.run/'model_3270.pt')},a.output/'old_policy_3270_trajectories.pt')
    names=tuple(old)
    recent_source=torch.load(a.recent_data,map_location='cpu',weights_only=False)
    datasets={'old_policy_3270':old,'recent_policy_3330':{k:recent_source[k].cuda() for k in names}}
    del actor,ck,states,actions,dones,labels,frames,masks,recent_source

    def windows(data,length,lo,hi):
        ti=torch.arange(steps-length+1,device='cuda')[:,None].expand(-1,hi-lo).reshape(-1)
        ei=torch.arange(lo,hi,device='cuda')[None].expand(steps-length+1,-1).reshape(-1)
        valid=data['masks'][ti,ei]
        return ti[valid],ei[valid]

    probes={}
    for name,data in datasets.items():
        g=torch.Generator(device='cuda').manual_seed(7301)
        def subset(pair):
            ix=torch.randperm(len(pair[0]),device='cuda',generator=g)[:512]
            return pair[0][ix],pair[1][ix]
        # Match the prior independent trajectory split and RNG draw order exactly.
        subset(windows(data,8,0,256));p8=subset(windows(data,8,256,384))
        subset(windows(data,16,0,256));p16=subset(windows(data,16,256,384))
        probes[name]={'8':p8,'16':p16}
    torch.save({name:{k:tuple(x.cpu() for x in v) for k,v in pairs.items()} for name,pairs in probes.items()},a.output/'probe_indices.pt')
    out={'description':'Retrospective fixed old/new policy probe. Trajectories newly generated with saved actor 3270 and saved actor 3330, not original replay contents. Old/new independent validation uses env ids 256..383 only; same recorded actions and same windows for every WM. Cannot attribute any regression to buffer capacity alone.',
         'cycles':[3270,3280,3290,3300,3310,3320,3330],'segment_seconds':[.6,1.4],
         'no_updates':True,'records':[]}
    for cycle in out['cycles']:
        ck=torch.load(a.run/f'model_{cycle}.pt',map_location='cpu',weights_only=False)
        model=Dynamics(kinematics=env.kinematics if cfg.wm_foot_features else None).cuda().eval()
        model.load_state_dict(ck['model']);row={'wm_cycle':cycle,'datasets':{}}
        for name,data in datasets.items():
            st=data['states'];ac=data['actions'];dn=data['dones'];cx=data['context']
            ti,ei=probes[name]['8'];pred=st[ti,ei];live=torch.ones(len(ei),device='cuda');loss=live.new_zeros(())
            for t in range(8):
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(pred,ac[ti+t,ei],cx[ei])
                target=st[ti+t+1,ei].clone()
                target[:,3:7]=torch.where((pred[:,3:7]*target[:,3:7]).sum(-1,keepdim=True)<0,-target[:,3:7],target[:,3:7])
                err=((pred-target)/model.normalizer.scale).square().mean(-1)
                loss+=(err*live).sum()/live.sum().clamp_min(1);live*=~dn[ti+t,ei]
            result={'loss8':(loss/8).item(),'physical':{}}
            ti,ei=probes[name]['16'];pred=st[ti,ei];live=torch.ones(len(ei),device='cuda',dtype=torch.bool)
            for t in range(16):
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(pred,ac[ti+t,ei],cx[ei])
                if t+1 in (1,8,16):
                    error=(pred-st[ti+t+1,ei])[live]
                    angle=rotation_vector(pred[live,3:7],st[ti+t+1,ei][live,3:7]).norm(dim=-1)
                    rms=lambda lo,hi:error[:,lo:hi].square().mean().sqrt().item()
                    result['physical'][t+1]={'samples':len(error),
                        'root_position_rms_cm':error[:,:3].square().sum(-1).mean().sqrt().item()*100,
                        'root_rotation_rms_deg':torch.rad2deg(angle.square().mean().sqrt()).item(),
                        'joint_position_rmse_deg':math.degrees(rms(7,19)),
                        'linear_velocity_rmse_m_s':rms(19,22),'angular_velocity_rmse_rad_s':rms(22,25),
                        'joint_velocity_rmse_rad_s':rms(25,37)}
                live &= ~dn[ti+t,ei]
            row['datasets'][name]=result
        out['records'].append(row)
        with (a.output/'history_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('HISTORY',cycle,json.dumps({k:v['physical'][16] for k,v in row['datasets'].items()}),flush=True)
        del model,ck
    out['elapsed_seconds']=time.monotonic()-start
    (a.output/'history.json').write_text(json.dumps(out,indent=2))
    print('DONE',a.output/'history.json',flush=True)

if __name__=='__main__':main()
