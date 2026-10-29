"""同一起点比较闭环想象与真实仿真，并核对 Critic 的有限回合折扣回报。"""
import argparse
from pathlib import Path
import json
import math
import torch
from ...play_policy import load_config
from ...warp_env import WarpGo2Env
from .networks import MotionPolicy
from .state_model import Dynamics, StateValue, build_state, obs_from_state
from .state_train import goal, soft_termination
from .rewards import rotation_vector


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=32)
    p.add_argument('--horizon',type=int,default=16)
    p.add_argument('--seed',type=int,default=123)
    args=p.parse_args()
    if min(args.num_envs,args.horizon)<=0:p.error('环境数和 horizon 必须为正')
    torch.manual_seed(args.seed)
    ck=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    if ck.get('format')!='go2_core_normalized_v3':raise ValueError('需要归一化 37 维检查点')
    cfg=load_config(ck);cfg.num_envs=args.num_envs;cfg.task.observation_noise=False
    env=WarpGo2Env(cfg,Path(ck['motion']));env.enable_cuda_graph()
    actor=MotionPolicy().cuda().eval();actor.load_state_dict(ck['actor'])
    model=Dynamics().cuda().eval();model.load_state_dict(ck['model'])
    critic=StateValue().cuda().eval();critic.load_state_dict(ck['critic'])
    s=build_state(env);history=env.history.clone();frame=env.frame.clone();limit=env.frame_limit.clone()
    prev=env.previous_action.clone();ctx=torch.zeros(args.num_envs,16,device=env.device);ctx[:,0]=.5
    initial_value=critic(s,goal(env,frame,limit),ctx)
    survive=torch.ones(args.num_envs,device=env.device);alive=survive.bool()
    imagined=torch.zeros_like(survive);short_real=torch.zeros_like(survive);full_real=torch.zeros_like(survive)
    errors=[];action_errors=[];failures=0
    print(f'评估 cycle={ck["cycle"]}；同起点、无观测/动作噪声；{args.num_envs}环境，预测{args.horizon}步',flush=True)
    for t in range(max(args.horizon,math.ceil(cfg.episode_seconds/env.dt)+1)):
        action=actor(env.observation())
        if t<args.horizon:
            ia=actor(torch.cat((history.flatten(1),goal(env,frame,limit)),-1))
            with torch.autocast('cuda',dtype=torch.bfloat16):ns=model(s,ia,ctx)
            advance=math.floor((t+1)*env.dt*env.ref.fps)-math.floor(t*env.dt*env.ref.fps)
            nf=torch.minimum(frame+advance,limit)
            a=env.ref.sample(nf);b=env.ref.sample(frame)
            ref={'pos':a['body_pos'],'quat':a['body_quat'],'linear_vel':(a['body_pos']-b['body_pos'])/env.dt,
                 'angular_vel':rotation_vector(a['body_quat'],b['body_quat'])/env.dt}
            r,_=env.task.reward(env.kinematics.transition(s,ns,env.dt),ref,ia,prev,ns[:,7:19],env.joint_limits,env.dt)
            d=1-(1-soft_termination(env,ns,nf))*(nf<limit).float()
            imagined+=.99**t*survive*r;survive*=1-d
            history=torch.cat((history[:,1:],obs_from_state(ns,ia,env.default_q)[:,None]),1)
            s,frame,prev=ns,nf,ia
        _,reward,terminated,truncated,_=env.step(action)
        full_real+=.99**t*alive*reward
        if t<args.horizon:
            short_real+=.99**t*alive*reward
            real=build_state(env)
            if alive.any():
                errors.append((s[alive,:3]-real[alive,:3]).square().mean().sqrt().item())
                action_errors.append((ia[alive]-action[alive]).square().mean().sqrt().item())
        failures+=int((alive&terminated).sum())
        alive &= ~(terminated|truncated)
        if not alive.any() and t+1>=args.horizon:break
    result={'cycle':ck['cycle'],'real_short_return':short_real.mean().item(),
            'imagined_short_return_soft_termination':imagined.mean().item(),
            'initial_critic_value':initial_value.mean().item(),'real_episode_discounted_return':full_real.mean().item(),
            'critic_mae':(initial_value-full_real).abs().mean().item(),
            'failed_envs':failures,'root_position_rmse_by_step':errors,'action_rmse_by_step':action_errors}
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    print('短期预测使用软终止、真实执行使用硬终止；闭环误差包含动作分叉。这里是无噪声评估，非训练分布上的无偏价值检验。',flush=True)


if __name__=='__main__':main()
