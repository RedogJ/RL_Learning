import math, json, torch
from pathlib import Path
from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.state_model import Dynamics,StateValue,TerminationHead,build_state,obs_from_state
from go2_mujoco_mimic.tasks.mimic.state_train import goal,termination_probability,policy_input
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector,estimate_torque_squared

@torch.no_grad()
def main():
    folder=Path('/workspace/go2_mujoco_mimic/logs/backflip_v19_bce_2026-11-05_04-25-36_439009')
    ck=torch.load(folder/'model_3280.pt',map_location='cuda',weights_only=False)
    old=torch.load(folder/'model_3270.pt',map_location='cpu',weights_only=False)['actor']
    cfg=load_config(ck);cfg.num_envs=96;cfg.task.observation_noise=False
    cfg.task.random_reference_start=True
    env=WarpGo2Env(cfg,Path(ck['motion']));env.enable_cuda_graph()
    actor=MotionPolicy(heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval()
    base=ck['actor']; diff={k:base[k]-old[k].to('cuda') for k in base}
    model=Dynamics(kinematics=env.kinematics if cfg.wm_foot_features else None).cuda().eval();model.load_state_dict(ck['model'])
    head=TerminationHead(cfg.task.future_length*19).cuda().eval();head.load_state_dict(ck['termination_head'])
    critic=StateValue().cuda().eval();critic.load_state_dict(ck['critic'])
    print('cycle=3280; direction=3280-3270; 96 reference resets, no noise; base actor continuation after 16; fixed WM/head/critic',flush=True)
    outputs={}
    for seed in (123,456):
      for alpha in (-0.25,0.,0.25):
        torch.manual_seed(seed);env.reset(torch.ones(cfg.num_envs,device=env.device,dtype=torch.bool))
        actor.load_state_dict({k:base[k]+alpha*diff[k] for k in base})
        s=build_state(env);history=env.history.clone();frame=env.frame.clone();limit=env.frame_limit.clone();prev=env.previous_action.clone()
        ctx=torch.zeros(cfg.num_envs,16,device=env.device);ctx[:,0]=.5
        survive=torch.ones(cfg.num_envs,device=env.device);alive=survive.bool()
        short=torch.zeros_like(survive); penalty=torch.zeros_like(survive);realshort=torch.zeros_like(survive);real=torch.zeros_like(survive)
        failures=torch.zeros_like(alive);realcritic=torch.zeros_like(survive);alive16=torch.zeros_like(survive)
        for t in range(max(16,math.ceil(cfg.episode_seconds/env.dt)+1)):
            if t == 16: actor.load_state_dict(base)
            action=actor(env.observation())
            if t<16:
                ia=actor(policy_input(env,s,history,frame,limit))
                with torch.autocast('cuda',dtype=torch.bfloat16): ns=model(s,ia,ctx)
                adv=math.floor((t+1)*env.dt*env.ref.fps)-math.floor(t*env.dt*env.ref.fps)
                nf=torch.minimum(frame+adv,limit);a=env.ref.sample(nf);b=env.ref.sample(frame)
                ref={'pos':a['body_pos'],'quat':a['body_quat'],'linear_vel':(a['body_pos']-b['body_pos'])/env.dt,'joint_pos':a['joints'],'angular_vel':rotation_vector(a['body_quat'],b['body_quat'])/env.dt}
                tq=env.default_q+env.action_scales*ia
                if cfg.clamp_joint_targets:tq=tq.clamp(env.joint_limits[:,0],env.joint_limits[:,1])
                torque=None
                if cfg.task.reward.torque_l2_weight and cfg.task.reward.torque_mode=='substep_mean':
                    torque=estimate_torque_squared(s[:,7:19],s[:,25:37],ns[:,7:19],ns[:,25:37],tq,cfg.kp,cfg.kd,cfg.torque_limit,env.substeps)
                reward,_=env.task.reward(env.kinematics.transition(s,ns,env.dt),ref,ia,prev,ns[:,7:19],env.joint_limits,env.dt,joint_vel=ns[:,25:37],prev_joint_vel=s[:,25:37],target=tq,torque_squared_mean=torque,kp=cfg.kp,kd=cfg.kd,torque_limit=cfg.torque_limit)
                f=termination_probability(env,model,head,ns,nf,limit)
                short+=.99**t*survive*reward
                penalty+=.99**t*survive*cfg.task.reward.failure_penalty*f
                survive*=((1-f)*(nf<limit).float())
                history=torch.cat((history[:,1:],obs_from_state(ns,ia,env.default_q)[:,None]),1)
                s,frame,prev=ns,nf,ia
            _,r,terminated,truncated,_=env.step(action)
            real+=.99**t*alive*r
            if t<16:realshort+=.99**t*alive*r
            failures|=alive&terminated
            alive &= ~(terminated|truncated)
            if t==15:
                alive16=alive.float().clone()
                realcritic=.99**16*alive16*critic(build_state(env),goal(env,env.frame,limit),ctx)
            if t>=15 and not alive.any():break
        tail=.99**16*survive*critic(s,goal(env,frame,limit),ctx)
        obj=short-penalty+tail
        vals={'objective':obj,'imag_reward':short,'imag_failure_penalty':penalty,'imag_tail':tail,'real_short':realshort,'real_tail':real-realshort,'real_return':real,'real_critic_tail':realcritic,'completion':(~failures).float(),'real_fail':failures.float(),'soft_survival':survive,'alive16':alive16}
        outputs[f'{seed}:{alpha}']={k:v.cpu().tolist() for k,v in vals.items()}
        print(json.dumps({'seed':seed,'alpha':alpha,**{k:round(v.float().mean().item(),5) for k,v in vals.items()}},ensure_ascii=False),flush=True)
    with open('/tmp/go2_direction_audit_fixed_continuation.json','w') as f:json.dump(outputs,f)
if __name__=='__main__':main()
