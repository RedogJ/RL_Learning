"""同一起点比较闭环想象与真实仿真，并核对 Critic 的有限回合折扣回报。"""
import argparse
from pathlib import Path
import json
import math
import torch
from ...play_policy import load_config
from ...warp_env import WarpGo2Env
from .networks import MotionPolicy
from .state_model import Dynamics, StateValue, TerminationHead, build_state, obs_from_state, critic_remaining_time
from .state_train import goal, termination_probability, policy_input
from .rewards import rotation_vector, estimate_torque_squared


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=32)
    p.add_argument('--horizon',type=int,default=16)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--same-actions',action='store_true',help='世界模型逐步接收真实仿真 Actor 的动作，排除闭环动作分叉；不更新任何模型')
    args=p.parse_args()
    if min(args.num_envs,args.horizon)<=0:p.error('环境数和 horizon 必须为正')
    torch.manual_seed(args.seed)
    ck=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    if ck.get('format')!='go2_core_normalized_v3':raise ValueError('需要归一化 37 维检查点')
    cfg=load_config(ck);cfg.num_envs=args.num_envs;cfg.task.observation_noise=False
    env=WarpGo2Env(cfg,Path(ck['motion']));env.enable_cuda_graph()
    actor=MotionPolicy(heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval();actor.load_state_dict(ck['actor'])
    model=Dynamics(kinematics=env.kinematics if cfg.wm_foot_features else None).cuda().eval();model.load_state_dict(ck['model'])
    termination_head=None
    if ck.get('termination_head') is not None:
        termination_head=TerminationHead(cfg.task.future_length*19).cuda().eval()
        termination_head.load_state_dict(ck['termination_head'])
    # 无配置元数据的旧 checkpoint 使用旧 243 维 Critic，保持原始估值。
    critic=StateValue(**ck.get('critic_config',{})).cuda().eval();critic.load_state_dict(ck['critic'])
    s=build_state(env);history=env.history.clone();frame=env.frame.clone();limit=env.frame_limit.clone()
    prev=env.previous_action.clone();ctx=torch.zeros(args.num_envs,16,device=env.device);ctx[:,0]=.5
    initial_value=critic(s,goal(env,frame,limit),ctx,history,critic_remaining_time(env,frame,limit))
    survive=torch.ones(args.num_envs,device=env.device);alive=survive.bool()
    imagined=torch.zeros_like(survive);short_real=torch.zeros_like(survive);full_real=torch.zeros_like(survive)
    errors=[];action_errors=[];failures=0
    prediction_errors=[]
    failure_scores=[];failure_labels=[];true_state_scores=[]
    print(f'评估 cycle={ck["cycle"]}；同起点、无观测/动作噪声；{args.num_envs}环境，预测{args.horizon}步',flush=True)
    print('模式：'+('同一动作序列，检查动力学多步误差' if args.same_actions else '闭环想象，包含动作分叉'),flush=True)
    print('终止：'+('学习式 BCE' if termination_head is not None else '解析 sigmoid'),flush=True)
    for t in range(max(args.horizon,math.ceil(cfg.episode_seconds/env.dt)+1)):
        action=actor(env.observation())
        if termination_head is not None:
            real_before=build_state(env)
        if t<args.horizon:
            ia=action if args.same_actions else actor(policy_input(env,s,history,frame,limit))
            with torch.autocast('cuda',dtype=torch.bfloat16):ns=model(s,ia,ctx)
            advance=math.floor((t+1)*env.dt*env.ref.fps)-math.floor(t*env.dt*env.ref.fps)
            nf=torch.minimum(frame+advance,limit)
            a=env.ref.sample(nf);b=env.ref.sample(frame)
            ref={'pos':a['body_pos'],'quat':a['body_quat'],'linear_vel':(a['body_pos']-b['body_pos'])/env.dt,
                 'joint_pos':a['joints'],
                 'angular_vel':rotation_vector(a['body_quat'],b['body_quat'])/env.dt}
            tq=env.default_q+env.action_scales*ia
            if env.cfg.clamp_joint_targets: tq=tq.clamp(env.joint_limits[:,0],env.joint_limits[:,1])
            torque_sq = None
            if cfg.task.reward.torque_l2_weight and cfg.task.reward.torque_mode == 'substep_mean':
                torque_sq=estimate_torque_squared(s[:,7:19],s[:,25:37],ns[:,7:19],ns[:,25:37],
                                                  tq,cfg.kp,cfg.kd,cfg.torque_limit,env.substeps)
            r,_=env.task.reward(env.kinematics.transition(s,ns,env.dt),ref,ia,prev,ns[:,7:19],env.joint_limits,env.dt,
                                joint_vel=ns[:,25:37],prev_joint_vel=s[:,25:37],target=tq,torque_squared_mean=torque_sq,
                                kp=cfg.kp,kd=cfg.kd,torque_limit=cfg.torque_limit)
            failure=termination_probability(env,model,termination_head,ns,nf,limit)
            r=r-env.cfg.task.reward.failure_penalty*failure
            d=1-(1-failure)*(nf<limit).float()
            imagined+=.99**t*survive*r;survive*=1-d
            history=torch.cat((history[:,1:],obs_from_state(ns,ia,env.default_q)[:,None]),1)
            s,frame,prev=ns,nf,ia
        _,reward,terminated,truncated,_=env.step(action)
        if termination_head is not None:
            # 独立仿真验证，按每步真实起点预测，不混入长闭环动作分叉。
            with torch.autocast('cuda',dtype=torch.bfloat16):pred_next=model(real_before,action,ctx)
            prob=termination_probability(env,model,termination_head,pred_next,env.frame,limit)
            prob_true=termination_probability(env,model,termination_head,build_state(env),env.frame,limit)
            failure_scores.append(prob[alive]);true_state_scores.append(prob_true[alive])
            failure_labels.append(terminated[alive].float())
        full_real+=.99**t*alive*reward
        if t<args.horizon:
            short_real+=.99**t*alive*reward
            real=build_state(env)
            if alive.any():
                errors.append((s[alive,:3]-real[alive,:3]).square().mean().sqrt().item())
                action_errors.append((ia[alive]-action[alive]).square().mean().sqrt().item())
                prediction_errors.append({
                    'step':t+1, 'samples':int(alive.sum()),
                    'root_position_rmse_m':errors[-1],
                    'root_rotation_rms_deg':torch.rad2deg(rotation_vector(s[alive,3:7],real[alive,3:7]).square().sum(-1).mean().sqrt()).item(),
                    'joint_position_rmse_rad':(s[alive,7:19]-real[alive,7:19]).square().mean().sqrt().item(),
                    'root_linear_velocity_rmse_m_s':(s[alive,19:22]-real[alive,19:22]).square().mean().sqrt().item(),
                    'root_angular_velocity_rmse_rad_s':(s[alive,22:25]-real[alive,22:25]).square().mean().sqrt().item(),
                    'joint_velocity_rmse_rad_s':(s[alive,25:37]-real[alive,25:37]).square().mean().sqrt().item(),
                })
        failures+=int((alive&terminated).sum())
        alive &= ~(terminated|truncated)
        if not alive.any() and t+1>=args.horizon:break
    result={'cycle':ck['cycle'],'real_short_return':short_real.mean().item(),
            'imagined_short_return_soft_termination':imagined.mean().item(),
            'initial_critic_value':initial_value.mean().item(),'real_episode_discounted_return':full_real.mean().item(),
            'critic_mae':(initial_value-full_real).abs().mean().item(),
            'failed_envs':failures,'root_position_rmse_by_step':errors,'action_rmse_by_step':action_errors,
            'prediction_mode':'same_actions' if args.same_actions else 'closed_loop',
            'prediction_errors_by_step':prediction_errors}
    if termination_head is not None:
        labels=torch.cat(failure_labels);positive=labels.bool();negative=~positive
        result['termination_validation']={'samples':labels.numel(),'failures':int(positive.sum()),
                                         'normal_steps':int(negative.sum())}
        for name,items in [('predicted_states',failure_scores),('true_states',true_state_scores)]:
            prob=torch.cat(items);classified=prob>=.5
            result['termination_validation'][name]={
                'bce':torch.nn.functional.binary_cross_entropy(prob.clamp(1e-7,1-1e-7),labels).item(),
                'brier':(prob-labels).square().mean().item(),
                'recall':classified[positive].float().mean().item() if positive.any() else None,
                'false_positive_rate':classified[negative].float().mean().item() if negative.any() else None,
                'mean_probability_on_failures':prob[positive].mean().item() if positive.any() else None,
                'mean_probability_on_normal_steps':prob[negative].mean().item() if negative.any() else None}
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    print('短期预测使用软终止、真实执行使用硬终止；同动作模式排除动作分叉，闭环模式包含动作分叉。误差仅统计真实轨迹尚存活的环境；这是独立仿真诊断，不是训练 loss，也不是训练分布上的无偏价值检验。',flush=True)


if __name__=='__main__':main()
