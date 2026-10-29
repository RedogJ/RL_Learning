"""连续状态世界模型训练：真实采集、八步状态拟合、模型想象与 actor/critic。"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path

import torch
from .train import Replay, finite_grad
from .networks import MotionPolicy
from .state_model import Dynamics, StateValue, LAYOUT, build_state, obs_from_state
from .rewards import gravity_local, rotation_vector
from ...warp_env import EnvCfg, WarpGo2Env


def goal(env, frame, frame_limit):
    offsets = torch.arange(1, env.cfg.task.future_length+1, device=env.device)
    idx = torch.minimum(frame[:, None] + torch.floor(offsets*env.dt*env.ref.fps).long(), frame_limit[:, None])
    r = env.ref.sample(idx)
    return torch.cat((r['joints'], r['pos'], r['quat']), -1).flatten(1)


def soft_termination(env, state, frame):
    r = env.ref.sample(frame)
    height = (state[:,2]-r['pos'][:,2]).abs()
    tilt = (gravity_local(state[:,3:7])[:,2]-gravity_local(r['quat'])[:,2]).abs()
    # 解析平滑近似，仅用于想象中的可微存活权重。
    a = torch.sigmoid(40*(height-env.cfg.task.max_height_error))
    b = torch.sigmoid(40*(tilt-env.cfg.task.max_tilt_error))
    return 1-(1-a)*(1-b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--action-scale', type=float, default=1.0, help='动作到关节目标偏移的尺度，单位 rad')
    p.add_argument('--cycles', type=int, default=100)
    p.add_argument('--collect-steps', type=int, default=32)
    p.add_argument('--buffer-steps', type=int, default=320)
    p.add_argument('--wm-batch', type=int, default=128)
    p.add_argument('--actor-batch', type=int, default=128)
    p.add_argument('--wm-updates', type=int, default=50)
    p.add_argument('--warmup-cycles', type=int, default=20, help='先采集和拟合动力学，暂不更新 Actor/Critic')
    p.add_argument('--window', type=int, default=8)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--save-interval', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--motion', type=Path, default=Path('/workspace/go2_mujoco_mimic/data/kine2go/ai4_dog_canter/motion.npy'))
    p.add_argument('--run-name', default='state_world_model')
    p.add_argument('--resume', type=Path)
    p.add_argument('--actor-checkpoint', type=Path, help='只初始化旧版训练的 Actor；不加载旧世界模型')
    args = p.parse_args()
    if args.warmup_cycles < 0: p.error('warmup-cycles 不可为负')
    if not math.isfinite(args.action_scale) or args.action_scale <= 0:
        p.error('action-scale 必须是有限正数')
    if min(args.num_envs,args.cycles,args.collect_steps,args.wm_batch,args.actor_batch,args.wm_updates,args.window,args.horizon,args.save_interval) <= 0 or args.buffer_steps < max(args.window,args.horizon):
        p.error('参数应为正数且缓冲容量不小于窗口与想象长度')
    torch.manual_seed(args.seed); torch.backends.cuda.matmul.allow_tf32 = True
    cfg = EnvCfg(num_envs=args.num_envs, action_scale=args.action_scale)
    cfg.task.random_reference_start = True
    env = WarpGo2Env(cfg,args.motion); env.enable_cuda_graph()
    actor, model, critic = MotionPolicy().to(env.device), Dynamics().to(env.device), StateValue().to(env.device)
    opt_m = torch.optim.Adam(model.parameters(),lr=2e-4)
    opt_a = torch.optim.Adam(actor.parameters(),lr=2e-4,betas=(.9,.95))
    opt_v = torch.optim.Adam(critic.parameters(),lr=2e-4)
    start=0
    if args.resume and args.actor_checkpoint: p.error('--resume 与 --actor-checkpoint 不能同时使用')
    if args.actor_checkpoint:
        old = torch.load(args.actor_checkpoint,map_location=env.device,weights_only=False)
        if old.get('motion') != str(args.motion): raise ValueError('Actor 检查点参考动作不一致')
        actor.load_state_dict(old.get('policy', old.get('actor')))
        print('已加载旧 Actor；连续状态世界模型与 critic 从头初始化。', flush=True)
    if args.resume:
        ck = torch.load(args.resume,map_location=env.device,weights_only=False)
        if ck.get('format') != 'go2_core_normalized_v3': raise ValueError('需要 37 维 go2_core_normalized_v3 检查点；旧版未归一化世界模型不兼容')
        if ck['env'] != asdict(cfg) or ck['motion'] != str(args.motion): raise ValueError('续训配置或动作不同')
        actor.load_state_dict(ck['actor']);model.load_state_dict(ck['model']);critic.load_state_dict(ck['critic'])
        opt_a.load_state_dict(ck['opt_a']);opt_m.load_state_dict(ck['opt_m']);opt_v.load_state_dict(ck['opt_v']);start=ck['cycle']
    run=Path('/workspace/go2_mujoco_mimic/logs')/f'{args.run_name}_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}'
    run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps({'args':vars(args),'env':asdict(cfg),'state_size':LAYOUT.size,'context':'固定：地面摩擦 0.5、标定偏差/质心偏移为零'},default=str,ensure_ascii=False,indent=2))
    # 摩擦、12 个关节偏移、机身质心 xyz；当前 MuJoCo 未做 domain randomization。
    context=torch.zeros((args.num_envs,16),device=env.device);context[:,0]=0.5
    replay=Replay(args.buffer_steps,args.num_envs,env.device)
    state=build_state(env)
    print(f'连续状态={LAYOUT.size}维；环境={args.num_envs}；日志={run}；身体奖励由可微运动学计算；世界模型输入65维、输出37维',flush=True)
    print(f'缓冲装满后每轮新数据比例={min(1,args.collect_steps/args.buffer_steps):.1%}；Actor想象={args.actor_batch}条×{args.horizon}步',flush=True)
    for cycle in range(start,start+args.cycles):
        rewards=[]; fails=torch.zeros((),device=env.device,dtype=torch.long)
        for _ in range(args.collect_steps):
            with torch.no_grad():
                obs=env.observation(); frame=env.frame.clone(); history=env.history.clone(); previous=env.previous_action.clone()
                action=(actor(obs)+torch.randn_like(previous)*.1).clamp(-1,1)
                _,reward,terminated,truncated,_=env.step(action)
                next_state=build_state(env)
                done=terminated|truncated
                replay.add((state,history,frame,previous,action,next_state,done.float(),context,env.frame_limit.clone()))
                rewards.append(reward.mean());fails+=terminated.sum()
                env.reset(done)
                if done.any():
                    initial=build_state(env)
                    state=torch.where(done[:,None],initial,next_state)
                else: state=next_state
        if replay.count < max(args.window,args.horizon): continue
        if not bool(model.normalizer.ready):
            count = min(replay.count, replay.capacity)
            calibration = torch.cat((replay.storage[0][:count].flatten(0, 1),
                                     replay.storage[5][:count].flatten(0, 1)), 0)
            model.normalizer.calibrate(calibration)
            critic.normalizer.load_state_dict(model.normalizer.state_dict())
            (run/'normalization.json').write_text(json.dumps({
                'method':'first_collection_fixed_mean_std', 'samples':len(calibration),
                'mean':model.normalizer.mean.tolist(), 'scale':model.normalizer.scale.tolist()},indent=2))
            print('状态归一化已校准并冻结；state_loss 为归一化误差，每轮取全部 WM 更新的平均。',flush=True)
        wm_losses=[]
        for _ in range(args.wm_updates):
            states,hist,frames,previous,actions,targets,dones,contexts,limits=replay.sample(args.wm_batch,args.window)
            pred=states[:,0];live=torch.ones(args.wm_batch,device=env.device);loss=torch.zeros((),device=env.device)
            for t in range(args.window):
                with torch.autocast('cuda',dtype=torch.bfloat16): pred=model(pred,actions[:,t],contexts[:,t])
                step=((pred-targets[:,t])/model.normalizer.scale).square().mean(-1)
                loss+= (step*live).sum()/live.sum().clamp_min(1)
                live=live*(1-dones[:,t])
            loss/=args.window
            opt_m.zero_grad(set_to_none=True);loss.backward()
            if not finite_grad(model.parameters()): raise RuntimeError('连续状态模型梯度非有限值')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt_m.step()
            wm_losses.append(loss.detach())
        mean_wm_loss=torch.stack(wm_losses).mean().item()
        if cycle < args.warmup_cycles:
            m={'cycle':cycle+1,'real_reward':torch.stack(rewards).mean().item(),'failures':fails.item(),'state_loss':mean_wm_loss,'phase':'world_model_warmup'}
            with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(m)+'\n')
            print(f"[{cycle+1}] 预训练 reward={m['real_reward']:.4f} fallen={m['failures']} state_loss={m['state_loss']:.5f}",flush=True)
            if (cycle+1)%args.save_interval==0 or cycle+1==start+args.cycles:
                torch.save({'format':'go2_core_normalized_v3','cycle':cycle+1,'actor':actor.state_dict(),'model':model.state_dict(),'critic':critic.state_dict(),'opt_a':opt_a.state_dict(),'opt_m':opt_m.state_dict(),'opt_v':opt_v.state_dict(),'env':asdict(cfg),'motion':str(args.motion)},run/'latest.pt')
            continue
        states,hist,frames,previous,actions,targets,dones,contexts,limits=replay.sample(args.actor_batch,args.horizon)
        s=states[:,0].detach(); history=hist[:,0].detach(); frame=frames[:,0].clone(); prev=previous[:,0].detach();ctx=contexts[:,0];frame_limit=limits[:,0]
        survive=torch.ones(args.actor_batch,device=env.device);ret=torch.zeros_like(survive)
        ss=[];gg=[];rr=[];dd=[]
        for param in model.parameters():param.requires_grad_(False)
        for param in critic.parameters():param.requires_grad_(False)
        for t in range(args.horizon):
            g=goal(env,frame,frame_limit);ss.append(s);gg.append(g)
            action=actor(torch.cat((history.flatten(1),g),-1))
            with torch.autocast('cuda',dtype=torch.bfloat16): next_s=model(s,action,ctx)
            advance=math.floor((t+1)*env.dt*env.ref.fps)-math.floor(t*env.dt*env.ref.fps)
            nf=torch.minimum(frame+int(advance),frame_limit)
            target=env.ref.sample(nf); previous_target=env.ref.sample(frame)
            reference={'pos':target['body_pos'],'quat':target['body_quat'],
                       'linear_vel':(target['body_pos']-previous_target['body_pos'])/env.dt,
                       'angular_vel':rotation_vector(target['body_quat'],previous_target['body_quat'])/env.dt}
            reward,_=env.task.reward(env.kinematics.transition(s,next_s,env.dt),reference,action,prev,next_s[:,7:19],env.joint_limits,env.dt)
            failure=soft_termination(env,next_s,nf)
            end_of_reference = (nf >= frame_limit).float()
            failure = 1-(1-failure)*(1-end_of_reference)
            rr.append(reward);dd.append(failure)
            ret+=.99**t*survive*reward;survive=survive*(1-failure)
            new_obs=obs_from_state(next_s,action,env.default_q,noise=True)
            history=torch.cat((history[:,1:],new_obs[:,None]),1)
            s=next_s;frame=nf;prev=action
        last_goal=goal(env,frame,frame_limit)
        bootstrap=(.99**args.horizon)*survive*critic(s,last_goal,ctx)
        objective=ret+bootstrap
        opt_a.zero_grad(set_to_none=True);(-objective.mean()).backward()
        if not finite_grad(actor.parameters()):raise RuntimeError('策略梯度非有限值')
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
        for param in actor.parameters():
            if param.grad is not None:param.grad.clamp_(-.1,.1)
        opt_a.step()
        for param in model.parameters():param.requires_grad_(True)
        for param in critic.parameters():param.requires_grad_(True)
        with torch.no_grad():
            target_v=critic(s.detach(),last_goal,ctx)
            returns=[]
            for t in reversed(range(args.horizon)):
                vnext=critic(s.detach(),last_goal,ctx) if t==args.horizon-1 else critic(ss[t+1].detach(),gg[t+1],ctx)
                target_v=rr[t].detach()+.99*(1-dd[t].detach())*((1-.95)*vnext+.95*target_v)
                returns.append(target_v)
            returns=torch.stack(list(reversed(returns)),1).flatten()
        values=torch.stack([critic(x.detach(),g,ctx) for x,g in zip(ss,gg)],1).flatten()
        value_loss=(values-returns).square().mean();opt_v.zero_grad(set_to_none=True);value_loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);opt_v.step()
        m={'cycle':cycle+1,'real_reward':torch.stack(rewards).mean().item(),'failures':fails.item(),'state_loss':mean_wm_loss,'imagined_return':objective.mean().item(),'imagined_reward_sum':ret.mean().item(),'critic_bootstrap':bootstrap.mean().item(),'value_loss':value_loss.item()}
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(m)+'\n')
        print(f"[{cycle+1}] reward={m['real_reward']:.4f} fallen={m['failures']} state_loss={m['state_loss']:.5f} 想象累计奖励={m['imagined_reward_sum']:.4f} Critic尾项={m['critic_bootstrap']:.4f} 想象总目标={m['imagined_return']:.4f}",flush=True)
        if (cycle+1)%args.save_interval==0 or cycle+1==start+args.cycles:
            torch.save({'format':'go2_core_normalized_v3','cycle':cycle+1,'actor':actor.state_dict(),'model':model.state_dict(),'critic':critic.state_dict(),'opt_a':opt_a.state_dict(),'opt_m':opt_m.state_dict(),'opt_v':opt_v.state_dict(),'env':asdict(cfg),'motion':str(args.motion)},run/'latest.pt')


if __name__=='__main__':main()
