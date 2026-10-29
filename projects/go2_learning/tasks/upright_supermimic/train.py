"""滚动世界模型训练。仅在真实仿真验证通过后启用想象策略更新。"""
import argparse, json, os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from tensordict import TensorDict
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from rsl_rl.runners import OnPolicyRunner
from go2_learning.paths import LOG_ROOT
from go2_learning.tasks.upright_supermimic.configuration import task_from_snapshot
from go2_learning.training.checkpoints import resolve_checkpoint
from go2_learning.tasks.upright_supermimic.dynamics import Dynamics, Critic
from go2_learning.tasks.upright_supermimic.sequence_buffer import SequenceBuffer
from go2_learning.tasks.upright_supermimic.state_bridge import observe, analytical_reward, enforce_action_state, enforce_context, POLICY_DIM, STATE_DIM
from go2_learning.tasks.upright_supermimic.reward_cfg import REWARD_OVERRIDES
from go2_learning.tasks.upright.reward_cfg import RewardCfg
from dataclasses import fields


def td(x): return TensorDict({'policy': x}, batch_size=[x.shape[0]])


def state_error(pred, target, std):
    # 随机推力未来会重新抽样，不将其当作可由确定性模型预测的机器人状态。
    error=((pred-target)/std).square()
    return torch.cat((error[...,:80],error[...,88:]),-1).mean()


def wm_updates(model, optimizer, buffer, ids, batch, horizon, iterations, failure_penalty, reward_scale, alpha, dt):
    for _ in range(iterations):
        obs, action, next_obs = buffer.windows(batch, horizon, ids)
        predicted = obs[:, 0]; dyn_loss = 0.
        for k in range(horizon):
            nxt, _, _ = model(predicted, action[:, k])
            predicted = enforce_action_state(nxt, predicted, action[:, k], alpha)
            predicted = enforce_context(predicted, obs[:,k], dt, next_obs[:,k])
            dyn_loss = dyn_loss + state_error(predicted,next_obs[:,k],model.delta_std)
        o,a,n,r,fail,done,_,_ = buffer.transitions(batch,ids)
        _, reward_hat, fail_logit = model(o,a)
        nonterminal = (r + fail.float()*failure_penalty) / reward_scale
        loss = dyn_loss/horizon + F.mse_loss(reward_hat,nonterminal) + F.binary_cross_entropy_with_logits(fail_logit,fail.float())
        if not torch.isfinite(loss):
            raise RuntimeError('世界模型损失非有限数，停止以保护已保存的检查点')
        optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    return float(loss.detach())


def validation(model, buffer, train_ids, ids, batch, horizon, failure_penalty, reward_scale, env):
    with torch.no_grad():
        obs, actions, actual = buffer.windows(batch,horizon,ids)
        pred=obs[:,0]; baseline=pred; errors=[]; baselines=[]
        for k in range(horizon):
            nxt,_,_ = model(pred,actions[:,k])
            pred = enforce_action_state(nxt, pred, actions[:,k], env.action_filter_alpha)
            pred = enforce_context(pred, obs[:,k], env.dt, actual[:,k])
            errors.append(state_error(pred,actual[:,k],model.delta_std).item())
            baselines.append(state_error(baseline,actual[:,k],model.delta_std).item())
        o,a,n,r,fail,done,remain,next_remain=buffer.transitions(batch,ids)
        _,_,_,train_r,train_fail,_,_,_=buffer.transitions(max(batch,1024),train_ids)
        _,rw,fl=model(o,a)
        target=(r + fail.float()*failure_penalty)/reward_scale
        predicted_next,_,_=model(o,a)
        predicted_next=enforce_action_state(predicted_next,o,a,env.action_filter_alpha)
        analytical=analytical_reward(predicted_next,a,env,o)/reward_scale
        normal = ~done
        analytical_mae=(analytical[normal]-target[normal]).abs().mean().item() if normal.any() else float('inf')
        exact=analytical_reward(n,a,env,o)/reward_scale
        parity_mae=(exact[normal]-target[normal]).abs().mean().item() if normal.any() else float('inf')
        valid_slots=buffer.active[:,ids]
        unique_failures=(buffer.fail[:,ids] & valid_slots).sum().item()
        train_target=(train_r + train_fail.float()*failure_penalty)/reward_scale
        prior=train_fail.float().mean()
        probability=fl.sigmoid()
        positives=probability[fail]; negatives=probability[~fail]
        separation=(positives[:,None] > negatives[None,:]).float().mean().item() if len(positives) and len(negatives) else 0.0
        return {'wm_one_step_mse':errors[0], 'wm_horizon_mse':errors[-1], 'baseline_horizon_mse':baselines[-1],
                'reward_mae_scaled':(rw-target).abs().mean().item(), 'failure_bce':F.binary_cross_entropy_with_logits(fl,fail.float()).item(),
                'reward_baseline_mae_scaled':(train_target.mean()-target).abs().mean().item(),
                'analytical_reward_mae_scaled':analytical_mae,
                'reward_parity_mae_scaled':parity_mae,
                'failure_brier':(probability-fail.float()).square().mean().item(),
                'failure_baseline_brier':(prior-fail.float()).square().mean().item(),
                'validation_failures_sampled':int(fail.sum().item()),
                'validation_unique_failure_events':int(unique_failures),
                'failure_auc_sampled':separation,
                'failure_probability_mean':probability.mean().item()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='latest',help='upright 专家模型')
    p.add_argument('--resume',default=None,help='本实验的训练检查点，重新建立仿真并恢复缓冲及优化器')
    p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--cycles',type=int,default=1000,help='本次额外运行的循环次数')
    p.add_argument('--collect-steps',type=int,default=40)
    p.add_argument('--wm-updates',type=int,default=50)
    p.add_argument('--warmup-updates',type=int,default=300)
    p.add_argument('--actor-updates',type=int,default=1)
    p.add_argument('--critic-updates',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--horizon',type=int,default=5)
    p.add_argument('--replay-steps',type=int,default=512)
    p.add_argument('--noise',type=float,default=.08)
    p.add_argument('--actor-lr',type=float,default=None,help='Actor 学习率；续训默认沿用检查点')
    p.add_argument('--teacher-weight',type=float,default=1.0)
    p.add_argument('--max-wm-mse',type=float,default=1.0,help='Actor 更新允许的多步归一化误差上限')
    p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--reward-scale',type=float,default=1000.)
    p.add_argument('--save-interval',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--randomization',action=argparse.BooleanOptionalAction,default=True)
    add_launcher_args(p); args=p.parse_args()
    if args.num_envs < 4 or min(args.cycles,args.collect_steps,args.wm_updates,args.warmup_updates,args.actor_updates,args.critic_updates,args.batch_size,args.horizon,args.save_interval) <= 0 or args.replay_steps < args.horizon or args.noise < 0 or not 0 < args.gamma <= 1 or args.reward_scale <= 0:
        p.error('环境数至少为 4；采样、训练和折扣参数必须有效')
    torch.manual_seed(args.seed)
    resume_data=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    if args.actor_lr is None:
        args.actor_lr=resume_data['args']['actor_lr'] if resume_data else 1e-6
    if args.actor_lr <= 0: p.error('actor-lr 必须为正数')
    ckpt=Path(resume_data['teacher']) if resume_data else resolve_checkpoint(args.checkpoint,'upright',LOG_ROOT)
    saved_cfg=(resume_data.get('task_config') if resume_data else None) or json.loads(ckpt.with_name('run.json').read_text())
    if not resume_data:
        saved_cfg=deepcopy(saved_cfg)
        unknown=set(REWARD_OVERRIDES)-{field.name for field in fields(RewardCfg)}
        if unknown: raise ValueError(f'未知奖励配置: {sorted(unknown)}')
        saved_cfg['reward_cfg'].update(REWARD_OVERRIDES)
    task=task_from_snapshot(saved_cfg)
    if resume_data:
        previous=resume_data['args']
        for k in ('num_envs','replay_steps','horizon','randomization','reward_scale','gamma'):
            if getattr(args,k) != previous[k]: raise ValueError(f'续训参数 {k} 必须与检查点一致')
    sim_cfg=SimulationCfg(dt=task.cfg.environment.physics_dt,physics=PhysxCfg())
    with launch_simulation(sim_cfg,args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        sim=SimulationContext(sim_cfg)
        try:
            env=make_env(sim,args.num_envs,task.cfg.randomization if args.randomization else None,args.seed,task=task)
            if list(env.robot.joint_names) != saved_cfg['joint_names']: raise ValueError('关节顺序不匹配')
            for k,v in [('default_joint_pos',env.default_joint_pos[0]),('action_joint_limits',env.action_joint_limits[0])]:
                if not torch.allclose(torch.tensor(saved_cfg[k],device=env.device),v,atol=1e-6,rtol=0): raise ValueError(f'{k} 不匹配')
            env.diagnostics_interval=24; env.detailed_diagnostics=False
            env.finish_initial_settling()
            runner=OnPolicyRunner(env=env,train_cfg=task.cfg.agent,log_dir=None,device=str(sim.device))
            if not resume_data or 'teacher_actor' not in resume_data:
                runner.load(str(ckpt),map_location=str(sim.device))
            actor=runner.get_inference_policy(device=str(sim.device)); teacher=deepcopy(actor).eval()
            if resume_data and 'teacher_actor' in resume_data:
                teacher.load_state_dict(resume_data['teacher_actor'])
            for q in teacher.parameters(): q.requires_grad_(False)
            obs=env.get_observations(); sim_state=observe(env,obs['policy']); dim=sim_state.shape[-1]
            if dim != STATE_DIM: raise ValueError(f'世界模型状态维数不符：{dim} != {STATE_DIM}')
            model=Dynamics(dim,env.num_actions).to(sim.device)
            critic=Critic(dim).to(sim.device)
            opt_m=torch.optim.Adam(model.parameters(),lr=3e-4)
            opt_a=torch.optim.Adam(actor.parameters(),lr=args.actor_lr)
            opt_c=torch.optim.Adam(critic.parameters(),lr=3e-4)
            buffer=SequenceBuffer(args.replay_steps,args.num_envs,dim,env.num_actions,sim.device)
            train_count=max(1,min(int(args.num_envs*.6),args.num_envs-3))
            eval_count=max(1,(args.num_envs-train_count)//4)
            val_end=args.num_envs-2*eval_count
            train_ids=torch.arange(train_count,device=sim.device)
            val_ids=torch.arange(train_count,val_end,device=sim.device)
            teacher_ids=torch.arange(val_end,val_end+eval_count,device=sim.device)
            student_ids=torch.arange(val_end+eval_count,args.num_envs,device=sim.device)
            train_mask=torch.zeros(args.num_envs,dtype=torch.bool,device=sim.device);train_mask[train_ids]=True
            val_mask=torch.zeros_like(train_mask);val_mask[val_ids]=True
            teacher_mask=torch.zeros_like(train_mask);teacher_mask[teacher_ids]=True
            student_mask=torch.zeros_like(train_mask);student_mask[student_ids]=True
            start=0
            if resume_data:
                model.load_state_dict(resume_data['model']);actor.load_state_dict(resume_data['actor']);critic.load_state_dict(resume_data['critic'])
                opt_m.load_state_dict(resume_data['opt_m']);opt_a.load_state_dict(resume_data['opt_a']);opt_c.load_state_dict(resume_data['opt_c'])
                for group in opt_a.param_groups: group['lr']=args.actor_lr
                for name, value in resume_data['buffer'].items():
                    if torch.is_tensor(value): getattr(buffer,name).copy_(value.to(sim.device))
                    else: setattr(buffer,name,value)
                # 重启仿真与旧进程末尾不是连续轨迹；插入无效屏障，禁止窗口跨进程拼接。
                zero=torch.zeros(args.num_envs,device=sim.device)
                barrier=torch.zeros(args.num_envs,dtype=torch.bool,device=sim.device)
                buffer.add(sim_state,env.standing_actions,sim_state,zero,barrier,barrier,barrier,zero,zero)
                torch.set_rng_state(resume_data['rng']); torch.cuda.set_rng_state_all(resume_data['cuda_rng'])
                start=resume_data['cycle']; print('续训恢复模型、优化器和经验缓冲；仿真环境重新初始化。',flush=True)
            run=LOG_ROOT/'upright_supermimic'/('train_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f'))
            run.mkdir(parents=True)
            (run/'config.json').write_text(json.dumps({'args':vars(args),'teacher':str(ckpt),'task_config':saved_cfg},ensure_ascii=False,indent=2,default=str))
            print(f'输出：{run}；教师：{ckpt}；世界模型奖励与失败为预测，需看真实评估',flush=True)
            failure_penalty=task.cfg.rewards.failure_penalty
            for cycle in range(start,start+args.cycles):
                falls=0; active_count=0; reward_sum=0.
                eval_totals=torch.zeros((2,3),device=sim.device)
                with torch.no_grad():
                    for _ in range(args.collect_steps):
                        active=env.policy_active.clone(); old=sim_state.clone()
                        remaining=(env.max_episode_length-env.episode_length_buf).clamp_min(0)/env.max_episode_length
                        actions=env.standing_actions.clone()
                        noisy=active & (train_mask | val_mask)
                        expert_eval=active & teacher_mask
                        student_eval=active & student_mask
                        if noisy.any(): actions[noisy]=actor(obs[noisy])+args.noise*torch.randn_like(actions[noisy])
                        if expert_eval.any(): actions[expert_eval]=teacher(obs[expert_eval])
                        if student_eval.any(): actions[student_eval]=actor(obs[student_eval])
                        obs,reward,done,extras=env.step(actions)
                        failure=done.bool() & ~extras['time_outs'].bool()
                        sim_state=observe(env,obs['policy'])
                        next_remaining=(remaining-1/env.max_episode_length).clamp_min(0)
                        buffer.add(old,actions,sim_state,reward,failure,done.bool(),active,remaining,next_remaining)
                        falls+=int(failure.sum().item()); active_count+=int(active.sum().item()); reward_sum+=float(reward.sum().item())
                        for j,group in enumerate((expert_eval,student_eval)):
                            eval_totals[j,0]+=group.sum();eval_totals[j,1]+=(failure & group).sum();eval_totals[j,2]+=reward[group].sum()
                if buffer.count < args.horizon: print('等待缓冲中连续样本',flush=True); continue
                if cycle==0 and not resume_data:
                    o,a,n=buffer.windows(min(4096,args.batch_size*4),1,train_ids); model.calibrate(o[:,0],a[:,0],n[:,0])
                wm_loss=wm_updates(model,opt_m,buffer,train_ids,args.batch_size,args.horizon,args.warmup_updates if cycle==0 and not resume_data else args.wm_updates,failure_penalty,args.reward_scale,env.action_filter_alpha,env.dt)
                stats=validation(model,buffer,train_ids,val_ids,max(args.batch_size,2048),args.horizon,failure_penalty,args.reward_scale,env)
                # 论文：在想象轨迹上计算 TD(lambda)，Critic 无 EMA 目标网络。
                model_ready=(stats['wm_horizon_mse'] < stats['baseline_horizon_mse']
                             and stats['wm_horizon_mse'] < args.max_wm_mse
                             and stats['analytical_reward_mae_scaled'] < stats['reward_baseline_mae_scaled']
                             and stats['validation_unique_failure_events'] >= 3
                             and stats['failure_auc_sampled'] > .6
                             and stats['reward_parity_mae_scaled'] < .001
                             and stats['failure_brier'] < stats['failure_baseline_brier']
                             and torch.isfinite(torch.tensor(list(stats.values()),dtype=torch.float32)).all().item())
                actor_loss=None; critic_loss=None
                if model_ready:
                    for q in model.parameters():q.requires_grad_(False)
                    for q in critic.parameters():q.requires_grad_(False)
                    imagined_states=None
                    for _ in range(args.actor_updates):
                        x,_,_,_,_,_,rem,_=buffer.transitions(args.batch_size,train_ids)
                        if env.randomization is not None:
                            # 想象期即将新抽取的未知随机推力不应被当成确定性的预测。
                            safe=(x[:,81]>0) | (x[:,80]>args.horizon*env.dt)
                            x,rem=x[safe],rem[safe]
                            if not len(x): continue
                        x=x.detach(); rem=rem.detach(); survival=torch.ones(len(x),device=sim.device)
                        objective=torch.zeros_like(survival); penalty=torch.zeros_like(survival)
                        imagined_states=[]; imagined_rem=[]; imagined_rewards=[]; imagined_fail=[]
                        for k in range(args.horizon):
                            imagined_states.append(x); imagined_rem.append(rem)
                            action=actor(td(x[:,:POLICY_DIM]))
                            with torch.no_grad(): expert=teacher(td(x[:,:POLICY_DIM].detach()))
                            penalty+=(action-expert).square().mean(-1)
                            old=x; nxt,_,fail_logit=model(x,action)
                            x=enforce_action_state(nxt,old,action,env.action_filter_alpha)
                            x=enforce_context(x,old,env.dt)
                            chance=fail_logit.sigmoid()
                            reward_hat=analytical_reward(x,action,env,old)/args.reward_scale-chance*failure_penalty/args.reward_scale
                            objective+=(args.gamma**k)*survival*reward_hat
                            imagined_rewards.append(reward_hat.detach()); imagined_fail.append(chance.detach())
                            survives_timeout=(rem > (1.0/env.max_episode_length + 1e-7)).float()
                            survival=survival*(1-chance)*survives_timeout; rem=(rem-1/env.max_episode_length).clamp_min(0)
                        objective+=(args.gamma**args.horizon)*survival*critic(x,rem)
                        imagined_states.append(x); imagined_rem.append(rem)
                        a_loss=(-objective+args.teacher_weight*penalty).mean()
                        opt_a.zero_grad(set_to_none=True); a_loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(),.5);opt_a.step();actor_loss=float(a_loss.detach())
                    for q in model.parameters():q.requires_grad_(True)
                    for q in critic.parameters():q.requires_grad_(True)
                    if imagined_states is not None:
                        with torch.no_grad():
                            values=[critic(s.detach(),t.detach()) for s,t in zip(imagined_states,imagined_rem)]
                            returns=[]; next_return=values[-1]
                            for k in reversed(range(args.horizon)):
                                cont=(1-imagined_fail[k])*(imagined_rem[k]>(1.0/env.max_episode_length + 1e-7)).float()
                                next_return=imagined_rewards[k]+args.gamma*cont*((1-.95)*values[k+1]+.95*next_return)
                                returns.append(next_return)
                            targets=torch.stack(list(reversed(returns))).detach()
                        for _ in range(args.critic_updates):
                            values_pred=torch.stack([critic(s.detach(),t.detach()) for s,t in zip(imagined_states[:-1],imagined_rem[:-1])])
                            loss_c=F.mse_loss(values_pred,targets)
                            opt_c.zero_grad(set_to_none=True);loss_c.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.0);opt_c.step()
                            critic_loss=float(loss_c.detach())
                data={'cycle':cycle+1,'replay_slots':min(buffer.count,args.replay_steps)*args.num_envs,
                      'train_valid_transitions':int(buffer.active[:,train_ids].sum().item()),
                      'validation_valid_transitions':int(buffer.active[:,val_ids].sum().item()),
                      'real_failures':falls,'real_active_steps':active_count,'real_reward_per_active_step':reward_sum/max(active_count,1),'wm_loss':wm_loss,**stats,'critic_loss':critic_loss,'actor_updated':actor_loss is not None,'actor_loss':actor_loss}
                for j,label in enumerate(('teacher','student')):
                    count,failures,total_reward=eval_totals[j].cpu().tolist()
                    data[f'eval_{label}_active_steps']=int(count)
                    data[f'eval_{label}_failures']=int(failures)
                    data[f'eval_{label}_reward_per_step']=total_reward/max(count,1)
                with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
                print(json.dumps(data,ensure_ascii=False),flush=True)
                if (cycle+1)%args.save_interval==0 or cycle+1==start+args.cycles:
                    keys=('obs','action','next_obs','reward','fail','done','active','remaining','next_remaining','step_id')
                    state={'teacher':str(ckpt),'teacher_actor':teacher.state_dict(),'task_config':saved_cfg,'args':vars(args),'cycle':cycle+1,'model':model.state_dict(),'actor':actor.state_dict(),'critic':critic.state_dict(),'opt_m':opt_m.state_dict(),'opt_a':opt_a.state_dict(),'opt_c':opt_c.state_dict(),'buffer':{**{k:getattr(buffer,k).cpu() for k in keys},'count':buffer.count},'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'metrics':data}
                    tmp=run/'checkpoint.tmp';torch.save(state,tmp);os.replace(tmp,run/'latest.pt')
            print(f'训练文件：{run}',flush=True)
        finally: SimulationContext.clear_instance()

if __name__=='__main__': main()
