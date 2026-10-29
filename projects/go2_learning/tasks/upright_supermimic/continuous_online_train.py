"""Start isolated online training from a single expert and a pretrained world model."""
import argparse
import json
import math
import os
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
from go2_learning.tasks.upright_supermimic.paper_context import PaperContext, configure_ground, configure_paper_physics
from go2_learning.tasks.upright_supermimic.paper_model import PaperCritic, enforce_executed_action
from go2_learning.tasks.upright_supermimic.continuous_world_model import ContinuousWorldModel, disable_contact_rewards, soft_upright_tilt_failure
from go2_learning.tasks.upright_supermimic.paper_buffer import PaperBuffer
from go2_learning.tasks.upright_supermimic.state_bridge import (
    STATE_DIM, POLICY_DIM, observe, next_policy_action, analytical_reward,
)
from go2_learning.training.checkpoints import resolve_checkpoint


def clean_state(env, obs):
    state=observe(env,obs['policy'])
    # 论文不含突发推力；此处保留旧状态布局但显式移除推力槽。
    return torch.cat((state[:,:80],torch.zeros_like(state[:,80:88]),state[:,88:]),-1)


def noisy_observation(state):
    """论文 A.1 的传感器噪声；Go2 的关节速度观测已缩放 0.1。"""
    obs=state[:,:POLICY_DIM].clone()
    for start,end,scale in ((3,6,.2),(6,9,.05),(9,21,.01),(21,33,.05)):
        obs[:,start:end]+=(2*torch.rand_like(obs[:,start:end])-1)*scale
    return obs


def td(obs):
    return TensorDict({'policy':obs},batch_size=[obs.shape[0]])


def wm_loss(model,buf,ids,batch,k):
    states,actions,contexts,next_states,fail,_,_=buf.windows(batch,k,ids)
    pred=states[:,0,:55];dyn=torch.zeros((),device=pred.device)
    for i in range(k):
        pred=enforce_executed_action(model(pred,actions[:,i],contexts[:,i]),actions[:,i])
        diff=(pred-next_states[:,i,:55])/model.out_norm.std().clamp_min(1e-3)
        dyn+=diff.square().mean()
    return dyn/k


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default=None,help='单策略世界模型预训练检查点')
    p.add_argument('--resume',default=None,help='本入口的在线检查点；恢复模型、优化器和 replay')
    p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--cycles',type=int,default=100,help='追加在线训练轮数')
    p.add_argument('--collect-steps',type=float,default=2.5,
                   help='每次更新平均新增的每环境控制步；2.5 对应 UTD 0.4')
    p.add_argument('--wm-batch',type=int,default=2048,help='每次世界模型更新采样的轨迹窗口数')
    p.add_argument('--actor-batch',type=int,default=128)
    p.add_argument('--wm-updates',type=int,default=10);p.add_argument('--wm-horizon',type=int,default=8)
    p.add_argument('--horizon',type=int,default=16);p.add_argument('--replay-steps',type=int,default=5000)
    p.add_argument('--critic-warmup-cycles',type=int,default=0,help='前几轮只训练世界模型和 Critic，暂不更新 Actor；默认关闭')
    p.add_argument('--actor-lr',type=float,default=None,help='默认 2e-4；续训默认继承')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save-interval',type=int,default=10)
    add_launcher_args(p);args=p.parse_args()
    if args.num_envs<4 or args.critic_warmup_cycles<0 or not math.isfinite(args.collect_steps) or args.collect_steps<1 or min(args.cycles,args.wm_batch,args.actor_batch,args.wm_updates,
                              args.wm_horizon,args.horizon,args.replay_steps,args.save_interval)<=0:
        p.error('环境至少 4 个，各训练参数必须为正')
    if args.replay_steps<args.wm_horizon:
        p.error('缓冲长度必须不少于世界模型窗口')
    torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=True
    resume=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    pretrained=None;dataset=None;source=None
    if not resume:
        if not args.checkpoint:p.error('首次训练需要 --checkpoint 指定预训练世界模型')
        pretrained=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if pretrained.get('format')!='continuous_pretrain_v1' or pretrained['args'].get('model','mlp')!='mlp':raise ValueError('需要连续状态 MLP 预训练检查点')
        pretrain_config=json.loads(Path(args.checkpoint).with_name('config.json').read_text())
        dataset=torch.load(pretrained['args']['checkpoint'],map_location='cpu',weights_only=False)
        if dataset.get('world_model_format')!='deeponet_dataset_v1' or dataset['collection_args'].get('policy')!='expert':
            raise ValueError('初始 replay 必须是单一专家策略数据')
        if dataset['source_checkpoint']!=pretrain_config['source']:
            raise ValueError('预训练与初始 replay 的策略来源不一致')
        if dataset['args']['num_envs']!=args.num_envs or dataset['args']['replay_steps']!=args.replay_steps:
            raise ValueError('在线 num-envs 和 replay-steps 必须与初始数据一致')
        if len(dataset['teacher_env_ids'])!=args.num_envs:
            raise ValueError('初始数据并非全部来自专家策略')
        source=torch.load(dataset['source_checkpoint'],map_location='cpu',weights_only=False)
        if source.get('world_model_format')!='paper_context_v3' or abs(source.get('control_dt',-1.)-.02)>1e-9:
            raise ValueError('源检查点不是 50 Hz paper_context_v3')
    if args.actor_lr is None:args.actor_lr=resume['args']['actor_lr'] if resume else 2e-4
    if args.actor_lr<=0:p.error('Actor 学习率必须为正')
    if resume:
        if resume.get('world_model_format')!='continuous_mlp_v1' or resume.get('online_pretrained_format')!='continuous_online_v1':raise ValueError('续训需要本入口产生的检查点')
        if abs(resume.get('control_dt',-1.)-0.02)>1e-9:raise ValueError('续训检查点控制频率与论文 50 Hz 不一致')
        for name in ('num_envs','replay_steps','wm_horizon','horizon','collect_steps','critic_warmup_cycles'):
            if getattr(args,name)!=resume['args'][name]:raise ValueError('续训必须匹配原参数 '+name)
    ckpt=Path(resume['teacher'] if resume else source['teacher'])
    saved_cfg=deepcopy(resume['task_config'] if resume else source['task_config'])
    task=task_from_snapshot(deepcopy(saved_cfg));disable_contact_rewards(task.cfg.rewards)
    # 奖励快照与实验真实环境一致，保留真实非脚掌接触失败条件。
    from dataclasses import asdict
    saved_cfg['reward_cfg'].update(asdict(task.cfg.rewards))
    configure_paper_physics(task);configure_ground(task)
    sim_cfg=SimulationCfg(dt=task.cfg.environment.physics_dt,physics=PhysxCfg())
    with launch_simulation(sim_cfg,args):
        from isaaclab.sim import SimulationContext
        from go2_learning.simulation.env import make_env
        sim=SimulationContext(sim_cfg)
        try:
            env=make_env(sim,args.num_envs,task.cfg.randomization,args.seed,task=task)
            if list(env.robot.joint_names)!=saved_cfg['joint_names']:raise ValueError('关节顺序与专家不匹配')
            for name,value in [('default_joint_pos',env.default_joint_pos[0]),('action_joint_limits',env.action_joint_limits[0])]:
                if not torch.allclose(torch.tensor(saved_cfg[name],device=env.device),value,atol=1e-6,rtol=0):raise ValueError(name+' 不匹配')
            context=PaperContext(env,args.seed+101)
            context_check=context.verify()
            if max(context_check['friction_max_abs_error'],context_check['com_max_abs_error_m'])>1e-5:
                raise RuntimeError('PhysX 回读 context 与记录不一致')
            env.diagnostics_interval=24;env.detailed_diagnostics=False
            env.finish_initial_settling()
            runner=OnPolicyRunner(env=env,train_cfg=task.cfg.agent,log_dir=None,device=str(sim.device))
            actor=runner.get_inference_policy(device=str(sim.device));teacher=deepcopy(actor).eval()
            if resume:
                teacher.load_state_dict(resume['teacher_actor']);actor.load_state_dict(resume['actor'])
            else:
                actor.load_state_dict(source['teacher_actor']);teacher.load_state_dict(source['teacher_actor'])
            for v in teacher.parameters():v.requires_grad_(False)
            model=ContinuousWorldModel(55,env.num_actions).to(sim.device)
            critic=PaperCritic(STATE_DIM).to(sim.device)
            opt_m=torch.optim.Adam(model.parameters(),lr=2e-4)
            opt_a=torch.optim.Adam(actor.parameters(),lr=args.actor_lr,betas=(.9,.95))
            opt_c=torch.optim.Adam(critic.parameters(),lr=2e-4)
            buf=PaperBuffer(args.replay_steps,args.num_envs,STATE_DIM,env.num_actions,sim.device)
            # 论文的采集环境都用于 replay；真实泛化评估通过独立的 eval 进程完成。
            train_ids=torch.arange(args.num_envs,device=sim.device)
            # env.step 自动 reset；在 reset 前保留真实终止状态，避免把新回合初态当成下一状态。
            original_reset_finished=env._reset_finished_envs
            terminal={}
            def capture(dones):
                terminal['state']=clean_state(env,env.get_observations()).clone()
                terminal['executed_action']=env.actions.clone()
                return original_reset_finished(dones)
            env._reset_finished_envs=capture
            obs=env.get_observations();state=clean_state(env,obs)
            start=0;collect_carry=0.
            if not resume:
                model.load_state_dict(pretrained['model'])
                # 在线阶段从预训练权重开始；优化器状态单独初始化。
                for name,value in dataset['buffer'].items():
                    if torch.is_tensor(value):getattr(buf,name).copy_(value.to(sim.device))
                    else:setattr(buf,name,value)
                zero=torch.zeros(args.num_envs,device=sim.device);mask=torch.zeros(args.num_envs,dtype=torch.bool,device=sim.device)
                buf.add(state,env.standing_actions,context.value,state,mask,mask,mask,zero)
                print('已加载预训练世界模型、v32 单策略和专家采集 replay；新仿真回合之间已插入屏障。',flush=True)
            if resume:
                model.load_state_dict(resume['model']);critic.load_state_dict(resume['critic'])
                opt_m.load_state_dict(resume['opt_m']);opt_a.load_state_dict(resume['opt_a']);opt_c.load_state_dict(resume['opt_c'])
                for name,value in resume['buffer'].items():
                    if torch.is_tensor(value):getattr(buf,name).copy_(value.to(sim.device))
                    else:setattr(buf,name,value)
                # 重启后进入新回合，插入无效屏障，禁止和旧轨迹拼接。
                zero=torch.zeros(args.num_envs,device=sim.device);mask=torch.zeros(args.num_envs,dtype=torch.bool,device=sim.device)
                buf.add(state,env.standing_actions,context.value,state,mask,mask,mask,zero)
                torch.set_rng_state(resume['rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
                context.generator.set_state(resume['context_rng'])
                if env.randomization is not None and 'randomization_rng' in resume:
                    env.randomization.generator.set_state(resume['randomization_rng'])
                if env.reset_orientation_sampler is not None and 'orientation_rng' in resume:
                    env.reset_orientation_sampler.generator.set_state(resume['orientation_rng'])
                start=resume['cycle']
                collect_carry=resume.get('collect_carry',0.)
                print('已恢复世界模型、策略、价值网络、优化器和 replay；仿真回合重新初始化。',flush=True)
            name='online_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
            run=LOG_ROOT/'upright_supermimic'/'continuous_online'/name;run.mkdir(parents=True)
            info={'args':vars(args),'teacher':str(ckpt),'task_config':saved_cfg,'context_check':context_check,
                  'initial_world_model':args.checkpoint if not resume else resume.get('initial_world_model'),
                  'initial_dataset':pretrained['args']['checkpoint'] if not resume else resume.get('initial_dataset'),
                  'critic_warmup_cycles':args.critic_warmup_cycles,
                  'context_layout':['friction','12 joint actuator calibration offsets','base COM xyz'],
                  'go2_adaptations':['upright task and 52-dim single-frame pretrained actor',
                                     'joint calibration is actuator position-target offset',
                                     'replay stores action after Go2 clamping and low-pass filter',
                                     'contact rewards disabled; imagined tilt-only soft termination, real contact termination retained',
                                     '50 Hz Isaac PhysX instead of MuJoCo Warp'],
                  'physics_dt':task.cfg.environment.physics_dt,'control_dt':env.dt}
            (run/'config.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,default=str))
            print('连续 MLP 在线训练输出：',run,'物理参数回读：',context_check,flush=True)
            print('接触奖励关闭；真实接触仍重置；想象中仅近似倾斜失败，未预测接触失败。',flush=True)
            for cycle in range(start,start+args.cycles):
                # 论文 Table A.5：25 万轮余弦衰减到 3%，之后保持。
                fraction=min(cycle/250000.,1.)
                factor=.03+.97*.5*(1.+math.cos(math.pi*fraction))
                for optim,base in ((opt_m,2e-4),(opt_a,args.actor_lr),(opt_c,2e-4)):
                    for group in optim.param_groups:group['lr']=base*factor
                collect_carry+=args.collect_steps
                steps=int(collect_carry);collect_carry-=steps
                # 初次需要填满一个 k 步窗口；此预热开销纳入计数。
                steps=max(steps,args.wm_horizon-buf.count)
                collected_failures=torch.zeros((),device=sim.device)
                collected_active=torch.zeros((),device=sim.device)
                collected_reward=torch.zeros((),device=sim.device)
                with torch.no_grad():
                    for step in range(steps):
                        active=env.policy_active.clone();old=state.clone();ctx=context.value.clone()
                        remain=(env.max_episode_length-env.episode_length_buf).clamp_min(0)/env.max_episode_length
                        actions=env.standing_actions.clone()
                        if active.any():actions[active]=actor(td(noisy_observation(old[active])))+.1*torch.randn_like(actions[active])
                        obs,r,done,extras=env.step(actions);state=clean_state(env,obs)
                        failed=done.bool() & ~extras['time_outs'].bool()
                        collected_failures+=(failed & active).sum()
                        collected_active+=active.sum()
                        collected_reward+=r[active].sum()
                        true_next=torch.where(done.bool()[:,None],terminal['state'],state)
                        if step==0 and active.any():
                            expected=next_policy_action(old[active,33:45],actions[active],env.action_filter_alpha)
                            if not torch.allclose(expected,terminal['executed_action'][active],atol=1e-6,rtol=0):
                                raise RuntimeError('replay 动作与实际执行动作不一致')
                        buf.add(old,terminal['executed_action'],ctx,true_next,failed,done.bool(),active,remain)
                        if active.any():
                            model.update_statistics(old[active,:55],terminal['executed_action'][active],ctx[active],true_next[active,:55])
                for _ in range(args.wm_updates):
                    dyn=wm_loss(model,buf,train_ids,args.wm_batch,args.wm_horizon)
                    loss=dyn
                    if not torch.isfinite(loss):raise RuntimeError('世界模型损失非有限数')
                    opt_m.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt_m.step()
                with torch.no_grad():
                    vd=wm_loss(model,buf,train_ids,max(128,args.wm_batch),args.wm_horizon)
                for q in model.parameters():q.requires_grad_(False)
                for q in critic.parameters():q.requires_grad_(False)
                x,_,ctx,_,_,_,rem=buf.transitions(args.actor_batch,train_ids)
                x=x.detach();ctx=ctx.detach();rem=rem.detach();survival=torch.ones(len(x),device=sim.device)
                objective=torch.zeros_like(survival);states=[];contexts=[];remains=[];rewards=[];fails=[]
                for k in range(args.horizon):
                    states.append(x);contexts.append(ctx);remains.append(rem)
                    action=actor(td(noisy_observation(x)))
                    old=x;executed=next_policy_action(old[:,33:45],action,env.action_filter_alpha)
                    nxt=model(x[:,:55],executed,ctx)
                    x=torch.cat((enforce_executed_action(nxt,executed),torch.zeros_like(x[:,55:])),dim=-1)
                    chance=soft_upright_tilt_failure(x[:,6:9],task.cfg.termination.maximum_tilt_degrees)
                    reward=analytical_reward(x,action,env,old)/1000.0-chance*task.cfg.rewards.failure_penalty/1000.0
                    objective+=(.99**k)*survival*reward
                    rewards.append(reward.detach());fails.append(chance.detach())
                    survival=survival*(1-chance)*(rem>1.0/env.max_episode_length).float()
                    rem=(rem-1.0/env.max_episode_length).clamp_min(0)
                objective+=(.99**args.horizon)*survival*critic(x,ctx,rem)
                states.append(x);contexts.append(ctx);remains.append(rem)
                a_loss=-objective.mean()
                if not torch.isfinite(a_loss):raise RuntimeError('策略损失非有限数')
                actor_updated=cycle>=args.critic_warmup_cycles
                if actor_updated:
                    opt_a.zero_grad(set_to_none=True);a_loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1.)
                    for q in actor.parameters():
                        if q.grad is not None:q.grad.clamp_(-.1,.1)
                    opt_a.step()
                for q in model.parameters():q.requires_grad_(True)
                for q in critic.parameters():q.requires_grad_(True)
                with torch.no_grad():
                    vals=[critic(s.detach(),c.detach(),r.detach()) for s,c,r in zip(states,contexts,remains)]
                    ret=vals[-1];targets=[]
                    for k in reversed(range(args.horizon)):
                        cont=(1-fails[k])*(remains[k]>1.0/env.max_episode_length).float()
                        ret=rewards[k]+.99*cont*(.05*vals[k+1]+.95*ret);targets.append(ret)
                    targets=torch.stack(list(reversed(targets))).flatten().detach()
                    cs=torch.stack([s.detach() for s in states[:-1]]).flatten(0,1)
                    cc=torch.stack([c.detach() for c in contexts[:-1]]).flatten(0,1)
                    cr=torch.stack([r.detach() for r in remains[:-1]]).flatten()
                # 论文：5 个 epoch，每个 epoch 4 个洗牌小批次。
                for _ in range(5):
                    perm=torch.randperm(len(targets),device=sim.device)
                    for ids in perm.tensor_split(4):
                        if not len(ids):continue
                        c_loss=F.mse_loss(critic(cs[ids],cc[ids],cr[ids]),targets[ids])
                        opt_c.zero_grad(set_to_none=True);c_loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.);opt_c.step()
                data={'cycle':cycle+1,'wm_state_loss':float(dyn.detach()),
                      'replay_state_loss':float(vd),'actor_loss':float(a_loss.detach()),
                      'imagined_survival':float(survival.detach().mean()),'imagined_failure_probability':float(torch.stack(fails).mean()),
                      'critic_loss':float(c_loss.detach()),'buffer_steps':buf.count,
                      'context_friction_min':float(context.value[:,0].min()),'context_friction_max':float(context.value[:,0].max()),
                      'learning_rate_factor':factor,'collected_steps_per_env':steps,
                      'actor_updated':actor_updated,'collected_active_transitions':int(collected_active),
                      'collected_failures':int(collected_failures),
                      'collected_reward_per_active_step':float(collected_reward/collected_active.clamp_min(1))}
                with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
                print(json.dumps(data,ensure_ascii=False),flush=True)
                if (cycle+1)%args.save_interval==0 or cycle+1==start+args.cycles:
                    state_out={'teacher':str(ckpt),'teacher_actor':teacher.state_dict(),'task_config':saved_cfg,
                               'args':vars(args),'cycle':cycle+1,'model':model.state_dict(),'actor':actor.state_dict(),
                               'critic':critic.state_dict(),'opt_m':opt_m.state_dict(),'opt_a':opt_a.state_dict(),
                               'opt_c':opt_c.state_dict(),'metrics':data,'world_model_format':'continuous_mlp_v1','control_dt':env.dt,
                               'online_pretrained_format':'continuous_online_v1',
                               'initial_world_model':info['initial_world_model'],'initial_dataset':info['initial_dataset'],
                               'buffer':{**{k:getattr(buf,k).cpu() for k in ('state','next_state','action','context','fail','done','active','remaining','step_id')},'count':buf.count},
                               'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'context_rng':context.generator.get_state(),
                               'collect_carry':collect_carry}
                    if env.randomization is not None:state_out['randomization_rng']=env.randomization.generator.get_state()
                    if env.reset_orientation_sampler is not None:state_out['orientation_rng']=env.reset_orientation_sampler.generator.get_state()
                    tmp=run/'checkpoint.tmp';torch.save(state_out,tmp);os.replace(tmp,run/'latest.pt')
                    os.link(run/'latest.pt',run/f'cycle_{cycle+1}.pt')
            print('训练文件：',run,flush=True)
        finally:SimulationContext.clear_instance()


if __name__=='__main__':main()
