"""Middle SuperMimic：真实采集 → WM监督 → 闭环想象 → Actor/Critic。"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone,timedelta
import json
import hashlib
import os
from pathlib import Path
import random
import time
import numpy as np
import torch

from .config import Config,from_dict
from .environment import Environment
from .kinematics import Kinematics
from .models import Actor,Dynamics,Critic
from .replay import Replay
from .learner import fit_world,imagine,validate_world
from .tasks.standing.task import StandingTask
from .tasks.standing.reward_cfg import RewardCfg
from .build_model import JOINTS,OUTPUT

PROJECT=Path(__file__).resolve().parent


def resolve_checkpoint(value):
    if value=='latest':
        paths=list((PROJECT/'logs/standing').glob('*/latest.pt'))
        if not paths: raise FileNotFoundError('没有 standing 检查点')
        return max(paths,key=lambda p:p.stat().st_mtime_ns)
    path=Path(value).expanduser()
    if path.is_dir():path=path/'latest.pt'
    if not path.is_file():raise FileNotFoundError(path)
    return path


def save(run,payload):
    numbered=run/f'model_{payload["cycle"]}.pt'
    temporary=run/'.checkpoint.tmp'
    torch.save(payload,temporary);os.replace(temporary,numbered)
    latest=run/'.latest.tmp'
    if latest.exists():latest.unlink()
    os.link(numbered,latest);os.replace(latest,run/'latest.pt')
    print(f'已保存：{numbered}',flush=True)


def make_modules(cfg,env):
    fk=Kinematics(env.model).to(env.device)
    task=StandingTask(cfg.env,cfg.reward,fk,env.reference,env.limits,env.context)
    actor=Actor(cfg).to(env.device)
    wm=Dynamics(cfg,env.default,env.limits).to(env.device)
    critic=Critic(cfg).to(env.device)
    return task,actor,wm,critic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['standing'],default='standing')
    parser.add_argument('--backend',choices=['warp','cpu'],default='warp')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume')
    parser.add_argument('--run-name',default='standing')
    parser.add_argument('--cycles',type=int,default=1000,help='本次追加的 cycle 数')
    for name in ['num-envs','collect-steps','buffer-steps','wm-batch','wm-window','wm-updates',
                 'wm-warmup','critic-warmup','actor-batch','actor-rounds','horizon','wm-width',
                 'critic-epochs','critic-minibatches','save-interval','seed']:
        parser.add_argument('--'+name,type=int)
    parser.add_argument('--actor-lr',type=float)
    parser.add_argument('--wm-lr',type=float)
    parser.add_argument('--no-cuda-graph',action='store_true')
    args=parser.parse_args()
    checkpoint=None
    if args.resume:
        cp=resolve_checkpoint(args.resume)
        checkpoint=torch.load(cp,map_location=args.device,weights_only=False)
        if checkpoint.get('format')!='middle_supermimic_v1':raise ValueError('不是 middle 检查点')
        digest=hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
        if checkpoint.get('model_sha256',digest)!=digest:raise ValueError('MJCF 已变化，不能静默续训旧世界模型')
        cfg=from_dict(checkpoint['config'])
        cfg.reward=RewardCfg()  # 续训使用当前奖励，旧配置仍保存在上一段历史中。
    else:cfg=Config()
    cfg.train.cycles=args.cycles
    for key,value in vars(args).items():
        if value is None:continue
        if key=='num_envs':cfg.env.num_envs=value
        elif key=='wm_width':cfg.model.wm_width=value
        elif hasattr(cfg.train,key):setattr(cfg.train,key,value)
    if checkpoint and cfg.model.wm_width != checkpoint['config']['model']['wm_width']:
        raise ValueError('续训不能改变 WM 宽度')
    if args.no_cuda_graph:cfg.env.cuda_graph=False
    tc=cfg.train
    for name in ['cycles','collect_steps','buffer_steps','wm_batch','wm_window','wm_updates',
                 'actor_batch','actor_rounds','horizon','critic_epochs','critic_minibatches','save_interval']:
        if getattr(tc,name)<1:raise ValueError(f'{name} 必须为正')
    if tc.buffer_steps<tc.wm_window:raise ValueError('buffer_steps 不能小于 wm_window')
    if not 0<tc.gamma<1 or not 0<=tc.lam<=1:raise ValueError('gamma/lam 范围不正确')
    torch.set_num_threads(4)
    random.seed(tc.seed);np.random.seed(tc.seed);torch.manual_seed(tc.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    print(f'初始化真实仿真：{args.backend}，envs={cfg.env.num_envs}；首次 Warp 内核编译可能较慢。',flush=True)
    env=Environment(cfg.env,args.backend,args.device)
    task,actor,wm,critic=make_modules(cfg,env)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=tc.actor_lr,betas=(.9,.95))
    wm_opt=torch.optim.Adam(wm.parameters(),lr=tc.wm_lr)
    critic_opt=torch.optim.Adam(critic.parameters(),lr=tc.critic_lr)
    start_cycle=0;history=[]
    reward_changed=False
    if checkpoint:
        if checkpoint['joint_names']!=list(JOINTS):
            raise ValueError('关节顺序不匹配')
        if not torch.allclose(checkpoint['reference'].to(env.device),env.reference,atol=1e-6):
            raise ValueError('模型站姿已改变，不能静默恢复旧参考')
        actor.load_state_dict(checkpoint['actor']);wm.load_state_dict(checkpoint['world_model'])
        reward_changed=checkpoint['config']['reward']!=asdict(cfg.reward)
        if not reward_changed:
            critic.load_state_dict(checkpoint['critic']);critic_opt.load_state_dict(checkpoint['critic_optimizer'])
        actor_opt.load_state_dict(checkpoint['actor_optimizer']);wm_opt.load_state_dict(checkpoint['wm_optimizer'])
        start_cycle=checkpoint['cycle'];history=checkpoint.get('history',[])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        np.random.set_state(checkpoint['numpy_rng']);random.setstate(checkpoint['python_rng'])
        if env.device.type=='cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['cuda_rng']])
        print(f'从 cycle {start_cycle} 续训；使用当前 reward_cfg；奖励变化={reward_changed}；Replay 重新采集',flush=True)
    for optimizer,lr in [(actor_opt,tc.actor_lr),(wm_opt,tc.wm_lr),(critic_opt,tc.critic_lr)]:
        for group in optimizer.param_groups:group['lr']=lr
    if reward_changed:cfg.train.critic_warmup=max(cfg.train.critic_warmup,5)
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d_%H-%M-%S_%f')
    run=PROJECT/'logs/standing'/f'{args.run_name}_{stamp}'
    run.mkdir(parents=True)
    config=asdict(cfg)
    history=history+[{'start_cycle':start_cycle+1,'end_cycle':None,'config':config,
                     'source_checkpoint':str(cp) if checkpoint else None}]
    (run/'run.json').write_text(json.dumps({'config':config,'history':history},ensure_ascii=False,indent=2)+'\n')
    replay=Replay(tc.buffer_steps,env.device)
    # 至少四个环境时，最后约 20% 仅用于 WM 验证；不进入监督 batch 或想象起点。
    heldout=max(1,cfg.env.num_envs//5) if cfg.env.num_envs>=4 else 0
    training_envs=cfg.env.num_envs-heldout
    episode_steps=round(cfg.env.episode_seconds*cfg.env.control_hz)
    episode_return=torch.zeros(cfg.env.num_envs,device=env.device)
    print(f'Middle SuperMimic | task=standing | {args.backend} {env.device} | envs={cfg.env.num_envs}\n'
          f'collect={tc.collect_steps}, WM={tc.wm_batch}×{tc.wm_window}×{tc.wm_updates}, '
          f'imagination={tc.actor_batch}×{tc.horizon}×{tc.actor_rounds}\n日志：{run}',flush=True)
    print('这是 PD 站姿参考的模仿训练；不是 Go2 动作，不是纯 PPO。',flush=True)
    try:
        for local_cycle in range(1,tc.cycles+1):
            cycle=start_cycle+local_cycle;begin=time.monotonic()
            complete_count=torch.zeros((),device=env.device);complete_return=complete_count.clone()
            complete_time=complete_count.clone();fallen=complete_count.clone();reward_sum=complete_count.clone()
            term_sums={}
            fresh_states=[];fresh_actions=[];fresh_obs=[];fresh_goals=[]
            for _ in range(tc.collect_steps):
                with torch.no_grad():
                    state=env.state();old_history=env.history.clone();last_action=env.last_action.clone()
                    old_step=env.steps.clone()
                    goals=task.goal(state)
                    action=(actor(old_history,goals)+tc.exploration*torch.randn_like(last_action)).clamp(-1,1)
                    next_state=env.step(action)
                    terms=task.reward(state,next_state,action,last_action)
                    reward=sum(terms.values())
                    failed=task.termination(next_state)
                    truncated=env.steps>=episode_steps;done=failed|truncated
                    replay.add({'state':state,'next_state':next_state,'action':action,
                                'previous_action':last_action,'history':old_history,'step':old_step,'done':done})
                    episode_return+=reward
                    complete_count+=done.sum();complete_return+=(episode_return*done).sum()
                    complete_time+=(env.steps*done).sum()*env.dt
                    fallen+=failed.sum();reward_sum+=reward.mean()
                    for k,v in terms.items():term_sums[k]=term_sums.get(k,0)+v.mean()
                    fresh_states.append(state);fresh_actions.append(action)
                    fresh_obs.append(old_history[:,-1]);fresh_goals.append(goals)
                    episode_return.masked_fill_(done,0)
                    env.reset(done)
            fresh_states=torch.stack(fresh_states);fresh_actions=torch.stack(fresh_actions)
            if not torch.isfinite(fresh_states).all() or not torch.isfinite(env.state()).all():
                raise RuntimeError('真实仿真出现非有限状态，停止优化')
            wm.update_normalization(fresh_states[:,:training_envs],fresh_actions[:,:training_envs])
            actor.proprio_normalizer.update(torch.stack(fresh_obs))
            actor.goal_normalizer.update(torch.stack(fresh_goals))
            wm_losses=[];policy_logs={};validation={}
            ready=replay.count>=tc.wm_window
            if ready:
                for _ in range(tc.wm_updates):
                    wm_losses.append(fit_world(wm,wm_opt,replay.sample(tc.wm_batch,tc.wm_window,training_envs),task,cfg))
                if heldout:
                    validation=validate_world(wm,replay.sample(32,tc.wm_window,heldout,training_envs),task)
            stage='WM warmup'
            if ready and local_cycle>tc.wm_warmup:
                update_actor=local_cycle>tc.wm_warmup+tc.critic_warmup
                stage='Actor/Critic' if update_actor else 'Critic warmup'
                for _ in range(tc.actor_rounds):
                    policy_logs=imagine(actor,critic,wm,task,replay.sample(tc.actor_batch,env_count=training_envs),cfg,
                                        actor_opt,critic_opt,update_actor)
            if env.device.type=='cuda':torch.cuda.synchronize(env.device)
            count=int(complete_count.item())
            record={'cycle':cycle,'stage':stage,'reward_rate':reward_sum.item()/tc.collect_steps/env.dt,
                    'episode_return':complete_return.item()/count if count else None,
                    'episode_seconds':complete_time.item()/count if count else None,
                    'completed_episodes':count,'fallen':int(fallen.item()),
                    'world_model_loss':torch.stack(wm_losses).mean().item() if wm_losses else None,
                    'replay_transitions':min(replay.count,tc.buffer_steps)*cfg.env.num_envs,
                    'seconds':time.monotonic()-begin,
                    'reward_terms':{k:v.item()/tc.collect_steps/env.dt for k,v in term_sums.items()},
                    **{k:v.item() for k,v in policy_logs.items()},**validation}
            with (run/'metrics.jsonl').open('a') as log:log.write(json.dumps(record,ensure_ascii=False)+'\n')
            ret=f'{record["episode_return"]:.3f}' if count else '--'
            duration=f'{record["episode_seconds"]:.2f}s' if count else '--'
            objective=record.get('imagined_objective')
            imag=f'{objective:.3f}' if objective is not None else '--'
            wm_text=f'{record["world_model_loss"]:.4f}' if wm_losses else '--'
            validation_key=f'wm_eval_{tc.wm_window}step_joint_deg'
            val_error=validation.get(validation_key)
            validation_text=f' 验证关节={val_error:.2f}deg' if val_error is not None else ''
            print(f'[{cycle}] {stage} | reward_rate={record["reward_rate"]:.3f} | '
                  f'\033[1mreturn={ret} episode={duration}\033[0m | fallen={record["fallen"]} '
                  f'WM={wm_text}{validation_text} 想象总目标={imag} {record["seconds"]:.2f}s',flush=True)
            if local_cycle%tc.save_interval==0 or local_cycle==tc.cycles:
                history[-1]['end_cycle']=cycle
                save(run,{'format':'middle_supermimic_v1','cycle':cycle,'config':config,'history':history,
                          'model_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                          'joint_names':list(JOINTS),'reference':env.reference,'actor':actor.state_dict(),
                          'world_model':wm.state_dict(),'critic':critic.state_dict(),
                          'actor_optimizer':actor_opt.state_dict(),'wm_optimizer':wm_opt.state_dict(),
                          'critic_optimizer':critic_opt.state_dict(),'torch_rng':torch.get_rng_state(),
                          'cuda_rng':torch.cuda.get_rng_state_all() if env.device.type=='cuda' else None,
                          'numpy_rng':np.random.get_state(),'python_rng':random.getstate()})
                (run/'run.json').write_text(json.dumps({'config':config,'history':history},ensure_ascii=False,indent=2)+'\n')
    except KeyboardInterrupt:
        print('已停止；最近完成保存的检查点保留。',flush=True)


if __name__=='__main__':
    try:
        main()
    finally:
        # 在 Warp 的 atexit 清理 CUDA context 之前排空 Torch 的跨流延迟释放。
        import gc
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
