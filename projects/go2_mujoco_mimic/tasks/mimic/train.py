"""MuJoCo Warp 数据采集、世界模型拟合和可微想象策略更新。"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from .networks import MotionPolicy
from .world_model import WorldModel, Value
from ...warp_env import EnvCfg, WarpGo2Env


# 兼容历史导入；当前训练入口直接使用 replay.py。
from .replay import Replay


def finite_grad(parameters):
    return all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--cycles', type=int, default=100)
    p.add_argument('--collect-steps', type=int, default=32)
    p.add_argument('--wm-updates', type=int, default=5)
    p.add_argument('--policy-updates', type=int, default=1)
    p.add_argument('--wm-batch', type=int, default=2048)
    p.add_argument('--actor-batch', type=int, default=128)
    p.add_argument('--wm-amp', choices=('bf16', 'fp32'), default='bf16')
    p.add_argument('--save-interval', type=int, default=50)
    p.add_argument('--window', type=int, default=8)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--buffer-steps', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--motion', type=Path, default=Path('/workspace/go2_mujoco_mimic/data/kine2go/ai4_dog_canter/motion.npy'))
    p.add_argument('--run-name', default='world_model')
    p.add_argument('--resume', type=Path)
    p.add_argument('--allow-reference-cycle-change', action='store_true', help='允许从旧参考周期配置续训；其他环境配置仍须一致')
    p.add_argument('--no-physics-graph', action='store_true', help='关闭物理子步 CUDA Graph，使用逐步调用')
    p.add_argument('--profile', action='store_true', help='同步计时采集、世界模型、策略和保存阶段；会增加少量开销')
    args = p.parse_args()
    if min(args.num_envs,args.cycles,args.collect_steps,args.wm_updates,args.policy_updates,args.wm_batch,args.actor_batch,args.window,args.horizon,args.save_interval) < 1 or args.buffer_steps < max(args.window,args.horizon):
        p.error('所有数量须为正，buffer-steps 须不小于窗口和想象步数')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.wm_amp == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU 不支持 BF16；可指定 --wm-amp fp32')
    cfg = EnvCfg(num_envs=args.num_envs)
    env = WarpGo2Env(cfg, args.motion)
    if not args.no_physics_graph:
        print('录制物理子步 CUDA Graph...', flush=True)
        env.enable_cuda_graph()
    policy = MotionPolicy().to(env.device)
    model = WorldModel().to(env.device)
    critic = Value().to(env.device)
    opt_m = torch.optim.Adam(model.parameters(), lr=3e-4)
    opt_p = torch.optim.Adam(policy.parameters(), lr=1e-4)
    opt_v = torch.optim.Adam(critic.parameters(), lr=3e-4)
    begin = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=env.device, weights_only=False)
        if saved['motion'] != str(args.motion):
            raise ValueError('续训的动作路径与检查点不一致')
        from copy import deepcopy
        saved_env = deepcopy(saved['env_cfg'])
        saved_env['task'].setdefault('terminate_on_tracking_error', True)
        saved_num_envs = saved_env['num_envs']
        saved_env['num_envs'] = cfg.num_envs
        if saved_num_envs != cfg.num_envs:
            print(f'并行环境数变化：{saved_num_envs} -> {cfg.num_envs}；Replay 将重新采集。', flush=True)
        saved_cycles = saved_env['task'].get('reference_cycles', 1)
        saved_env['task']['reference_cycles'] = cfg.task.reference_cycles
        if saved_env != asdict(cfg):
            raise ValueError('续训的环境配置与检查点不一致（参考周期之外）')
        if saved_cycles != cfg.task.reference_cycles:
            if not args.allow_reference_cycle_change:
                raise ValueError(f'参考周期从 {saved_cycles} 改为 {cfg.task.reference_cycles}；若有意继续，请加 --allow-reference-cycle-change')
            print(f'任务配置变化：参考周期 {saved_cycles} -> {cfg.task.reference_cycles}；会新建日志并重新采集 replay。', flush=True)
        policy.load_state_dict(saved['policy']); model.load_state_dict(saved['model']); critic.load_state_dict(saved['critic'])
        opt_p.load_state_dict(saved['opt_p']); opt_m.load_state_dict(saved['opt_m']); opt_v.load_state_dict(saved['opt_v'])
        begin = saved['cycle']
        print('续训：恢复模型和优化器；采样缓冲重新收集。', flush=True)
    run = Path('/workspace/go2_mujoco_mimic/logs') / f'{args.run_name}_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}'
    run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps({'args': vars(args), 'env': asdict(cfg)}, default=str, ensure_ascii=False, indent=2))
    replay = Replay(args.buffer_steps, args.num_envs, env.device)
    obs = env.observation()
    print(f'训练目录={run} 环境={args.num_envs} 数据={args.motion}；世界模型首版，非论文完整实现', flush=True)
    for cycle in range(begin, begin + args.cycles):
        torch.cuda.synchronize(); t0 = time.monotonic(); real_rewards = []; failures = torch.zeros((), device=env.device, dtype=torch.long)
        for _ in range(args.collect_steps):
            with torch.no_grad():
                action = (policy(obs) + torch.randn((args.num_envs, 12), device=env.device)*0.1).clamp(-1,1)
                nxt, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                replay.add((obs, action, nxt, reward, terminated.float(), done.float()))
                real_rewards.append(reward.mean()); failures += terminated.sum()
                env.reset(done)
                obs = env.observation()
        if min(replay.count, replay.capacity) < max(args.window,args.horizon):
            print(f'cycle={cycle+1} 采样预热={min(replay.count, replay.capacity)}步',flush=True); continue
        if args.profile:
            torch.cuda.synchronize(); collect_end = time.monotonic()
        for _ in range(args.wm_updates):
            batch = replay.sample(args.wm_batch, args.window)
            ob, act, next_ob, rew, terminal, done = batch
            pred_obs = ob[:,0]
            live = torch.ones(args.wm_batch, device=env.device)
            total = torch.zeros((), device=env.device)
            for j in range(args.window):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.wm_amp == 'bf16'):
                    nxt_prop, aux, logit = model(pred_obs, act[:,j])
                nxt_prop, aux, logit = nxt_prop.float(), aux.float(), logit.float()
                target_prop = next_ob[:,j,378:408]
                # 观测含传感器噪声，此处拟合带噪标签；不作为精度结论。
                loss_state = (nxt_prop[:,:30]-target_prop).square().mean(-1)
                # aux 前 1 维预测实际每步奖励；其余为预留，不参与训练。
                loss_reward = (aux[:,0]-rew[:,j]).square()
                loss_done = F.binary_cross_entropy_with_logits(logit,terminal[:,j],reduction='none')
                total = total + ((loss_state + loss_reward + loss_done)*live).sum()/live.sum().clamp_min(1)
                live = live * (1-done[:,j])
                pred_obs = torch.cat((pred_obs[:,42:420], nxt_prop, next_ob[:,j,420:]),-1)
            opt_m.zero_grad(set_to_none=True); (total/args.window).backward()
            if not finite_grad(model.parameters()): raise RuntimeError('世界模型梯度非有限值')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt_m.step()
        if args.profile:
            torch.cuda.synchronize(); model_end = time.monotonic()
        for _ in range(args.policy_updates):
            ob, _, next_ob, _, _, done = replay.sample(args.actor_batch,args.horizon)
            imagined = ob[:,0].detach(); survival = torch.ones(args.actor_batch,device=env.device); returns = torch.zeros_like(survival)
            for param in model.parameters(): param.requires_grad_(False)
            for j in range(args.horizon):
                action = policy(imagined)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.wm_amp == 'bf16'):
                    prop, aux, logit = model(imagined,action)
                prop, aux, logit = prop.float(), aux.float(), logit.float()
                returns = returns + (0.99**j)*survival*aux[:,0]
                survival = survival*(1-torch.sigmoid(logit))*(1-done[:,j])
                imagined = torch.cat((imagined[:,42:420],prop,next_ob[:,j,420:]),-1)
            for param in critic.parameters(): param.requires_grad_(False)
            objective = returns + (0.99**args.horizon)*survival*critic(imagined)
            opt_p.zero_grad(set_to_none=True); (-objective.mean()).backward()
            if not finite_grad(policy.parameters()): raise RuntimeError('策略梯度非有限值')
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt_p.step()
            for param in list(model.parameters())+list(critic.parameters()): param.requires_grad_(True)
            value_loss = (critic(ob[:,0])-objective.detach()).square().mean()
            opt_v.zero_grad(set_to_none=True); value_loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(),1.0); opt_v.step()
        if args.profile:
            torch.cuda.synchronize(); policy_end = time.monotonic()
        metrics = {'cycle':cycle+1,'real_reward':torch.stack(real_rewards).mean().item(),'failures':int(failures.item()),'model_loss':(total/args.window).item(),'imagined_return':objective.mean().item(),'value_loss':value_loss.item()}
        if args.profile:
            metrics.update({'collect_seconds': collect_end-t0, 'world_model_seconds': model_end-collect_end,
                            'policy_seconds': policy_end-model_end})
        with (run/'metrics.jsonl').open('a') as file: file.write(json.dumps(metrics)+'\n')
        torch.cuda.synchronize()
        print(f"[{cycle+1}] reward={metrics['real_reward']:.4f} fallen={metrics['failures']} WM_loss={metrics['model_loss']:.4f} 想象回报={metrics['imagined_return']:.4f} 用时={time.monotonic()-t0:.1f}s",flush=True)
        if args.profile:
            print(f"  分段(秒): 采集={metrics['collect_seconds']:.3f} 世界模型={metrics['world_model_seconds']:.3f} 策略/critic={metrics['policy_seconds']:.3f}", flush=True)
        if (cycle+1) % args.save_interval == 0 or cycle+1 == begin+args.cycles:
            save_start = time.monotonic()
            torch.save({'cycle':cycle+1,'policy':policy.state_dict(),'model':model.state_dict(),'critic':critic.state_dict(),'opt_p':opt_p.state_dict(),'opt_m':opt_m.state_dict(),'opt_v':opt_v.state_dict(),'motion':str(args.motion),'env_cfg':asdict(cfg)},run/'latest.pt')
            if args.profile:
                print(f'  保存检查点={time.monotonic()-save_start:.3f}s', flush=True)


if __name__ == '__main__': main()
