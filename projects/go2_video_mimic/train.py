"""Shared SuperMimic networks/physics with an independent anchor-video task.

Real MuJoCo Warp collection -> multi-step physical WM supervision -> closed-loop
differentiable imagination -> Actor and fixed lambda-return Critic updates.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.replay import Replay
from go2_mujoco_mimic.tasks.mimic.state_model import (
    Dynamics, StateValue, build_state, obs_from_state, critic_remaining_time)
from go2_mujoco_mimic.tasks.mimic.state_train import fit_critic
from go2_mujoco_mimic.tasks.mimic.rewards import (
    gravity_local, rotation_vector, estimate_torque_squared)
from .config import PROJECT, MOTION, TrainCfg
from .environment import AnchorEnv, make_config, goal, ACTION_MAPPING

FORMAT = 'go2_anchor_supermimic_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample(replay, batch, length, env_ids):
    size = min(replay.count, replay.capacity)
    if size < length:
        raise ValueError('Replay too short')
    start = torch.randint(replay.count-size, replay.count-length+1, (batch,), device=replay.device)
    env = env_ids[torch.randint(len(env_ids), (batch,), device=replay.device)]
    slots = (start[:, None] + torch.arange(length, device=replay.device)) % replay.capacity
    return tuple(col[slots, env[:, None]] for col in replay.storage)


def reference_transition(env, before, after):
    a, b = env.ref.sample(before), env.ref.sample(after)
    return dict(pos=b['body_pos'], quat=b['body_quat'], joint_pos=b['joints'],
                linear_vel=(b['body_pos']-a['body_pos'])/env.dt,
                angular_vel=rotation_vector(b['body_quat'], a['body_quat'])/env.dt)


def soft_failure(env, state, frame):
    ref = env.ref.sample(frame)
    height = (state[:, 2]-ref['pos'][:, 2]).abs()
    tilt = (gravity_local(state[:, 3:7])[:, 2]-gravity_local(ref['quat'])[:, 2]).abs()
    a = torch.sigmoid(60*(height-env.cfg.task.max_height_error))
    b = torch.sigmoid(40*(tilt-env.cfg.task.max_tilt_error))
    # Smooth nonfoot-height proxy for imagination; real failures use exact
    # MuJoCo nonfoot/ground collision flags, as well as height and tilt.
    body, _ = env.kinematics(state)
    knee = body[:, [3, 7, 11, 15], 2].amin(-1)
    contact = torch.sigmoid(100*(.025-knee))
    return 1-(1-a)*(1-b)*(1-contact)


def optimize(optimizer, loss, parameters):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    return float(norm.detach())


def fit_world(model, optimizer, batch):
    states, _, _, _, actions, targets, dones, _, _, _ = batch
    pred = states[:, 0]
    context = batch[7][:, 0]
    live = torch.ones(len(pred), device=pred.device)
    numerator, denominator = pred.new_zeros(()), pred.new_zeros(())
    for t in range(actions.shape[1]):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(pred, actions[:, t], context)
        target = targets[:, t]
        quat = torch.where((pred[:, 3:7]*target[:, 3:7]).sum(-1, keepdim=True) < 0,
                           -target[:, 3:7], target[:, 3:7])
        target = torch.cat((target[:, :3], quat, target[:, 7:]), -1)
        error = ((pred-target)/model.normalizer.scale).square().mean(-1)
        numerator = numerator+(live*error).sum()
        denominator = denominator+live.sum()
        # Failure transitions are supervised; subsequent cross-reset steps aren't.
        live = live*(1-dones[:, t])
    loss = numerator/denominator.clamp_min(1)
    optimize(optimizer, loss, model.parameters())
    return float(loss.detach())


@torch.no_grad()
def validate_world(model, batch):
    state, _, _, _, actions, targets, dones, contexts, _, _ = batch
    pred, live = state[:, 0], torch.ones(len(state), dtype=torch.bool, device=state.device)
    result = {}
    for t in range(actions.shape[1]):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(pred, actions[:, t], contexts[:, t])
        if t == actions.shape[1]-1 and live.any():
            result = dict(wm_holdout_samples=int(live.sum()),
                          wm_holdout_joint_rmse_rad=float((pred[live, 7:19]-targets[live, t, 7:19]).square().mean().sqrt()),
                          wm_holdout_root_rmse_m=float((pred[live, :3]-targets[live, t, :3]).square().sum(-1).mean().sqrt()))
        live &= ~dones[:, t].bool()
    return result


def imagine(env, actor, model, critic, opt_a, opt_v, batch, cfg, update_actor):
    states, histories, frames, previous, _, _, _, contexts, limits, _ = batch
    s, history, frame = states[:, 0].detach(), histories[:, 0].detach(), frames[:, 0].clone()
    prev, ctx, limit = previous[:, 0], contexts[:, 0], limits[:, 0]
    survive, objective = s.new_ones(len(s)), s.new_zeros(len(s))
    ss, gg, hh, tt, rr, dd, valid = [], [], [], [], [], [], []
    model.requires_grad_(False)
    critic.requires_grad_(False)
    for t in range(cfg.horizon):
        g = goal(env, frame, limit, s)
        ss.append(s.detach()); gg.append(g.detach()); hh.append(history.detach())
        tt.append(critic_remaining_time(env, frame, limit).detach())
        valid.append((frame < limit).detach())
        action = actor(torch.cat((history.flatten(1), g), -1))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ns = model(s, action, ctx)
        nf = torch.minimum(frame+1, limit)
        reference = reference_transition(env, frame, nf)
        target = (env.default_q+env.action_scales*action).clamp(env.joint_limits[:, 0], env.joint_limits[:, 1])
        torque = estimate_torque_squared(s[:, 7:19], s[:, 25:37], ns[:, 7:19], ns[:, 25:37],
                                        target, env.cfg.kp, env.cfg.kd, env.cfg.torque_limit, env.substeps)
        reward, _ = env.task.reward(env.kinematics.transition(s, ns, env.dt), reference,
                                   action, prev, ns[:, 7:19], env.joint_limits, env.dt,
                                   joint_vel=ns[:, 25:37], prev_joint_vel=s[:, 25:37], target=target,
                                   torque_squared_mean=torque, kp=env.cfg.kp, kd=env.cfg.kd,
                                   torque_limit=env.cfg.torque_limit)
        failure = soft_failure(env, ns, nf)
        reward = reward-env.cfg.task.reward.failure_penalty*failure
        done = 1-(1-failure)*(nf < limit).float()
        rr.append(reward.detach()); dd.append(done.detach())
        objective = objective+cfg.gamma**t*survive*reward
        survive = survive*(1-done)
        history = torch.cat((history[:, 1:], obs_from_state(ns, action, env.default_q)[:, None]), 1)
        s, frame, prev = ns, nf, action
    tail_goal, tail_time = goal(env, frame, limit, s), critic_remaining_time(env, frame, limit)
    max_value = 4.8*env.dt/(1-cfg.gamma)
    tail = critic(s, tail_goal, ctx, history, tail_time).clamp(max=max_value)
    objective = objective+cfg.gamma**cfg.horizon*survive*tail
    grad = optimize(opt_a, -objective.mean(), actor.parameters()) if update_actor else 0.
    # Fixed, detached lambda-return targets; Critic never backpropagates into WM/Actor.
    critic.requires_grad_(True)
    stack_s, stack_g = torch.stack(ss), torch.stack(gg)
    stack_h, stack_t = torch.stack(hh), torch.stack(tt)
    stack_c = ctx.expand(cfg.horizon, -1, -1)
    with torch.no_grad():
        values = critic(stack_s, stack_g, stack_c, stack_h, stack_t)
        tail_value = critic(s.detach(), tail_goal, ctx, history.detach(), tail_time)
        running, returns = tail_value, []
        for i in reversed(range(cfg.horizon)):
            next_value = tail_value if i == cfg.horizon-1 else values[i+1]
            running = rr[i]+cfg.gamma*(1-dd[i])*((1-cfg.lambda_)*next_value+cfg.lambda_*running)
            returns.append(running)
        targets = torch.stack(returns[::-1])
    mask = torch.stack(valid).flatten()
    def flat(x):
        return x.flatten(0, 1)[mask]
    if mask.any():
        count = int(mask.sum())
        loss, before, after = fit_critic(critic, opt_v, flat(stack_s), flat(stack_g), flat(stack_c),
                                       targets.flatten()[mask], cfg.critic_epochs,
                                       min(cfg.critic_minibatches, count), histories=flat(stack_h),
                                       remaining_times=flat(stack_t))
    else:
        loss = before = after = 0.
    model.requires_grad_(True)
    return dict(imagined_objective=float(objective.detach().mean()), actor_grad_norm=grad,
                actor_updated=update_actor, critic_loss=loss,
                critic_loss_before=before, critic_loss_after=after)


def save(run, cycle, cfg, env, motion, actor, model, critic, optimizers):
    model_path = PROJECT.parents[1]/'assets/unitree_model/Go2/mjcf/go2.xml'
    payload = dict(format=FORMAT, cycle=cycle, train=asdict(cfg), env=asdict(env.cfg),
                   action_mapping=ACTION_MAPPING,
                   actor=actor.state_dict(), model=model.state_dict(), critic=critic.state_dict(),
                   optimizers=[o.state_dict() for o in optimizers], motion=str(motion),
                   reference_timing=json.loads(motion.with_name('clip.json').read_text()),
                   motion_sha256=sha(motion), robot_sha256=sha(model_path),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    path = run/f'model_{cycle}.pt'
    temporary = run/'.checkpoint.tmp'
    torch.save(payload, temporary)
    os.replace(temporary, path)
    latest = run/'.latest.tmp'
    if latest.exists(): latest.unlink()
    os.link(path, latest)
    os.replace(latest, run/'latest.pt')
    print(f'Saved {path}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--run-name', default='anchor_side_run_megasam_world_3x')
    for name in asdict(TrainCfg()):
        if name == 'actor_hidden': continue
        kind = float if isinstance(getattr(TrainCfg(), name), float) else int
        p.add_argument('--'+name.rstrip('_').replace('_', '-'), dest=name, type=kind)
    args = p.parse_args()
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if checkpoint and checkpoint.get('format') != FORMAT:
        raise ValueError('This project only resumes its own anchor checkpoints')
    if checkpoint and checkpoint.get('action_mapping') != ACTION_MAPPING:
        raise ValueError('Action mapping changed; start a new experiment')
    cfg = TrainCfg(**checkpoint['train']) if checkpoint else TrainCfg()
    for name in asdict(cfg):
        if getattr(args, name, None) is not None: setattr(cfg, name, getattr(args, name))
    if min(cfg.num_envs, cfg.cycles, cfg.collect_steps, cfg.wm_batch, cfg.wm_updates,
           cfg.window, cfg.actor_batch, cfg.horizon, cfg.wm_width, cfg.save_interval,
           cfg.critic_epochs, cfg.critic_minibatches) < 1 or cfg.num_envs < 4:
        p.error('Positive sizes and at least 4 environments are required')
    if cfg.buffer_steps < max(cfg.window, cfg.horizon) or min(cfg.warmup, cfg.critic_warmup) < 0:
        p.error('Invalid replay size/warmup')
    if not 0 < cfg.gamma < 1 or not 0 <= cfg.lambda_ <= 1 or cfg.actor_lr <= 0:
        p.error('Invalid discount/lambda/learning rate')
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):
        p.error('run-name must contain letters, digits, underscores or hyphens')
    motion = (args.motion or (Path(checkpoint['motion']) if checkpoint else MOTION)).resolve()
    if checkpoint and checkpoint['motion_sha256'] != sha(motion):
        raise ValueError('Reference changed; start a new experiment')
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    metadata = json.loads(motion.with_name('clip.json').read_text())
    env_cfg = make_config(cfg.num_envs, metadata['duration_s']+.02)
    env = AnchorEnv(env_cfg, motion)
    env.enable_cuda_graph()
    actor = MotionPolicy(hidden=cfg.actor_hidden).to(env.device)
    model = Dynamics(width=cfg.wm_width).to(env.device)
    critic = StateValue(history_length=10, goal_dim=190).to(env.device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, betas=(.9, .95))
    opt_m = torch.optim.Adam(model.parameters(), lr=2e-4)
    opt_v = torch.optim.Adam(critic.parameters(), lr=2e-4)
    start = 0
    if checkpoint:
        expected_env = checkpoint['env'].copy(); expected_env['num_envs'] = cfg.num_envs
        if expected_env != json.loads(json.dumps(asdict(env_cfg))):
            raise ValueError('Environment/reward changed; start a new experiment')
        if checkpoint['robot_sha256'] != sha(PROJECT.parents[1]/'assets/unitree_model/Go2/mjcf/go2.xml'):
            raise ValueError('Robot model changed')
        for module, key in ((actor, 'actor'), (model, 'model'), (critic, 'critic')):
            module.load_state_dict(checkpoint[key])
        for opt, saved in zip((opt_a, opt_m, opt_v), checkpoint['optimizers']): opt.load_state_dict(saved)
        for group in opt_a.param_groups: group['lr'] = cfg.actor_lr
        torch.set_rng_state(checkpoint['torch_rng']); torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        start = checkpoint['cycle']
    run = PROJECT/'logs'/f'{args.run_name}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(dict(train=asdict(cfg), env=asdict(env_cfg),
         motion=str(motion), motion_sha256=sha(motion), reference_timing=metadata, action_mapping=ACTION_MAPPING,
         shared_stack='go2_mujoco_mimic: WarpGo2Env, Dynamics, MotionPolicy, StateValue, Replay, fit_critic'), indent=2))
    replay = Replay(cfg.buffer_steps, cfg.num_envs, env.device)
    holdout_count = max(1, cfg.num_envs//5)
    train_ids = torch.arange(holdout_count, cfg.num_envs, device=env.device)
    holdout_ids = torch.arange(holdout_count, device=env.device)
    context = torch.zeros(cfg.num_envs, 16, device=env.device); context[:, 0] = .5
    state = build_state(env)
    episode_return = state.new_zeros(cfg.num_envs)
    print(f'Anchor SuperMimic: {cfg.num_envs} envs, 50Hz, {len(env.ref.frames)} noncyclic frames\nLog: {run}', flush=True)
    for cycle in range(start, start+cfg.cycles):
        begin = time.monotonic()
        rewards, completed_returns, completed_seconds = [], [], []
        failures = 0
        for _ in range(cfg.collect_steps):
            with torch.no_grad():
                history, frame, prev = env.history.clone(), env.frame.clone(), env.previous_action.clone()
                action = (actor(env.observation())+.1*torch.randn_like(prev)).clamp(-1, 1)
                _, reward, terminated, truncated, _ = env.step(action)
                next_state = build_state(env)
                done = terminated | truncated
                replay.add((state, history, frame, prev, action, next_state,
                            done.float(), context, env.frame_limit.clone(), env.frame.clone()))
                episode_return += reward
                completed_returns.extend(episode_return[done].tolist())
                completed_seconds.extend((env.steps[done]*env.dt).tolist())
                episode_return = torch.where(done, 0., episode_return)
                rewards.append(float(reward.mean())); failures += int(terminated.sum())
                env.reset(done)
                state = torch.where(done[:, None], build_state(env), next_state)
        if replay.count < max(cfg.window, cfg.horizon): continue
        if not bool(model.normalizer.ready):
            count = min(replay.count, replay.capacity)
            calibration = torch.cat((replay.storage[0][:count, train_ids].flatten(0, 1),
                                     replay.storage[5][:count, train_ids].flatten(0, 1)))
            model.normalizer.calibrate(calibration)
            critic.normalizer.load_state_dict(model.normalizer.state_dict())
        losses = [fit_world(model, opt_m, sample(replay, cfg.wm_batch, cfg.window, train_ids))
                  for _ in range(cfg.wm_updates)]
        metrics = dict(cycle=cycle+1, real_reward_rate=float(np.mean(rewards)/env.dt),
                       failures=failures, wm_loss=float(np.mean(losses)),
                       completed_episodes=len(completed_returns),
                       episode_return_mean=float(np.mean(completed_returns)) if completed_returns else None,
                       episode_duration_mean_s=float(np.mean(completed_seconds)) if completed_seconds else None,
                       actor_updated=False)
        if cycle >= cfg.warmup:
            metrics.update(imagine(env, actor, model, critic, opt_a, opt_v,
                                   sample(replay, cfg.actor_batch, cfg.horizon, train_ids), cfg,
                                   update_actor=cycle >= cfg.warmup+cfg.critic_warmup))
        if (cycle+1) % 10 == 0:
            metrics.update(validate_world(model, sample(replay, 32, cfg.window, holdout_ids)))
        metrics['seconds'] = time.monotonic()-begin
        with (run/'metrics.jsonl').open('a') as f: f.write(json.dumps(metrics)+'\n')
        duration = f'{metrics["episode_duration_mean_s"]:.2f}s' if completed_seconds else '--'
        print(f'[{cycle+1}] reward/s={metrics["real_reward_rate"]:.3f} WM={metrics["wm_loss"]:.4f} '
              f'fallen={failures} episode={duration} actor={metrics["actor_updated"]} '
              f'{metrics["seconds"]:.2f}s', flush=True)
        if (cycle+1) % cfg.save_interval == 0 or cycle+1 == start+cfg.cycles:
            save(run, cycle+1, cfg, env, motion, actor, model, critic, (opt_a, opt_m, opt_v))


if __name__ == '__main__':
    main()
