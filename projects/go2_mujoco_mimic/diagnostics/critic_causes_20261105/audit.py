"""Frozen-checkpoint audit; writes diagnostics only, never updates any weights."""
import argparse
import json
import math
from pathlib import Path

import torch

from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.state_model import Dynamics, StateValue, TerminationHead, build_state, obs_from_state
from go2_mujoco_mimic.tasks.mimic.state_train import goal, policy_input, termination_probability
from go2_mujoco_mimic.tasks.mimic.adaptive_rsi import AdaptiveRSI
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector, estimate_torque_squared


def error(a, b):
    d = a - b
    return dict(n=d.numel(), bias=d.mean().item(), mae=d.abs().mean().item(),
                rmse=d.square().mean().sqrt().item(), prediction_mean=a.mean().item(),
                target_mean=b.mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.manual_seed(9132)
    torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = load_config(ck)
    cfg.num_envs = 128
    cfg.task.observation_noise = True
    cfg.task.random_reference_start = True
    env = WarpGo2Env(cfg, Path(ck['motion']))
    env.enable_cuda_graph()
    actor = MotionPolicy(history_length=cfg.task.history_length, future_length=cfg.task.future_length,
                         heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval()
    actor.load_state_dict(ck['actor'])
    model = Dynamics(kinematics=env.kinematics if cfg.wm_foot_features else None).cuda().eval()
    model.load_state_dict(ck['model']); model.requires_grad_(False)
    critic = StateValue().cuda().eval(); critic.load_state_dict(ck['critic']); critic.requires_grad_(False)
    head = TerminationHead(cfg.task.future_length*19).cuda().eval()
    head.load_state_dict(ck['termination_head']); head.requires_grad_(False)
    del ck
    def context(n):
        x = torch.zeros(n, 16, device='cuda'); x[:, 0] = .5
        return x
    ctx = context(cfg.num_envs)
    H, gamma, lam = 16, .99, .95

    def imagine(pack, noise=True, steps=None):
        s, hist, frame, lim, prev = pack
        ctx = context(len(s))
        nsteps = int((lim-frame).max().item()) if steps is None else steps
        nsteps = max(H, nsteps)
        survive = torch.ones(len(s), device='cuda')
        ret = torch.zeros_like(survive)
        values, rewards, dones, masks, survivals, states, histories, frames = [], [], [], [], [], [], [], []
        pose = None
        for t in range(nsteps):
            values.append(critic(s, goal(env, frame, lim), ctx))
            masks.append(frame < lim); survivals.append(survive)
            states.append(s); histories.append(hist); frames.append(frame)
            action = actor(policy_input(env, s, hist, frame, lim))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ns = model(s, action, ctx)
            advance = math.floor((t+1)*env.dt*env.ref.fps)-math.floor(t*env.dt*env.ref.fps)
            nf = torch.minimum(frame+advance, lim)
            a, b = env.ref.sample(nf), env.ref.sample(frame)
            ref = dict(pos=a['body_pos'], quat=a['body_quat'], joint_pos=a['joints'],
                       linear_vel=(a['body_pos']-b['body_pos'])/env.dt,
                       angular_vel=rotation_vector(a['body_quat'], b['body_quat'])/env.dt)
            actual, pose = env.kinematics.transition_with_pose(pose, s, ns, env.dt)
            tq = env.default_q + env.action_scales*action
            if cfg.clamp_joint_targets:
                tq = tq.clamp(env.joint_limits[:, 0], env.joint_limits[:, 1])
            tq2 = None
            if cfg.task.reward.torque_l2_weight and cfg.task.reward.torque_mode == 'substep_mean':
                tq2 = estimate_torque_squared(s[:, 7:19], s[:, 25:37], ns[:, 7:19], ns[:, 25:37],
                                             tq, cfg.kp, cfg.kd, cfg.torque_limit, env.substeps)
            rew, _ = env.task.reward(actual, ref, action, prev, ns[:, 7:19], env.joint_limits, env.dt,
                                    joint_vel=ns[:, 25:37], prev_joint_vel=s[:, 25:37], target=tq,
                                    torque_squared_mean=tq2, kp=cfg.kp, kd=cfg.kd, torque_limit=cfg.torque_limit)
            f = termination_probability(env, model, head, ns, nf, lim)
            rew = rew-cfg.task.reward.failure_penalty*f
            done = 1-(1-f)*(nf < lim).float()
            rewards.append(rew); dones.append(done)
            ret = ret+(gamma**t)*survive*rew
            survive = survive*(1-done)
            hist = torch.cat((hist[:, 1:], obs_from_state(ns, action, env.default_q, noise=noise)[:, None]), 1)
            s, frame, prev = ns, nf, action
            if t == H-1:
                hvalue = critic(s, goal(env, frame, lim), ctx)
                hret, hsurvive = ret, survive
                hs, hh, hf = s, hist, frame
        # Full model return is an internal consistency diagnostic, not simulator ground truth.
        target = hvalue
        for t in reversed(range(H)):
            next_v = values[t+1] if t+1 < H else hvalue
            target = rewards[t]+gamma*(1-dones[t])*((1-lam)*next_v+lam*target)
        return dict(full=ret, reward=hret, tail=gamma**H*hsurvive*hvalue,
                    tail_value=gamma**H*hsurvive.detach()*hvalue,
                    tail_survival=gamma**H*hsurvive*hvalue.detach(),
                    lambda_target=target, value=values[0], hvalue=hvalue, hsurvive=hsurvive,
                    hs=hs, hh=hh, hf=hf, masks=torch.stack(masks, 1),
                    survivals=torch.stack(survivals, 1), states=states, histories=histories, frames=frames)

    rsi = AdaptiveRSI(len(env.ref.frames)-1, env.ref.fps, env.device, bin_seconds=.2, control_dt=env.dt)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not cfg.task.reference_loop: rsi.gmax = 1.
    rsi.load_state_dict(ck['adaptive_rsi']); del ck
    mask = torch.ones(cfg.num_envs, device='cuda', dtype=torch.bool)
    columns = {k: [] for k in ('s', 'hist', 'frame', 'limit', 'prev', 'value', 'reward', 'done', 'alive', 'fail')}
    with torch.no_grad():
        torch.manual_seed(9132)
        env.reset(mask, rsi.probabilities(), rsi.frames_per_bin, start_probability=.2)
        alive = mask.clone()
        for t in range(math.ceil(cfg.episode_seconds/env.dt)+2):
            s, hist, frame, lim, prev = build_state(env).clone(), env.history.clone(), env.frame.clone(), env.frame_limit.clone(), env.previous_action.clone()
            value = critic(s, goal(env, frame, lim), ctx)
            _, reward, failed, truncated, _ = env.step(actor(env.observation()))
            done = failed | truncated
            for key, x in dict(s=s, hist=hist, frame=frame, limit=lim, prev=prev, value=value,
                               reward=reward, done=done, alive=alive.clone(), fail=failed).items():
                columns[key].append(x.clone())
            alive &= ~done
            if not alive.any(): break
        if alive.any(): raise RuntimeError('Some physical episodes did not finish')
        c = {k: torch.stack(v) for k, v in columns.items()}
        T, N = c['alive'].shape
        mc = torch.zeros_like(c['reward']); running = torch.zeros(N, device='cuda')
        for t in reversed(range(T)):
            running = c['reward'][t]+gamma*(~c['done'][t]).float()*running
            mc[t] = running
        eligible = c['alive'].nonzero()
        torch.manual_seed(9876)
        idx = eligible[torch.randperm(len(eligible), device='cuda')[:512]]
        ti, ei = idx.unbind(1)
        selected = {k: v[ti, ei].clone() for k, v in c.items()}
        pack = tuple(selected[k] for k in ('s', 'hist', 'frame', 'limit', 'prev'))
        torch.manual_seed(5151)
        im = imagine(pack, noise=True)
        report = dict(checkpoint=args.checkpoint, settings=dict(envs=N, sampled_states=len(idx),
                       observation_noise=True, action_noise=False, horizon=H, gamma=gamma, lam=lam,
                       real_valid_states=len(eligible), real_failed=int((c['alive'] & c['fail']).sum())),
                      overall={}, bins={}, tail_bins={}, exact_goal_aliases=[])
        for name, a, b in (
            ('critic_vs_actual_mc', selected['value'], mc[ti, ei]),
            ('critic_vs_imag_lambda', im['value'], im['lambda_target']),
            ('critic_vs_full_model_return', im['value'], im['full']),
            ('model_full_return_vs_actual_mc', im['full'], mc[ti, ei]),
            ('model_h_plus_v_vs_actual_mc', im['reward']+im['tail'], mc[ti, ei]),
            ('model_lambda_vs_actual_mc', im['lambda_target'], mc[ti, ei]),
        ):
            report['overall'][name] = error(a, b)
        report['overall']['normalizer_equal'] = bool(torch.equal(model.normalizer.mean, critic.normalizer.mean) and torch.equal(model.normalizer.scale, critic.normalizer.scale))
        # Unweighted critic fits conditional values even for low-survival soft-failure paths.
        valid = im['masks'][:, :H]; sv = im['survivals'][:, :H]
        report['overall']['fit_survival_below_01_fraction'] = ((sv < .1) & valid).sum().item()/valid.sum().item()
        for b in range(9):
            m = selected['frame']//6 == b
            if m.any():
                report['bins'][str(b)] = dict(critic_actual=error(im['value'][m], mc[ti, ei][m]),
                    critic_model=error(im['value'][m], im['full'][m]),
                    model_actual=error(im['full'][m], mc[ti, ei][m]))
        jt = (ti+H).clamp(max=T-1)
        real_alive_h = (ti+H < T) & c['alive'][jt, ei]
        tail_valid = (selected['frame']+H < selected['limit']) & real_alive_h
        for b in range(9):
            m = tail_valid & (im['hf']//6 == b)
            if m.any():
                report['tail_bins'][str(b)] = dict(
                    real_value_vs_real_mc=error(c['value'][jt, ei][m], mc[jt, ei][m]),
                    imag_value_vs_real_mc=error(im['hvalue'][m], mc[jt, ei][m]),
                    imag_value_minus_real_value=error(im['hvalue'][m], c['value'][jt, ei][m]),
                    state_normalized_rmse=(((im['hs'][m]-c['s'][jt, ei][m])/model.normalizer.scale).square().mean().sqrt().item()))
        fs = torch.arange(len(env.ref.frames)-1, device='cuda')
        lims = torch.full_like(fs, len(env.ref.frames)-1)
        gs = goal(env, fs, lims)
        for i in range(len(fs)):
            for j in range(i+1, len(fs)):
                if torch.equal(gs[i], gs[j]): report['exact_goal_aliases'].append([i, j])
        # Plausible reset-style history vs the actual history at unchanged physical state.
        repeated = selected['hist'][:, -1:, :].expand_as(selected['hist']).clone()
        a = actor(policy_input(env, *pack[:2], pack[2], pack[3]))
        alt = actor(policy_input(env, pack[0], repeated, pack[2], pack[3]))
        da = (a-alt).abs()
        report['history_action_counterfactual'] = dict(mean=da.mean().item(), max=da.max().item(),
            p95_per_state=torch.quantile(da.mean(-1), .95).item())
        print('VALUE_AUDIT', json.dumps(report), flush=True)
        # Record a small tensor dataset for local follow-up diagnostics; no checkpoints copied.
        torch.save(dict(pack=tuple(x.cpu() for x in pack), real_mc=mc[ti, ei].cpu(),
                        actual_value=selected['value'].cpu(), sample_t=ti.cpu(), sample_env=ei.cpu()),
                   args.output.replace('.json', '_samples.pt'))

    # Initial states from real trajectories, early enough for nonzero tail. No noise so gradient probes reproducible.
    pick = ((selected['frame'] <= 24).nonzero(as_tuple=True)[0])[:32]
    gradpack = tuple(x[pick] for x in pack)
    torch.manual_seed(999)
    vals = imagine(gradpack, noise=False)
    parts = dict(reward=vals['reward'], tail=vals['tail'], tail_value=vals['tail_value'],
                 tail_survival=vals['tail_survival'], model_remaining=vals['full']-vals['reward'])
    params = list(actor.parameters())
    grads = {}
    for name, val in parts.items():
        grads[name] = [g.detach() for g in torch.autograd.grad(val.mean(), params, retain_graph=True)]
    def dot(a, b): return sum((x*y).sum() for x, y in zip(a, b))
    norms = {k: dot(v, v).sqrt().item() for k, v in grads.items()}
    cos = {}
    for a, b in (('tail', 'reward'), ('tail', 'model_remaining'), ('tail_value', 'tail_survival'),
                 ('tail_value', 'model_remaining'), ('tail_survival', 'model_remaining')):
        cos[a+':'+b] = dot(grads[a], grads[b]).item()/max(norms[a]*norms[b], 1e-20)
    report['gradient'] = dict(n=len(pick), frames=selected['frame'][pick].tolist(),
        norms=norms, cosines=cos, means={k: v.mean().item() for k, v in parts.items()})
    print('GRADIENT_AUDIT', json.dumps(report['gradient']), flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
