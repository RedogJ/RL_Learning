"""Use the existing RSI reset to test identical Critic inputs at distinct finite horizons."""
import argparse
import json
from pathlib import Path

import torch

from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.state_model import StateValue, build_state
from go2_mujoco_mimic.tasks.mimic.state_train import goal


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.manual_seed(734)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = load_config(ck); cfg.num_envs = 128; cfg.task.observation_noise = False
    env = WarpGo2Env(cfg, Path(ck['motion'])); env.enable_cuda_graph()
    actor = MotionPolicy(history_length=cfg.task.history_length, future_length=cfg.task.future_length,
                         heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval()
    actor.load_state_dict(ck['actor'])
    critic = StateValue().cuda().eval(); critic.load_state_dict(ck['critic'])
    probs = torch.zeros(len(env.ref.frames)-1, device='cuda'); probs[41:53] = 1/12
    env.reset(torch.ones(cfg.num_envs, dtype=torch.bool, device='cuda'), probs, 1)
    frame, state, hist = env.frame.clone(), build_state(env).clone(), env.history.clone()
    goals = goal(env, frame, env.frame_limit)
    context = torch.zeros(cfg.num_envs, 16, device='cuda'); context[:, 0] = .5
    value = critic(state, goals, context)
    ret = torch.zeros(cfg.num_envs, device='cuda'); failed = torch.zeros_like(ret, dtype=torch.bool)
    alive = ~failed
    reward_series = []; state_series = []
    for t in range(15):
        _, r, f, tr, _ = env.step(actor(env.observation()))
        ret += .99**t*alive*r; failed |= alive & f
        alive &= ~(f | tr)
        reward_series.append(r.clone()); state_series.append(build_state(env).clone())
        if not alive.any(): break
    report = dict(checkpoint=args.checkpoint,
                  state_max_pairwise_spread=(state.max(0).values-state.min(0).values).abs().max().item(),
                  goal_max_pairwise_spread=(goals.max(0).values-goals.min(0).values).abs().max().item(),
                  history_max_pairwise_spread=(hist.max(0).values-hist.min(0).values).abs().max().item(),
                  critic_min=value.min().item(), critic_max=value.max().item(),
                  actual_return_min=ret.min().item(), actual_return_max=ret.max().item(),
                  failures=failed.sum().item(), frames={})
    equal = frame >= 42
    report['frames_42_through_52'] = dict(
        state_max_pairwise_spread=(state[equal].max(0).values-state[equal].min(0).values).abs().max().item(),
        goal_max_pairwise_spread=(goals[equal].max(0).values-goals[equal].min(0).values).abs().max().item(),
        history_max_pairwise_spread=(hist[equal].max(0).values-hist[equal].min(0).values).abs().max().item())
    for f in frame.unique().tolist():
        m = frame == f
        report['frames'][str(f)] = dict(n=m.sum().item(), remaining_steps=53-f,
                  critic=value[m].mean().item(), actual_return=ret[m].mean().item(),
                  actual_return_std=ret[m].std(unbiased=False).item())
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
