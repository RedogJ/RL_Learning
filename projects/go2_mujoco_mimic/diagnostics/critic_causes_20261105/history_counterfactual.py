"""Paired physics rollouts: same physical state/phase, actual vs repeated last observation history."""
import json
from pathlib import Path

import mujoco_warp as mjw
import torch
import warp as wp

from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.warp_env import WarpGo2Env, warp_stream_ordered
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.state_model import StateValue
from go2_mujoco_mimic.tasks.mimic.state_train import goal


@warp_stream_ordered
def install(env, state, hist, frame, lim, prev):
    mask = torch.ones(len(state), dtype=torch.bool, device='cuda')
    mjw.reset_data(env.model, env.data, wp.from_torch(mask, dtype=wp.bool))
    env.qpos[:, :7] = state[:, :7]
    env.qpos[:, env.qids] = state[:, 7:19]
    env.qvel[:, :6] = state[:, 19:25]
    env.qvel[:, env.vids] = state[:, 25:37]
    env.previous_joint_vel.copy_(state[:, 25:37])
    env.previous_action.copy_(prev)
    env.history.copy_(hist)
    env.frame.copy_(frame); env.start_frame.copy_(frame); env.frame_limit.copy_(lim)
    env.steps.zero_()
    mjw.forward(env.model, env.data)


@torch.no_grad()
def main():
    checkpoint = '/workspace/go2_mujoco_mimic/logs/backflip_tail_0.5_2026-11-05_07-41-15_227013/model_3400.pt'
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = load_config(ck); cfg.num_envs = 64; cfg.task.observation_noise = False
    env = WarpGo2Env(cfg, Path(ck['motion'])); env.enable_cuda_graph()
    actor = MotionPolicy(history_length=cfg.task.history_length, future_length=cfg.task.future_length,
                         heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).cuda().eval()
    actor.load_state_dict(ck['actor'])
    critic = StateValue().cuda().eval(); critic.load_state_dict(ck['critic'])
    ds = torch.load('/tmp/go2_critic_causes_3400_samples.pt', map_location='cuda', weights_only=False)
    pack = ds['pack']
    # Sample across phases; duplicate every physical state exactly for each history choice.
    ids = torch.arange(32, device='cuda')*13
    s, h, f, lim, prev = [x[ids].clone() for x in pack]
    hist = torch.cat((h, h[:, -1:, :].expand_as(h)), 0)
    state, frame, limits, previous = [torch.cat((x, x), 0) for x in (s, f, lim, prev)]
    install(env, state, hist, frame, limits, previous)
    ctx = torch.zeros(cfg.num_envs, 16, device='cuda'); ctx[:, 0] = .5
    value = critic(state, goal(env, frame, limits), ctx)
    first_action = actor(env.observation())
    ret = torch.zeros(cfg.num_envs, device='cuda'); failed = torch.zeros_like(ret, dtype=torch.bool)
    alive = ~failed
    for t in range(54):
        _, r, fl, tr, _ = env.step(actor(env.observation()))
        ret += .99**t*alive*r; failed |= alive & fl; alive &= ~(fl | tr)
        if not alive.any(): break
    delta = ret[32:]-ret[:32]
    report = dict(checkpoint=checkpoint, n_pairs=32, future_noise=False,
                  counterfactual='Replace recorded 10-frame history by 10 copies of its latest frame; same previous action and physical state',
                  critic_pair_max_diff=(value[32:]-value[:32]).abs().max().item(),
                  first_action_mean_abs_diff=(first_action[32:]-first_action[:32]).abs().mean().item(),
                  return_mean_abs_diff=delta.abs().mean().item(), return_mean_diff=delta.mean().item(),
                  return_max_abs_diff=delta.abs().max().item(),
                  actual_history_failed=int(failed[:32].sum()), repeated_history_failed=int(failed[32:].sum()),
                  frames=f.tolist(), actual_history_returns=ret[:32].tolist(), repeated_history_returns=ret[32:].tolist(),
                  critic=value[:32].tolist())
    Path('/tmp/go2_critic_history.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
