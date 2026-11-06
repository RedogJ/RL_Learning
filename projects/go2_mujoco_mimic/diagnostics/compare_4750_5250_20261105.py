"""Read-only policy comparison under the same v23 reward and initial conditions."""
import json
from pathlib import Path

import torch

from go2_mujoco_mimic.play_policy import load_config
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector
from go2_mujoco_mimic.warp_env import WarpGo2Env

ROOT = Path('/workspace/go2_mujoco_mimic')
CASES = {
    '4750': ROOT / 'logs/backflip_v22_joint_reward_2026-11-05_10-30-30_673384/model_4750.pt',
    '5250': ROOT / 'logs/backflip_v23_landing_refine_2026-11-05_12-26-04_609466/model_5250.pt',
}


@torch.no_grad()
def main():
    base = torch.load(CASES['5250'], map_location='cpu', weights_only=False)
    cfg = load_config(base)
    cfg.num_envs = 16
    cfg.task.random_reference_start = False
    cfg.task.observation_noise = False
    torch.manual_seed(123)
    env = WarpGo2Env(cfg, Path(base['motion']))
    env.enable_cuda_graph()
    actor = MotionPolicy(history_length=cfg.task.history_length, future_length=cfg.task.future_length,
                         heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).to(env.device).eval()
    output = dict(num_envs=16, seed=123, noise=False, reward_from=str(CASES['5250']),
                  phase_frames=dict(flight=[19, 28], landing=[29, 40], hold=[41, 53]), cases={})
    for label, path in CASES.items():
        ck = torch.load(path, map_location='cpu', weights_only=False)
        actor.load_state_dict(ck['actor'])
        torch.manual_seed(123)
        env.reset(torch.ones(cfg.num_envs, device=env.device, dtype=torch.bool))
        active = torch.ones(cfg.num_envs, device=env.device, dtype=torch.bool)
        completed = torch.zeros_like(active)
        returns = torch.zeros(cfg.num_envs, device=env.device)
        records = []
        for _ in range(61):
            _, reward, failed, truncated, terms = env.step(actor(env.observation()))
            ref = env.ref.sample(env.frame)
            returns += torch.where(active, reward, 0)
            records.append(dict(active=active.clone(), frame=env.frame.clone(),
                joint=(env.qpos[:, env.qids]-ref['joints']).clone(),
                rotation=rotation_vector(env.qpos[:, 3:7], ref['quat']).norm(dim=-1),
                position=(env.qpos[:, :3]-ref['pos']).norm(dim=-1),
                height=(env.qpos[:, 2]-ref['pos'][:, 2]).abs(),
                terms={k: v.clone() for k, v in terms.items()}))
            completed |= active & truncated & ~failed
            active &= ~(failed | truncated)
            if not active.any():
                break
        if active.any():
            raise RuntimeError('Incomplete evaluation')
        frames = torch.stack([r['frame'] for r in records])
        valid = torch.stack([r['active'] for r in records])
        values = {k: torch.stack([r[k] for r in records]) for k in ('joint', 'rotation', 'position', 'height')}
        phases = {}
        for name, (low, high) in dict(all=[1, 53], **output['phase_frames']).items():
            mask = valid & (frames >= low) & (frames <= high)
            joint = values['joint'][mask]
            phases[name] = dict(samples=int(mask.sum()),
                joint_rms_deg=torch.rad2deg(joint.square().mean().sqrt()).item(),
                rear_thigh_rms_deg=torch.rad2deg(joint[:, [7, 10]].square().mean().sqrt()).item(),
                per_joint_rms_deg=torch.rad2deg(joint.square().mean(0).sqrt()).tolist(),
                root_rotation_rms_deg=torch.rad2deg(values['rotation'][mask].square().mean().sqrt()).item(),
                root_position_mean_m=values['position'][mask].mean().item(),
                height_abs_mean_m=values['height'][mask].mean().item(),
                reward_rate={k: torch.stack([r['terms'][k] for r in records])[mask].mean().item()
                             for k in records[0]['terms']})
        output['cases'][label] = dict(checkpoint=str(path), completed=int(completed.sum()),
                                     return_mean=returns.mean().item(), phases=phases)
        print(label, json.dumps(output['cases'][label]), flush=True)
    (ROOT / 'diagnostics/compare_4750_5250_20261105.json').write_text(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
