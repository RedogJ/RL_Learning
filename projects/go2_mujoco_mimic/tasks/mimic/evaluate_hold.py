"""完整播放一次动作，再持续跟踪参考末态，评估落地后的站立稳定性。"""
import argparse
import json
import math
from pathlib import Path

import torch

from ...play_policy import load_config
from ...warp_env import WarpGo2Env
from .networks import MotionPolicy
from .rewards import rotation_vector


def hold_frame_limit(original_limit, fps, motion_seconds, hold_seconds):
    """hold_seconds 为动作结束后的总保持时间；不会缩短原来的参考。"""
    if not all(math.isfinite(x) for x in (fps, motion_seconds, hold_seconds)):
        raise ValueError('帧率与时长必须有限')
    if fps <= 0 or motion_seconds < 0 or hold_seconds < 0:
        raise ValueError('帧率必须为正，时长必须非负')
    return max(original_limit, math.ceil((motion_seconds + hold_seconds) * fps))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--hold-seconds', type=float, default=2., help='原动作结束后的总保持时间，含参考已有保持段')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--no-noise', action='store_true', help='关闭观测噪声；默认沿用 checkpoint')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.num_envs <= 0 or not math.isfinite(args.hold_seconds) or args.hold_seconds < 0:
        parser.error('环境数必须为正，hold-seconds 必须有限且非负')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = load_config(checkpoint)
    if cfg.task.reference_loop:
        parser.error('此评估只支持单次、不循环的参考动作')
    motion = Path(checkpoint['motion'])
    clip = json.loads(motion.with_name('clip.json').read_text())
    if 'motion_duration_s' not in clip:
        parser.error('clip.json 需要 motion_duration_s，才能区分原动作和已有保持段')
    motion_seconds = float(clip['motion_duration_s'])
    cfg.num_envs = args.num_envs
    cfg.task.random_reference_start = False
    if args.no_noise:
        cfg.task.observation_noise = False
    # 给参考末态评估留足回合时间；原参考文件与 checkpoint 均不改写。
    cfg.episode_seconds = max(cfg.episode_seconds, motion_seconds + args.hold_seconds + 1.)
    torch.manual_seed(args.seed)
    env = WarpGo2Env(cfg, motion)
    env.enable_cuda_graph()
    original_limit = len(env.ref.frames) - 1
    limit = hold_frame_limit(original_limit, env.ref.fps, motion_seconds, args.hold_seconds)
    cfg.episode_seconds = max(cfg.episode_seconds, limit / env.ref.fps + env.dt)
    actor = MotionPolicy(history_length=cfg.task.history_length, future_length=cfg.task.future_length,
                         heading_dim=2*int(cfg.task.heading_observation)+2*int(cfg.task.longitudinal_observation)).to(env.device).eval()
    actor.load_state_dict(checkpoint.get('actor', checkpoint.get('policy')))
    torch.manual_seed(args.seed)
    env.reset(torch.ones(args.num_envs, device=env.device, dtype=torch.bool))
    # Reference.sample 对超出原轨迹的帧返回最后一帧，策略继续执行，不做中途 reset。
    env.frame_limit.fill_(limit)
    active = torch.ones(args.num_envs, device=env.device, dtype=torch.bool)
    completed = torch.zeros_like(active)
    reached_original_end = torch.zeros_like(active)
    failed_before = torch.zeros_like(active)
    failed_after = torch.zeros_like(active)
    records = []
    print(f'模型 cycle={checkpoint["cycle"]}；原参考 {original_limit/env.ref.fps:.3f}s；'
          f'评估至 {limit/env.ref.fps:.3f}s；总末态保持约 {limit/env.ref.fps-motion_seconds:.3f}s；'
          f'环境={args.num_envs}，观测噪声={cfg.task.observation_noise}', flush=True)
    for _ in range(math.ceil(limit / env.ref.fps / env.dt) + 2):
        _, _, failed, truncated, _ = env.step(actor(env.observation()))
        reference = env.ref.sample(env.frame)
        # 使用现有真实仿真的失败判据；额外记录姿态/位置误差，不把没触发失败当作完全站稳。
        reached_original_end |= active & (env.frame >= original_limit) & ~failed
        completed |= active & (env.frame >= limit) & truncated & ~failed
        failed_before |= active & failed & (env.frame <= original_limit)
        failed_after |= active & failed & (env.frame > original_limit)
        records.append(dict(
            active=active.clone(), frame=env.frame.clone(),
            joint_error=(env.qpos[:, env.qids] - reference['joints']).clone(),
            rotation=rotation_vector(env.qpos[:, 3:7], reference['quat']).norm(dim=-1),
            position=(env.qpos[:, :3] - reference['pos']).norm(dim=-1),
            height=(env.qpos[:, 2] - reference['pos'][:, 2]).abs(),
            speed=env.qvel[:, :3].norm(dim=-1),
        ))
        active &= ~(failed | truncated)
        if not active.any():
            break
    if active.any():
        raise RuntimeError('评估未到达终点，检查参考帧率和控制步数')
    frames = torch.stack([r['frame'] for r in records])
    live = torch.stack([r['active'] for r in records])
    def summarize(mask):
        count = int(mask.sum())
        if not count:
            return {'samples': 0}
        values = {key: torch.stack([r[key] for r in records])[mask]
                  for key in ('joint_error', 'rotation', 'position', 'height', 'speed')}
        return dict(samples=count,
                    joint_rms_deg=torch.rad2deg(values['joint_error'].square().mean().sqrt()).item(),
                    root_rotation_rms_deg=torch.rad2deg(values['rotation'].square().mean().sqrt()).item(),
                    root_position_mean_m=values['position'].mean().item(),
                    height_abs_mean_m=values['height'].mean().item(),
                    root_speed_mean_m_s=values['speed'].mean().item())
    result = dict(checkpoint=str(args.checkpoint), cycle=checkpoint['cycle'], seed=args.seed,
                  num_envs=args.num_envs, observation_noise=cfg.task.observation_noise,
                  original_end_s=original_limit/env.ref.fps, extended_end_s=limit/env.ref.fps,
                  hold_seconds=limit/env.ref.fps-motion_seconds,
                  reached_original_end=int(reached_original_end.sum()), completed=int(completed.sum()),
                  failed_before_original_end=int(failed_before.sum()), failed_during_extension=int(failed_after.sum()),
                  original_hold=summarize(live & (frames/env.ref.fps >= motion_seconds) & (frames <= original_limit)),
                  extended_hold=summarize(live & (frames > original_limit)),
                  final_half_second=summarize(live & (frames/env.ref.fps > limit/env.ref.fps-.5)))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
