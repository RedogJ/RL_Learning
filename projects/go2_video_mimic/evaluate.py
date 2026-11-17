"""Evaluate real MuJoCo physics and optionally render policy/anchor reference."""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from go2_mujoco_mimic.environment import Go2Environment
from go2_mujoco_mimic.warp_env import Reference
from go2_mujoco_mimic.tasks.mimic.networks import MotionPolicy
from go2_mujoco_mimic.tasks.mimic.rewards import rotation_vector
from .anchor import yaw_quat, features
from .config import PROJECT, MOTION
from .environment import AnchorEnv, make_config, ACTION_MAPPING
from .train import FORMAT, sha, reference_transition


def timing_label(meta, prefix):
    speed = meta.get('playback_speed', 1.)
    if meta.get('speed_status') == 'user_selected_normal_speed':
        return f'{prefix}: NORMAL SPEED {speed:g}x - user selected'
    if speed == 1:
        return f'{prefix}: ORIGINAL SLOW-MOTION BASELINE'
    return f'{prefix}: ESTIMATED {speed:g}x - original speed unverified'


def render(records, reference_poses, path, fps=50., reference_only=False, speed_label=''):
    cpu = Go2Environment(600, 50, checkerboard=True)
    m, d = cpu.model, cpu.data
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, 960)
    m.vis.global_.offheight = max(m.vis.global_.offheight, 544)
    ghost = mujoco.MjData(m)
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1  # procedural ground
    opt.geomgroup[2] = 1  # robot visuals, collision group 3 stays hidden
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 1.65, 100, -17
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = os.environ.get('VIDEO_FFMPEG') or shutil.which('ffmpeg')
    if not encoder or not Path(encoder).is_file():
        raise RuntimeError('Run through run.sh with --video to prepare the encoder cache')
    command = [encoder, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1920x544', '-r', str(fps), '-i', 'pipe:0', '-an',
               '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(path)]
    try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    except OSError: font = ImageFont.load_default(size=22)
    with mujoco.Renderer(m, height=544, width=960) as renderer:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as writer:
            for i in range(len(records)):
                d.qpos[:] = records[i]
                mujoco.mj_forward(m, d)
                cam.lookat[:] = d.qpos[:3]; cam.lookat[2] += .02
                renderer.update_scene(d, camera=cam, scene_option=opt)
                real = renderer.render().copy()
                ghost.qpos[:] = reference_poses[i]
                mujoco.mj_forward(m, ghost)
                renderer.update_scene(ghost, camera=cam, scene_option=opt)
                # Blue reference uses render-scene colors, without changing assets.
                for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                    if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid >= 0 and m.geom_bodyid[geom.objid]:
                        geom.matid = -1; geom.rgba[:] = [.12, .5, .95, 1.]
                ref = renderer.render().copy()
                frame = Image.fromarray(np.concatenate((real, ref), axis=1))
                draw = ImageDraw.Draw(frame)
                labels = ('KINEMATIC REFERENCE' if reference_only else 'POLICY - REAL PHYSICS',
                          'ANCHOR-ALIGNED REFERENCE')
                for x, text in zip((12, 972), labels):
                    draw.rectangle((x-4, 8, x+350, 42), fill='black')
                    draw.text((x, 12), text, fill='white', font=font)
                if speed_label:
                    draw.rectangle((8, 502, 1300, 538), fill='black')
                    draw.text((12, 506), speed_label, fill='white', font=font)
                writer.stdin.write(np.asarray(frame).tobytes())
            writer.stdin.close()
            if writer.wait() != 0: raise RuntimeError('Video encoding failed')
    left = 'kinematic reference' if reference_only else 'real physics'
    print(f'Video: {path}; left={left}, right=anchor-aligned reference; tracking view', flush=True)


def aligned_reference(env, index=0):
    ref = env.ref.sample(env.frame[index:index+1])
    actual = env.qpos[index:index+1, 3:7]
    # Gauge-fixed reference has zero yaw; align it to the actual robot's yaw.
    from go2_mujoco_mimic.tasks.mimic.rewards import quat_mul
    q = quat_mul(yaw_quat(actual), ref['quat'])[0].cpu().numpy()
    qp = env.qpos[index].cpu().numpy().copy()
    qp[:2] = env.qpos[index, :2].cpu().numpy()
    qp[2] = float(ref['pos'][0, 2])
    qp[3:7] = q
    qp[env.qids.cpu().numpy()] = ref['joints'][0].cpu().numpy()
    return qp


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--motion', type=Path, default=MOTION)
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--random-start', action='store_true')
    p.add_argument('--reference-only', action='store_true')
    p.add_argument('--video', type=Path)
    args = p.parse_args()
    if args.num_envs < 1: p.error('num-envs must be positive')
    if args.reference_only:
        cpu = Go2Environment(600, 50)
        frames = np.load(args.motion)
        records = []
        for row in frames:
            qp = cpu.data.qpos.copy(); qp[:3] = row[48:51]; qp[3:7] = row[51:55]
            qp[cpu.qpos_ids] = row[6:18].reshape(3, 4).T.reshape(12)
            records.append(qp)
        meta = json.loads(args.motion.with_name('clip.json').read_text())
        speed = meta.get('playback_speed', 1.)
        label = timing_label(meta, 'REFERENCE TIMING')
        name = (f'reference_megasam_{speed:g}x.mp4' if meta.get('camera_compensation') == 'MegaSaM' else
                f'reference_{speed:g}x.mp4' if meta.get('speed_status') == 'user_selected_normal_speed'
                else 'reference_slow1x.mp4' if speed == 1 else f'reference_est{speed:g}x.mp4')
        default_video = PROJECT/'videos'/name
        render(np.array(records), np.array(records), args.video or default_video,
               reference_only=True, speed_label=label)
        print('Reference-only kinematic playback, not a policy result.')
        return
    if args.checkpoint is None: p.error('--checkpoint is required for policy evaluation')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('format') != FORMAT: raise ValueError('Not an anchor-video checkpoint')
    if ck.get('action_mapping') != ACTION_MAPPING: raise ValueError('Action mapping mismatch')
    motion = Path(ck['motion'])
    if sha(motion) != ck['motion_sha256']: raise ValueError('Reference hash mismatch')
    meta = json.loads(motion.with_name('clip.json').read_text())
    cfg = make_config(args.num_envs, meta['duration_s']+.02)
    cfg.task.random_reference_start = args.random_start
    cfg.task.observation_noise = False
    torch.manual_seed(args.seed)
    env = AnchorEnv(cfg, motion); env.enable_cuda_graph()
    actor = MotionPolicy(hidden=ck['train']['actor_hidden']).to(env.device).eval()
    actor.load_state_dict(ck['actor'])
    live = torch.ones(args.num_envs, device=env.device, dtype=torch.bool)
    total, seconds = torch.zeros(args.num_envs, device=env.device), torch.zeros(args.num_envs, device=env.device)
    completed = torch.zeros_like(live)
    expected = (env.frame_limit-env.start_frame).float()/env.ref.fps
    records, refs = [env.qpos[0].cpu().numpy().copy()], [aligned_reference(env)]
    errors, joint_errors, physical_distance = [], [], []
    start_xy = env.qpos[:, :2].clone()
    for _ in range(int(np.ceil(cfg.episode_seconds/env.dt))+2):
        previous_frame = env.frame.clone()
        _, reward, failed, truncated, _ = env.step(actor(env.observation()))
        total += torch.where(live, reward, 0.)
        seconds += live*env.dt
        completed |= live & truncated & ~failed
        actual_pos, actual_q = env.xpos[:, env.bids], env.xquat[:, env.bids]
        ref = env.ref.sample(env.frame)
        # Relative body-position diagnostic, independent of global XY/yaw.
        zero = torch.zeros_like(actual_pos)
        af = features(dict(pos=actual_pos, quat=actual_q, linear_vel=zero, angular_vel=zero))
        rf = features(dict(pos=ref['body_pos'], quat=ref['body_quat'], linear_vel=zero, angular_vel=zero))
        if live.any():
            errors.extend((af['pos'][live]-rf['pos'][live]).square().sum(-1).mean(-1).sqrt().tolist())
            joint_errors.extend((env.qpos[live][:, env.qids]-ref['joints'][live]).square().mean(-1).tolist())
        if bool(live[0]):
            records.append(env.qpos[0].cpu().numpy().copy()); refs.append(aligned_reference(env))
        done = failed | truncated
        physical_distance.extend((env.qpos[live & done, :2]-start_xy[live & done]).norm(dim=-1).tolist())
        live &= ~done
        if not live.any(): break
        env.reset(done)
    result = dict(cycle=ck['cycle'], num_envs=args.num_envs, seed=args.seed,
                  reference_speed=meta.get('playback_speed', 1.),
                  speed_status=meta.get('speed_status', 'source_slow_motion_unadjusted'),
                  random_reference_start=args.random_start, completion_rate=float(completed.float().mean()),
                  episode_duration_mean_s=float(seconds.mean()), expected_reference_span_mean_s=float(expected.mean()),
                  episode_return_mean=float(total.mean()), relative_body_position_rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
                  joint_position_rmse_rad=float(np.sqrt(np.mean(joint_errors))),
                  endpoint_displacement_mean_m=float(np.mean(physical_distance)) if physical_distance else None,
                  notes='Real physics; no exploration or observation noise. Completion is reference-end without failure. Deterministic start trials share the same initial state.')
    out = args.checkpoint.parent/f'evaluation_{"random" if args.random_start else "start"}_{args.seed}.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),
                        qpos=np.array(records), reference_qpos=np.array(refs), fps=50.)
    print(json.dumps(result, indent=2), flush=True)
    if args.video:
        label = timing_label(meta, 'TRAINING REFERENCE')
        render(np.array(records), np.array(refs), args.video, speed_label=label)


if __name__ == '__main__':
    main()
