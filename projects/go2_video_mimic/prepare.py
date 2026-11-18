"""Convert our trusted dog2go2 PKL to a noncyclic anchor training reference."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .config import PROJECT, DEFAULT_SPEED, DEFAULT_SOURCE


def convert(motion, fps=50., speed=1., preserve_world=False):
    if not all(np.isfinite(x) and x > 0 for x in (fps, speed)):
        raise ValueError('fps and speed must be positive finite values')
    if motion['robot_type'] != 'unitree_go2':
        raise ValueError('Expected our Go2 motion export')
    q = np.asarray(motion['dof_pos'], dtype=float)
    p = np.asarray(motion['root_pos'], dtype=float)
    quat = np.asarray(motion['root_rot'], dtype=float)
    if q.shape != (len(p), 12) or quat.shape != (len(p), 4) or len(p) < 4:
        raise ValueError('Invalid motion shapes')
    if not all(np.isfinite(x).all() for x in (q, p, quat)):
        raise ValueError('Motion contains nonfinite values')
    src_fps = float(motion['fps'])
    if not np.isfinite(src_fps) or src_fps <= 0:
        raise ValueError('Invalid source fps')
    t = np.arange(len(p))/src_fps
    times = np.arange(int(np.floor(t[-1]/speed*fps+1e-6))+1)/fps
    if len(times) < 4:
        raise ValueError('Retimed reference must contain at least four control frames')
    query = times*speed
    # PKL is FR FL RR RL, xyz + xyzw; native RL Go2 is FL FR RL RR + wxyz.
    q = q.reshape(-1, 4, 3)[:, [1, 0, 3, 2]].reshape(-1, 12)
    joint = np.stack([np.interp(query, t, col) for col in q.T], -1)
    r = Slerp(t, Rotation.from_quat(quat))(query)
    yaw = np.arctan2(r.as_matrix()[:, 1, 0], r.as_matrix()[:, 0, 0])
    if not preserve_world:
        r = Rotation.from_euler('z', -yaw[:, None])*r
    orientation = r.as_quat()[:, [3, 0, 1, 2]]
    for i in range(1, len(orientation)):
        if np.dot(orientation[i-1], orientation[i]) < 0:
            orientation[i] *= -1
    out = np.zeros((len(times), 61), dtype=np.float32)
    out[:, 6:18] = joint.reshape(-1, 4, 3).transpose(0, 2, 1).reshape(-1, 12)
    if preserve_world:
        out[:, 48:50] = np.stack([np.interp(query, t, col) for col in p[:, :2].T], -1)
    out[:, 50] = np.interp(query, t, p[:, 2])
    out[:, 51:55] = orientation
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--output', type=Path)
    p.add_argument('--speed', type=float,
                   help='Additional retiming factor; default 1 for the already-3x MegaSaM export, 3 for the bundled original slow source')
    p.add_argument('--speed-note')
    args = p.parse_args()
    compensated = args.source.resolve() == DEFAULT_SOURCE.resolve()
    old_source = args.source.resolve() == (PROJECT/'data/source/side_run_go2.pkl').resolve()
    if args.speed is None:
        args.speed = DEFAULT_SPEED if old_source else 1.
    source_playback_speed = DEFAULT_SPEED if compensated else 1.
    effective_speed = source_playback_speed * args.speed
    # Only read locally generated, trusted PKL files, never arbitrary downloads.
    with args.source.open('rb') as f:
        motion = pickle.load(f)
    out = convert(motion, speed=args.speed, preserve_world=compensated)
    selected_normal = (compensated and args.speed == 1.) or (old_source and args.speed == DEFAULT_SPEED)
    status = ('user_selected_normal_speed' if selected_normal else
              'source_slow_motion_unadjusted' if old_source and args.speed == 1 else
              'source_timing_preserved' if args.speed == 1 else 'estimated_unverified')
    note = args.speed_note or (
        ('User selected 3x normal speed; MegaSaM PKL already has this timing. No additional retiming.' if compensated else
         'User selected 3x playback as normal speed for this clip; source capture FPS was not supplied.')
        if selected_normal else 'Original slow-motion timeline retained.' if args.speed == 1 else
        'Source capture FPS unknown; multiplier is an explicit estimate.')
    if args.output is None:
        name = ('side_run_megasam_world_3x' if compensated and args.speed == 1 else
                f'side_run_megasam_world_est{effective_speed:g}x' if compensated else
                'side_run_3x' if selected_normal else 'side_run' if args.speed == 1
                else f'side_run_est{args.speed:g}x')
        args.output = PROJECT/'data'/name
    tracking = 'world_anchor' if compensated else 'relative_only'
    previous = args.output/'clip.json'
    if previous.exists():
        prior = json.loads(previous.read_text())
        if (prior.get('tracking', 'relative_only') != tracking or prior.get('playback_speed', 1.) != effective_speed or
                prior.get('camera_compensation', 'none') != ('MegaSaM' if compensated else 'none')):
            raise ValueError('Use a separate output directory when changing the reference or time scale')
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output/'motion.npy', out)
    metadata = dict(format='go2_anchor_video_v1', fps=50., frames=len(out),
                    duration_s=(len(out)-1)/50., cyclic=False,
                    playback_speed=effective_speed, retime_factor=args.speed,
                    source_playback_speed=source_playback_speed,
                    camera_compensation='MegaSaM' if compensated else 'none',
                    tracking=tracking, reference_id=args.output.name,
                    speed_status=status, speed_note=note,
                    source_span_s=(len(motion['root_pos'])-1)/float(motion['fps']),
                    source_used_span_s=(len(out)-1)/50.*args.speed,
                    source_trimmed_tail_s=max(0., (len(motion['root_pos'])-1)/float(motion['fps'])-(len(out)-1)/50.*args.speed),
                    source=str(args.source), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    joint_order='FL FR RL RR, each hip/thigh/calf', quaternion='wxyz',
                    notes=[('Source has MegaSaM camera compensation; retain world root pose and velocities for anchor tracking.' if compensated else
                            'Reference XY and yaw are gauge-fixed, not motion-compensated.'),
                           'Height, tilt and joints are retained.',
                           'Candidate reference is not certified physically feasible; inspect the source quality report.',
                           'Reference is resampled on a new 50Hz physical timeline, not merely encoded faster.',
                           'Velocities are recomputed from retimed FK; accelerations are audited separately.',
                           'Any sub-control-step tail is explicitly trimmed; endpoints are not loop-closed.',
                           'Unused legacy 61-column fields remain zero; body poses and velocities use model FK.'])
    (args.output/'clip.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
