"""Audit reference time scaling and an open-loop real-physics PD baseline."""
import json
import numpy as np
import mujoco
from scipy.signal import find_peaks
from go2_mujoco_mimic.environment import Go2Environment
from .config import PROJECT, DEFAULT_SPEED


def poses(cpu, frames):
    out = np.tile(cpu.data.qpos, (len(frames), 1))
    out[:, :3], out[:, 3:7] = frames[:, 48:51], frames[:, 51:55]
    out[:, cpu.qpos_ids] = frames[:, 6:18].reshape(-1, 3, 4).transpose(0, 2, 1).reshape(-1, 12)
    return out


def velocities(model, qp, fps):
    v = np.empty((len(qp), model.nv))
    for i in range(len(qp)):
        a, b = max(0, i-1), min(len(qp)-1, i+1)
        mujoco.mj_differentiatePos(model, v[i], (b-a)/fps, qp[a], qp[b])
    return v


def slow_period(joints, fps):
    x = joints-joints.mean(0)
    x = x / np.maximum(x.std(0), .03)
    lo, hi = int(.5*fps), min(int(2.5*fps), len(x)//2)
    lags = np.arange(lo, hi+1)
    scores = np.array([(x[:-k]*x[k:]).mean() for k in lags])
    peaks, _ = find_peaks(scores)
    if len(peaks) == 0: return None
    best = peaks[np.argmax(scores[peaks])]
    return float(lags[best]/fps)


def pd_probe(cpu, qp, velocity):
    m, d = cpu.model, cpu.data
    mujoco.mj_resetData(m, d)
    d.qpos[:], d.qvel[:] = qp[0], velocity[0]
    mujoco.mj_forward(m, d)
    ground = m.geom('ground').id
    feet = {m.geom(leg).id for leg in ('FL', 'FR', 'RL', 'RR')}
    reason = 'reference_end'
    errors, saturation = [], []
    for i in range(1, len(qp)):
        for _ in range(cpu.substeps):
            tau = 25*(qp[i, cpu.qpos_ids]-d.qpos[cpu.qpos_ids])-.5*d.qvel[cpu.dof_ids]
            d.ctrl[cpu.actuator_ids] = np.clip(tau, -23.7, 23.7)
            saturation.append(np.mean(np.abs(tau)>23.7))
            mujoco.mj_step(m, d)
            bad = any((c.geom1 == ground and c.geom2 not in feet)
                      or (c.geom2 == ground and c.geom1 not in feet) for c in d.contact)
            if bad:
                reason = 'nonfoot_ground_contact'
                break
        errors.append(np.mean((d.qpos[cpu.qpos_ids]-qp[i, cpu.qpos_ids])**2))
        if reason != 'reference_end': break
        if not np.isfinite(d.qpos).all():
            reason = 'nonfinite'; break
        if abs(d.qpos[2]-qp[i, 2]) > .18 or d.xmat[m.body('base_link').id].reshape(3, 3)[2, 2] < .4:
            reason = 'height_or_tilt'; break
    return dict(seconds=float(d.time), end_reason=reason,
                joint_rmse_rad=float(np.sqrt(np.mean(errors))),
                torque_saturation_fraction=float(np.mean(saturation)),
                note='Open-loop reference joint PD, not an RL result or proof of RL infeasibility.')


def main():
    results = []
    period = None
    for directory in ('side_run', 'side_run_3x', 'side_run_est5x'):
        path = PROJECT/'data'/directory/'motion.npy'
        frames = np.load(path)
        metadata = json.loads(path.with_name('clip.json').read_text())
        fps = float(metadata['fps'])
        cpu = Go2Environment(600, 50)
        qp = poses(cpu, frames)
        v = velocities(cpu.model, qp, fps)
        acceleration = np.gradient(v, 1/fps, axis=0)
        if directory == 'side_run': period = slow_period(qp[:, cpu.qpos_ids], fps)
        speed = metadata.get('playback_speed', 1.)
        out = dict(reference=directory, speed=speed,
                   status=metadata.get('speed_status', 'source_slow_motion_unadjusted'),
                   frames=len(frames), duration_s=(len(frames)-1)/fps,
                   inferred_joint_period_s=period/speed if period else None,
                   peak_joint_velocity_rad_s=float(np.abs(v[:, cpu.dof_ids]).max()),
                   peak_joint_acceleration_rad_s2=float(np.abs(acceleration[:, cpu.dof_ids]).max()),
                   quaternion_norm_max_error=float(np.abs(np.linalg.norm(qp[:, 3:7], axis=-1)-1).max()),
                   joint_limit_excess_rad=float(np.maximum(np.maximum(cpu.model.jnt_range[cpu.joint_ids, 0]-qp[:, cpu.qpos_ids],
                       qp[:, cpu.qpos_ids]-cpu.model.jnt_range[cpu.joint_ids, 1]), 0).max()),
                   physics_dt_s=float(cpu.model.opt.timestep), gravity=cpu.model.opt.gravity.tolist(),
                   pd_baseline=pd_probe(cpu, qp, v))
        results.append(out)
    report = dict(source_speed_verified=False,
                  adopted_normal_speed=DEFAULT_SPEED, speed_selection='user_selected_normal_speed',
                  five_x_assumption='~120fps capture played at ~24fps; capture FPS unavailable, not a verified recovery.',
                  results=results)
    path = PROJECT/'logs/timing_comparison.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
