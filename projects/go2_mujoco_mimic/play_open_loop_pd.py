'''可视化开环 PD 跟踪后空翻参考，不使用策略；蓝色为参考，实体为开环 PD 结果。'''
import argparse
import time
from pathlib import Path
from threading import Event
import mujoco
import numpy as np
from mujoco import viewer as mj_viewer
from .environment import Go2Environment

DEFAULT = Path('/workspace/go2_mujoco_mimic/data/solo12_backflip/reference_30hz/motion.npy')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion', type=Path, default=DEFAULT)
    p.add_argument('--speed', type=float, default=0.25, help='播放倍速，例如 0.25 表示四分之一速度')
    p.add_argument('--hold', type=float, default=1.0, help='参考播完后停留的秒数（真实时间），再重新开始')
    args = p.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0 or args.hold < 0:
        p.error('--speed 必须是有限正数，--hold 不可为负')
    ref = np.load(args.motion, allow_pickle=False)
    env = Go2Environment(checkerboard=True)
    m, d = env.model, env.data
    ghost = mujoco.MjData(m)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[2] = 1
    dt = 1 / 30

    def set_q(data, row):
        data.qpos[:3] = row[48:51]
        data.qpos[3:7] = row[51:55]
        data.qpos[env.qpos_ids] = row[6:18].reshape(3, 4).T.reshape(-1)

    def reset():
        mujoco.mj_resetData(m, d)
        set_q(d, ref[0]); start = d.qpos.copy()
        set_q(d, ref[1]); end = d.qpos.copy()
        set_q(d, ref[0])
        mujoco.mj_differentiatePos(m, d.qvel, dt, start, end)
        mujoco.mj_forward(m, d)

    def show_ghost(v, idx):
        set_q(ghost, ref[idx])
        mujoco.mj_forward(m, ghost)
        with v.lock():
            scn = v.user_scn
            scn.ngeom = 0
            mujoco.mjv_addGeoms(m, ghost, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_ALL, scn)
            for i in range(scn.ngeom):
                g = scn.geoms[i]
                g.matid = -1
                g.rgba[:] = (0.08, 0.45, 1.0, 0.30)

    paused = Event()

    def on_key(keycode):
        if keycode == 32:
            if paused.is_set():
                paused.clear()
                print('[播放] 继续', flush=True)
            else:
                paused.set()
                print('[播放] 暂停；再次按空格继续', flush=True)

    v = mj_viewer.launch_passive(m, d, key_callback=on_key)
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.5, 135, -20
    print(f'开环 PD 回放：{len(ref)} 帧 {dt*(len(ref)-1):.2f}s；倍速={args.speed:g}；蓝色=参考，实体=开环 PD；空格暂停，关闭窗口退出', flush=True)
    try:
        while v.is_running():
            reset()
            show_ghost(v, 0)
            v.sync()
            for idx in range(1, len(ref)):
                while paused.is_set() and v.is_running():
                    v.sync()
                    time.sleep(0.02)
                if not v.is_running():
                    break
                t0 = time.monotonic()
                raw = ref[idx, 6:18].reshape(3, 4).T.reshape(-1)
                target = env.default_q + np.clip(raw - env.default_q, -1.0, 1.0)
                with v.lock():
                    for _ in range(env.substeps):
                        torque = 25 * (target - d.qpos[env.qpos_ids]) - 0.5 * d.qvel[env.dof_ids]
                        d.ctrl[env.actuator_ids] = np.clip(torque, -23.7, 23.7)
                        mujoco.mj_step(m, d)
                show_ghost(v, idx)
                v.sync()
                if idx % 6 == 0:
                    print(f'参考帧={idx} 机身高度 实际={d.qpos[2]:.3f} 参考={ref[idx, 50]:.3f}m', flush=True)
                time.sleep(max(0.0, dt / args.speed - (time.monotonic() - t0)))
            else:
                print(f'本轮结束：末帧机身高度 实际={d.qpos[2]:.3f}m 参考={ref[-1, 50]:.3f}m', flush=True)
                end = time.monotonic() + args.hold
                while time.monotonic() < end and v.is_running():
                    v.sync()
                    time.sleep(0.02)
    finally:
        v.close()


if __name__ == '__main__':
    main()
