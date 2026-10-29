"""检查 MuJoCo Go2 场景；当前只是 PD 站姿检查。"""
import argparse
import time

import mujoco
import numpy as np

from .environment import Go2Environment, JOINT_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--seconds', type=float, default=5.)
    args = parser.parse_args()
    if args.seconds <= 0 or not np.isfinite(args.seconds):
        parser.error('seconds 必须为正有限值')
    env = Go2Environment()
    print('MuJoCo', mujoco.__version__, '| 关节顺序:', ', '.join(JOINT_NAMES), flush=True)
    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(env.model, env.data)
    try:
        steps = round(args.seconds / env.control_dt)
        for step in range(steps):
            if viewer is not None and not viewer.is_running():
                break
            start = time.perf_counter()
            state = env.step(env.default_q)
            if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all():
                raise FloatingPointError('仿真产生非有限状态')
            if step % 30 == 0:
                print(f"t={(step+1)*env.control_dt:.2f}s height={state['root_position'][2]:.3f}m", flush=True)
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0., env.control_dt - (time.perf_counter() - start)))
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == '__main__':
    main()
