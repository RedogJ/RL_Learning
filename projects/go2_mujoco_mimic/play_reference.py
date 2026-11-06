"""按 60 Hz 播放 Kine2Go 参考运动学轨迹，不进行动力学控制。"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .environment import Go2Environment


DEFAULT = Path(__file__).parent / 'data/kine2go/ai4_dog_canter/motion.npy'


def yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def cycle_pose(frame, first, last, cycle):
    """用首尾的水平位移与航向变化拼接；保持轨迹自身的高度和倾斜。"""
    delta = (yaw(last[51:55]) - yaw(first[51:55]) + np.pi) % (2*np.pi) - np.pi
    dp = last[48:50].astype(float) - first[48:50]
    if abs(delta) < 1e-10:
        shift = cycle * dp
    else:
        z = complex(*dp) * (1 - np.exp(1j * cycle * delta)) / (1 - np.exp(1j * delta))
        shift = np.array([z.real, z.imag])
    angle = cycle * delta
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    position = frame[48:51].astype(float).copy()
    position[:2] = first[48:50] + shift + rotation @ (frame[48:50] - first[48:50])
    w, x, y, z = frame[51:55]
    c, s = np.cos(angle/2), np.sin(angle/2)
    quaternion = np.array([c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w])
    return position, quaternion


def apply_frame(env, frame, pose=None):
    position, quaternion = pose if pose is not None else (frame[48:51], frame[51:55])
    env.data.qpos[:3] = position
    env.data.qpos[3:7] = quaternion  # wxyz
    # 本文件的关节按 hip FL/FR/RL/RR、thigh FL/FR/RL/RR、calf FL/FR/RL/RR 排列。
    # 由参考足部世界坐标对 MuJoCo 正运动学逐帧核验，不同于数据卡列出的次序。
    env.data.qpos[env.qpos_ids] = frame[6:18].reshape(3, 4).T.reshape(-1)
    mujoco.mj_forward(env.model, env.data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion', type=Path, default=DEFAULT)
    p.add_argument('--fps', type=float, default=None)
    p.add_argument('--speed', type=float, default=1.0)
    p.add_argument('--once', action='store_true')
    p.add_argument('--start', type=int, default=None, help='起始帧（包含）；默认使用作者配置')
    p.add_argument('--end', type=int, default=None, help='结束帧（不包含）；默认使用作者配置')
    p.add_argument('--check', action='store_true', help='无窗口检查参考足部正运动学误差')
    p.add_argument('--checkerboard', action='store_true', help='显示格子地面')
    args = p.parse_args()
    frames = np.load(args.motion, allow_pickle=False)
    if frames.ndim != 2 or frames.shape[1] != 61 or len(frames) == 0 or not np.isfinite(frames).all():
        raise ValueError('轨迹应为有限的 (T, 61) 数组')
    metadata = args.motion.with_name('clip.json')
    info = json.loads(metadata.read_text()) if metadata.is_file() else {}
    config_path = args.motion.with_name('config.json')
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    interval = config.get('policy_cfg', {}).get('motion_range', [0, None])
    start = args.start if args.start is not None else (interval[0] or 0)
    end = args.end if args.end is not None else interval[1]
    # 作者使用 Python 切片；-1 表示排除最后一帧，而不是取到末尾。
    end = len(frames) if end is None else (len(frames) + end if end < 0 else end)
    if not 0 <= start < end <= len(frames):
        p.error(f'帧范围无效：[{start}, {end})，文件共有 {len(frames)} 帧')
    frames = frames[start:end]
    fps = args.fps if args.fps is not None else info.get('fps', 60)
    if not np.isfinite([fps, args.speed]).all() or fps <= 0 or args.speed <= 0:
        raise ValueError('fps 和 speed 必须为正有限值')
    env = Go2Environment(checkerboard=args.checkerboard)
    print(f"动作：{args.motion}；帧 [{start}, {end})，{len(frames)} 帧，{fps:g} Hz，{len(frames)/fps:.2f} 秒；运动学回放（不施力、不运行策略）", flush=True)
    if args.check:
        ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f'{leg}_foot') for leg in ('FL','RL','FR','RR')]
        errors = []
        for frame in frames:
            apply_frame(env, frame)
            errors.append(np.linalg.norm(env.data.xpos[ids] - frame[36:48].reshape(4,3), axis=-1))
        print(f'脚掌位置误差：平均 {np.mean(errors):.8f} m，最大 {np.max(errors):.8f} m', flush=True)
        return
    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 135, -20
        started = time.monotonic()
        while viewer.is_running():
            frame_no = int((time.monotonic()-started)*fps*args.speed)
            if args.once and frame_no >= len(frames):
                break
            cycle, index = divmod(frame_no, len(frames))
            apply_frame(env, frames[index], cycle_pose(frames[index], frames[0], frames[-1], cycle))
            viewer.sync()
            time.sleep(1/120)


if __name__ == '__main__':
    main()
