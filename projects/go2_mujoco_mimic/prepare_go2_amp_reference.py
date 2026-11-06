"""将 Go2 AMP JSON 转成仅供本工程运动学播放的 61 列参考文件。"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--fps', type=float, help='重采样频率；省略时保留原频率')
    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    source = np.asarray(data['Frames'], dtype=np.float64)
    dt = float(data['FrameDuration'])
    if source.ndim != 2 or source.shape[1] != 49 or len(source) < 2 or not np.isfinite(source).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError('轨迹需要至少两帧有限的 49 维数据和正时间间隔')
    if args.fps is not None:
        if not np.isfinite(args.fps) or args.fps <= 0:
            parser.error('fps 必须为正的有限值')
        from scipy.spatial.transform import Rotation, Slerp
        times = np.arange(len(source)) * dt
        sample = np.minimum(np.arange(int(np.ceil(times[-1] * args.fps)) + 1) / args.fps, times[-1])
        interpolated = np.stack([np.interp(sample, times, source[:, i]) for i in range(source.shape[1])], axis=1)
        interpolated[:, 3:7] = Slerp(times, Rotation.from_quat(source[:, 3:7]))(sample).as_quat()
        source, dt = interpolated, 1.0 / args.fps
    quat = source[:, 3:7]
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    if (norm < 1e-8).any():
        raise ValueError('四元数不能为零')
    quat = quat / norm  # 输入 xyzw，输出 wxyz
    # q 和 -q 表示同一旋转，但策略直接读取四元数，需要相邻帧符号连续。
    for index in range(1, len(quat)):
        if np.dot(quat[index - 1], quat[index]) < 0:
            quat[index] *= -1
    result = np.zeros((len(source), 61), dtype=np.float32)
    result[:, 6:18] = source[:, 7:19].reshape(-1, 4, 3).transpose(0, 2, 1).reshape(-1, 12)
    result[:, 48:51] = source[:, :3]
    result[:, 51:55] = quat[:, [3, 0, 1, 2]]
    # 原数据是机身坐标系 FL/FR/RL/RR；播放器的检查字段是世界坐标系 FL/RL/FR/RR。
    feet = source[:, 19:31].reshape(-1, 4, 3)
    vector = quat[:, None, :3]
    rotated = feet + 2 * np.cross(vector, np.cross(vector, feet) + quat[:, None, 3:4] * feet)
    result[:, 36:48] = (rotated + source[:, None, :3])[:, [0, 2, 1, 3]].reshape(-1, 12)
    args.destination.mkdir(parents=True, exist_ok=True)
    np.save(args.destination / 'motion.npy', result)
    metadata = {'fps': 1.0 / dt, 'source': str(args.source), 'mode': 'kinematic_reference',
                'frames': len(source), 'duration_s': (len(source)-1)*dt,
                'notes': 'Root and joints define reference; current Warp environment computes body poses and velocities from them.'}
    (args.destination / 'clip.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f"转换 {len(source)} 帧，{1/dt:g} Hz，时长 {(len(source)-1)*dt:.2f}s：{args.destination / 'motion.npy'}")


if __name__ == '__main__':
    main()
