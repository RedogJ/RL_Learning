"""将 80Hz 单次后空翻运动学预览重采样成 30Hz 参考；不改变源文件。"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('projects/go2_mujoco_mimic/data/solo12_backflip/go2_landing_preview.npz'))
    p.add_argument('--out-dir',type=Path,default=Path('projects/go2_mujoco_mimic/data/solo12_backflip/reference_30hz'))
    p.add_argument('--fps',type=float,default=30.)
    p.add_argument('--hold',type=float,default=.4)
    args=p.parse_args()
    if args.fps <= 0 or args.hold < 0: p.error('fps 必须为正，hold 不可为负')
    with np.load(args.source,allow_pickle=False) as z:
        times=z['time'].astype(float);pos=z['position'].astype(float);q=z['quaternion_wxyz'].astype(float);joint=z['joints'].astype(float)
    if times.ndim!=1 or len(times)<2 or np.any(np.diff(times)<=0) or pos.shape!=(len(times),3) or q.shape!=(len(times),4) or joint.shape!=(len(times),12):
        raise ValueError('参考轨迹形状或时间无效')
    # 最后帧及静止保持延伸至最近的控制帧；旋转在最后帧之后保持不变。
    end_step=int(np.ceil((times[-1]+args.hold)*args.fps))
    grid=np.arange(end_step+1)/args.fps
    sample=np.clip(grid,times[0],times[-1])
    xyz=np.stack([np.interp(sample,times,pos[:,i]) for i in range(3)],axis=1)
    j=np.stack([np.interp(sample,times,joint[:,i]) for i in range(12)],axis=1)
    quat=Slerp(times,Rotation.from_quat(q[:,[1,2,3,0]]))(sample).as_quat()[:,[3,0,1,2]]
    # 解除四元数符号跳变（旋转保持相同）。
    for i in range(1,len(quat)):
        if np.dot(quat[i-1],quat[i])<0:quat[i]*=-1
    frames=np.zeros((len(grid),61),dtype=np.float32)
    frames[:,6:18]=j.reshape(-1,4,3).transpose(0,2,1).reshape(-1,12)
    frames[:,48:51]=xyz;frames[:,51:55]=quat
    args.out_dir.mkdir(parents=True,exist_ok=True)
    np.save(args.out_dir/'motion.npy',frames)
    (args.out_dir/'clip.json').write_text(json.dumps({'name':'go2_solo12_backflip_kinematic_preview','fps':args.fps,'mode':'once','motion_duration_s':float(times[-1]),'hold_s':args.hold,'notes':'Kinematic approximation; no dynamic feasibility validation.','source':str(args.source)},indent=2))
    print(f'生成 {len(grid)} 帧 {args.fps:g}Hz；单次时长 {grid[-1]:.3f}s，其中原动作 {times[-1]:.3f}s；文件 {args.out_dir / "motion.npy"}')

if __name__=='__main__':main()
