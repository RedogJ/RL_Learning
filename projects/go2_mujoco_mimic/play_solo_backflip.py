"""Solo12 后空翻轨迹映射到 Go2 的运动学预览；不施力、不运行策略。"""
import argparse
from pathlib import Path
import time
import h5py
import mujoco
import numpy as np
from .environment import Go2Environment

DEFAULT = Path('/workspace/go2_mujoco_mimic/data/solo12_backflip/backflip_v4.hdf5')


def load(path):
    with h5py.File(path, 'r') as f:
        q = f['q'][:]
        durations = f['knot_durations'][:]
    if q.ndim != 2 or q.shape[1] != 18 or len(q) != len(durations) or not np.isfinite(q).all() or not np.isfinite(durations).all() or (durations <= 0).any():
        raise ValueError('轨迹格式或时间不合法')
    # 作者使用 modified Rodrigues parameters，q=[(1-r²),2r]/(1+r²)，遇影子切换须连续化符号。
    r=q[:,3:6];r2=(r*r).sum(axis=1,keepdims=True)
    quat=np.concatenate(((1-r2)/(1+r2),2*r/(1+r2)),axis=1)
    for i in range(1,len(quat)):
        if np.dot(quat[i-1],quat[i]) < 0: quat[i] *= -1
    # Pinocchio Solo12 关节顺序按 URDF 的 FL、FR、HL、HR，每条 HAA、HFE、KFE；HL/HR 对应 RL/RR。
    return q[:,:3].copy(),quat,q[:,6:18].copy(),np.concatenate(([0.],np.cumsum(durations[:-1])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,default=DEFAULT)
    p.add_argument('--speed',type=float,default=.5)
    p.add_argument('--check',action='store_true')
    p.add_argument('--mapping',choices=('retargeted','limited','raw'),default='retargeted',help='retargeted 为落地预览，limited 为限位近似，raw 为原始角度')
    args=p.parse_args()
    if args.speed<=0 or not np.isfinite(args.speed):p.error('speed 必须是有限正数')
    pos,quat,joints,times=load(args.motion)
    env=Go2Environment(checkerboard=True)
    limits=env.model.jnt_range[env.joint_ids]
    raw_joints = joints.copy()
    if args.mapping == 'retargeted':
        path=args.motion.with_name('go2_landing_preview.npz')
        if not path.is_file(): raise FileNotFoundError(f'未找到重定向预览 {path}')
        with np.load(path,allow_pickle=False) as data:
            pos=data['position'].copy();quat=data['quaternion_wxyz'].copy();joints=data['joints'].copy();times=data['time'].copy()
        if joints.shape != raw_joints.shape or pos.shape != (len(joints),3) or quat.shape != (len(joints),4) or not all(np.isfinite(x).all() for x in (pos,quat,joints,times)):
            raise ValueError('重定向轨迹无效')
    if args.mapping == 'limited':
        # 保留两段腿的中线方向：等长两连杆的中线角是 thigh + calf/2。
        # 这是运动学近似，不保证原足端位置、动态可行性或无穿模。
        for leg in range(4):
            thigh, calf = 3*leg+1, 3*leg+2
            limited_calf=np.clip(-np.abs(joints[:,calf]),limits[calf,0],limits[calf,1])
            joints[:,thigh] += .5*(joints[:,calf]-limited_calf)
            joints[:,calf] = limited_calf
        joints=np.clip(joints,limits[:,0],limits[:,1])
    violation=(joints<limits[:,0])|(joints>limits[:,1])
    excess=np.maximum(limits[:,0]-joints,0)+np.maximum(joints-limits[:,1],0)
    # 仅把首帧最低脚掌移到地面；保留后续原始高度变化，不做动力学修正。
    env.data.qpos[:3]=pos[0];env.data.qpos[3:7]=quat[0];env.data.qpos[env.qpos_ids]=joints[0];mujoco.mj_forward(env.model,env.data)
    ids=[mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,f'{leg}_foot') for leg in ('FL','FR','RL','RR')]
    z_shift=0. if args.mapping == 'retargeted' else .022-env.data.xpos[ids,2].min()
    pos[:,2]+=z_shift
    print(f'帧数={len(times)} 时间={times[-1]:.3f}s；MRP 姿态已连续化；首帧高度平移={z_shift:.3f}m',flush=True)
    print(f'映射={args.mapping}；平均关节角修正={np.abs(joints-raw_joints).mean():.3f}rad，最大修正={np.abs(joints-raw_joints).max():.3f}rad',flush=True)
    print(f'Go2 关节限位：越界样本={violation.sum()}/{violation.size} ({violation.mean():.1%}) 最大越界={excess.max():.3f}rad',flush=True)
    for i,name in enumerate(env.joint_ids):
        if violation[:,i].any():
            print(f'  {mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_JOINT,int(name))}: {violation[:,i].sum()}帧，范围 [{joints[:,i].min():.3f},{joints[:,i].max():.3f}]，限制 [{limits[i,0]:.3f},{limits[i,1]:.3f}]',flush=True)
    print('这是运动学预览；未经动力学或策略跟踪验证，不可直接作为可行训练数据。',flush=True)
    if args.check:return
    from mujoco import viewer
    with viewer.launch_passive(env.model,env.data) as v:
        v.cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,'base_link')
        v.cam.distance=2.;v.cam.azimuth=135;v.cam.elevation=-15
        start=time.monotonic()
        while v.is_running():
            t=((time.monotonic()-start)*args.speed)%(times[-1]+.8)
            index=min(int(np.searchsorted(times,t,side='right')-1),len(times)-1)
            env.data.qpos[:3]=pos[index];env.data.qpos[3:7]=quat[index];env.data.qpos[env.qpos_ids]=joints[index]
            mujoco.mj_forward(env.model,env.data);v.sync();time.sleep(1/120)


if __name__=='__main__':main()
