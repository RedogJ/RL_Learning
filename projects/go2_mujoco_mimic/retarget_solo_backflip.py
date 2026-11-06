"""约束 Go2 足端落地的运动学近似；不输出动力学可行性结论。"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from .environment import Go2Environment
from .play_solo_backflip import load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('projects/go2_mujoco_mimic/data/solo12_backflip/backflip_v4.hdf5'))
    parser.add_argument('--output',type=Path,default=Path('projects/go2_mujoco_mimic/data/solo12_backflip/go2_landing_preview.npz'))
    args=parser.parse_args()
    pos,quat,q,t=load(args.source);env=Go2Environment();model,data=env.model,env.data
    lim=model.jnt_range[env.joint_ids];ground=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'ground')
    feet=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'{leg}_foot') for leg in ('FL','FR','RL','RR')]
    original=q.copy()
    for leg in range(4):
        thigh,calf=3*leg+1,3*leg+2
        new=np.clip(-np.abs(q[:,calf]),lim[calf,0],lim[calf,1])
        q[:,thigh]+=.5*(q[:,calf]-new);q[:,calf]=new
    q=np.clip(q,lim[:,0]+1e-7,lim[:,1]-1e-7)
    data.qpos[:3]=pos[0];data.qpos[3:7]=quat[0];data.qpos[env.qpos_ids]=q[0];mujoco.mj_forward(model,data)
    pos[:,2]+=.022-data.xpos[feet,2].min()
    base_q=q.copy();base_pos=pos.copy()
    prev=None
    # 后 0.35s 渐进约束四足，避免突然跳到落地姿态。
    for i in range(len(t)):
        blend=np.clip((t[i]-(t[-1]-.35))/.25,0,1)
        if blend <= 0:continue
        target=base_q[i].copy();z0=base_pos[i,2]; prior=target if prev is None else prev
        def residual(x):
            data.qpos[:3]=[pos[i,0],pos[i,1],x[-1]];data.qpos[3:7]=quat[i];data.qpos[env.qpos_ids]=x[:-1]
            mujoco.mj_kinematics(model,data)
            footz=data.xpos[feet,2]
            return np.r_[20*blend*(footz-.022), .8*(x[:-1]-target), 1.2*blend*(x[:-1]-prior), 2*(x[-1]-z0)]
        low=np.r_[lim[:,0]+1e-7,z0-.3];high=np.r_[lim[:,1]-1e-7,z0+.3]
        x0=np.clip(np.r_[target,z0],low+1e-8,high-1e-8)
        result=least_squares(residual,x0,bounds=(low,high),max_nfev=40,ftol=1e-6)
        q[i]=result.x[:-1];pos[i,2]=result.x[-1];prev=q[i].copy()
    # 对所有帧，将接触地面的几何体提至地面以上（平面上的支持点）。
    # 使用 mj_geomDistance 检查所有碰撞 geom；该操作只调整根部高度。
    geomids=[i for i in range(model.ngeom) if i != ground and (model.geom_contype[i] or model.geom_conaffinity[i])]
    shifts=[];minimum=[];feet_heights=[]
    for i in range(len(t)):
        data.qpos[:3]=pos[i];data.qpos[3:7]=quat[i];data.qpos[env.qpos_ids]=q[i];mujoco.mj_forward(model,data)
        ds=[mujoco.mj_geomDistance(model,data,ground,g,10.,None) for g in geomids]
        delta=max(0.,-min(ds)+1e-4);pos[i,2]+=delta;shifts.append(delta);minimum.append(min(ds)+delta)
        feet_heights.append(data.xpos[feet,2].copy()+delta-.022)
    dz=np.diff(pos[:,2]);dq=np.diff(q,axis=0);feet_heights=np.array(feet_heights)
    report={'frames':len(t),'duration_s':float(t[-1]),'max_ground_lift_m':float(max(shifts)),
            'min_collision_distance_m':float(min(minimum)),
            'max_frame_root_height_change_m':float(np.abs(dz).max()),
            'max_frame_joint_change_rad':float(np.abs(dq).max()),
            'last_feet_clearance_m':feet_heights[-1].tolist(),
            'max_deviation_from_raw_rad':float(np.abs(q-original).max()),
            'note':'Kinematic preview only; dynamically feasibility, contact forces and in-between-frame penetration not verified.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,position=pos,quaternion_wxyz=quat,joints=q,time=t)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
