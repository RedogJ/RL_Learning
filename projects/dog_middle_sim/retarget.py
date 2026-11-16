"""Go2 行走参考 → middle 足端约束 IK；只生成独立数据，不修改模型。"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation, Slerp

from .build_model import ROOT, SOURCE, OUTPUT, JOINTS, LEGS, stl_vertices, xyz

DEFAULT_SOURCE=ROOT/'projects/go2_mujoco_mimic/data/kine2go/ai4_dog_canter/motion.npy'
DEFAULT_OUTPUT=Path(__file__).resolve().parent/'data/go2_canter'
# Go2 前方+X/左方+Y → middle 前方-Y/左方+X；保持右手系和竖直方向。
B=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])


def quaternion(rotation):
    return rotation.as_quat()[...,[3,0,1,2]]


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[...,[1,2,3,0]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--frames',type=int,nargs=2,help='覆盖旧配置中的[start,end)')
    parser.add_argument('--stride-scale',type=float,default=0.65,help='腿长缩放外，水平步幅再缩放')
    parser.add_argument('--lift-scale',type=float,default=0.4,help='腿长缩放外，抬脚高度再缩放')
    parser.add_argument('--speed',type=float,default=1.0,help='时间倍率；0.5 将整个动作放慢一倍')
    args=parser.parse_args()
    args.source=args.source.expanduser().resolve();args.output=args.output.expanduser().resolve()
    if not all(np.isfinite(v) and v>0 for v in (args.stride_scale,args.lift_scale,args.speed)):
        raise ValueError('缩放系数必须为正有限数')
    frames=np.load(args.source,allow_pickle=False)
    if frames.ndim!=2 or frames.shape[1]!=61 or not np.isfinite(frames).all():
        raise ValueError('预期 Go2 (T,61) 有限数组')
    config_path=args.source.with_name('config.json');clip_path=args.source.with_name('clip.json')
    old_config=json.loads(config_path.read_text()) if config_path.exists() else {}
    clip=json.loads(clip_path.read_text()) if clip_path.exists() else {}
    start,end=args.frames or old_config.get('policy_cfg',{}).get('motion_range',[0,len(frames)])
    if not 0<=start<end<=len(frames) or end-start<4:raise ValueError('参考范围无效')
    frames=frames[start:end].astype(np.float64)
    fps=float(clip.get('fps',60))*args.speed
    if not np.isfinite(fps) or fps<=0:raise ValueError('参考 fps 必须为正有限值')
    t=np.arange(len(frames))/fps
    phase=np.linspace(0,1,len(frames))
    joint=frames[:,6:18].reshape(-1,3,4).transpose(0,2,1).reshape(-1,12)
    # 原始片段近似周期：分摊端点角差，使循环时姿态不跳变。
    joint-=phase[:,None]*(joint[-1]-joint[0])
    raw_rotation=rotation(frames[:,51:55])
    first_yaw=raw_rotation[0].as_euler('ZYX')[0]
    heading=Rotation.from_euler('z',-first_yaw)
    source_rot=heading*raw_rotation
    yaw0=source_rot[0].as_euler('ZYX')[0];yaw1=source_rot[-1].as_euler('ZYX')[0]
    yaw_delta=np.arctan2(np.sin(yaw1-yaw0),np.cos(yaw1-yaw0))
    desired_end=Rotation.from_euler('z',yaw_delta)*source_rot[0]
    correction=desired_end*source_rot[-1].inv()
    corrections=Slerp([0,1],Rotation.concatenate([Rotation.identity(),correction]))(phase)
    source_rot=corrections*source_rot
    source_pos=heading.apply(frames[:,48:51]-np.r_[frames[0,48:50],0.0])
    source_pos[:,2]-=phase*(source_pos[-1,2]-source_pos[0,2])

    path=ROOT/'assets/unitree_model/Go2/mjcf/go2.xml'
    xml=ET.parse(path).getroot();xml.find('compiler').set('meshdir',str(path.parent/'assets'))
    go=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));gd=mujoco.MjData(go)
    go_joint_names=[f'{leg}_{part}_joint' for leg in LEGS for part in ('hip','thigh','calf')]
    gq=np.array([go.jnt_qposadr[go.joint(n).id] for n in go_joint_names])
    gfeet=[go.geom(leg).id for leg in LEGS]
    gd.qpos[:7]=[0,0,0,1,0,0,0];gd.qpos[gq]=np.tile([0,.8,-1.6],4)
    mujoco.mj_kinematics(go,gd);go_neutral=gd.geom_xpos[gfeet].copy()
    source_feet=[];source_local=[]
    for i in range(len(t)):
        gd.qpos[:3]=source_pos[i];gd.qpos[3:7]=quaternion(source_rot[i]);gd.qpos[gq]=joint[i]
        mujoco.mj_kinematics(go,gd)
        feet=gd.geom_xpos[gfeet].copy()
        source_feet.append(feet);source_local.append(source_rot[i].inv().apply(feet-source_pos[i]))
    source_feet=np.array(source_feet);source_local=np.array(source_local)
    # 将步态摆动中心对齐 middle 自己的站姿；不把 Go2 默认关节角造成的姿态偏置带过来。
    go_neutral=source_local[:-1].mean(axis=0)
    foot_radius=np.array([go.geom_size[i,0] for i in gfeet])
    source_bottom=source_feet[:,:,2]-foot_radius

    model=mujoco.MjModel.from_xml_path(str(OUTPUT));data=mujoco.MjData(model)
    nominal=model.key_qpos[model.key('stand').id].copy()
    qids=np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINTS])
    jids=np.array([model.joint(n).id for n in JOINTS])
    limits=model.jnt_range[jids]
    site_ids=[model.site(f'{leg}_sole').id for leg in LEGS]
    data.qpos[:]=nominal;mujoco.mj_kinematics(model,data)
    neutral_rotation=rotation(nominal[3:7])
    middle_neutral=neutral_rotation.inv().apply(data.site_xpos[site_ids]-nominal[:3])
    # 以关节轴到膝轴、膝轴到脚底的长度估计形态比例，不使用总外观长度。
    lower=[model.body(f'{leg}_LOW').id for leg in LEGS]
    middle_lengths=np.array([np.linalg.norm(model.body_pos[bid])+np.linalg.norm(model.site_pos[sid])
                             for bid,sid in zip(lower,site_ids)])
    go_lengths=np.array([np.linalg.norm(go.body_pos[go.body(f'{leg}_calf').id])
                         +np.linalg.norm(go.geom_pos[gid]) for leg,gid in zip(LEGS,gfeet)])
    scale=float(np.median(middle_lengths/go_lengths))
    local_targets=middle_neutral[None]+scale*args.stride_scale*(source_local-go_neutral[None])@B.T
    root_pos=nominal[:3]+scale*args.stride_scale*source_pos@B.T
    root_pos[:,2]=nominal[2]+scale*(source_pos[:,2]-source_pos[0,2])
    root_rot=Rotation.from_matrix(B@source_rot.as_matrix()@B.T)*neutral_rotation
    target_feet=np.stack([root_rot[i].apply(local_targets[i])+root_pos[i] for i in range(len(t))])
    target_feet[:,:,2]=scale*args.lift_scale*np.maximum(source_bottom,0)

    # 缓存原始碰撞网格的 link 局部顶点，用实际旋转后的最低点约束脚底高度。
    urdf=ET.parse(SOURCE).getroot();links={l.get('name'):l for l in urdf.findall('link')}
    vertices={}
    for name,link in links.items():
        col=link.find('collision')
        if col is None:continue
        vertices[name]=stl_vertices(SOURCE.parent/col.find('geometry/mesh').get('filename'))+xyz(col.find('origin'))
    def min_height(name):
        bid=model.body(name).id
        height=float((vertices[name]@data.xmat[bid].reshape(3,3)[2,:]+data.xpos[bid,2]).min())
        if name in tuple(f'{leg}_LOW' for leg in LEGS):
            gid=model.geom(f'{name[:2]}_foot_sphere').id
            height=min(height,float(data.geom_xpos[gid,2]-model.geom_size[gid,0]))
        return height

    qposes=[];errors=[];last=nominal[qids].copy()
    for i in range(len(t)-1):
        def put(x):
            data.qpos[:3]=root_pos[i];data.qpos[2]=x[-1]
            data.qpos[3:7]=quaternion(root_rot[i]);data.qpos[qids]=x[:12]
            mujoco.mj_kinematics(model,data)
        def residual(x):
            put(x)
            xy=(data.site_xpos[site_ids,:2]-target_feet[i,:,:2]).ravel()
            z=np.array([min_height(f'{leg}_LOW') for leg in LEGS])-target_feet[i,:,2]
            return np.r_[25*xy,35*z,0.04*(x[:12]-nominal[qids]),0.12*(x[:12]-last),15*(x[-1]-root_pos[i,2])]
        low=np.r_[limits[:,0]+.005,max(.12,root_pos[i,2]-.12)]
        high=np.r_[limits[:,1]-.005,root_pos[i,2]+.10]
        x0=np.clip(np.r_[last,root_pos[i,2]],low+1e-7,high-1e-7)
        fit=least_squares(residual,x0,bounds=(low,high),max_nfev=100,
                          ftol=1e-7,xtol=1e-7,gtol=1e-7)
        put(fit.x);last=fit.x[:12].copy()
        qposes.append(data.qpos.copy())
        errors.append(np.linalg.norm(data.site_xpos[site_ids,:2]-target_feet[i,:,:2],axis=-1))
    qposes=np.array(qposes)
    # 再做整周期联合优化：每帧同时考虑前后帧，避免顺序 IK 在换支撑腿时跳分支。
    count=len(qposes)
    initial=np.column_stack((qposes[:,qids],qposes[:,2]))
    plane_bodies=[name for name in vertices if name!='BODY']
    rows_per_frame=50+len(plane_bodies)
    def cyclic_residual(flat):
        values=flat.reshape(count,13)
        result=[]
        for i in range(count):
            q=values[i,:12]
            data.qpos[:3]=root_pos[i];data.qpos[2]=values[i,-1]
            data.qpos[3:7]=quaternion(root_rot[i]);data.qpos[qids]=q
            mujoco.mj_kinematics(model,data)
            minz=np.array([min_height(f'{leg}_LOW') for leg in LEGS])
            penetration=np.array([min_height(name) for name in plane_bodies])
            before=values[(i-1)%count];after=values[(i+1)%count]
            result.extend(np.r_[25*(data.site_xpos[site_ids,:2]-target_feet[i,:,:2]).ravel(),
                                35*(minz-target_feet[i,:,2]),
                                .04*(q-nominal[qids]),
                                .02*fps*(q-before[:12]),
                                .0002*fps**2*(after[:12]-2*q+before[:12]),
                                15*(values[i,-1]-root_pos[i,2]),
                                .2*fps*(values[i,-1]-before[-1]),
                                100*np.minimum(penetration,0)])
        return np.asarray(result)
    sparsity=lil_matrix((rows_per_frame*count,13*count),dtype=np.int8)
    for i in range(count):
        for j in ((i-1)%count,i,(i+1)%count):
            sparsity[i*rows_per_frame:(i+1)*rows_per_frame,j*13:(j+1)*13]=1
    low=np.column_stack((np.tile(limits[:,0]+.005,(count,1)),np.maximum(.12,root_pos[:count,2]-.12)))
    high=np.column_stack((np.tile(limits[:,1]-.005,(count,1)),root_pos[:count,2]+.10))
    print('开始整周期平滑 IK…',flush=True)
    combined=least_squares(cyclic_residual,initial.ravel(),bounds=(low.ravel(),high.ravel()),
                           jac_sparsity=sparsity.tocsr(),max_nfev=50,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    refined=combined.x.reshape(count,13)
    qposes[:,qids]=refined[:,:12];qposes[:,2]=refined[:,-1]
    errors=[]
    for i,qp in enumerate(qposes):
        data.qpos[:]=qp;mujoco.mj_kinematics(model,data)
        errors.append(np.linalg.norm(data.site_xpos[site_ids,:2]-target_feet[i,:,:2],axis=-1))
    # 尾帧是下一周期起点：关节/高度重复，根 xy+yaw 保留累计位移。
    endpose=qposes[0].copy();endpose[:2]=root_pos[-1,:2]
    endpose[3:7]=quaternion(Rotation.from_euler('z',yaw_delta)*rotation(qposes[0,3:7]))
    qposes=np.vstack((qposes,endpose))
    ground_lift=[];lowest=[];feet=[]
    for i,qp in enumerate(qposes):
        data.qpos[:]=qp;mujoco.mj_kinematics(model,data)
        minimum=min(min_height(name) for name in vertices)
        lift=max(0.,-minimum)
        qposes[i,2]+=lift
        data.qpos[:]=qposes[i];mujoco.mj_kinematics(model,data)
        ground_lift.append(lift);lowest.append([min_height(f'{leg}_LOW') for leg in LEGS])
        feet.append(data.site_xpos[site_ids].copy())
    feet=np.array(feet);lowest=np.array(lowest)
    # 用跨周期邻帧差分计算速度，避免端点直接设零。
    yaw=Rotation.from_euler('z',yaw_delta)
    origin=qposes[0,:3].copy();delta=qposes[-1,:3]-origin
    def shift(qp,back=False):
        out=qp.copy()
        if back:
            out[:3]=origin+yaw.inv().apply(qp[:3]-origin-delta)
            out[3:7]=quaternion(yaw.inv()*rotation(qp[3:7]))
        else:
            out[:3]=origin+delta+yaw.apply(qp[:3]-origin)
            out[3:7]=quaternion(yaw*rotation(qp[3:7]))
        return out
    velocities=[]
    for i in range(len(t)):
        before=qposes[i-1] if i else shift(qposes[-2],True)
        after=qposes[i+1] if i<len(t)-1 else shift(qposes[1])
        v=np.zeros(model.nv)
        mujoco.mj_differentiatePos(model,v,2/fps,before,after)
        velocities.append(v)
    velocities=np.array(velocities)
    velocity_limits=np.array([float(urdf.find(f"joint[@name='{name}']/limit").get('velocity')) for name in JOINTS])
    errors=np.array(errors)
    try: source_name=str(args.source.relative_to(ROOT))
    except ValueError:source_name=str(args.source)
    report={'format':'middle_motion_v1','source':source_name,
            'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'source_frames':[start,end],'source_clip':clip,'fps':fps,'frames':len(t),
            'duration_seconds':float(t[-1]),'joint_names':list(JOINTS),
            'model_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            'morphology_scale':scale,'stride_scale':args.stride_scale,'lift_scale':args.lift_scale,
            'axis_mapping':B.tolist(),'cycle_delta_xy_m':(qposes[-1,:2]-qposes[0,:2]).tolist(),
            'source_foot_center_local_m':go_neutral.tolist(),
            'cycle_delta_yaw_rad':float(yaw_delta),
            'foot_xy_rms_error_m':np.sqrt(np.mean(errors**2,axis=0)).tolist(),
            'foot_xy_max_error_m':errors.max(0).tolist(),
            'ground_lift_max_m':float(max(ground_lift)),
            'foot_ground_min_m':float(lowest.min()),
            'peak_joint_velocity_rad_s':np.max(np.abs(velocities[:,6:]),axis=0).tolist(),
            'urdf_velocity_limits_rad_s':velocity_limits.tolist(),
            'peak_velocity_to_urdf_limit_ratio':float((np.abs(velocities[:,6:])/velocity_limits).max()),
            'joint_cycle_gap_rad':float(np.max(abs(qposes[0,qids]-qposes[-1,qids]))),
            'cyclic_optimization_evaluations':int(combined.nfev),
            'cyclic_optimization_converged':bool(combined.success),
            'notes':['足端IK映射，非Go2策略权重迁移','当前仅验证运动学，不保证动力学可行',
                     '原始canter周期端点做了姿态闭合；抬脚/步幅按middle范围缩小',
                     '碰撞网格逐帧最低点抬升，避免参考回放穿地；不代表真实接触力解']}
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'motion.npz',qpos=qposes,qvel=velocities,fps=fps,
                        joint_names=np.array(JOINTS),target_feet=target_feet,actual_feet=feet,
                        foot_ground_height=lowest,times=t)
    (args.output/'clip.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f'已生成：{args.output}/motion.npz\n{len(t)}帧，{fps:g}Hz，周期{t[-1]:.3f}s，根部位移{report["cycle_delta_xy_m"]}',flush=True)
    print('足端XY RMS误差(m)：',report['foot_xy_rms_error_m'])
    print('最大地面修正(m)：',report['ground_lift_max_m'],'最大关节速度(rad/s)：',float(np.abs(velocities[:,6:]).max()))


if __name__=='__main__':main()
