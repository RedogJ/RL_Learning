"""验证可微运动学与 MuJoCo 位姿、差分速度及奖励一致，并检查梯度。"""
import numpy as np
import torch
import mujoco
from ...environment import Go2Environment
from .kinematics import BodyKinematics
from .rewards import compute_reward, rotation_vector
from .reward_cfg import RewardCfg


def main():
    cpu = Go2Environment()
    names = ['base_link'] + [f'{leg}_{part}' for leg in ('FL','FR','RL','RR') for part in ('hip','thigh','calf','foot')]
    ids = [mujoco.mj_name2id(cpu.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    fk = BodyKinematics(cpu.model, ids, cpu.joint_ids).double()
    rng = np.random.default_rng(42)
    states, positions, quats = [], [], []
    for _ in range(32):
        q = cpu.data.qpos
        q[:3] = rng.normal(size=3)
        q[3:7] = rng.normal(size=4); q[3:7] /= np.linalg.norm(q[3:7])
        q[cpu.qpos_ids] = rng.uniform(cpu.model.jnt_range[cpu.joint_ids,0], cpu.model.jnt_range[cpu.joint_ids,1])
        mujoco.mj_kinematics(cpu.model,cpu.data)
        states.append(np.concatenate((q[:7],q[cpu.qpos_ids],rng.normal(size=18))))
        positions.append(cpu.data.xpos[ids].copy()); quats.append(cpu.data.xquat[ids].copy())
    s=torch.tensor(np.array(states),dtype=torch.double)
    pos=torch.tensor(np.array(positions)); quat=torch.tensor(np.array(quats))
    p,q=fk(s)
    pe=(p-pos).abs().max().item(); qe=rotation_vector(q,quat).abs().max().item()
    assert pe<1e-6 and qe<1e-6,(pe,qe)
    actual=fk.transition(s[:16],s[16:],cpu.control_dt)
    reference={'pos':pos[16:],'quat':quat[16:],'linear_vel':(pos[16:]-pos[:16])/cpu.control_dt,
               'angular_vel':rotation_vector(quat[16:],quat[:16])/cpu.control_dt}
    for key in actual: assert torch.allclose(actual[key],reference[key],atol=1e-5,rtol=1e-5),key
    a=torch.zeros(16,12,dtype=torch.double); limits=torch.tensor(cpu.model.jnt_range[cpu.joint_ids])
    r1,_=compute_reward(RewardCfg(),actual,reference,a,a,s[16:,7:19],limits,cpu.control_dt)
    r2,_=compute_reward(RewardCfg(),reference,reference,a,a,s[16:,7:19],limits,cpu.control_dt)
    assert torch.allclose(r1,r2,atol=1e-8)
    x=s[:1].clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda x: fk(x), (x,),eps=1e-6,atol=1e-5,rtol=1e-3)
    y=(s[:1]+torch.randn_like(s[:1])*.02).requires_grad_()
    ref={k:v[:1].detach() for k,v in fk.transition(s[:1],y.detach(),cpu.control_dt).items()}
    ref['pos']=ref['pos']+.01
    def reward_fn(x):
        return compute_reward(RewardCfg(),fk.transition(s[:1],x,cpu.control_dt),ref,a[:1],a[:1],x[:,7:19],limits,cpu.control_dt)[0]
    assert torch.autograd.gradcheck(reward_fn,(y,),eps=1e-6,atol=1e-5,rtol=1e-3)
    print(f'PASS: 32 随机姿态；位置最大误差={pe:.3g}m；朝向误差={qe:.3g}rad；差分速度/奖励一致；位姿及奖励梯度检查通过',flush=True)


if __name__=='__main__': main()
