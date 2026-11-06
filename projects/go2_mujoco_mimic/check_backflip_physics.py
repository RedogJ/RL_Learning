"""开环 PD 跟踪后空翻参考的动力学诊断，不等于可训练性证明。"""
import json
import numpy as np
import mujoco
from .environment import Go2Environment


def main():
    ref=np.load('projects/go2_mujoco_mimic/data/solo12_backflip/reference_30hz/motion.npy')
    env=Go2Environment();m,d=env.model,env.data
    def set_q(row):
        d.qpos[:3]=row[48:51];d.qpos[3:7]=row[51:55];d.qpos[env.qpos_ids]=row[6:18].reshape(3,4).T.reshape(-1)
    set_q(ref[0]);start=d.qpos.copy();set_q(ref[1]);end=d.qpos.copy();set_q(ref[0]);mujoco.mj_differentiatePos(m,d.qvel,1/30,start,end);mujoco.mj_forward(m,d)
    qs=[];ps=[];qsat=0;torque_sat=0;steps=0;tip=0
    for idx in range(1,len(ref)):
        raw=ref[idx,6:18].reshape(3,4).T.reshape(-1)
        qsat+=np.count_nonzero(np.abs(raw-env.default_q)>1.)
        target=env.default_q+np.clip(raw-env.default_q,-1.,1.)
        for _ in range(env.substeps):
            torque=25*(target-d.qpos[env.qpos_ids])-.5*d.qvel[env.dof_ids]
            torque_sat+=np.count_nonzero(np.abs(torque)>=23.7);steps+=12
            d.ctrl[env.actuator_ids]=np.clip(torque,-23.7,23.7);mujoco.mj_step(m,d)
        qs.append(np.mean((d.qpos[env.qpos_ids]-raw)**2));ps.append(np.linalg.norm(d.qpos[:3]-ref[idx,48:51]));tip+=int(d.qpos[2]<.15)
    print(json.dumps({'frames':len(ref),'duration_s':(len(ref)-1)/30,'joint_rmse_rad':float(np.sqrt(np.mean(qs))),
                      'root_pos_error_final_m':float(ps[-1]),'root_pos_error_max_m':float(max(ps)),
                      'target_saturation_fraction':qsat/((len(ref)-1)*12),'torque_saturation_fraction':torque_sat/steps,
                      'frames_root_below_0.15m':tip,'final_root_height_m':float(d.qpos[2])},indent=2))
    print('开环 PD 跟踪用于排查参考与控制器差距，失败不证明 RL 无法学会。')
if __name__=='__main__':main()
