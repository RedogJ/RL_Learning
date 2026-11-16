"""供 MuJoCo Warp CUDA Graph 使用的固定形状 PD kernel。"""
import warp as wp


@wp.kernel
def pd_control(qpos: wp.array2d(dtype=wp.float32),qvel: wp.array2d(dtype=wp.float32),
               ctrl: wp.array2d(dtype=wp.float32),target: wp.array2d(dtype=wp.float32),
               qids: wp.array(dtype=wp.int64),vids: wp.array(dtype=wp.int64),
               kp: float,kd: float,limit: float):
    world,joint=wp.tid()
    torque=kp*(target[world,joint]-qpos[world,qids[joint]])-kd*qvel[world,vids[joint]]
    ctrl[world,joint]=wp.clamp(torque,-limit,limit)
