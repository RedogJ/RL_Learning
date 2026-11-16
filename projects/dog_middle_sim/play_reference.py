"""Middle 映射参考的运动学循环回放；不执行策略或施力。"""
import argparse
import json
import time

import mujoco
import numpy as np

from .build_model import OUTPUT
from .reference import Motion
from .retarget import DEFAULT_OUTPUT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion',default=str(DEFAULT_OUTPUT/'motion.npz'))
    parser.add_argument('--speed',type=float,default=1)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--cycles',type=int,default=3,help='无窗口检查的周期数')
    parser.add_argument('--show-targets',action='store_true',help='蓝色小球显示IK脚端目标')
    parser.add_argument('--show-foot-diagnostics',action='store_true',
                        help='显示 URDF 足端、推算脚底及检测到的地面碰撞点')
    args=parser.parse_args()
    if not np.isfinite(args.speed) or args.speed<=0 or args.cycles<1:raise ValueError('speed/cycles 必须为正')
    motion=Motion(args.motion)
    model=mujoco.MjModel.from_xml_path(str(OUTPUT));data=mujoco.MjData(model)
    print(f'Middle参考：{motion.path}\n周期={motion.duration:.3f}s，速度={args.speed:g}倍，'
          'SE(2)连续循环（不reset回起点）。\n运动学回放，不执行PD/策略，不代表动作已训练成功。',flush=True)
    def put(t):
        qpos,qvel,target=motion.sample(t)
        data.qpos[:]=qpos;data.qvel[:]=qvel;data.time=t
        mujoco.mj_forward(model,data)
        return target
    if args.headless:
        maximum_penetration=0.0
        for t in np.linspace(0,motion.duration*args.cycles,(len(motion.times)-1)*args.cycles+1):
            put(t)
            for contact in data.contact:
                maximum_penetration=max(maximum_penetration,-float(contact.dist))
        print(json.dumps({'cycles':args.cycles,'root_end':data.qpos[:3].tolist(),
                          'max_contact_penetration_m':maximum_penetration,
                          'joint_cycle_gap_rad':motion.metadata['joint_cycle_gap_rad']},indent=2))
        return
    from mujoco import viewer as mj_viewer
    paused=[False]
    def on_key(key):
        if key==32:paused[0]=not paused[0]
    put(0)
    if args.show_foot_diagnostics:
        print('足端诊断：红色=URDF 足端，绿色=推算脚底，黄色=当前检测到的地面碰撞点。'
              '黄色点只在碰撞检测到接触时出现；运动学回放不代表真实接触力。',flush=True)
    with mj_viewer.launch_passive(model,data,key_callback=on_key) as viewer:
        viewer.cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid=model.body('BODY').id
        viewer.cam.distance=1.8;viewer.cam.azimuth=130;viewer.cam.elevation=-18
        viewer.opt.geomgroup[3]=0;viewer.opt.sitegroup[4]=0
        elapsed=0.;previous=time.monotonic()
        print('空格暂停/继续；左键旋转、右键平移、滚轮缩放；关窗退出。',flush=True)
        while viewer.is_running():
            now=time.monotonic()
            if not paused[0]:elapsed+=(now-previous)*args.speed
            previous=now
            targets=put(elapsed)
            with viewer.lock():
                viewer.user_scn.ngeom=0
                def marker(point, color, radius):
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                        return
                    geom=viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,
                                       np.array([radius,0,0]),point,np.eye(3).ravel(),
                                       np.asarray(color,dtype=np.float32))
                    viewer.user_scn.ngeom+=1
                if args.show_targets:
                    for point in targets:
                        marker(point,[.05,.4,1.,.7],.012)
                if args.show_foot_diagnostics:
                    for name in ('left_front_foot','right_front_foot','left_foot','right_foot'):
                        marker(data.site(name).xpos,[1.,.1,.1,1.],.009)
                    for leg in ('FL','FR','RL','RR'):
                        marker(data.site(f'{leg}_sole').xpos,[.1,1.,.2,1.],.009)
                    ground_id=model.geom('ground').id
                    for contact in data.contact:
                        if ground_id in (contact.geom1,contact.geom2):
                            marker(contact.pos,[1.,.85,0.,1.],.012)
            viewer.sync()
            time.sleep(1/120)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('参考回放已退出。')
