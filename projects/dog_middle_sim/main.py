"""middle 的 MuJoCo PD 仿真；没有策略网络或训练流程。"""
import argparse
import time

import mujoco
import numpy as np

from .build_model import OUTPUT, JOINTS, build
from .mouse import MouseInteraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--kp", type=float, default=40)
    parser.add_argument("--kd", type=float, default=1)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.rebuild or not OUTPUT.exists():
        build()
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    data = mujoco.MjData(model)
    qids = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINTS])
    vids = np.array([model.jnt_dofadr[model.joint(n).id] for n in JOINTS])
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_forward(model, data)
    target = data.qpos[qids].copy()
    limits = model.actuator_ctrlrange
    print(f"middle: {model.nu} 电机，质量 {model.body_mass.sum():.3f} kg，"
          f"物理频率 {1/model.opt.timestep:g} Hz，PD={args.kp:g}/{args.kd:g}", flush=True)
    peak_speed = 0.0
    def step():
        nonlocal peak_speed
        torque = args.kp * (target - data.qpos[qids]) - args.kd * data.qvel[vids]
        data.ctrl[:] = np.clip(torque, limits[:, 0], limits[:, 1])
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("出现非有限状态")
        peak_speed = max(peak_speed, float(np.max(np.abs(data.qvel[vids]))))
    def report():
        up = float(data.xmat[model.body("BODY").id].reshape(3, 3)[2, 2])
        print(f"t={data.time:.1f}s height={data.qpos[2]:.3f}m up={up:.3f} "
              f"contacts={data.ncon} joint_error_rms={np.sqrt(np.mean((target-data.qpos[qids])**2)):.4f}rad "
              f"peak_joint_speed={peak_speed:.2f}rad/s", flush=True)
    if args.headless:
        next_log = 1.0
        for _ in range(round(args.seconds / model.opt.timestep)):
            step()
            if data.time >= next_log:
                report()
                next_log += 1
        report()
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = (0, -0.05, 0.1)
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 130
            viewer.cam.elevation = -20
            viewer.opt.geomgroup[3] = 0
            viewer.opt.sitegroup[4] = 0
            mouse=MouseInteraction(viewer,model,data)
            next_log = 1.0
            while viewer.is_running():
                start = time.monotonic()
                # 先同步输入，避免鼠标外力到下一显示周期才进入物理。
                mouse.sync()
                for _ in range(10):
                    step()
                if data.time >= next_log:
                    report()
                    print(f'[鼠标外力] {mouse.force_norm():.2f} N',flush=True)
                    next_log = data.time + 1
                time.sleep(max(0, 10 * model.opt.timestep - (time.monotonic() - start)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Middle 仿真已退出。")
