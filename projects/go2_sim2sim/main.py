"""独立 MuJoCo 平地播放；默认读取固定的 Isaac Lab Actor 检查点。"""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
import torch

from .config import DEFAULT_CHECKPOINT, KD, KP, MODEL_ROOT, PHYSICS_HZ, PROJECT_ROOT, REPO_ROOT, TORQUE_LIMIT
from .env import Go2Sim
from .keyboard import Keyboard
from .policy import Policy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("running", "upright"), default="running")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--headless", action="store_true", help="无窗口加速测试")
    parser.add_argument("--duration", type=float, default=0., help="仿真秒数；窗口默认持续，无窗口默认 30 秒")
    parser.add_argument("--vx", type=float, default=0.)
    parser.add_argument("--vy", type=float, default=0.)
    parser.add_argument("--yaw", type=float, default=0.)
    parser.add_argument("--physics-hz", type=int, default=PHYSICS_HZ)
    parser.add_argument("--pd-only", action="store_true", help="不用 Actor，只维持默认关节站姿")
    parser.add_argument("--no-auto-reset", action="store_true", help="诊断时连续运行，不自动重置")
    parser.add_argument("--record", action="store_true", help="额外逐控制步保存观测/动作/关节状态，便于跨仿真对照")
    parser.add_argument("--run-name", default="play")
    args = parser.parse_args()
    if args.duration < 0 or args.physics_hz <= 0 or not np.isfinite([args.duration, args.vx, args.vy, args.yaw]).all():
        parser.error("时长必须非负，频率必须为正，指令必须有限")
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_name):
        parser.error("run-name 只能包含字母、数字、下划线和连字符")
    if args.headless and args.duration == 0:
        args.duration = 30.
    if args.checkpoint is None:
        if args.task == "upright":
            candidates = [p for p in (REPO_ROOT / "projects/go2_learning/logs/upright").glob("*/latest.pt")
                          if p.with_name("run.json").is_file() and json.loads(p.with_name("run.json").read_text()).get("task_name") == "upright"]
            if not candidates:
                parser.error("没有找到 upright 的 latest.pt，请用 --checkpoint 指定")
            args.checkpoint = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        else:
            args.checkpoint = DEFAULT_CHECKPOINT
    return args


def run(args):
    torch.set_num_threads(1)
    policy = Policy(args.checkpoint)
    if policy.task != args.task and not (args.task == "running" and policy.task == "standing"):
        raise ValueError(f"检查点任务为 {policy.task}，请使用 --task {policy.task}")
    env = Go2Sim(policy, args.physics_hz)
    keyboard = Keyboard([args.vx, args.vy, args.yaw], policy.cfg["command_cfg"])
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M-%S_%f")
    log_dir = PROJECT_ROOT / "logs" / f"{timestamp}_{args.run_name}"
    log_dir.mkdir(parents=True)
    metadata = {"checkpoint": str(policy.path), "checkpoint_sha256": policy.sha256, "iteration": policy.iteration,
                "source_config": policy.cfg, "mujoco_version": mujoco.__version__,
                "physics_hz": args.physics_hz, "control_dt": policy.control_dt,
                "kp": KP, "kd": KD, "torque_limit": TORQUE_LIMIT,
                "pd_only": args.pd_only, "initial_command": keyboard.command.tolist(),
                "joint_passive_parameters": {
                    "joint_names": policy.joint_names,
                    "damping": env.model.dof_damping[env.dof_ids].tolist(),
                    "frictionloss": env.model.dof_frictionloss[env.dof_ids].tolist(),
                    "armature": env.model.dof_armature[env.dof_ids].tolist(),
                },
                "model_manifest": json.loads((MODEL_ROOT / "download_manifest.json").read_text())}
    (log_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"MuJoCo {mujoco.__version__} | 任务 {policy.task} | Actor {policy.path}\n"
          f"控制 {1 / policy.control_dt:g} Hz / 物理 {args.physics_hz} Hz | PD={KP:g}/{KD:g} | "
          f"力矩上限 {TORQUE_LIMIT:g} N·m\n结果：{log_dir}", flush=True)
    if policy.task == "upright":
        print(f"upright：固定初始方向（不采样姿态分布），动作滤波 alpha={policy.filter_alpha:g}；"
              "仅倾角及超时重置，未实现非脚掌接触重置。", flush=True)
    viewer = None
    viewer_thread = None
    if not args.headless:
        import glfw
        from mujoco import viewer as mujoco_viewer

        keyboard.enable_hold_control()
        before_threads = {t.ident for t in threading.enumerate()}
        viewer = mujoco_viewer.launch_passive(env.model, env.data, key_callback=keyboard.press)
        viewer_thread = next((t for t in threading.enumerate()
                              if t.ident not in before_threads
                              and getattr(t, "_target", None) is mujoco_viewer._launch_internal), None)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.base_id
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 135., -20.
        print("按住方向键移动/转向，松键停车；窗口失焦停车；R 重置。", flush=True)
    fallen = timeouts = 0
    tilt_seen = timeout_seen = False
    elapsed_steps = 0
    active_steps = 0
    sums = np.zeros(4)
    min_height = float("inf")
    next_print = 0.
    started = time.perf_counter()
    metrics_file = (log_dir / "metrics.jsonl").open("w")
    trace_file = (log_dir / "trace.jsonl").open("w") if args.record else None
    try:
        while (viewer is None or viewer.is_running()) and (args.duration == 0 or elapsed_steps * policy.control_dt < args.duration - 1e-9):
            tick = time.perf_counter()
            env.command[:], reset = keyboard.read()
            if reset:
                env.reset()
                tilt_seen = timeout_seen = False
            obs = env.observation()
            active = env.active
            action = policy(obs) if active and not args.pd_only else env.hold_action
            state, done = env.step(action)
            elapsed_steps += 1
            sim_time = elapsed_steps * policy.control_dt
            roll, pitch = env.observations.angles(state["gravity"])
            if active:
                active_steps += 1
                speed_error = np.linalg.norm(state["linear_velocity"][:2] - env.command[:2])
                sums += [state["height"], abs(roll), abs(pitch), speed_error]
                min_height = min(min_height, state["height"])
            record = {"t": sim_time, "active": active, "height": state["height"],
                      "roll_deg": float(np.degrees(roll)), "pitch_deg": float(np.degrees(pitch)),
                      "velocity": state["linear_velocity"].tolist(), "command": env.command.tolist(),
                      "upright_alignment": float(-state["gravity"][0]),
                      "raw_action_max": float(np.abs(action).max()),
                      "clipped_fraction": float(np.mean(np.abs(action) > 1.)),
                      "max_torque": float(np.abs(env.last_torque).max()), "done": done}
            if sim_time >= next_print:
                attitude = (f"upright={record['upright_alignment']:+.3f}" if policy.task == "upright"
                            else f"roll={np.degrees(roll):6.1f}° pitch={np.degrees(pitch):6.1f}°")
                print(f"t={sim_time:5.1f}s | z={state['height']:.3f}m | "
                      f"{attitude} | "
                      f"vx={state['linear_velocity'][0]:+.2f}/{env.command[0]:+.2f}m/s | "
                      f"fallen={fallen} timeout={timeouts}", flush=True)
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                next_print += 1.
            if trace_file is not None:
                trace_file.write(json.dumps({**record, "observation": obs.tolist(), "raw_action": action.tolist(),
                                             "q": state["q"].tolist(), "dq": state["dq"].tolist(),
                                             "target": policy.target(env.last_action).tolist(),
                                             "torque": env.last_torque.tolist()}) + "\n")
            if done:
                if done == "tilt" and not tilt_seen:
                    fallen += 1
                    tilt_seen = True
                if done == "timeout" and not timeout_seen:
                    timeouts += 1
                    timeout_seen = True
                if not args.no_auto_reset:
                    print(f"重置：{done}，本回合 {env.episode_steps * policy.control_dt:.2f}s", flush=True)
                    env.reset()
                    tilt_seen = timeout_seen = False
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0., policy.control_dt - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        keyboard.close()
        metrics_file.close()
        if trace_file is not None:
            trace_file.close()
        if viewer is not None:
            viewer.close()
            # 3.7 的 close() 只请求退出；等待本次渲染线程释放 GL，防止进程先终止 GLFW。
            if viewer_thread is not None:
                viewer_thread.join(timeout=3.)
        averages = sums / max(active_steps, 1)
        summary = {"simulation_seconds": elapsed_steps * policy.control_dt,
                   "wall_seconds": time.perf_counter() - started, "fallen": fallen, "timeouts": timeouts,
                   "active_steps": active_steps, "mean_height": averages[0],
                   "mean_abs_roll_deg": float(np.degrees(averages[1])),
                   "mean_abs_pitch_deg": float(np.degrees(averages[2])),
                   "mean_xy_speed_error": averages[3], "min_height": min_height if active_steps else None}
        (log_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY " + json.dumps(summary), flush=True)
    return log_dir


if __name__ == "__main__":
    run(parse_args())
