"""用 MuJoCo Warp 回放策略，并用透明蓝色模型叠加参考动作。"""
import argparse
from dataclasses import fields
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .environment import Go2Environment
from .tasks.mimic.networks import MotionPolicy
from .warp_env import EnvCfg, WarpGo2Env


DEFAULT = Path('/workspace/go2_mujoco_mimic/logs/mimic_optimized_2026-10-29_07-31-02_685069/latest.pt')


def load_config(saved):
    cfg = EnvCfg()
    raw = saved['env_cfg'] if 'env_cfg' in saved else saved['env']
    for f in fields(cfg):
        if f.name not in ('num_envs', 'task'):
            setattr(cfg, f.name, raw[f.name])
    for key, value in raw['task'].items():
        if key == 'reward':
            for name, val in value.items():
                setattr(cfg.task.reward, name, val)
        else:
            setattr(cfg.task, key, value)
    cfg.num_envs = 1
    return cfg


def update_reference(cpu, ghost, ref, index):
    pose = ref.sample(torch.tensor(index, device=ref.frames.device))
    ghost.qpos[:3] = pose['pos'].cpu().numpy()
    ghost.qpos[3:7] = pose['quat'].cpu().numpy()
    ghost.qpos[cpu.qpos_ids] = pose['joints'].cpu().numpy()
    mujoco.mj_forward(cpu.model, ghost)


def update_ghost(viewer, cpu, ghost, opt, offset):
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        mujoco.mjv_addGeoms(cpu.model, ghost, opt, mujoco.MjvPerturb(),
                            mujoco.mjtCatBit.mjCAT_ALL, scene)
        for i in range(scene.ngeom):
            geom = scene.geoms[i]
            geom.pos[:] += offset
            geom.matid = -1
            geom.rgba[:] = (0.08, 0.45, 1.0, 0.30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT)
    p.add_argument('--ghost-offset-y', type=float, default=0.0, help='参考模型横向偏移（米）；默认原位叠加')
    p.add_argument('--no-noise', action='store_true', help='关闭观测噪声以便观察')
    p.add_argument('--headless', action='store_true', help='无窗口短测')
    p.add_argument('--steps', type=int, default=0, help='控制步数；0 表示持续播放')
    args = p.parse_args()
    if args.steps < 0 or (args.headless and args.steps == 0):
        p.error('headless 模式需要正数 --steps')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = load_config(checkpoint)
    if args.no_noise:
        cfg.task.observation_noise = False
    env = WarpGo2Env(cfg, Path(checkpoint['motion']))
    env.enable_cuda_graph()
    policy = MotionPolicy(history_length=cfg.task.history_length, future_length=cfg.task.future_length).to(env.device)
    policy.load_state_dict(checkpoint['policy'] if 'policy' in checkpoint else checkpoint['actor'])
    policy.eval()
    cpu = Go2Environment(cfg.physics_hz, cfg.control_hz)
    ghost = mujoco.MjData(cpu.model)
    visual_opt = mujoco.MjvOption()
    visual_opt.geomgroup[:] = 0
    visual_opt.geomgroup[2] = 1
    offset = np.array([0., args.ghost_offset_y, 0.])
    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(cpu.model, cpu.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(cpu.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 135, -20
    print(f"模型 cycle={checkpoint['cycle']} | 参考={checkpoint['motion']} | 蓝色=参考，实体=策略 | 偏移={args.ghost_offset_y:g}m", flush=True)
    print(f'参考连续播放 {cfg.task.reference_cycles} 次；失败、最终片段结束或超时后一起重置。Ctrl+C 退出。', flush=True)
    steps = 0
    try:
        while (viewer is None or viewer.is_running()) and (args.steps == 0 or steps < args.steps):
            start = time.monotonic()
            with torch.no_grad():
                action = policy(env.observation())
                _, reward, terminated, truncated, _ = env.step(action)
                # 先展示终止前的实际状态和同一帧参考，再进行重置。
                cpu.data.qpos[:] = env.qpos[0].cpu().numpy()
                cpu.data.qvel[:] = env.qvel[0].cpu().numpy()
                mujoco.mj_forward(cpu.model, cpu.data)
                index = int(env.frame[0].item())
                update_reference(cpu, ghost, env.ref, index)
                if viewer is not None:
                    update_ghost(viewer, cpu, ghost, visual_opt, offset)
                    viewer.sync()
                if steps % 30 == 0:
                    print(f'step={steps} 参考帧={index} reward={reward.item():.4f} failed={bool(terminated.item())}', flush=True)
                if (terminated | truncated).item():
                    print(f'重置: failed={bool(terminated.item())} truncated={bool(truncated.item())} 控制步={int(env.steps.item())}', flush=True)
                env.reset(terminated | truncated)
            steps += 1
            if viewer is not None:
                time.sleep(max(0, env.dt-(time.monotonic()-start)))
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == '__main__':
    main()
