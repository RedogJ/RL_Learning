# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""对照入口：同一个双杆场景，将手写 PPO 循环替换为 OnPolicyRunner。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg

# 原 env.py 内部使用 `from scene import ...`，因此把原工程目录加入导入路径。
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from rsl_rl_version.agent_cfg import make_agent_cfg

LOG_ROOT = PROJECT_DIR.parents[1] / "logs" / "my_cartpole_rsl_rl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("train", "play"), default="train")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_updates", type=int, default=300)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--run_name", default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=0, help="播放步数；0 表示运行到关闭窗口或 Ctrl+C")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs is None:
        args.num_envs = 64 if args.control == "train" else 1
    if args.num_envs < 1 or args.max_updates < 1 or args.steps < 0:
        parser.error("num_envs/max_updates 必须为正数，steps 不能为负数")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not args.run_name or any(c not in allowed for c in args.run_name):
        parser.error("run_name 只允许英文字母、数字、下划线和连字符")
    if args.control == "train" and args.checkpoint != "latest":
        parser.error("本对照版 train 从头训练；checkpoint 参数用于 play")
    return args


def resolve_checkpoint(value: str) -> Path:
    if value == "latest":
        candidates = list(LOG_ROOT.glob("*/latest.pt"))
        if not candidates:
            raise FileNotFoundError("没有 RSL-RL 权重，请先运行本版本的 train")
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    args = parse_args()
    version = importlib.metadata.version("rsl-rl-lib")
    if version != "5.5.1":
        raise RuntimeError(
            f"本对照版按 RSL-RL 5.5.1 验证，当前是 {version}；请先核对接口，不要直接升级环境"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = None
    metadata = None
    agent_cfg = make_agent_cfg()
    if args.control == "play":
        checkpoint = resolve_checkpoint(args.checkpoint)
        # 从训练记录恢复网络配置，避免训练后修改 agent_cfg 造成播放不一致。
        metadata = json.loads(checkpoint.with_name("run.json").read_text())
        if metadata["rsl_rl_version"] != version:
            raise ValueError("checkpoint 的 RSL-RL 版本与当前运行版本不一致")
        agent_cfg = metadata["agent_cfg"]

    sim_cfg = SimulationCfg(dt=1 / 120, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from rsl_rl.runners import OnPolicyRunner

        from env import CartpoleEnv
        from scene import DropSceneCfg, configure_cartpole_joints, configure_cartpole_scene_cfg
        from rsl_rl_version.vec_env import CartpoleVecEnv

        sim = sim_utils.SimulationContext(sim_cfg)
        try:
            # 与原 main.py 相同的场景创建顺序和关节修正。
            scene_cfg = DropSceneCfg(num_envs=args.num_envs, env_spacing=2.0, replicate_physics=False)
            configure_cartpole_scene_cfg(scene_cfg)
            scene = InteractiveScene(scene_cfg)
            configure_cartpole_joints(scene)
            sim.set_camera_view(eye=(5.0, 5.0, 5.0), target=(0.0, 0.0, 3.5))
            sim.reset()
            scene.reset()

            # 对照重点 1：把原来的物理环境适配成 RSL-RL VecEnv。
            env = CartpoleVecEnv(CartpoleEnv(sim, scene, scene_cfg))
            if metadata is not None and metadata["environment"] != env.cfg:
                raise ValueError("当前环境参数与训练记录不一致，请检查 env.py 和仿真步长")
            print(f"RSL-RL {version} | 双杆 | {env.num_envs} 个环境 | 6 维观测 | 1 维动作")

            log_dir = None
            if args.control == "train":
                run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                log_dir = LOG_ROOT / f"{run_id}_{args.run_name}"
                log_dir.mkdir(parents=True, exist_ok=False)
                metadata = {
                    "rsl_rl_version": version,
                    "seed": args.seed,
                    "num_envs": env.num_envs,
                    "max_updates": args.max_updates,
                    "environment": env.cfg,
                    "agent_cfg": agent_cfg,
                }
                (log_dir / "run.json").write_text(json.dumps(metadata, indent=2))
                print(f"训练记录：{log_dir}")

            # 对照重点 2：Runner 根据配置创建 Actor、Critic、PPO、Storage。
            runner = OnPolicyRunner(
                env, agent_cfg, log_dir=str(log_dir) if log_dir else None, device=str(sim.device)
            )
            if args.control == "train":
                # 对照重点 3：替代 collect_rollout + compute_gae + trainer.update。
                runner.learn(num_learning_iterations=args.max_updates, init_at_random_ep_len=False)
                checkpoint = log_dir / "latest.pt"
                runner.save(str(checkpoint), infos={"task": "my_double_cartpole"})
                print(f"训练完成，权重：{checkpoint}")
            else:
                runner.load(str(checkpoint), map_location=str(sim.device))
                policy = runner.get_inference_policy(device=str(sim.device))
                observation = env.get_observations()
                print(f"播放：{checkpoint}；Ctrl+C 退出")
                step = 0
                while sim.is_headless_or_exist_active_visualizer() and (args.steps == 0 or step < args.steps):
                    started = time.perf_counter()
                    with torch.inference_mode():
                        actions = policy(observation)
                        observation, _, _, _ = env.step(actions)
                    step += 1
                    time.sleep(max(0.0, sim_cfg.dt - (time.perf_counter() - started)))
        finally:
            sim_utils.SimulationContext.clear_instance()


if __name__ == "__main__":
    main()
