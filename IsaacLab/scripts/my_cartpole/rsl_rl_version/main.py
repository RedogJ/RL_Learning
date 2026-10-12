# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""标准 RSL-RL 入口：创建环境 → 读取配置 → 创建 Runner → 训练或播放。"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg

from rsl_rl_version.agent_cfg import make_agent_cfg

LOG_ROOT = Path(__file__).resolve().parent / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("train", "play"), default="train")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_updates", type=int, default=300)
    parser.add_argument("--checkpoint", default="latest", help="播放权重路径，或 latest")
    parser.add_argument("--run_name", default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=0, help="播放步数，0 表示一直播放")
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
        parser.error("train 从头训练；checkpoint 参数用于 play")
    return args


def resolve_checkpoint(value: str) -> Path:
    """只在本版本日志中查找最近完成的训练，也可以指定某个权重文件。"""
    if value != "latest":
        return Path(value).expanduser().resolve()
    checkpoints = list(LOG_ROOT.glob("*/latest.pt"))
    if not checkpoints:
        raise FileNotFoundError("没有 RSL-RL 权重，请先运行 train")
    return max(checkpoints, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    sim_cfg = SimulationCfg(dt=1 / 120, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        from isaaclab.sim import SimulationContext
        from rsl_rl.runners import OnPolicyRunner

        from rsl_rl_version.env import make_env

        sim = SimulationContext(sim_cfg)
        try:
            # 1. 创建环境：物理、奖励、重置都由环境负责。
            env = make_env(sim, args.num_envs)

            # 2. 读取配置：Python 字典和官方示例的 YAML 最终提供同样的接口。
            train_cfg = make_agent_cfg()
            if args.control == "train":
                run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                log_dir = LOG_ROOT / f"{run_id}_{args.run_name}"
                log_dir.mkdir(parents=True)
                # 保存网络配置，使以后修改 agent_cfg.py 也能播放旧模型。
                (log_dir / "run.json").write_text(json.dumps({"agent_cfg": train_cfg}, indent=2))
            else:
                checkpoint = resolve_checkpoint(args.checkpoint)
                train_cfg = json.loads(checkpoint.with_name("run.json").read_text())["agent_cfg"]
                log_dir = None

            # 3. 创建标准 Runner，由库创建网络、PPO 和采样存储。
            runner = OnPolicyRunner(
                env=env,
                train_cfg=train_cfg,
                log_dir=str(log_dir) if log_dir else None,
                device=str(sim.device),
            )

            # 4. 训练，或者加载模型进行推理。
            if args.control == "train":
                print(f"训练环境：{env.num_envs}；结果目录：{log_dir}")
                runner.learn(num_learning_iterations=args.max_updates)
                runner.save(str(log_dir / "latest.pt"))
            else:
                runner.load(str(checkpoint), map_location=str(sim.device))
                policy = runner.get_inference_policy(device=str(sim.device))
                obs = env.get_observations()
                print(f"播放：{checkpoint}；Ctrl+C 退出")
                step = 0
                with torch.inference_mode():
                    while sim.is_headless_or_exist_active_visualizer() and (args.steps == 0 or step < args.steps):
                        started = time.perf_counter()
                        actions = policy(obs)
                        obs, _, _, _ = env.step(actions)
                        step += 1
                        time.sleep(max(0.0, sim_cfg.dt - (time.perf_counter() - started)))
        finally:
            SimulationContext.clear_instance()


if __name__ == "__main__":
    main()
