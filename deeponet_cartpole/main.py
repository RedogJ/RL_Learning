# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from pathlib import Path

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg

# 读取启动参数
parser = argparse.ArgumentParser()
parser.add_argument(
    "--control",
    choices=["pd", "sample", "train", "play"],
    default="pd",
)
parser.add_argument(
    "--max_updates",
    type=int,
    default=300,
)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path(__file__).resolve().parent / "checkpoints" / "latest_deeponet.pt",
)
parser.add_argument("--history_steps", type=int, default=5)
parser.add_argument("--control_steps", type=int, default=5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--rollout_steps", type=int, default=128, help="每轮采样的控制段数")
parser.add_argument("--seed", type=int, default=42)

add_launcher_args(parser)
args = parser.parse_args()
if min(args.history_steps, args.control_steps, args.max_updates, args.num_envs, args.rollout_steps) < 1:
    parser.error("历史帧数、控制段步数、更新次数、环境数量和采样段数必须大于零")
if not 0.0 <= args.gamma <= 1.0:
    parser.error("gamma 必须在 [0, 1] 内")
if args.control == "play" and not args.checkpoint.is_file():
    parser.error(f"找不到权重文件：{args.checkpoint}")
torch.manual_seed(args.seed)

# 明确选择 Isaac Sim PhysX
sim_cfg = SimulationCfg(
    dt=1 / 120,
    physics=PhysxCfg(),
)

with launch_simulation(sim_cfg, args):
    # 在 Isaac Sim 运行环境准备好之后导入场景相关功能
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    from controller import compute_pd_force
    from env import CartpoleEnv
    from networks import PolicyNetwork, ValueNetwork
    from scene import DropSceneCfg, configure_cartpole_joints
    from rollout import collect_rollout
    from ppo import compute_gae, PPOTrainer

    sim = sim_utils.SimulationContext(sim_cfg)

    try:
        scene_cfg = DropSceneCfg(
            num_envs=args.num_envs,
            env_spacing=2.0,
            # 每个环境有不同的世界锚点，由 PhysX 读取各自修正后的 USD。
            replicate_physics=False,
        )

        scene_cfg.ball.spawn.cart_half_height = scene_cfg.cart.spawn.size[2] / 2
        # 根据小车上表面和杆半长计算初始位置，确保两端锚点重合。
        cart_pos = scene_cfg.cart.init_state.pos
        scene_cfg.ball.init_state.pos = (
            cart_pos[0],
            cart_pos[1],
            cart_pos[2] + scene_cfg.cart.spawn.size[2] / 2 + scene_cfg.ball.spawn.size[2] / 2,
        )
        scene = InteractiveScene(scene_cfg)
        configure_cartpole_joints(scene)

        sim.set_camera_view(
            eye=(4.0, 4.0, 4.0),
            target=(0.0, 0.0, 2.8),
        )
        sim.reset()
        scene.reset()

        print("PhysX 滑块仿真已初始化：仅允许沿 X 移动，行程为 ±2 米")

        checkpoint = None
        if args.control == "play":
            checkpoint = torch.load(args.checkpoint, map_location=sim.device, weights_only=True)
            if checkpoint.get("architecture") != "deeponet":
                raise ValueError("这是旧 MLP 权重；DeepONet 需要重新训练并使用 latest_deeponet.pt")
            args.history_steps = checkpoint["history_steps"]
            args.control_steps = checkpoint["control_steps"]
            args.gamma = checkpoint["gamma"]
            if abs(checkpoint["step_dt"] - sim.get_physics_dt()) > 1e-12:
                raise ValueError("当前物理步长与训练权重不一致")

        env = CartpoleEnv(sim, scene, scene_cfg, history_steps=args.history_steps)
        observation = env.reset()
        history = env.get_history()

        if args.control in ("sample", "train", "play"):
            policy = PolicyNetwork(
                history_steps=args.history_steps,
                control_steps=args.control_steps,
                step_dt=env.step_dt,
            ).to(sim.device)
            print(
                f"DeepONet：历史 {args.history_steps} 帧，控制段 {args.control_steps} 步，"
                f"正常段的策略重新预测频率 {1 / policy.control_period:.1f} Hz，"
                f"控制指令执行频率 {1 / env.step_dt:.1f} Hz"
            )
        if args.control in ("sample", "train"):
            critic = ValueNetwork(history_steps=args.history_steps).to(sim.device)

        if args.control == "play":
            policy.load_state_dict(checkpoint["policy"])
            policy.eval()
            env.force_limit = checkpoint["force_limit"]
            print(f"已加载权重：{args.checkpoint}，训练更新次数：{checkpoint['updates']}")
            print("正在播放训练好的策略；按 Ctrl+C 退出")

        if args.control == "train":
            trainer = PPOTrainer(policy, critic)
            update_count = 0
            total_physics_steps = 0

        step = 0

        while sim.is_headless_or_exist_active_visualizer():
            if args.control in ("sample", "train"):
                # num_steps 现在是控制段数量，每段最多推进 control_steps 个物理步。
                rollout, history = collect_rollout(
                    env,
                    policy,
                    critic,
                    history,
                    num_steps=args.rollout_steps,
                    gamma=args.gamma,
                )
                advantages, returns = compute_gae(rollout, gamma=args.gamma)

                if args.control == "train":
                    losses = trainer.update(rollout, advantages, returns)
                    update_count += 1
                    total_physics_steps += rollout["durations"].sum().item()

                    print(
                        f"更新：{update_count}/{args.max_updates}，"
                        f"策略损失：{losses['policy_loss']:.4f}，"
                        f"价值损失：{losses['value_loss']:.4f}，"
                        f"失败次数：{rollout['terminated'].sum().item()}，"
                        f"累计有效物理步：{total_physics_steps}"
                    )

                    if update_count >= args.max_updates:
                        checkpoint_path = args.checkpoint
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                        torch.save(
                            {
                                "architecture": "deeponet",
                                "history_steps": args.history_steps,
                                "control_steps": args.control_steps,
                                "step_dt": env.step_dt,
                                "gamma": args.gamma,
                                "seed": args.seed,
                                "total_physics_steps": total_physics_steps,
                                "policy": policy.state_dict(),
                                "critic": critic.state_dict(),
                                "updates": update_count,
                                "force_limit": env.force_limit,
                            },
                            checkpoint_path,
                        )
                        print("权重已保存：", checkpoint_path)
                        break

                else:
                    print(
                        "优势形状：", advantages.shape,
                        "平均优势：", advantages.mean().item(),
                    )

                continue


            if args.control == "play":
                # Branch 只更新一次；固定的时间网格可以一次查询并缓存整段控制。
                with torch.no_grad():
                    features = policy.encode_history(env.get_history())
                    actions = policy.evaluate_control(features, policy.control_times)
                observation, rewards, terminated, truncated, info = env.step_sequence(
                    actions, gamma=args.gamma, real_time=True
                )
                step += int(info["durations"].max().item())
            else:
                theta = observation[:, 2]
                theta_dot = observation[:, 3]

                control_force = compute_pd_force(
                    theta=theta,
                    theta_dot=theta_dot,
                    kp=90.0,
                    kd=15.0,
                )

                # 每轮开始的前 0.2 秒，保留 1 N 扰动
                disturbance = (
                    env.episode_steps < round(0.2 / env.step_dt)
                ).float()

                # 环境接收归一化动作：[环境数量, 1]
                actions = (
                    (control_force + disturbance) / env.force_limit
                ).unsqueeze(-1)

                observation, rewards, terminated, truncated, _ = env.step(actions)
                step += 1

            if step % 48 == 0:
                print(
                    f"摆角：{torch.rad2deg(observation[0, 2]).item():.2f} 度，"
                    f"奖励：{rewards[0].item():.4f}"
                )

            done_ids = (terminated | truncated).nonzero(
                as_tuple=False
            ).flatten()

            if done_ids.numel() > 0:
                if args.control == "play":
                    env_id = done_ids[0].item()
                    episode_time = env.episode_steps[env_id].item() * env.step_dt
                    reason = "越界或杆倾角超过 30 度" if terminated[env_id].item() else "达到 5 秒上限"
                    print(f"环境 {env_id} 本轮持续 {episode_time:.2f} 秒：{reason}")
                observation = env.reset(done_ids)
                print("本轮结束，已重置")

        print("仿真运行结束")
    finally:
        sim_utils.SimulationContext.clear_instance()
