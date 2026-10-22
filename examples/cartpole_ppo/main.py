import argparse
import time
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg
from pathlib import Path

# 读取启动参数
parser = argparse.ArgumentParser()
parser.add_argument(
    "--control",
    choices=["zero", "pd", "sample", "train", "play"],
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
    default=Path(__file__).resolve().parent / "checkpoints" / "latest_double.pt",
)
parser.add_argument("--num_envs", type=int, default=64)

add_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.max_updates < 1:
    parser.error("num_envs 和 max_updates 必须大于零")
if args.control == "play" and not args.checkpoint.is_file():
    parser.error(f"找不到权重文件：{args.checkpoint}")

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
    from scene import DropSceneCfg, configure_cartpole_joints, configure_cartpole_scene_cfg
    from env import CartpoleEnv

    from networks import PolicyNetwork, ValueNetwork
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

        # 两个接头的两端锚点在初始状态下分别重合。
        configure_cartpole_scene_cfg(scene_cfg)
        scene = InteractiveScene(scene_cfg)
        configure_cartpole_joints(scene)

        sim.set_camera_view(
            eye=(5.0, 5.0, 5.0),
            target=(0.0, 0.0, 3.5),
        )
        sim.reset()
        scene.reset()

    
        print(f"PhysX 二阶倒立摆已初始化：{scene_cfg.num_envs} 个环境，小车行程为 ±2 米")

        env = CartpoleEnv(sim, scene, scene_cfg)
        observation = env.reset()

        if args.control in ("sample", "train", "play"):
            policy = PolicyNetwork(observation_dim=env.observation_dim).to(sim.device)
        if args.control in ("sample", "train"):
            critic = ValueNetwork(observation_dim=env.observation_dim).to(sim.device)

        if args.control == "play":
            checkpoint = torch.load(
                args.checkpoint, map_location=sim.device, weights_only=True
            )
            if checkpoint.get("environment") != "double_cartpole" or checkpoint.get("observation_dim") != 6:
                raise ValueError("双杆环境需要重新训练的 6 维观测权重；旧单杆权重不能直接加载")
            policy.load_state_dict(checkpoint["policy"])
            policy.eval()
            env.force_limit = checkpoint["force_limit"]
            print(f"已加载权重：{args.checkpoint}，训练更新次数：{checkpoint['updates']}")
            print("正在播放训练好的策略；按 Ctrl+C 退出")

        if args.control == "train":
            trainer = PPOTrainer(policy, critic)
            update_count = 0
        elif args.control == "pd":
            print("PD 模式仅用于双杆物理检查，当前增益未针对双杆调节，不保证平衡")

        step = 0

        while sim.is_headless_or_exist_active_visualizer():
            if args.control in ("sample", "train"):
                rollout, observation = collect_rollout(
                    env,
                    policy,
                    critic,
                    observation,
                    num_steps=128,
                )
                advantages, returns = compute_gae(rollout)

                if args.control == "train":
                    losses = trainer.update(rollout, advantages, returns)
                    update_count += 1

                    print(
                        f"更新：{update_count}/{args.max_updates}，"
                        f"策略损失：{losses['policy_loss']:.4f}，"
                        f"价值损失：{losses['value_loss']:.4f}，"
                        f"失败次数：{rollout['terminated'].sum().item()}"
                    )

                    if update_count >= args.max_updates:
                        checkpoint_path = args.checkpoint
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                        torch.save(
                            {
                                "environment": "double_cartpole",
                                "observation_dim": env.observation_dim,
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
                step_start = time.perf_counter()
                # 展示策略的平均动作，不加入训练时的探索噪声。
                with torch.no_grad():
                    actions = policy(observation)
            elif args.control == "zero":
                # 无推力模式用于观察两个关节的被动物理行为。
                actions = torch.zeros((env.num_envs, 1), device=sim.device)
            else:
                theta = observation[:, 2]
                theta_dot = observation[:, 3]

                control_force = compute_pd_force(
                    theta=theta,
                    theta_dot=theta_dot,
                    kp=90.0,
                    kd=15.0,
                )
                control_force += compute_pd_force(
                    theta=observation[:, 4],
                    theta_dot=observation[:, 5],
                    kp=60.0,
                    kd=10.0,
                )

                # 每轮开始的前 0.2 秒，保留 1 N 扰动
                disturbance = (
                    env.episode_steps < round(0.2 / env.step_dt)
                ).float()

                # 环境接收归一化动作：[环境数量, 1]
                actions = (
                    (control_force + disturbance) / env.force_limit
                ).unsqueeze(-1)

            observation, rewards, terminated, truncated, _ = env.step(
                actions
            )
            step += 1

            if step % 48 == 0:
                print(
                    f"第一杆：{torch.rad2deg(observation[0, 2]).item():.2f} 度，"
                    f"第二杆：{torch.rad2deg(observation[0, 4]).item():.2f} 度，"
                    f"奖励：{rewards[0].item():.4f}"
                )

            done_ids = (terminated | truncated).nonzero(
                as_tuple=False
            ).flatten()

            if done_ids.numel() > 0:
                if args.control == "play":
                    env_id = done_ids[0].item()
                    episode_time = env.episode_steps[env_id].item() * env.step_dt
                    reason = "小车越界或任一杆倾角超过 30 度" if terminated[env_id].item() else "达到 5 秒上限"
                    print(f"环境 {env_id} 本轮持续 {episode_time:.2f} 秒：{reason}")
                observation = env.reset(done_ids)
                print("本轮结束，已重置")

            if args.control == "play":
                # 只限制播放速度，训练和采样仍按计算能力推进。
                time.sleep(max(0.0, env.step_dt - (time.perf_counter() - step_start)))

        print("仿真运行结束")
    finally:
        sim_utils.SimulationContext.clear_instance()
