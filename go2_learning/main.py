# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""标准 RSL-RL 入口：创建环境 → 读取配置 → 创建 Runner → 训练或播放。"""

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from dataclasses import asdict
from pathlib import Path

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_physx.physics import PhysxCfg

from go2_learning.agent_cfg import make_agent_cfg
from go2_learning.randomization import RandomizationCfg
from go2_learning.commands import CommandCfg, TASK_VERSION
from go2_learning.rewards import RewardCfg
from go2_learning.attitude_integral import IntegralCfg
from go2_learning.training_history import TrainingHistoryMixin, inherited_history, validate_network_config

LOG_ROOT = Path(__file__).resolve().parent / "logs"
LOG_TIMEZONE = timezone(timedelta(hours=8))  # 日志目录固定使用北京时间，不依赖 Docker 时区。


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("view", "train", "play"), default="view")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_updates", type=int, default=300, help="本次训练执行的 PPO 更新数；续训时为额外增加的轮数")
    parser.add_argument("--checkpoint", default="latest", help="播放权重路径，或 latest")
    parser.add_argument("--resume", default=None, help="加载网络权重并继承训练历史；奖励和训练参数使用当前代码")
    parser.add_argument("--run_name", default="baseline")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--randomization", action=argparse.BooleanOptionalAction, default=None,
                        help="初始关节角随机化和随机推力；训练默认开启，view/play 默认关闭")
    parser.add_argument("--steps", type=int, default=0, help="播放步数，0 表示一直播放")
    parser.add_argument("--keyboard_test", action="store_true", help="在 GL 窗口检测上下左右键，仅打印输入")
    parser.add_argument("--keyboard", action="store_true", help="播放时用上下键控制前后速度，左右键控制转向")
    parser.add_argument("--verbose_logs", action="store_true", help="训练时在终端显示 RSL-RL 完整日志表")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = 42
    if args.resume is not None and args.control != "train":
        parser.error("--resume 仅用于 train；播放请选择 --checkpoint")
    if args.keyboard_test and args.control == "train":
        parser.error("键盘检测用于 view 或 play 模式")
    if args.keyboard and (args.control != "play" or args.keyboard_test):
        parser.error("--keyboard 用于 play，且不能与 --keyboard_test 同时使用")
    if args.keyboard:
        parser.error("当前为纯站立阶段，请用 play 播放；检测键盘可用 view --keyboard_test")
    if args.randomization is None:
        args.randomization = args.control == "train"
    if args.num_envs is None:
        args.num_envs = 2048 if args.control == "train" else 1
    if args.num_envs < 1 or args.max_updates < 1 or args.steps < 0:
        parser.error("num_envs/max_updates 必须为正数，steps 不能为负数")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not args.run_name or any(c not in allowed for c in args.run_name):
        parser.error("run_name 只允许英文字母、数字、下划线和连字符")
    if args.control == "train" and args.checkpoint != "latest":
        parser.error("--checkpoint 用于 play；继续训练请使用 --resume")
    return args


def resolve_checkpoint(value: str) -> Path:
    """只在本版本日志中查找最近完成的训练，也可以指定某个权重文件。"""
    if value != "latest":
        return Path(value).expanduser().resolve()
    checkpoints = []
    for path in LOG_ROOT.glob("*/latest.pt"):
        metadata = path.with_name("run.json")
        if metadata.is_file() and json.loads(metadata.read_text()).get("task_version") == TASK_VERSION:
            checkpoints.append(path)
    if not checkpoints:
        raise FileNotFoundError("没有完整关节范围的站立策略，请先运行 train；旧版 ±0.25 rad 动作权重不能直接加载。")
    return max(checkpoints, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    args = parse_args()
    command_cfg, reward_cfg = CommandCfg(), RewardCfg()
    integral_cfg = IntegralCfg()
    train_cfg = make_agent_cfg()
    randomization_cfg = RandomizationCfg() if args.randomization else None
    run_cfg = None
    if args.control == "play" or args.resume is not None:
        checkpoint = resolve_checkpoint(args.resume if args.resume is not None else args.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"检查点必须是具体的 .pt 文件：{checkpoint}")
        run_cfg = json.loads(checkpoint.with_name("run.json").read_text())
        if run_cfg.get("task_version") != TASK_VERSION:
            raise ValueError("该权重与当前完整关节范围的动作映射不兼容，请重新训练。")
        if args.control == "play":
            # 播放按原模型配置重建网络和观测，避免修改默认值影响旧模型。
            train_cfg = run_cfg["agent_cfg"]
            command_cfg = CommandCfg(**run_cfg["command_cfg"])
            saved_rewards = {key: value for key, value in run_cfg["reward_cfg"].items()
                             if key not in ("roll_integral_weight", "pitch_integral_weight", "height_integral_weight",
                                            "tracking_linear_weight", "linear_sigma", "tracking_yaw_weight", "yaw_sigma",
                                            "vertical_velocity_weight")}
            reward_cfg = RewardCfg(**{
                "tracking_reward_type": "quadratic", "only_positive_rewards": False,
                "action_limit_outside_floor": False,
                "roll_velocity_weight": 0.0, "pitch_velocity_weight": 0.0,
                "horizontal_velocity_weight": 0.0, "yaw_velocity_weight": 0.0,
                "undesired_contact_weight": 0.0,
                "foot_support_weight": 0.0,
                "action_rate_weight": 0.0, **saved_rewards,
            })
            integral_cfg = IntegralCfg(**run_cfg["integral_cfg"])
        else:
            # 续训只继承网络参数和轮数历史，所有训练配置使用当前代码/命令行。
            validate_network_config(train_cfg, run_cfg["agent_cfg"])
    torch.manual_seed(args.seed)
    sim_cfg = SimulationCfg(dt=1 / 120, physics=PhysxCfg())
    with launch_simulation(sim_cfg, args):
        from isaaclab.sim import SimulationContext
        from rsl_rl.runners import OnPolicyRunner

        from go2_learning.env import make_env

        sim = SimulationContext(sim_cfg)
        mouse_push = None
        keyboard_input = None
        try:
            # 1. 创建环境：物理、奖励、重置都由环境负责。
            env = make_env(
                sim, args.num_envs, randomization_cfg, args.seed,
                command_cfg=command_cfg, random_commands=args.control == "train", reward_cfg=reward_cfg,
                integral_cfg=integral_cfg,
            )
            if run_cfg is not None:
                # 默认角决定观测和姿态奖励的参考，机械限位决定动作映射。
                if list(env.robot.joint_names) != run_cfg["joint_names"]:
                    raise ValueError("当前关节名称/顺序与检查点不一致，不能直接加载")
                saved_pose = torch.tensor(run_cfg["default_joint_pos"], device=env.device)
                if saved_pose.shape != env.default_joint_pos[0].shape or not torch.allclose(
                    saved_pose, env.default_joint_pos[0], atol=1e-6, rtol=0.0,
                ):
                    raise ValueError("当前默认关节站姿与检查点不一致，请恢复对应的 scene.py 配置后加载")
                saved_limits = torch.tensor(run_cfg["action_joint_limits"], device=env.device)
                if saved_limits.shape != env.action_joint_limits[0].shape or not torch.allclose(
                    saved_limits, env.action_joint_limits[0], atol=1e-6, rtol=0.0,
                ):
                    raise ValueError("当前关节机械限位与检查点不一致，不能直接加载")
            print(f"随机初始关节角 / 随机推力：{'开启' if args.randomization else '关闭'}")
            print(f"每回合重置后等待 {env.reset_settle_seconds:.2f}s（仿真时间）；"
                  "等待期间维持默认站姿，不执行策略、不计奖励或回合时间")
            env.finish_initial_settling()
            if args.control in ("view", "play"):
                from go2_learning.interaction import attach_mouse_push

                mouse_push = attach_mouse_push(env)
                if mouse_push is not None:
                    env.before_physics_step = mouse_push.apply
                    env.on_reset = mouse_push.clear

            if args.keyboard_test or args.keyboard:
                from go2_learning.keyboard import attach_keyboard_input

                keyboard_input = attach_keyboard_input(sim, test_only=not args.keyboard)

            if args.control == "view":
                # 使用默认站姿对应的归一化动作；此模式没有神经网络参与。
                print(f"Go2 场景已加载：{env.num_envs} 台，{env.num_actions} 个关节；Ctrl+C 退出")
                actions = env.standing_actions.clone()
                step = 0
                with torch.inference_mode():
                    while sim.is_headless_or_exist_active_visualizer() and (args.steps == 0 or step < args.steps):
                        started = time.perf_counter()
                        env.step(actions, auto_reset=False)
                        step += 1
                        if "none" not in (args.visualizer or []):
                            time.sleep(max(0.0, env.dt - (time.perf_counter() - started)))
                return

            # 2. 读取配置：Python 字典和官方示例的 YAML 最终提供同样的接口。
            if args.control == "train":
                run_id = datetime.now(LOG_TIMEZONE).strftime("%Y-%m-%d_%H-%M-%S_%f")
                log_dir = LOG_ROOT / f"{run_id}_{args.run_name}"
                log_dir.mkdir(parents=True)
                # 保存网络配置，使以后修改 agent_cfg.py 也能播放旧模型。
                current_run_cfg = {
                    "agent_cfg": train_cfg,
                    "seed": args.seed,
                    "num_envs": args.num_envs,
                    "randomization_cfg": asdict(randomization_cfg) if randomization_cfg else None,
                    "task_version": TASK_VERSION,
                    "command_cfg": asdict(command_cfg),
                    "reward_cfg": asdict(reward_cfg),
                    "integral_cfg": asdict(integral_cfg),
                    "joint_names": list(env.robot.joint_names),
                    "default_joint_pos": env.default_joint_pos[0].tolist(),
                    "action_joint_limits": env.action_joint_limits[0].tolist(),
                    "reset_settle_seconds": env.reset_settle_seconds,
                    "reset_mode": "per_episode_delay",
                    "resumed_from": str(checkpoint) if args.resume is not None else None,
                    "resume_mode": "weights_only" if args.resume is not None else None,
                    "requested_updates": args.max_updates,
                }
            else:
                log_dir = None

            # 3. 创建标准 Runner，由库创建网络、PPO 和采样存储。
            class TrainingRunner(TrainingHistoryMixin, OnPolicyRunner):
                pass

            runner_class = TrainingRunner if args.control == "train" else OnPolicyRunner
            runner = runner_class(
                env=env,
                train_cfg=train_cfg,
                log_dir=str(log_dir) if log_dir else None,
                device=str(sim.device),
            )

            # 4. 训练，或者加载模型进行推理。
            if args.control == "train":
                from go2_learning.settling_training import SettlingLogger, SettlingPPO

                runner.alg = SettlingPPO(runner.alg, env)
                runner.logger = SettlingLogger(runner.logger)
                parent_history = []
                if args.resume is not None:
                    checkpoint_infos = runner.load(str(checkpoint), map_location=str(sim.device), load_cfg={
                        "actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False,
                    })
                    # 轮号用于衔接历史，优化器不恢复。iter 是刚完成的零起始轮号。
                    runner.current_learning_iteration += 1
                    parent_history = inherited_history(
                        checkpoint, run_cfg, runner.current_learning_iteration, checkpoint_infos, LOG_ROOT,
                    )
                    print(f"已加载网络权重：{checkpoint}；继承 {runner.current_learning_iteration} 轮历史，"
                          f"本次再训练 {args.max_updates} 轮；优化器重新初始化")
                runner.configure_history(log_dir, current_run_cfg, parent_history)
                print(f"本次奖励配置（当前代码）：{asdict(reward_cfg)}")
                if not args.verbose_logs:
                    from go2_learning.training_log import CompactTrainingLogger

                    runner.logger = CompactTrainingLogger(runner.logger, env.dt)
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
                        if args.keyboard:
                            env.commands.set_keyboard(keyboard_input.axes)
                            # 输入改变后重新构造观测，本次推理立即看到最新速度目标。
                            obs = env.get_observations()
                        actions = env.standing_actions.clone()
                        active = env.policy_active
                        if active.any():
                            actions[active] = policy(obs[active])
                        obs, _, _, _ = env.step(actions)
                        step += 1
                        time.sleep(max(0.0, env.dt - (time.perf_counter() - started)))
        finally:
            if keyboard_input is not None:
                keyboard_input.close()
            if mouse_push is not None:
                mouse_push.close()
            SimulationContext.clear_instance()


if __name__ == "__main__":
    main()
