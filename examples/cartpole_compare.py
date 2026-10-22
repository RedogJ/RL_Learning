"""Compare two RSL-RL Cartpole policies in one synchronized simulation."""

from __future__ import annotations

import argparse
import copy
import os
import sys

import torch

from isaaclab.app import launch_simulation
from isaaclab.utils.assets import retrieve_file_path

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args as parse_play_args
from isaaclab_rl.entrypoints.common import (
    apply_env_overrides,
    create_isaaclab_env,
    pre_launch_video_config,
    run_playback,
    show_run_summary,
    startup_screen,
)
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, argparse.Namespace]:
    """Parse comparison-only arguments, then reuse the normal RSL-RL play parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint-l2", required=True)
    parser.add_argument("--checkpoint-l3", required=True)
    comparison_args, play_argv = parser.parse_known_args(argv)
    return comparison_args, parse_play_args(play_argv)


def make_initial_states_identical(env_cfg) -> None:
    """Give both policies exactly the same state at every episode reset."""
    cart_reset = env_cfg.events.reset_cart_position.params
    cart_reset["position_range"] = (0.0, 0.0)
    cart_reset["velocity_range"] = (0.0, 0.0)

    pole_reset = env_cfg.events.reset_pole_position.params
    pole_reset["position_range"] = (0.35, 0.35)
    pole_reset["velocity_range"] = (0.0, 0.0)


def main(argv: list[str]) -> None:
    comparison_args, args_cli = parse_args(argv)
    installed_version = check_rsl_rl_version()

    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
        pre_launch_video_config(env_cfg, args_cli)
        show_run_summary(screen, args_cli, env_cfg, library="rsl_rl (L2 + L3)", action="compare")
        screen.stage("Launching simulation")

        with launch_simulation(env_cfg, args_cli):
            agent_cfg = cli_args_rsl_rl.update_rsl_rl_cfg(agent_cfg, args_cli)
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
            apply_env_overrides(args_cli, env_cfg)

            # Two environments are placed along the x axis. Looking along +y makes them appear side by side.
            env_cfg.scene.num_envs = 2
            env_cfg.scene.env_spacing = 8.0
            env_cfg.seed = agent_cfg.seed
            env_cfg.sim.default_visualizer_cfg.eye = (0.0, -16.0, 7.0)
            env_cfg.sim.default_visualizer_cfg.lookat = (0.0, 0.0, 1.0)
            make_initial_states_identical(env_cfg)

            checkpoint_l2 = retrieve_file_path(comparison_args.checkpoint_l2)
            checkpoint_l3 = retrieve_file_path(comparison_args.checkpoint_l3)
            for label, path in (("L2", checkpoint_l2), ("L3", checkpoint_l3)):
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")

            screen.stage("Creating two synchronized environments")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=False,
            )
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

            screen.stage("Loading L2 and L3 policies")
            runner_l2 = create_rsl_rl_runner(env, copy.deepcopy(agent_cfg))
            runner_l2.load(checkpoint_l2)
            policy_l2 = runner_l2.get_inference_policy(device=env.unwrapped.device)

            runner_l3 = create_rsl_rl_runner(env, copy.deepcopy(agent_cfg))
            runner_l3.load(checkpoint_l3)
            policy_l3 = runner_l3.get_inference_policy(device=env.unwrapped.device)

            obs = env.get_observations()
            origins = env.unwrapped.scene.env_origins.detach().cpu().tolist()
            print("\n[COMPARE] 左侧：L2 平方误差策略")
            print("[COMPARE] 右侧：L3 三次方误差策略")
            print(f"[COMPARE] L2 environment origin: {origins[1]}")
            print(f"[COMPARE] L3 environment origin: {origins[0]}")
            print("[COMPARE] 两侧初始状态：cart=0 m, pole=0.35 rad, velocities=0\n")

            def step() -> None:
                nonlocal obs
                # Environment 1 is at x=-4 (left); environment 0 is at x=+4 (right).
                action_l2 = policy_l2(obs[1:2])
                action_l3 = policy_l3(obs[0:1])
                actions = torch.cat((action_l3, action_l2), dim=0)
                obs, _, dones, _ = env.step(actions)
                policy_l2.reset(dones[1:2])
                policy_l3.reset(dones[0:1])

            screen.close()
            run_playback(step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)
            env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
