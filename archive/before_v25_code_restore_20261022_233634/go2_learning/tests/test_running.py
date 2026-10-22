"""移动任务：速度目标、奖励坐标系、键盘输入和任务/模型隔离。"""

import argparse
import ast
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from go2_learning.control.commands import VelocityCommands
from go2_learning.tasks import TASKS, get_task
from go2_learning.tasks.running.rewards import (
    tracking_linear_reward, tracking_yaw_reward, posture_reward,
)
from go2_learning.tests.test_reset_settling import load_env_class, Scene, Sim
from go2_learning.training.checkpoints import resolve_checkpoint, validate_checkpoint_task
from go2_learning.training.history import make_segment
from go2_learning.ui.keyboard import KeyboardInput


class RunningTests(unittest.TestCase):
    def test_posture_strengthens_only_zero_command_and_low_actual_speed(self):
        cfg = get_task("running").cfg.rewards
        cfg.posture_weight = 2.
        cfg.posture_stand_still_scale = 5.
        cfg.posture_velocity_threshold = .3
        state = {
            "joint_offset": torch.ones(8, 12),
            "commands": torch.tensor([[0., 0., 0.], [1.5, 0., 0.], [-1.5, 0., 0.],
                                      [0., 0., .6], [0., .5, 0.], [0., 0., 0.],
                                      [0., 0., 0.], [0., 0., 0.]]),
            "linear_velocity": torch.tensor([[0., 0., 9.], [0., 0., 0.], [0., 0., 0.],
                                             [0., 0., 0.], [0., 0., 0.], [.31, 0., 0.],
                                             [0., -.31, 0.], [.3, 0., 0.]]),
        }
        torch.testing.assert_close(posture_reward(state, cfg),
                                   torch.tensor([-120., -24., -24., -24., -24., -24., -24., -120.]))
        cfg.posture_stand_still_scale = 1.
        torch.testing.assert_close(posture_reward(state, cfg), torch.full((8,), -24.))

    def test_stationary_posture_config_round_trip_and_legacy_playback(self):
        task = get_task("running")
        task.cfg.rewards.posture_stand_still_scale = 7.
        task.cfg.rewards.posture_velocity_threshold = .2
        saved = task.run_config(False)
        restored = get_task("running")
        restored.load_playback_config(saved)
        self.assertEqual(restored.run_config(False), saved)
        saved["reward_cfg"].pop("posture_stand_still_scale")
        saved["reward_cfg"].pop("posture_velocity_threshold")
        restored.load_playback_config(saved)
        self.assertEqual(restored.cfg.rewards.posture_stand_still_scale, 1.)
        for key in ("posture_stand_still_scale", "posture_velocity_threshold"):
            for value in (-1., float("nan"), float("inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    type(task.cfg.rewards)(**{key: value})

    def test_stationary_posture_reward_is_reported_with_dt(self):
        task = get_task("running")
        task.cfg.environment.reset_settle_seconds = 0.
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        scene.robot.data.joint_pos.torch += .1
        _, _, _, extras = env.step(env.standing_actions)
        expected = -task.cfg.rewards.posture_weight * task.cfg.rewards.posture_stand_still_scale * 12 * .1**2 * env.dt
        self.assertAlmostEqual(extras["log"]["Reward/posture"].item(), expected, places=5)

    def test_running_is_independent_and_restores_its_own_saved_configuration(self):
        running, standing = get_task("running"), get_task("standing")
        self.assertNotEqual(running.version, standing.version)
        running.cfg.rewards.target_height = .33
        running.cfg.agent["actor"]["hidden_dims"][0] = 64
        self.assertNotEqual(running.cfg.rewards.target_height, standing.cfg.rewards.target_height)
        self.assertEqual(standing.cfg.agent["actor"]["hidden_dims"][0], 128)
        saved = running.run_config(True)
        restored = get_task("running")
        restored.load_playback_config(saved)
        self.assertEqual(restored.run_config(True), saved)
        with self.assertRaises(ValueError):
            validate_checkpoint_task(standing.run_config(False), running)
        saved["command_source"] = "episode_random"
        self.assertEqual(make_segment(saved, "test", 1, 2, "checkpoint")["command_source"], "episode_random")

    def test_linear_tracking_rewards_forward_motion_in_body_frame(self):
        cfg = get_task("running").cfg.rewards
        state = {
            "roll": torch.zeros(3),
            "commands": torch.tensor([[.4, 0., 0.]]).repeat(3, 1),
            "linear_velocity": torch.tensor([[.4, 0., 9.], [0., 0., 0.], [-.4, 0., 0.]]),
            # 狗转过 90 度后世界 Y 速度才是前进；此项必须仍读取机身速度。
            "linear_velocity_w": torch.tensor([[0., .4, 0.]]).repeat(3, 1),
        }
        rewards = tracking_linear_reward(state, cfg)
        self.assertAlmostEqual(rewards[0].item(), cfg.tracking_linear_weight)
        self.assertGreater(rewards[0], rewards[1])
        self.assertGreater(rewards[1], rewards[2])
        # 侧滑也属于跟踪误差；竖直速度不属于水平速度目标。
        state["linear_velocity"][0, 1] = .4
        self.assertLess(tracking_linear_reward(state, cfg)[0], rewards[0])

    def test_turning_reward_tracks_direction_and_zero_command_means_stop(self):
        cfg = get_task("running").cfg.rewards
        state = {"roll": torch.zeros(3), "commands": torch.tensor([[0., 0., .6]]).repeat(3, 1),
                 "angular_velocity": torch.tensor([[0., 0., .6], [0., 0., 0.], [0., 0., -.6]])}
        rewards = tracking_yaw_reward(state, cfg)
        self.assertAlmostEqual(rewards[0].item(), cfg.tracking_yaw_weight)
        self.assertGreater(rewards[0], rewards[1])
        self.assertGreater(rewards[1], rewards[2])
        state["commands"].zero_()
        stopped = tracking_yaw_reward(state, cfg)
        self.assertGreater(stopped[1], stopped[0])
        torch.testing.assert_close(stopped[0], stopped[2])

    def test_stationary_tracking_is_stricter_without_changing_movement_or_turning(self):
        cfg = get_task("running").cfg.rewards
        cfg.tracking_linear_weight, cfg.linear_sigma = 200., .6
        cfg.stationary_linear_sigma = .1
        commands = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                                 [1.5, 0., 0.], [-1.5, 0., 0.], [0., 0., .6], [0., .5, 0.]])
        velocity = commands.clone()
        velocity[:, 2] = 0.
        velocity[1:, 0] += .05
        velocity[2, 0] = -.05
        velocity.requires_grad_()
        state = {"commands": commands, "linear_velocity": velocity}
        rewards = tracking_linear_reward(state, cfg)
        error = (velocity[:, :2] - commands[:, :2]).square().sum(-1)
        torch.testing.assert_close(rewards[:3], 200 * torch.exp(-error[:3] / .1**2))
        torch.testing.assert_close(rewards[3:], 200 * torch.exp(-error[3:] / .6**2))
        gradient = torch.autograd.grad(rewards.sum(), velocity)[0]
        self.assertLess(gradient[1, 0], 0.)
        self.assertGreater(gradient[2, 0], 0.)
        cfg.stationary_linear_sigma = None
        torch.testing.assert_close(tracking_linear_reward(state, cfg), 200 * torch.exp(-error / .6**2))

    def test_stationary_sigma_validation_and_legacy_playback(self):
        reward_type = type(get_task("running").cfg.rewards)
        for value in (0., -1., float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reward_type(stationary_linear_sigma=value)
        saved = get_task("running").run_config(False)
        saved["reward_cfg"].pop("stationary_linear_sigma")
        restored = get_task("running")
        restored.load_playback_config(saved)
        self.assertIsNone(restored.cfg.rewards.stationary_linear_sigma)

    def test_contact_penalty_is_identical_for_standing_forward_backward(self):
        task = get_task("running")
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        state = env._read_state()
        state["commands"] = torch.tensor([[0., 0., 0.], [1.5, 0., 0.], [-1.5, 0., 0.]])
        state["foot_contact_ratio"] = torch.zeros(3, 4)
        terms = task.reward_components(state, env.standing_actions, torch.zeros(3, dtype=torch.bool), env.dt)
        expected = -task.cfg.rewards.foot_contact_ratio_weight * 4 * env.dt
        torch.testing.assert_close(terms["foot_contact_ratio"], torch.full((3,), expected))
        self.assertNotIn("foot_support", terms)

    def test_contact_ratio_penalty_decreases_continuously_to_full_support(self):
        task = get_task("running")
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        state = env._read_state()
        state["foot_contact_ratio"] = torch.tensor([0., .5, 1.])[:, None].repeat(1, 4)
        terms = task.reward_components(state, env.standing_actions, torch.zeros(3, dtype=torch.bool), env.dt)
        expected = -task.cfg.rewards.foot_contact_ratio_weight * 4 * env.dt
        torch.testing.assert_close(terms["foot_contact_ratio"], torch.tensor([expected, expected * .25, 0.]))

    def test_force_balance_is_population_variance_and_independent_of_command(self):
        task = get_task("running")
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        state = env._read_state()
        state["foot_mean_force_z"] = torch.tensor([[50., 50., 50., 50.],
                                                 [0., 100., 0., 100.], [100., 0., 100., 0.]])
        state["commands"] = torch.tensor([[0., 0., 0.], [1.5, 0., 0.], [-1.5, 0., 0.]])
        args = (state, env.standing_actions, torch.zeros(3, dtype=torch.bool), env.dt)
        terms = task.reward_components(*args)
        expected = -task.cfg.rewards.foot_force_balance_weight * 2500 * env.dt
        torch.testing.assert_close(terms["foot_force_balance"], torch.tensor([0., expected, expected]))
        task.cfg.rewards.foot_force_balance_weight = 0.
        del state["foot_mean_force_z"]
        self.assertEqual(task.reward_components(*args)["foot_force_balance"].sum(), 0.)

    def test_training_samples_fixed_speed_with_random_direction_per_episode(self):
        cfg = get_task("running").cfg.commands
        commands = VelocityCommands(256, "cpu", 1 / 30, cfg, random_sampling=True)
        commands.reset(torch.arange(256))
        values = commands.values.clone()
        self.assertTrue(((values[:, 0].abs() == 1.5) | (values[:, 0] == 0.)).all())
        self.assertAlmostEqual(cfg.standing_probability, 1 / 3)
        self.assertTrue((values[:, 2] == 0.).all())
        self.assertTrue((values[:, 1] == 0.).all())
        self.assertTrue((values[:, 0] < 0.).any())
        self.assertTrue((values[:, 0] > 0.).any())
        self.assertTrue((values == 0.).all(dim=-1).any())
        for _ in range(20):
            commands.step(torch.zeros(256, dtype=torch.bool))
        torch.testing.assert_close(commands.values, values)
        commands.reset(torch.arange(16))
        torch.testing.assert_close(commands.values[16:], values[16:])
        self.assertFalse(torch.equal(commands.values[:16], values[:16]))

    def test_arrow_keys_form_velocity_targets_and_release_means_zero(self):
        cfg = get_task("running").cfg.commands
        commands = VelocityCommands(1, "cpu", 1 / 30, cfg)
        keyboard = KeyboardInput.__new__(KeyboardInput)
        for pressed, expected in (
            ({"UP"}, [1.5, 0., 0.]), ({"DOWN"}, [-1.5, 0., 0.]),
            ({"LEFT"}, [0., 0., .6]), ({"RIGHT"}, [0., 0., -.6]),
            ({"UP", "RIGHT"}, [1.5, 0., -.6]),
            ({"UP", "DOWN", "LEFT", "RIGHT"}, [0., 0., 0.]), (set(), [0., 0., 0.]),
        ):
            keyboard.pressed = pressed
            commands.set_keyboard(keyboard.axes)
            torch.testing.assert_close(commands.values[0], torch.tensor(expected))

    def test_environment_observation_and_reward_receive_the_same_command(self):
        task = get_task("running")
        task.cfg.environment.reset_settle_seconds = 0.
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        env.commands.set_keyboard((1, -1))
        obs = env.get_observations()["policy"]
        self.assertEqual(obs.shape, (3, 52))
        torch.testing.assert_close(obs[:, 45:48], torch.tensor([[1.5, 0., -.6]]).repeat(3, 1))
        expected = task.reward_components(env._read_state(), env.standing_actions,
                                          torch.zeros(3, dtype=torch.bool), env.dt)
        _, _, _, extras = env.step(env.standing_actions)
        self.assertAlmostEqual(extras["log"]["Reward/tracking_linear"].item(), expected["tracking_linear"].mean().item())
        self.assertAlmostEqual(extras["log"]["Reward/tracking_yaw"].item(), expected["tracking_yaw"].mean().item())
        self.assertNotIn("Reward/foot_support", extras["log"])

    def test_latest_model_search_cannot_select_standing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("running", "standing"):
                folder = root / name / "run"
                folder.mkdir(parents=True)
                (folder / "latest.pt").touch()
                (folder / "run.json").write_text(json.dumps(get_task(name).run_config(False)))
            self.assertEqual(resolve_checkpoint("latest", "running", root), root / "running/run/latest.pt")

    def test_cli_uses_automatic_commands_in_train_and_keyboard_only_in_play(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args")

        def add_launcher_args(parser):
            parser.add_argument("--viz", dest="visualizer", type=lambda value: value.split(","))

        namespace = dict(argparse=argparse, TASKS=TASKS, get_task=get_task,
                         add_launcher_args=add_launcher_args, __doc__="")
        exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), namespace)
        for extra, expected in (
            (["--control", "train", "--viz", "none"], False),
            (["--control", "play", "--viz", "newton_gl"], True),
            (["--control", "play", "--viz", "none", "--no-keyboard"], False),
        ):
            with patch.object(sys, "argv", ["go2", "--task", "running", *extra]):
                self.assertEqual(namespace["parse_args"]().keyboard, expected)
        for terrain in ("rough", "stairs"):
            for mode in ("view", "play"):
                with patch.object(sys, "argv", ["go2", "--task", "running", "--control", mode,
                                              "--terrain", terrain, "--viz", "newton_gl"]):
                    args = namespace["parse_args"]()
                    self.assertEqual((args.terrain, args.num_envs), (terrain, 1))
            for extra in (["--control", "train"], ["--control", "view", "--num_envs", "2"]):
                with patch.object(sys, "argv", ["go2", "--terrain", terrain, *extra]):
                    with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                        namespace["parse_args"]()
        with patch.object(sys, "argv", ["go2", "--task", "running", "--control", "train", "--keyboard"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                namespace["parse_args"]()


if __name__ == "__main__":
    unittest.main()
