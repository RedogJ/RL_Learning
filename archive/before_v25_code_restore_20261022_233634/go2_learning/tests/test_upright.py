"""后腿直立任务：目标方向、支撑足区分、终止、环境接入与配置隔离。"""

import argparse
import ast
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from go2_learning.simulation.contacts import Go2Contacts, REQUIRED_BODY_NAMES
from go2_learning.tasks import TASKS, get_task
from go2_learning.tasks.upright.reward_cfg import RewardCfg
from go2_learning.tasks.upright.reset_cfg import ResetCfg
from go2_learning.tasks.upright.rewards import rear_symmetry_error, upright_alignment
from go2_learning.tests.test_reset_settling import load_env_class, Scene, Sim
from go2_learning.training.checkpoints import resolve_checkpoint, validate_checkpoint_task


class UprightTests(unittest.TestCase):
    def state(self, count=4):
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=get_task("upright"))
        template = env._read_state()
        # 奖励状态与环境一样只包含逐环境张量。
        return {key: value[:1].repeat(count, *([1] * (value.ndim - 1))) for key, value in template.items()}

    def test_near_upright_reset_uses_joint_names_and_xyzw_orientation(self):
        task = get_task("upright")
        names = list(reversed(Scene().robot.joint_names))
        root = torch.tensor([[0., 0., .4, 0., 0., 0., 1.]] * 2)
        joints = torch.zeros(2, 12)
        pose, actual, hold_target = task.reset_state(root.clone(), joints.clone(), names)
        angle = math.radians(task.cfg.reset.forward_elevation_deg)
        torch.testing.assert_close(pose[:, 2], torch.full((2,), task.cfg.reset.root_height))
        torch.testing.assert_close(pose[:, 3:], torch.tensor([[0., -math.sin(angle/2), 0., math.cos(angle/2)]] * 2))
        for i, name in enumerate(names):
            leg, part, _ = name.split("_")
            prefix = 'front' if leg.startswith('F') else 'rear'
            expected = getattr(task.cfg.reset, f"{prefix}_{part}")
            torch.testing.assert_close(actual[:, i], torch.full((2,), expected))
            offset = getattr(task.cfg.reset, f"{prefix}_hold_offset")[("hip", "thigh", "calf").index(part)]
            if prefix == "front" and task.cfg.lock_front_legs:
                offset = 0.
            torch.testing.assert_close(hold_target[:, i], torch.full((2,), expected + offset))
        # 传入副本，不改观测使用的默认参考。
        torch.testing.assert_close(root[:, 2], torch.full((2,), .4))
        self.assertEqual(joints.abs().sum(), 0.)
        with self.assertRaises(ValueError):
            ResetCfg(forward_elevation_deg=float("nan"))
        with self.assertRaises(ValueError):
            ResetCfg(rear_hold_offset=(0., float("nan"), 0.))

    def test_partial_upright_reset_restores_pose_velocity_and_hold_target(self):
        task = get_task("upright")
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        default_reference = scene.robot.data.default_joint_pos.torch.clone()
        torch.testing.assert_close(env.default_joint_pos, default_reference)
        torch.testing.assert_close(env.action_center + env.action_half_range * env.standing_actions, env.reset_joint_target)
        env.finish_initial_settling()
        data = scene.robot.data
        data.root_link_pose_w.torch[0] = torch.tensor([9., 8., .1, 0., 0., 0., 1.])
        data.root_com_vel_w.torch[0] = 5.
        data.joint_pos.torch[0] = 0.
        data.joint_vel.torch[0] = 3.
        others = data.root_link_pose_w.torch[1:].clone()
        env.reset(torch.tensor([0]))
        expected_pose = env.reset_root_pose[0].clone()
        expected_pose[:3] += scene.env_origins[0]
        torch.testing.assert_close(data.root_link_pose_w.torch[0], expected_pose)
        torch.testing.assert_close(data.root_link_pose_w.torch[1:], others)
        torch.testing.assert_close(data.joint_pos.torch[0], env.reset_joint_pos[0])
        torch.testing.assert_close(data.root_com_vel_w.torch[0], torch.zeros(6))
        torch.testing.assert_close(data.joint_vel.torch[0], torch.zeros(12))
        torch.testing.assert_close(scene.robot.targets[0], env.reset_joint_target[0])
        torch.testing.assert_close(env.default_joint_pos, default_reference)
        self.assertGreater(env.settle_steps_remaining[0], 0)
        saved = task.run_config(False)
        self.assertEqual(saved["reset_cfg"]["forward_elevation_deg"], 90.)
        restored = get_task("upright")
        restored.load_playback_config(json.loads(json.dumps(saved)))
        self.assertEqual(json.loads(json.dumps(restored.run_config(False))), json.loads(json.dumps(saved)))
        # 等待结束后，网络目标不再叠加等待期的抗重力偏置。
        env.finish_initial_settling()
        env._set_action(torch.zeros_like(env.actions))
        expected_target = env.action_center.clone()
        expected_target[:, env.fixed_joint_ids] = env.reset_joint_pos[:, env.fixed_joint_ids]
        torch.testing.assert_close(scene.robot.targets, expected_target)

    def test_rear_feet_airborne_penalty_uses_simultaneous_physics_frames(self):
        # 环境 0 交替支撑，环境 1 一半时间两脚悬空，环境 2 全程悬空。
        forces = torch.zeros(3, 4, 4, 3)
        forces[0, 0::2, 2, 2] = 10.
        forces[0, 1::2, 3, 2] = 10.
        forces[1, :2, 2:, 2] = 10.
        from go2_learning.simulation.contacts import contact_counts
        fraction = contact_counts(forces, [0, 1, 2, 3], [], 1., rear_ids=[2, 3])["rear_feet_airborne_fraction"]
        torch.testing.assert_close(fraction, torch.tensor([0., .5, 1.]))
        task = get_task("upright")
        state = self.state(3)
        state["rear_feet_airborne_fraction"] = fraction
        r = task.reward_components(state, torch.zeros(3, 12), torch.zeros(3, dtype=torch.bool), .1)
        torch.testing.assert_close(r["rear_feet_airborne"], -task.cfg.rewards.rear_feet_airborne_weight * fraction * .1)
        saved = task.run_config(False)
        restored = get_task("upright")
        restored.load_playback_config(saved)
        self.assertEqual(restored.cfg.rewards.rear_feet_airborne_weight, task.cfg.rewards.rear_feet_airborne_weight)
        saved["reward_cfg"].pop("rear_feet_airborne_weight")
        restored.load_playback_config(saved)
        self.assertEqual(restored.cfg.rewards.rear_feet_airborne_weight, 0.)

    def test_front_speed_penalty_uses_only_front_actual_joint_velocities(self):
        task = get_task("upright")
        task.cfg.rewards.front_joint_velocity_weight = 2.
        task.cfg.rewards.joint_velocity_weight = 0.
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        front_ids = [scene.robot.joint_names.index(f"{leg}_{part}_joint")
                     for part in ("hip", "thigh", "calf") for leg in ("FL", "FR")]
        scene.robot.data.joint_vel.torch.fill_(100.)
        scene.robot.data.joint_vel.torch[:, front_ids] = 0.
        state = env._read_state()
        actions = torch.zeros(3, 12)
        failed = torch.zeros(3, dtype=torch.bool)
        torch.testing.assert_close(task.reward_components(state, actions, failed, .1)["front_joint_velocity"], torch.zeros(3))
        scene.robot.data.joint_vel.torch[:, front_ids[:2]] = torch.tensor([1., -2.])
        state = env._read_state()
        torch.testing.assert_close(task.reward_components(state, actions, failed, .1)["front_joint_velocity"], torch.full((3,), -1.))
        task.cfg.rewards.front_joint_velocity_weight = 0.
        self.assertNotIn("front_joint_velocity", env._read_state())

    def test_front_target_reward_and_unmasked_actions(self):
        task = get_task("upright")
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        self.assertEqual(env.fixed_joint_ids, [])
        self.assertEqual(scene.robot.limit_writes, [])
        env.finish_initial_settling()
        actions = torch.full_like(env.actions, 0.3)
        env._set_action(actions)
        torch.testing.assert_close(env.raw_actions, actions)
        torch.testing.assert_close(scene.robot.targets, env.action_center + 0.3 * env.action_half_range)
        state = env._read_state()
        for part in ("hip", "thigh", "calf"):
            state[f"{part}_joint_pos"][:, :2] = getattr(task.cfg.rewards, f"front_target_{part}")
        failed = torch.zeros(env.num_envs, dtype=torch.bool)
        r = task.reward_components(state, actions, failed, 0.1)["front_target"]
        torch.testing.assert_close(r, torch.full_like(r, task.cfg.rewards.front_target_weight * 0.1))
        state["thigh_joint_pos"][:, 0] += 0.6
        r = task.reward_components(state, actions, failed, 0.1)["front_target"]
        expected = task.cfg.rewards.front_target_weight * math.exp(-0.6**2 / 6 / task.cfg.rewards.front_target_sigma**2) * 0.1
        torch.testing.assert_close(r, torch.full_like(r, expected))
        state["thigh_joint_pos"][:, 2:] += 1.0
        torch.testing.assert_close(task.reward_components(state, actions, failed, 0.1)["front_target"], r)
        task.cfg.rewards.reward_schema_version = 6
        legacy = task.reward_components(state, actions, failed, .1)["front_target"]
        torch.testing.assert_close(legacy, torch.full_like(legacy, -task.cfg.rewards.front_target_weight * .6**2 / 6 * .1))
        with self.assertRaises(ValueError):
            RewardCfg(front_target_sigma=0.)


    def test_front_joint_lock_masks_policy_and_reset_noise_without_changing_observation_reference(self):
        task = get_task("upright")
        task.cfg.lock_front_legs = True
        scene = Scene()
        before_limits = scene.robot.data.joint_pos_limits.torch.clone()
        before_reference = scene.robot.data.default_joint_pos.torch.clone()
        env = load_env_class()(Sim(scene), scene, task=task)
        ids = env.fixed_joint_ids
        self.assertEqual({scene.robot.joint_names[i] for i in ids}, set(task.fixed_joint_positions()))
        rear = [i for i in range(12) if i not in ids]
        expected = env.reset_joint_pos[:, ids]
        torch.testing.assert_close(scene.robot.data.joint_pos_limits.torch[:, ids],
                                   torch.stack((expected, expected), -1))
        torch.testing.assert_close(scene.robot.data.joint_pos_limits.torch[:, rear], before_limits[:, rear])
        torch.testing.assert_close(env.action_joint_limits, before_limits)
        torch.testing.assert_close(env.default_joint_pos, before_reference)
        torch.testing.assert_close(scene.robot.data.joint_stiffness.torch[:, ids], torch.full_like(expected, 1e6))
        torch.testing.assert_close(scene.robot.data.joint_stiffness.torch[:, rear],
                                   torch.full_like(env.reset_joint_pos[:, rear], 25.))
        env.finish_initial_settling()
        action = torch.full_like(env.actions, .3)
        action[:, ids] = 10.
        env._set_action(action)
        torch.testing.assert_close(scene.robot.targets[:, ids], expected)
        torch.testing.assert_close(env.raw_actions[:, ids], env.standing_actions[:, ids])
        torch.testing.assert_close(scene.robot.targets[:, rear],
                                   env.action_center[:, rear] + .3 * env.action_half_range[:, rear])
        self.assertTrue((action[:, ids] == 10.).all())  # 不原地修改 PPO 保存的采样动作。
        env.randomization = NS(reset=lambda env_ids, pos: pos + .03)
        env.reset(torch.tensor([1]))
        torch.testing.assert_close(scene.robot.data.joint_pos.torch[1, ids], expected[1])
        self.assertEqual(env.num_actions, 12)
        self.assertEqual(env.get_observations()["policy"].shape[-1], 52)
        self.assertTrue(task.run_config(False)["front_legs_fixed"])
        saved = task.run_config(False)
        restored = get_task("upright")
        restored.load_playback_config(saved)
        self.assertFalse(restored.cfg.lock_front_legs)
        self.assertEqual(restored.run_config(False)["front_lock_solver_iterations"], [8, 4])
        task = get_task("upright")
        task.cfg.lock_front_legs = False
        scene = Scene()
        disabled = load_env_class()(Sim(scene), scene, task=task)
        self.assertEqual(disabled.fixed_joint_ids, [])
        self.assertEqual(scene.robot.limit_writes, [])

    def test_reset_hold_target_must_respect_mechanical_limits(self):
        task = get_task("upright")
        task.cfg.reset.rear_hold_offset = (0., 0., 10.)
        scene = Scene()
        with self.assertRaisesRegex(ValueError, "等待期 PD 目标"):
            load_env_class()(Sim(scene), scene, task=task)

    def test_reward_distinguishes_rising_rear_stand_and_headstand(self):
        task = get_task("upright")
        task.cfg.rewards.rear_support_weight = 5.
        state = self.state()
        state["gravity"] = torch.tensor([[0., 0., -1.], [-.5, 0., -3**.5 / 2],
                                         [-1., 0., 0.], [1., 0., 0.]])
        state["height"].fill_(task.target_height)
        state["foot_support_per_foot"] = torch.tensor([[1., 1., 1., 1.], [0., 0., 1., 1.],
                                                       [0., 0., 1., 1.], [1., 1., 0., 0.]])
        state["rear_knee_and_foot_support_fraction"] = torch.tensor([1., 1., 1., 0.])
        terms = task.reward_components(state, torch.zeros(4, 12), torch.zeros(4, dtype=torch.bool), 1 / 30)
        total = sum(terms.values())
        self.assertTrue(torch.isfinite(total).all())
        self.assertLess(total[0], total[1])
        self.assertLess(total[1], total[2])
        self.assertLess(total[3], total[0])
        torch.testing.assert_close(upright_alignment(state), torch.tensor([0., .5, 1., -1.]))

    def test_fore_and_rear_support_are_not_interchangeable(self):
        task = get_task("upright")
        task.cfg.rewards.rear_support_weight = 5.
        state = self.state(2)
        state["gravity"][:] = torch.tensor([-1., 0., 0.])
        state["foot_support_per_foot"] = torch.tensor([[0., 0., 1., 1.], [1., 1., 0., 0.]])
        state["rear_knee_and_foot_support_fraction"] = torch.tensor([1., 0.])
        terms = task.reward_components(state, torch.zeros(2, 12), torch.zeros(2, dtype=torch.bool), .1)
        torch.testing.assert_close(terms["rear_support"], torch.tensor([.5, 0.]))
        torch.testing.assert_close(terms["front_lift"], torch.tensor([.5, 0.]))
        # 失败项与 dt 无关。
        terms = task.reward_components(state, torch.zeros(2, 12), torch.tensor([False, True]), .001)
        torch.testing.assert_close(terms["failure"], torch.tensor([0., -task.cfg.rewards.failure_penalty]))

    def test_contact_support_is_canonical_even_with_shuffled_sensor_order(self):
        names = list(reversed(REQUIRED_BODY_NAMES))
        history = torch.zeros(1, 4, len(names), 3)
        history[:, :, names.index("RL_foot"), 2] = 10.
        history[:, :2, names.index("RR_foot"), 2] = 10.
        sensor = NS(body_names=names, cfg=NS(history_length=4),
                    data=NS(net_normal_forces_w_history=NS(torch=history)))
        state = Go2Contacts(sensor, 1., 4).read()
        torch.testing.assert_close(state["foot_support_per_foot"], torch.tensor([[0., 0., 1., .5]]))

    def test_rear_support_excludes_same_leg_knee_support_at_each_physics_frame(self):
        task = get_task("upright")
        task.cfg.rewards.reward_schema_version = 4
        names = list(reversed(REQUIRED_BODY_NAMES))
        history = torch.zeros(6, 4, len(names), 3)
        def force(env_id, body, frames=slice(None)):
            history[env_id, frames, names.index(body), 2] = 10.
        # 仅脚掌；两膝共同支撑；左膝共同支撑；右膝支撑不影响左脚。
        for env_id in (0, 1, 2):
            force(env_id, "RL_foot")
            force(env_id, "RR_foot")
        force(1, "RL_calf")
        force(1, "RR_calf")
        force(2, "RL_calf")
        force(3, "RL_foot")
        force(3, "RR_calf")
        # 两者平均接触比例相同，但交错支撑应给分，同时支撑不给分。
        for env_id in (4, 5):
            force(env_id, "RL_foot", slice(0, 2))
        force(4, "RL_calf", slice(2, 4))
        force(5, "RL_calf", slice(0, 2))
        sensor = NS(body_names=names, cfg=NS(history_length=4),
                    data=NS(net_normal_forces_w_history=NS(torch=history)))
        state = self.state(6)
        state.update(Go2Contacts(sensor, 1., 4).read())
        actions, failed, dt = torch.zeros(6, 12), torch.zeros(6, dtype=torch.bool), .1
        terms = task.reward_components(state, actions, failed, dt)
        cfg = task.cfg.rewards
        expected_support = torch.tensor([1., 0., .5, .5, .25, 0.])
        torch.testing.assert_close(terms["rear_support"], cfg.rear_support_weight * expected_support * dt)
        # 脚掌与膝同时接地时，接触惩罚照常扣，不因支撑奖励被取消而丢失。
        torch.testing.assert_close(terms["rear_knee_contact"],
                                   -cfg.rear_knee_contact_weight * torch.tensor([0., 2., 1., 1., 1., 1.]) * dt)
        torch.testing.assert_close(terms["undesired_contact"], torch.zeros(6))
        torch.testing.assert_close(task.metrics(state)["Task/rear_support_count"], expected_support.mean() * 2)
        doubled = task.reward_components(state, actions, failed, 2 * dt)
        torch.testing.assert_close(doubled["rear_support"], 2 * terms["rear_support"])

    def test_version_three_playback_retains_original_foot_support_scoring(self):
        task = get_task("upright")
        saved = task.run_config(False)
        saved["reward_cfg"]["reward_schema_version"] = 3
        task.load_playback_config(saved)
        state = self.state(1)
        state["foot_support_per_foot"][:] = torch.tensor([0., 0., 1., 1.])
        state["rear_leg_supported_fraction"][:] = 1.
        actions, failed = torch.zeros(1, 12), torch.zeros(1, dtype=torch.bool)
        old_terms = task.reward_components(state, actions, failed, .1)
        torch.testing.assert_close(old_terms["rear_support"], torch.tensor([task.cfg.rewards.rear_support_weight * .1]))
        task.cfg.rewards.reward_schema_version = 4
        new_terms = task.reward_components(state, actions, failed, .1)
        torch.testing.assert_close(new_terms["rear_support"], torch.zeros(1))

    def test_horizontal_pose_fails_but_vertical_goal_does_not(self):
        task = get_task("upright")
        state = {"gravity": torch.tensor([[0., 0., -1.], [-1., 0., 0.], [1., 0., 0.], [-1., 0., 0.]]),
                 "height": torch.tensor([.3, .55, .3, .13])}
        self.assertEqual(task.tilt_failures(state).tolist(), [True, False, True, False])
        self.assertEqual(task.failures(state).tolist(), [True, False, True, True])
        self.assertTrue(get_task("standing").tilt_failures(state)[1])  # 不能直接沿用原任务的判定。

    def test_upright_tilt_boundary_in_both_directions_and_playback(self):
        task = get_task("upright")
        angles = torch.deg2rad(torch.tensor([0., 44.9, 45., 45.1, 90., 180.]))
        for axis in (1, 2):
            for sign in (-1., 1.):
                gravity = torch.zeros(6, 3)
                gravity[:, 0] = -angles.cos()
                gravity[:, axis] = sign * angles.sin()
                self.assertEqual(task.tilt_failures({"gravity": gravity}).tolist(),
                                 [False, False, False, True, True, True])
        saved = task.run_config(False)
        saved["termination_cfg"]["maximum_tilt_degrees"] = 120.
        task.load_playback_config(saved)
        self.assertEqual(task.cfg.termination.maximum_tilt_degrees, 45.)
        self.assertEqual(saved["termination_cfg"]["maximum_tilt_degrees"], 120.)

    def test_environment_logs_goal_and_masks_settling_without_horizontal_integrals(self):
        task = get_task("upright")
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        self.assertEqual(env.get_observations()["policy"].shape, (3, 52))
        _, reward, _, extras = env.step(env.standing_actions)
        self.assertEqual(reward.sum(), 0.)
        self.assertNotIn("Task/upright_alignment", extras["log"])
        env.finish_initial_settling()
        for name in ("RL_foot", "RR_foot"):
            scene.contact_forces.forces[:, scene.contact_forces.body_names.index(name), 2] = 10.
        scene.robot.data.projected_gravity_b.torch[:] = torch.tensor([-1., 0., 0.])
        obs, reward, done, extras = env.step(env.standing_actions)
        self.assertFalse(done.any())
        self.assertEqual(extras["log"]["Task/upright_alignment"], 1.)
        self.assertEqual(extras["log"]["Task/rear_support_count"], 2.)
        self.assertEqual(extras["log"]["Task/front_support_count"], 0.)
        self.assertEqual(extras["log"]["Contact/hip_body_count"], 0.)
        self.assertEqual(extras["log"]["Contact/rear_both_airborne_fraction"], 0.)
        self.assertEqual(extras["log"]["Contact/rear_knee_support_count"], 0.)
        self.assertEqual(extras["log"]["Contact/rear_leg_supported_fraction"], 0.)
        self.assertEqual(extras["log"]["Contact/rear_both_unsupported_fraction"], 1.)
        self.assertEqual(extras["log"]["Task/rear_symmetry_error_rad"], 0.)
        self.assertEqual(env.attitude_integral.values.abs().sum(), 0.)
        torch.testing.assert_close(reward, sum(env._reward_components.values()))
        self.assertTrue(torch.isfinite(obs["policy"]).all())
        env.reset(torch.tensor([0]))
        self.assertGreater(env.settle_steps_remaining[0], 0)
        self.assertEqual(env.settle_steps_remaining[1], 0)

    def test_configuration_is_independent_serialized_and_checkpoint_scope_is_separate(self):
        task = get_task("upright")
        original_height = task.target_height
        task.cfg.rewards.target_height = .6
        task.cfg.agent["actor"]["hidden_dims"][0] = 64
        self.assertEqual(get_task("upright").target_height, original_height)
        self.assertEqual(get_task("running").cfg.agent["actor"]["hidden_dims"][0], 128)
        saved = task.run_config(False)
        restored = get_task("upright")
        restored.load_playback_config(saved)
        self.assertEqual(restored.run_config(False), saved)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("standing", "running", "upright"):
                folder = root / name / "run"
                folder.mkdir(parents=True)
                (folder / "latest.pt").touch()
                (folder / "run.json").write_text(json.dumps(get_task(name).run_config(False)))
            self.assertEqual(resolve_checkpoint("latest", "upright", root), root / "upright/run/latest.pt")
        with self.assertRaises(ValueError):
            validate_checkpoint_task(get_task("running").run_config(False), task)

    def test_cli_selects_upright_defaults(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args")
        namespace = dict(argparse=argparse, TASKS=TASKS, get_task=get_task, __doc__="",
                         add_launcher_args=lambda parser: None)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), namespace)
        with patch.object(sys, "argv", ["go2", "--task", "upright", "--control", "train"]):
            args = namespace["parse_args"]()
        self.assertEqual(args.num_envs, 2048)
        self.assertFalse(args.randomization)
        self.assertFalse(args.keyboard)

    def test_height_and_front_lift_are_independent_of_body_orientation(self):
        task = get_task("upright")
        state = self.state(3)
        state["gravity"] = torch.tensor([[0., 0., -1.], [-.5, 0., -3**.5 / 2], [-1., 0., 0.]])
        state["height"].fill_(task.target_height)
        self.assertAlmostEqual(state["height"][0].item(), task.target_height)
        state["foot_support_per_foot"][:] = torch.tensor([0., 0., 1., 1.])
        dt = 1 / 30
        terms = task.reward_components(state, torch.zeros(3, 12), torch.zeros(3, dtype=torch.bool), dt)
        torch.testing.assert_close(terms["height"], torch.full((3,), task.cfg.rewards.height_weight * dt))
        torch.testing.assert_close(terms["front_lift"], torch.full((3,), task.cfg.rewards.front_lift_weight * dt))
        self.assertNotIn("upright_progress", terms)
        self.assertLess(terms["upright_hold"][0], terms["upright_hold"][1])
        self.assertLess(terms["upright_hold"][1], terms["upright_hold"][2])

    def test_per_leg_hip_angles_use_fixed_fl_fr_rl_rr_order(self):
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=get_task("upright"))
        env.finish_initial_settling()
        # 假机器人的关节顺序是 FL/FR/RL/RR × hip/thigh/calf；四个髋在 0–3。
        scene.robot.data.joint_pos.torch[:, :4] = torch.tensor([[.1, -.2, .3, -.4]] * 3)
        _, _, _, extras = env.step(env.standing_actions)
        log = extras["log"]
        self.assertAlmostEqual(log["Task/hip_FL_rad"].item(), .1)
        self.assertAlmostEqual(log["Task/hip_FR_rad"].item(), -.2)
        self.assertAlmostEqual(log["Task/hip_RL_rad"].item(), .3)
        self.assertAlmostEqual(log["Task/hip_RR_rad"].item(), -.4)
        self.assertAlmostEqual(log["Task/hip_abs_angle_rad"].item(), .25)

    def test_per_leg_thigh_and_calf_angles_use_fixed_order(self):
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=get_task("upright"))
        env.finish_initial_settling()
        # 假机器人关节顺序：FL/FR/RL/RR × hip/thigh/calf。
        scene.robot.data.joint_pos.torch[:, 4:8] = torch.tensor([[.4, .5, .6, .7]] * 3)
        scene.robot.data.joint_pos.torch[:, 8:12] = torch.tensor([[-.4, -.5, -.6, -.7]] * 3)
        _, _, _, extras = env.step(env.standing_actions)
        log = extras["log"]
        for leg, value in zip(("FL", "FR", "RL", "RR"), (.4, .5, .6, .7)):
            self.assertAlmostEqual(log[f"Task/thigh_{leg}_rad"].item(), value)
        for leg, value in zip(("FL", "FR", "RL", "RR"), (-.4, -.5, -.6, -.7)):
            self.assertAlmostEqual(log[f"Task/calf_{leg}_rad"].item(), value)

    def test_hip_contact_is_separate_from_other_nonfoot_contacts(self):
        task = get_task("upright")
        task.cfg.rewards.reward_schema_version = 4
        state = self.state(3)
        state["undesired_contact_count"] = torch.tensor([0., 2., 3.])
        state["hip_contact_count"] = torch.tensor([0., 1., 3.])
        dt = .1
        actions, failed = torch.zeros(3, 12), torch.zeros(3, dtype=torch.bool)
        terms = task.reward_components(state, actions, failed, dt)
        cfg = task.cfg.rewards
        torch.testing.assert_close(terms["hip_contact"], -cfg.hip_contact_weight * torch.tensor([0., 1., 3.]) * dt)
        torch.testing.assert_close(terms["undesired_contact"], -cfg.undesired_contact_weight * torch.tensor([0., 1., 0.]) * dt)
        # 新公式版本缺少该项时按当前总罚分率恢复。
        saved = task.run_config(False)
        saved["reward_cfg"].pop("hip_contact_weight")
        restored = get_task("upright")
        restored.load_playback_config(saved)
        self.assertEqual(restored.cfg.rewards.hip_contact_weight, RewardCfg().hip_contact_weight)

    def test_knee_clearance_is_restored_from_saved_reward_configuration(self):
        task = get_task("upright")
        saved = task.run_config(False)
        saved["reward_cfg"].update(rear_knee_clearance_weight=70., rear_knee_clearance_sigma=.12)
        task.load_playback_config(saved)
        state = self.state(3)
        state["rear_knee_height"] = torch.tensor([[0., 0.], [.12, .12], [.5, .2]])
        terms = task.reward_components(state, torch.zeros(3, 12), torch.zeros(3, dtype=torch.bool), .1)
        expected = 7 * torch.tensor([0., 1 - math.exp(-1), 1 - math.exp(-.2 / .12)])
        torch.testing.assert_close(terms["rear_knee_clearance"], expected)
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        env.finish_initial_settling()
        _, reward, _, extras = env.step(env.standing_actions)
        self.assertIn("rear_knee_height", env._read_state())
        self.assertIn("Task/rear_knee_height", extras["log"])
        torch.testing.assert_close(reward, sum(env._reward_components.values()))
        saved["reward_cfg"].pop("rear_knee_clearance_weight")
        saved["reward_cfg"].pop("rear_knee_clearance_sigma")
        task.load_playback_config(saved)
        self.assertEqual(task.cfg.rewards.rear_knee_clearance_weight, 0.)
        scene = Scene()
        self.assertNotIn("rear_knee_height", load_env_class()(Sim(scene), scene, task=task)._read_state())

    def test_all_rear_calf_contacts_are_excluded_from_generic_penalty(self):
        task = get_task("upright")
        task.cfg.rewards.reward_schema_version = 4
        state = self.state(2)
        state["undesired_contact_count"] = torch.tensor([2., 1.])
        state["rear_knee_contact_count"] = torch.tensor([2., 1.])
        # 无论支撑还是侧向接触，后小腿均由专用项负责。
        state["rear_knee_support_count"] = torch.tensor([2., 0.])
        dt = .1
        terms = task.reward_components(state, torch.zeros(2, 12), torch.zeros(2, dtype=torch.bool), dt)
        torch.testing.assert_close(terms["undesired_contact"], torch.zeros(2))

    def test_rear_knee_contact_penalizes_support_and_side_contact(self):
        task = get_task("upright")
        task.cfg.rewards.reward_schema_version = 4
        task.cfg.rewards.rear_knee_contact_weight = 60.
        state = self.state(4)
        state["rear_knee_contact_count"] = torch.tensor([0., 1., 2., 1.])
        state["undesired_contact_count"] = state["rear_knee_contact_count"].clone()
        state["rear_knee_support_count"] = torch.tensor([0., 1., 2., 0.])
        # 水平、直立和过渡姿态均扣分；通用项不再重复计算后小腿。
        state["gravity"] = torch.tensor([[0., 0., -1.], [0., 0., -1.],
                                         [-1., 0., 0.], [-.5, 0., -3**.5 / 2]])
        actions, failed = torch.zeros(4, 12), torch.zeros(4, dtype=torch.bool)
        terms = task.reward_components(state, actions, failed, .1)
        torch.testing.assert_close(terms["rear_knee_contact"], torch.tensor([0., -6., -12., -6.]))
        torch.testing.assert_close(terms["undesired_contact"], torch.zeros(4))
        doubled = task.reward_components(state, actions, failed, .2)
        torch.testing.assert_close(doubled["rear_knee_contact"], 2 * terms["rear_knee_contact"])
        task.cfg.rewards.rear_knee_contact_weight = 0.
        self.assertEqual(task.reward_components(state, actions, failed, .1)["rear_knee_contact"].sum(), 0.)

    def test_mixed_sensor_contacts_count_each_body_once(self):
        scene = Scene()
        task = get_task("upright")
        task.cfg.rewards.reward_schema_version = 4
        env = load_env_class()(Sim(scene), scene, task=task)
        env.finish_initial_settling()
        # 同时有机身、髋、后小腿、脚掌接触；脚掌不属于非足部。
        for name in ("base", "FL_hip", "RL_calf", "RL_foot"):
            scene.contact_forces.forces[:, scene.contact_forces.body_names.index(name), 2] = 10.
        _, reward, _, extras = env.step(env.standing_actions)
        log = extras["log"]
        cfg = task.cfg.rewards
        torch.testing.assert_close(log["Reward/undesired_contact"], torch.tensor(-cfg.undesired_contact_weight * env.dt))
        torch.testing.assert_close(log["Reward/hip_contact"], torch.tensor(-cfg.hip_contact_weight * env.dt))
        torch.testing.assert_close(log["Reward/rear_knee_contact"], torch.tensor(-cfg.rear_knee_contact_weight * env.dt))
        torch.testing.assert_close(reward, sum(env._reward_components.values()))

    def test_rear_knee_contact_log_is_included_in_environment_reward(self):
        scene = Scene()
        task = get_task("upright")
        task.cfg.rewards.rear_knee_contact_weight = 60.
        env = load_env_class()(Sim(scene), scene, task=task)
        env.finish_initial_settling()
        for name in ("RL_calf", "RR_calf"):
            scene.contact_forces.forces[:, scene.contact_forces.body_names.index(name), 2] = 10.
        _, reward, _, extras = env.step(env.standing_actions)
        self.assertEqual(extras["log"]["Contact/rear_knee_contact_count"], 2.)
        self.assertEqual(extras["log"]["Contact/rear_knee_support_count"], 2.)
        torch.testing.assert_close(extras["log"]["Reward/rear_knee_contact"], torch.tensor(-120. * env.dt))
        torch.testing.assert_close(reward, sum(env._reward_components.values()))

    def test_rear_knee_contact_config_round_trip_and_legacy_playback(self):
        task = get_task("upright")
        task.cfg.rewards.rear_knee_contact_weight = 123.
        saved = task.run_config(False)
        restored = get_task("upright")
        restored.load_playback_config(saved)
        self.assertEqual(restored.run_config(False), saved)
        saved["reward_cfg"].pop("rear_knee_contact_weight")
        restored.load_playback_config(saved)
        self.assertEqual(restored.cfg.rewards.rear_knee_contact_weight, 0.)
        with self.assertRaises(ValueError):
            RewardCfg(rear_knee_contact_weight=-1.)

    def test_hip_rear_reward_uses_product_over_rear_pair(self):
        task = get_task("upright")
        cfg = task.cfg.rewards
        state = self.state(3)
        # 只取后两列（RL、RR）；前髋取任意值都不影响该项。
        state["hip_joint_pos"] = torch.tensor([
            [9., 9., 0., 0.],
            [0., 0., cfg.hip_rear_sigma, cfg.hip_rear_sigma],
            [0., 0., -cfg.hip_rear_sigma / 2, cfg.hip_rear_sigma / 2],
        ])
        dt = .1
        terms = task.reward_components(state, torch.zeros(3, 12), torch.zeros(3, dtype=torch.bool), dt)
        decay = math.exp(-1)
        # 第三行两腿各偏 sigma/2，乘积为 exp(-0.25)² = e^-0.5。
        expected = cfg.hip_rear_weight * torch.tensor([1., decay**2, math.exp(-0.5)]) * dt
        torch.testing.assert_close(terms["hip_rear"], expected)
        # 收益始终非负，且单腿外撇就会压低整组（乘积而非平均）。
        self.assertTrue((terms["hip_rear"] >= 0).all())
        one_leg = self.state(1)
        one_leg["hip_joint_pos"] = torch.tensor([[0., 0., cfg.hip_rear_sigma, 0.]])
        one = task.reward_components(one_leg, torch.zeros(1, 12), torch.zeros(1, dtype=torch.bool), dt)
        torch.testing.assert_close(one["hip_rear"],
                                   torch.full((1,), cfg.hip_rear_weight * decay * dt))

    def test_hip_rear_sigma_must_be_positive(self):
        with self.assertRaises(ValueError):
            RewardCfg(hip_rear_sigma=0.)

    def test_rear_symmetry_reward_uses_signed_joint_pairing(self):
        task = get_task("upright")
        cfg = task.cfg.rewards
        state = self.state(3)
        # 对称：髋左负右正、大腿相等、小腿相等。
        state["hip_joint_pos"] = torch.tensor([[0., 0., -.3, .3]] * 3)
        state["thigh_joint_pos"] = torch.tensor([[.8, .8, 1.2, 1.2]] * 3)
        state["calf_joint_pos"] = torch.tensor([[-1.6, -1.6, -2.0, -2.0]] * 3)
        dt = .1
        torch.testing.assert_close(rear_symmetry_error(state), torch.zeros(3))
        terms = task.reward_components(state, torch.zeros(3, 12), torch.zeros(3, dtype=torch.bool), dt)
        torch.testing.assert_close(terms["rear_symmetry"],
                                   torch.full((3,), cfg.rear_symmetry_weight * dt))
        # 非对称：髋和 -.3、大腿差 -.8、小腿差 -.4，RMS 误差见下。
        state["hip_joint_pos"] = torch.tensor([[0., 0., -.3, 0.]] * 3)
        state["thigh_joint_pos"] = torch.tensor([[.8, .8, 1.2, 2.0]] * 3)
        state["calf_joint_pos"] = torch.tensor([[-1.6, -1.6, -2.0, -1.6]] * 3)
        error = math.sqrt((.3**2 + .8**2 + .4**2) / 3)
        torch.testing.assert_close(rear_symmetry_error(state), torch.full((3,), error))
        terms = task.reward_components(state, torch.zeros(3, 12), torch.zeros(3, dtype=torch.bool), dt)
        expected = cfg.rear_symmetry_weight * math.exp(-(error / cfg.rear_symmetry_sigma) ** 2) * dt
        torch.testing.assert_close(terms["rear_symmetry"], torch.full((3,), expected))
        self.assertLess(terms["rear_symmetry"][0], cfg.rear_symmetry_weight * dt)

    def test_rear_symmetry_sigma_must_be_positive(self):
        with self.assertRaises(ValueError):
            RewardCfg(rear_symmetry_sigma=0.)

    def test_foot_slide_penalty_only_counts_contacting_feet_in_world_xy(self):
        task = get_task("upright")
        state = self.state(3)
        state["foot_velocity_w"] = torch.tensor([
            [[.5, 0., 9.], [.5, 0., 9.], [.5, 0., 9.], [.5, 0., 9.]],
            [[.5, 0., 0.], [0., .3, 0.], [0., 0., 9.], [.4, 0., 0.]],
            [[1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.]],
        ])
        # 第三只脚在空中，即使速度很大也不扣分；竖直速度不参与计算。
        state["foot_contact"] = torch.tensor([[True] * 4, [True, True, False, True], [False] * 4])
        dt = .1
        terms = task.reward_components(state, torch.zeros(3, 12), torch.zeros(3, dtype=torch.bool), dt)
        expected = torch.tensor([2., 1.2, 0.])
        torch.testing.assert_close(terms["foot_slide"], -task.cfg.rewards.foot_slide_weight * expected * dt)
        # 权重为 0 时不再读取足部速度，兼容未启用该项的配置。
        task.cfg.rewards.foot_slide_weight = 0.
        del state["foot_velocity_w"], state["foot_contact"]
        terms = task.reward_components(state, torch.zeros(3, 12), torch.zeros(3, dtype=torch.bool), dt)
        self.assertEqual(terms["foot_slide"].sum(), 0.)

    def test_removed_thigh_rear_fields_in_old_model_do_not_break_loading(self):
        task = get_task("upright")
        saved = task.run_config(False)
        saved["reward_cfg"].update(thigh_rear_weight=.5, thigh_rear_target=-.35, thigh_rear_sigma=3.0)
        task.load_playback_config(saved)  # 不抛 TypeError 即通过。
        restored = task.run_config(False)
        for name in ("thigh_rear_weight", "thigh_rear_target", "thigh_rear_sigma"):
            self.assertNotIn(name, restored["reward_cfg"])

    def test_removed_reward_fields_in_old_model_do_not_break_loading(self):
        task = get_task("upright")
        saved = task.run_config(False)
        removed = {
            "hip_front_weight": 45., "hip_front_sigma": .5,
            "hip_knee_lift_weight": 30., "hip_knee_lift_sigma": .15,
            "rear_airborne_weight": 40., "rear_both_airborne_weight": 60.,
            "rear_knee_and_foot_support_weight": 10.,
            "front_pose_weight": 15., "front_pose_target_x": .55, "front_pose_target_y": .12,
            "front_pose_target_z": .1, "front_pose_sigma": .12,
        }
        saved["reward_cfg"].update(removed)
        task.load_playback_config(saved)  # 已删除的字段被忽略，不抛 TypeError。
        restored = task.run_config(False)["reward_cfg"]
        for name in removed:
            self.assertNotIn(name, restored)

    def test_removed_progress_field_in_old_model_does_not_break_loading(self):
        task = get_task("upright")
        saved = task.run_config(False)
        self.assertNotIn("upright_progress_weight", saved["reward_cfg"])
        saved["reward_cfg"]["upright_progress_weight"] = 20.
        task.load_playback_config(saved)
        restored = task.run_config(False)
        self.assertNotIn("upright_progress_weight", restored["reward_cfg"])
        self.assertEqual(saved["reward_cfg"]["upright_progress_weight"], 20.)

    def test_pitch_velocity_penalizes_body_y_both_signs_without_orientation_gating(self):
        task = get_task("upright")
        task.cfg.rewards.reward_schema_version = 5
        task.cfg.rewards.pitch_velocity_weight = 2.
        state = self.state(5)
        state["angular_velocity"] = torch.tensor([[9., 0., 9.], [0., 2., 0.], [0., -2., 0.],
                                                  [0., 2., 0.], [0., 2., 0.]])
        state["gravity"] = torch.tensor([[0., 0., -1.], [0., 0., -1.], [0., 0., -1.],
                                         [-.5, 0., -3**.5 / 2], [-1., 0., 0.]])
        actions, failed = torch.zeros(5, 12), torch.zeros(5, dtype=torch.bool)
        terms = task.reward_components(state, actions, failed, .1)
        torch.testing.assert_close(terms["pitch_velocity"], torch.tensor([0., -.8, -.8, -.8, -.8]))
        doubled = task.reward_components(state, actions, failed, .2)
        torch.testing.assert_close(doubled["pitch_velocity"], 2 * terms["pitch_velocity"])
        task.cfg.rewards.pitch_velocity_weight = 0.
        self.assertEqual(task.reward_components(state, actions, failed, .1)["pitch_velocity"].sum(), 0.)

    def test_body_xz_and_pitch_penalties_do_not_overlap(self):
        task = get_task("upright")
        task.cfg.rewards.reward_schema_version = 4
        state = self.state(4)
        # 直立时 body X 朝世界 Z，body Z 朝世界 -X；两坐标系速度一致。
        state["gravity"][:] = torch.tensor([-1., 0., 0.])
        state["angular_velocity"] = torch.tensor([[2., 0., 0.], [0., 2., 0.],
                                                  [0., 0., 2.], [1., 2., 3.]])
        state["angular_velocity_w"] = torch.tensor([[0., 0., 2.], [0., 2., 0.],
                                                    [-2., 0., 0.], [-3., 2., 1.]])
        terms = task.reward_components(state, torch.zeros(4, 12), torch.zeros(4, dtype=torch.bool), .1)
        cfg = task.cfg.rewards
        torch.testing.assert_close(terms["angular_velocity"], -cfg.angular_velocity_weight * .1 * torch.tensor([4., 0., 4., 10.]))
        torch.testing.assert_close(terms["pitch_velocity"], -cfg.pitch_velocity_weight * .1 * torch.tensor([0., 4., 0., 4.]))
        # 相同速度在水平姿态下也得到相同罚分，不再乘 hold。
        state["gravity"][:] = torch.tensor([0., 0., -1.])
        state["angular_velocity_w"] = state["angular_velocity"].clone()
        horizontal = task.reward_components(state, torch.zeros(4, 12), torch.zeros(4, dtype=torch.bool), .1)
        torch.testing.assert_close(horizontal["angular_velocity"], terms["angular_velocity"])
        torch.testing.assert_close(horizontal["pitch_velocity"], terms["pitch_velocity"])

    def test_version_eight_restores_velocity_penalties_but_keeps_front_pose_reward(self):
        task = get_task("upright")
        self.assertEqual(task.cfg.rewards.reward_schema_version, 8)
        self.assertEqual(task.cfg.rewards.upright_hold_weight, 500.)
        state = self.state(2)
        state["gravity"][:] = torch.tensor([-1., 0., 0.])
        state["linear_velocity_w"][:] = torch.tensor([.3, .4, 2.])
        state["angular_velocity_w"][:] = torch.tensor([1., 2., 3.])
        state["angular_velocity"][:] = torch.tensor([1., 2., 3.])
        actions, failed = torch.zeros(2, 12), torch.zeros(2, dtype=torch.bool)
        current = task.reward_components(state, actions, failed, .1)
        self.assertTrue((current["front_target"] > 0).all())
        task.cfg.rewards.reward_schema_version = 5
        legacy = task.reward_components(state, actions, failed, .1)
        for key in ("balance", "angular_velocity", "pitch_velocity"):
            torch.testing.assert_close(current[key], legacy[key])
            self.assertTrue((current[key] < 0).all())

    def test_version_six_positive_velocity_rewards(self):
        task = get_task("upright")
        task.cfg.rewards.reward_schema_version = 6
        cfg = task.cfg.rewards
        self.assertEqual(cfg.reward_schema_version, 6)
        state = self.state(3)
        state["gravity"][:] = torch.tensor([-1., 0., 0.])
        state["linear_velocity_w"] = torch.tensor([[0., 0., 99.], [cfg.horizontal_velocity_sigma, 0., 0.], [-cfg.horizontal_velocity_sigma, 0., 0.]])
        state["angular_velocity_w"] = torch.tensor([[0., 0., 0.], [cfg.angular_velocity_sigma, 0., 0.], [-cfg.angular_velocity_sigma, 0., 0.]])
        state["angular_velocity"] = torch.tensor([[0., 0., 0.], [0., cfg.pitch_velocity_sigma, 0.], [0., -cfg.pitch_velocity_sigma, 0.]])
        actions, failed = torch.zeros(3, 12), torch.zeros(3, dtype=torch.bool)
        terms = task.reward_components(state, actions, failed, .1)
        for key, weight in (("balance", cfg.balance_weight), ("angular_velocity", cfg.angular_velocity_weight), ("pitch_velocity", cfg.pitch_velocity_weight)):
            torch.testing.assert_close(terms[key], weight * .1 * torch.tensor([1., math.exp(-1), math.exp(-1)]))
            torch.testing.assert_close(task.reward_components(state, actions, failed, .2)[key], 2 * terms[key])
        state["gravity"][:] = torch.tensor([0., 0., -1.])
        horizontal = task.reward_components(state, actions, failed, .1)
        torch.testing.assert_close(horizontal["balance"], terms["balance"])
        torch.testing.assert_close(horizontal["pitch_velocity"], terms["pitch_velocity"])
        self.assertTrue((horizontal["angular_velocity"] < terms["angular_velocity"]).all())
        restored = get_task("upright")
        restored.load_playback_config(task.run_config(False))
        self.assertEqual(restored.cfg.rewards.reward_schema_version, 6)
        for name in ("horizontal_velocity_sigma", "angular_velocity_sigma", "pitch_velocity_sigma"):
            with self.assertRaises(ValueError):
                RewardCfg(**{name: 0.})

    def test_horizontal_velocity_penalty_uses_world_xy_and_ignores_z(self):
        task = get_task("upright")
        task.cfg.rewards.reward_schema_version = 5
        state = self.state(5)
        state["linear_velocity_w"] = torch.tensor([[0., 0., 0.], [1., 2., 0.],
                                                   [-1., -2., 0.], [0., 0., 2.], [1., 2., -2.]])
        # 故意设置不同的机身速度，确认读取世界坐标，不把直立时的机身 X 当作水平。
        state["linear_velocity"] = torch.full((5, 3), 99.)
        actions, failed = torch.zeros(5, 12), torch.zeros(5, dtype=torch.bool)
        cfg = task.cfg.rewards
        terms = task.reward_components(state, actions, failed, .1)
        torch.testing.assert_close(terms["balance"], -cfg.balance_weight * .1 * torch.tensor([0., 5., 5., 0., 5.]))
        self.assertNotIn("vertical_velocity", terms)
        state["gravity"][:] = torch.tensor([-1., 0., 0.])
        vertical_body = task.reward_components(state, actions, failed, .1)
        doubled = task.reward_components(state, actions, failed, .2)
        for name in ("balance",):
            torch.testing.assert_close(vertical_body[name], terms[name])
            torch.testing.assert_close(doubled[name], 2 * terms[name])
        cfg.balance_weight = 0.
        disabled = task.reward_components(state, actions, failed, .1)
        self.assertEqual(disabled["balance"].sum(), 0.)

    def test_version_five_changes_only_horizontal_balance(self):
        task = get_task("upright")
        task.cfg.rewards.reward_schema_version = 5
        state = self.state(2)
        state["gravity"] = torch.tensor([[0., 0., -1.], [-1., 0., 0.]])
        state["linear_velocity_w"][:] = torch.tensor([.3, .4, 2.])
        state["angular_velocity"][:] = torch.tensor([1., 2., 3.])
        state["angular_velocity_w"][:] = torch.tensor([1., 2., 3.])
        state["foot_support_per_foot"][:] = 1.
        state["rear_leg_supported_fraction"][:] = 1.
        state["undesired_contact_count"][:] = 3.
        state["hip_contact_count"][:] = 1.
        state["rear_knee_contact_count"][:] = 1.
        state["rear_knee_support_count"][:] = 1.
        actions, failed = torch.zeros(2, 12), torch.zeros(2, dtype=torch.bool)
        current = task.reward_components(state, actions, failed, .1)
        torch.testing.assert_close(current["balance"], torch.full((2,), -task.cfg.rewards.balance_weight * .25 * .1))
        task.cfg.rewards.reward_schema_version = 1
        legacy = task.reward_components(state, actions, failed, .1)
        self.assertGreater(legacy["balance"][1], 0.)
        for name in current:
            if name != "balance":
                torch.testing.assert_close(current[name], legacy[name], msg=name)

    def test_version_two_playback_keeps_exponential_balance_and_angular_gate(self):
        task = get_task("upright")
        saved = task.run_config(False)
        saved["reward_cfg"]["reward_schema_version"] = 2
        saved["reward_cfg"]["horizontal_velocity_sigma"] = .8
        task.load_playback_config(saved)
        state = self.state(2)
        state["gravity"] = torch.tensor([[0., 0., -1.], [-1., 0., 0.]])
        state["linear_velocity_w"][:] = torch.tensor([.3, .4, 2.])
        state["angular_velocity"][:] = torch.tensor([1., 2., 3.])
        cfg = task.cfg.rewards
        terms = task.reward_components(state, torch.zeros(2, 12), torch.zeros(2, dtype=torch.bool), .1)
        hold = torch.tensor([math.exp(-1 / cfg.upright_sigma**2), 1.])
        torch.testing.assert_close(terms["balance"], cfg.balance_weight * math.exp(-.25 / .8**2) * hold * .1)
        torch.testing.assert_close(terms["angular_velocity"], -cfg.angular_velocity_weight * 10 * hold * .1)
        self.assertNotIn("vertical_velocity", terms)

    def test_old_playback_keeps_legacy_angular_and_contact_formulas(self):
        task = get_task("upright")
        saved = task.run_config(False)
        saved["reward_cfg"].pop("reward_schema_version")
        saved["reward_cfg"]["hip_contact_weight"] = 60.
        task.load_playback_config(saved)
        self.assertEqual(task.cfg.rewards.reward_schema_version, 1)
        state = self.state(2)
        state["gravity"][:] = torch.tensor([-1., 0., 0.])
        state["angular_velocity_w"] = torch.tensor([[0., 2., 0.], [1., 2., 3.]])
        state["angular_velocity"] = torch.tensor([[0., 2., 0.], [3., 2., -1.]])
        state["undesired_contact_count"] = torch.tensor([3., 2.])
        state["hip_contact_count"] = torch.ones(2)
        state["rear_knee_contact_count"] = torch.ones(2)
        state["rear_knee_support_count"] = torch.tensor([1., 0.])
        cfg = task.cfg.rewards
        terms = task.reward_components(state, torch.zeros(2, 12), torch.zeros(2, dtype=torch.bool), .1)
        torch.testing.assert_close(terms["angular_velocity"], -.1 * cfg.angular_velocity_weight * torch.tensor([4., 14.]))
        torch.testing.assert_close(terms["pitch_velocity"], -.1 * cfg.pitch_velocity_weight * torch.tensor([4., 4.]))
        torch.testing.assert_close(terms["undesired_contact"], -.1 * cfg.undesired_contact_weight * torch.tensor([2., 2.]))
        torch.testing.assert_close(terms["hip_contact"], torch.full((2,), -6.))
        torch.testing.assert_close(terms["rear_knee_contact"], torch.full((2,), -.1 * cfg.rear_knee_contact_weight))
        saved["reward_cfg"].pop("hip_contact_weight")
        task.load_playback_config(saved)
        self.assertEqual(task.cfg.rewards.hip_contact_weight, 60.)
        with self.assertRaises(ValueError):
            RewardCfg(reward_schema_version=9)

    def test_deleted_reward_fields_are_ignored_when_loading_saved_config(self):
        task = get_task("upright")
        saved = task.run_config(False)
        removed = {"front_foot_position_weight": 1000., "vertical_velocity_weight": 30.,
                   "front_foot_target_left": [.16, .142, -.15],
                   "front_foot_target_right": [.16, -.142, -.15]}
        saved["reward_cfg"].update(removed)
        task.load_playback_config(saved)
        restored = task.run_config(False)["reward_cfg"]
        self.assertTrue(set(removed).isdisjoint(restored))
        self.assertTrue(set(removed).issubset(saved["reward_cfg"]))
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        env.finish_initial_settling()
        _, reward, _, extras = env.step(env.standing_actions)
        self.assertNotIn("front_foot_error_b", env._read_state())
        self.assertNotIn("front_foot_position", env._reward_components)
        self.assertNotIn("vertical_velocity", env._reward_components)
        self.assertNotIn("Task/front_foot_position_error_m", extras["log"])
        torch.testing.assert_close(reward, sum(env._reward_components.values()))

    def test_pitch_velocity_config_round_trip_and_old_model_playback(self):
        task = get_task("upright")
        task.cfg.rewards.pitch_velocity_weight = 3.
        saved = task.run_config(False)
        restored = get_task("upright")
        restored.load_playback_config(saved)
        self.assertEqual(restored.run_config(False), saved)
        saved["reward_cfg"].pop("pitch_velocity_weight")
        restored.load_playback_config(saved)
        self.assertEqual(restored.cfg.rewards.pitch_velocity_weight, 0.)

    def test_joint_velocity_penalty_uses_actual_speed_instead_of_action_changes(self):
        task = get_task("upright")
        task.cfg.rewards.joint_velocity_weight = 2.
        state = self.state(3)
        state["joint_velocity"].zero_()
        state["joint_velocity"][1, :3] = torch.tensor([1., -2., 3.])
        state["joint_velocity"][2, :3] = torch.tensor([-1., 2., -3.])
        state["previous_actions"].fill_(-1.)
        actions = torch.ones(3, 12)  # 动作变化很大，但静止关节不因此扣分。
        failed = torch.zeros(3, dtype=torch.bool)
        terms = task.reward_components(state, actions, failed, .1)
        torch.testing.assert_close(terms["joint_velocity"], torch.tensor([0., -2.8, -2.8]))
        self.assertNotIn("action_rate", terms)
        no_action_change = task.reward_components(state, state["previous_actions"], failed, .1)
        torch.testing.assert_close(no_action_change["joint_velocity"], terms["joint_velocity"])
        doubled = task.reward_components(state, actions, failed, .2)
        torch.testing.assert_close(doubled["joint_velocity"], 2 * terms["joint_velocity"])
        task.cfg.rewards.joint_velocity_weight = 0.
        state.pop("joint_velocity")
        torch.testing.assert_close(task.reward_components(state, actions, failed, .1)["joint_velocity"], torch.zeros(3))

    def test_environment_reads_unscaled_joint_speed_and_logs_new_penalty(self):
        task = get_task("upright")
        task.cfg.rewards.joint_velocity_weight = 2.
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        env.finish_initial_settling()
        scene.robot.data.joint_vel.torch.fill_(2.)
        torch.testing.assert_close(env._read_state()["joint_velocity"], torch.full((3, 12), 2.))
        # CPU 仿真替身在每物理步把关节速度设为 .02 rad/s。
        _, reward, _, extras = env.step(env.standing_actions)
        expected = torch.tensor(-2. * 12 * .02**2 * env.dt)
        torch.testing.assert_close(extras["log"]["Reward/joint_velocity"], expected)
        self.assertNotIn("Reward/action_rate", extras["log"])
        torch.testing.assert_close(reward, sum(env._reward_components.values()))

    def test_old_action_rate_weight_is_not_misinterpreted_as_joint_speed_weight(self):
        task = get_task("upright")
        saved = task.run_config(False)
        saved["reward_cfg"].pop("joint_velocity_weight")
        saved["reward_cfg"]["action_rate_weight"] = 10.
        task.load_playback_config(saved)
        self.assertEqual(task.cfg.rewards.joint_velocity_weight, 0.)
        self.assertNotIn("action_rate_weight", task.run_config(False)["reward_cfg"])
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=task)
        self.assertNotIn("joint_velocity", env._read_state())


if __name__ == "__main__":
    unittest.main()
