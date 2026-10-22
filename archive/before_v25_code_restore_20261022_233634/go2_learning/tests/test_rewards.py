"""CPU 回归检查：配置完整性、站姿奖励及移除速度/积分依赖，无需 Isaac Sim。"""

import ast
import json
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from go2_learning.tasks.standing.rewards import (
    RewardCfg, action_limit_penalty, action_rate_penalty, height_reward, locomotion_reward, pitch_reward,
    pitch_velocity_reward, reward_components, roll_pitch_from_gravity, roll_reward, roll_velocity_reward,
    horizontal_velocity_reward, yaw_velocity_reward, foot_support_reward,
)
from go2_learning.control.commands import VelocityCommands
from go2_learning.tasks.standing.command_cfg import CommandCfg


def quadratic_cfg(**kwargs):
    """固定旧奖励语义，检查兼容回放；不受新训练默认模式影响。"""
    return RewardCfg(**{"tracking_reward_type": "quadratic", "only_positive_rewards": False,
                        "roll_velocity_weight": 0., "pitch_velocity_weight": 0.,
                        "horizontal_velocity_weight": 0., "yaw_velocity_weight": 0.,
                        "undesired_contact_weight": 0., "foot_support_weight": 0., **kwargs})


class RewardTests(unittest.TestCase):
    def test_all_referenced_config_fields_exist(self):
        # 同时检查公式和环境日志，防止定义被删掉而使用处仍保留。
        project = Path(__file__).resolve().parents[1]
        fields = set(asdict(RewardCfg()))
        for filename in ("tasks/standing/rewards.py", "simulation/env.py"):
            tree = ast.parse((project / filename).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                owner = ast.unparse(node.value)
                if owner in ("cfg", "self.reward_cfg"):
                    self.assertIn(node.attr, fields, f"{filename}:{node.lineno}: 缺少 {node.attr}")

    def test_optional_hip_and_failure_penalties(self):
        state = {
            "height": torch.full((3,), 0.32),
            "gravity": torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1),
            "linear_velocity": torch.tensor([[0.4, 0.0, 0.0], [0.15, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
            "angular_velocity": torch.zeros(3, 3),
            "joint_offset": torch.zeros(3, 12),
            "hip_joint_pos": torch.full((3, 4), 0.1),
            "roll": torch.zeros(3),
            "pitch": torch.zeros(3),
            "attitude_integral": torch.zeros(3, 3),
            "previous_actions": torch.zeros(3, 12),
            "raw_actions": torch.zeros(3, 12),
        }
        actions = torch.zeros(3, 12)
        failed = torch.zeros(3, dtype=torch.bool)
        # 固定测试案例参数，避免用户调整默认权重时把测试预期一起改变。
        cfg = quadratic_cfg(hip_weight=2.0, alive=1.0, target_height=0.32, failure_penalty=3.0)
        reward = locomotion_reward(state, actions, failed, 1 / 30, cfg)
        expected = torch.full((3,), (1.0 - 2.0 * 4 * 0.1**2) / 30)
        torch.testing.assert_close(reward, expected)
        cfg.hip_weight = 0.0
        no_hip = locomotion_reward(state, actions, failed, 1 / 30, cfg)
        torch.testing.assert_close(no_hip, torch.full((3,), 1 / 30))
        fallen = locomotion_reward(state, actions, ~failed, 1 / 30, cfg)
        torch.testing.assert_close(fallen, no_hip - cfg.failure_penalty)

    def test_roll_pitch_recovery_and_separate_penalties(self):
        expected_roll = torch.tensor([0.0, 0.3, -0.3, 0.0, 0.0])
        expected_pitch = torch.tensor([0.0, 0.0, 0.0, 0.2, -0.2])
        gravity = torch.stack((expected_pitch.sin(),
                               -expected_roll.sin() * expected_pitch.cos(),
                               -expected_roll.cos() * expected_pitch.cos()), dim=-1)
        roll, pitch = roll_pitch_from_gravity(gravity)
        torch.testing.assert_close(roll, expected_roll)
        torch.testing.assert_close(pitch, expected_pitch)
        cfg = quadratic_cfg(roll_weight=4.0, pitch_weight=5.0)
        state = {"roll": roll, "pitch": pitch, "angular_velocity": torch.zeros(5, 3),
                 "attitude_integral": torch.zeros(5, 3)}
        torch.testing.assert_close(roll_reward(state, cfg), -4 * expected_roll.square())
        torch.testing.assert_close(pitch_reward(state, cfg), -5 * expected_pitch.square())
        state["angular_velocity"][:, 0] = 1.0
        state["angular_velocity"][:, 1] = 2.0
        state["angular_velocity"][:, 2] = 10.0  # yaw 不应进入 roll/pitch 项。
        torch.testing.assert_close(roll_reward(state, cfg), -4 * expected_roll.square())
        torch.testing.assert_close(pitch_reward(state, cfg), -5 * expected_pitch.square())

    def test_disabled_velocity_rewards_do_not_require_velocity_or_integral_state(self):
        state = {"height": torch.full((3,), .32), "roll": torch.zeros(3), "pitch": torch.zeros(3),
                 "joint_offset": torch.zeros(3, 12), "hip_joint_pos": torch.zeros(3, 4),
                 "previous_actions": torch.zeros(3, 12), "raw_actions": torch.zeros(3, 12)}
        cfg = RewardCfg(roll_velocity_weight=0., pitch_velocity_weight=0., undesired_contact_weight=0.,
                        horizontal_velocity_weight=0., yaw_velocity_weight=0., foot_support_weight=0.)
        args = (state, torch.zeros(3, 12), torch.zeros(3, dtype=torch.bool), 1 / 30, cfg)
        reference = locomotion_reward(*args)
        state["linear_velocity"] = torch.randn(3, 3) * 100
        state["angular_velocity"] = torch.randn(3, 3) * 100
        state["attitude_integral"] = torch.randn(3, 3) * 100
        torch.testing.assert_close(locomotion_reward(*args), reference)

    def test_config_json_round_trip(self):
        cfg = RewardCfg()
        self.assertEqual(RewardCfg(**json.loads(json.dumps(asdict(cfg)))), cfg)

    def test_standing_commands_stay_zero(self):
        commands = VelocityCommands(16, "cpu", 1 / 30, CommandCfg(), random_sampling=True)
        commands.reset(torch.arange(16))
        for _ in range(450):
            commands.step(torch.zeros(16, dtype=torch.bool))
        self.assertEqual(torch.count_nonzero(commands.values).item(), 0)
        commands.reset(torch.tensor([1, 5]))
        self.assertEqual(torch.count_nonzero(commands.values).item(), 0)

    def test_configured_standing_reward_prefers_level_and_target_height(self):
        # 分别只改变一个物理量，检查静态目标之间没有相互抵消。
        count = 5
        state = {
            "height": torch.full((count,), 0.32),
            "linear_velocity": torch.zeros(count, 3),
            "angular_velocity": torch.zeros(count, 3),
            "linear_velocity_w": torch.zeros(count, 3),
            "angular_velocity_w": torch.zeros(count, 3),
            "joint_offset": torch.zeros(count, 12),
            "hip_joint_pos": torch.zeros(count, 4),
            "roll": torch.zeros(count), "pitch": torch.zeros(count),
            "attitude_integral": torch.zeros(count, 3),
            "previous_actions": torch.zeros(count, 12),
            "raw_actions": torch.zeros(count, 12),
            "undesired_contact_count": torch.zeros(count),
            "foot_support_fraction": torch.ones(count), "commands": torch.zeros(count, 3),
        }
        state["roll"][1] = 0.2
        state["pitch"][2] = 0.2
        state["height"][3] = 0.28
        state["linear_velocity_w"][4, 0] = 0.2
        cfg = RewardCfg(roll_weight=1., pitch_weight=1., height_weight=1., target_height=.32)
        rewards = locomotion_reward(state, torch.zeros(count, 12), torch.zeros(count, dtype=torch.bool), 1 / 30, cfg)
        self.assertTrue(torch.all(rewards[0] > rewards[1:]))

    def test_roll_pitch_rewards_do_not_depend_on_integrals(self):
        state = {"roll": torch.zeros(3), "pitch": torch.zeros(3),
                 "angular_velocity": torch.zeros(3, 3),
                 "attitude_integral": torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.0]])}
        cfg = RewardCfg()
        roll_reference = roll_reward(state, cfg)
        pitch_reference = pitch_reward(state, cfg)
        # 奖励函数不再要求调用方提供积分状态。
        del state["attitude_integral"]
        torch.testing.assert_close(roll_reward(state, cfg), roll_reference)
        torch.testing.assert_close(pitch_reward(state, cfg), pitch_reference)

    def test_action_rate_changes_only_the_reward_for_action_changes(self):
        # 固定非零动作也可完全平滑；正反跳变比从零开始的变化惩罚更大。
        actions = torch.full((3, 12), .5)
        previous = torch.tensor([.5, 0., -.5]).unsqueeze(-1).repeat(1, 12)
        torch.testing.assert_close(action_rate_penalty(actions, previous), torch.tensor([0., 3., 12.]))
        state = {
            "height": torch.full((3,), .32), "roll": torch.zeros(3), "pitch": torch.zeros(3),
            "linear_velocity": torch.zeros(3, 3), "angular_velocity": torch.zeros(3, 3),
            "joint_offset": torch.zeros(3, 12), "hip_joint_pos": torch.zeros(3, 4),
            "attitude_integral": torch.zeros(3, 3), "previous_actions": previous, "raw_actions": actions,
        }
        cfg = quadratic_cfg(action_rate_weight=2., target_height=.32)
        args = (state, actions, torch.zeros(3, dtype=torch.bool), 1 / 30, cfg)
        with_penalty = locomotion_reward(*args)
        cfg.action_rate_weight = 0.
        without_penalty = locomotion_reward(*args)
        torch.testing.assert_close(without_penalty - with_penalty, torch.tensor([0., .2, .8]))

    def test_height_reward_uses_only_instantaneous_error(self):
        state = {"height": torch.full((3,), 0.32),
                 "attitude_integral": torch.tensor([[0.0, 0.0, -0.1], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]])}
        cfg = quadratic_cfg(target_height=0.32, height_weight=40.0)
        torch.testing.assert_close(height_reward(state, cfg), torch.zeros(3))
        state["height"][:] = 0.30
        del state["attitude_integral"]
        torch.testing.assert_close(height_reward(state, cfg), torch.full((3,), -0.016))

    def test_action_limit_penalty_detects_constant_raw_saturation(self):
        raw = torch.tensor([[-1., 0., 1.], [2., -3., 1.], [8., 8., 8.]])
        torch.testing.assert_close(action_limit_penalty(raw), torch.tensor([0., 5., 147.]))
        clipped = raw.clamp(-1, 1)
        torch.testing.assert_close(action_rate_penalty(clipped, clipped), torch.zeros(3))

    def test_action_limit_weight_zero_and_nonzero(self):
        raw = torch.full((3, 12), 8.)
        state = {"height": torch.full((3,), .32), "roll": torch.zeros(3), "pitch": torch.zeros(3),
                 "joint_offset": torch.zeros(3, 12), "hip_joint_pos": torch.zeros(3, 4),
                 "previous_actions": torch.ones(3, 12), "raw_actions": raw}
        cfg = quadratic_cfg(alive=1., target_height=.32, action_weight=0., action_limit_weight=0.)
        args = (state, raw.clamp(-1, 1), torch.zeros(3, dtype=torch.bool), 1 / 30, cfg)
        disabled = locomotion_reward(*args)
        torch.testing.assert_close(disabled, torch.full((3,), 1 / 30))
        cfg.action_limit_weight = .1
        enabled = locomotion_reward(*args)
        torch.testing.assert_close(disabled - enabled, torch.full((3,), .1 * 12 * 49 / 30))

    def test_components_match_actual_reward_and_failure_is_not_scaled_by_dt(self):
        actions = torch.tensor([[1., -.5], [.2, -.3]])
        state = {
            "roll": torch.tensor([.1, -.2]), "pitch": torch.tensor([.3, -.4]),
            "height": torch.tensor([.25, .35]), "joint_offset": torch.tensor([[.1, .2], [.3, .4]]),
            "hip_joint_pos": torch.tensor([[.2], [.3]]), "previous_actions": torch.zeros_like(actions),
            "raw_actions": torch.tensor([[2., -.5], [.2, -.3]]),
        }
        cfg = quadratic_cfg(alive=1., roll_weight=2., pitch_weight=3., height_weight=4., target_height=.3,
                        posture_weight=5., action_weight=6., action_rate_weight=7., hip_weight=8.,
                        action_limit_weight=9., failure_penalty=50.)
        failed = torch.tensor([False, True])
        dt = 1 / 30
        terms = reward_components(state, actions, failed, dt, cfg)
        expected = (cfg.alive - 2 * state["roll"].square() - 3 * state["pitch"].square()
                    - 4 * (state["height"] - .3).square() - 5 * state["joint_offset"].square().sum(-1)
                    - 6 * actions.square().sum(-1) - 7 * actions.square().sum(-1)
                    - 8 * state["hip_joint_pos"].square().sum(-1) - 9 * torch.tensor([1., 0.])) * dt
        expected -= 50 * failed.float()
        torch.testing.assert_close(sum(terms.values()), expected)
        torch.testing.assert_close(locomotion_reward(state, actions, failed, dt, cfg), expected)
        torch.testing.assert_close(terms["failure"], torch.tensor([0., -50.]))
        torch.testing.assert_close(terms["action_limit"], torch.tensor([-9 * dt, 0.]))

    def test_positive_tracking_rewards_have_correct_direction_and_bounds(self):
        cfg = RewardCfg(roll_weight=2., pitch_weight=3., height_weight=4., target_height=.3,
                        roll_sigma=.25, pitch_sigma=.25, height_sigma=.08)
        errors = torch.tensor([0., .5, 1., 2., -.5, -1., -2.])
        state = {"roll": errors * cfg.roll_sigma, "pitch": errors * cfg.pitch_sigma,
                 "height": cfg.target_height + errors * cfg.height_sigma}
        expected = torch.exp(-errors.square())
        for reward, weight in [(roll_reward(state, cfg), 2.), (pitch_reward(state, cfg), 3.),
                               (height_reward(state, cfg), 4.)]:
            torch.testing.assert_close(reward, weight * expected)
            self.assertTrue(torch.all((reward >= 0) & (reward <= weight)))
        # 用真实函数的导数确认低于目标应抬高，高于目标应降低。
        h = torch.tensor([.2, .25, .3, .35, .4], requires_grad=True)
        gradient = torch.autograd.grad(height_reward({"height": h}, cfg).sum(), h)[0]
        self.assertTrue(torch.all(gradient[:2] > 0))
        self.assertAlmostEqual(gradient[2].item(), 0.)
        self.assertTrue(torch.all(gradient[3:] < 0))

    def test_positive_floor_preserves_failure_and_component_accounting(self):
        cfg = RewardCfg(alive=1., roll_weight=1., pitch_weight=1., height_weight=1.,
                        target_height=.3, action_limit_weight=10., failure_penalty=50.,
                        action_limit_outside_floor=False, undesired_contact_weight=0.,
                        horizontal_velocity_weight=0., yaw_velocity_weight=0., foot_support_weight=0.)
        raw = torch.tensor([0., 8., 8.]).unsqueeze(-1).repeat(1, 12)
        actions = raw.clamp(-1, 1)
        state = {"roll": torch.zeros(3), "pitch": torch.zeros(3), "height": torch.full((3,), .3),
                 "angular_velocity": torch.zeros(3, 3),
                 "joint_offset": torch.zeros_like(raw), "hip_joint_pos": torch.zeros(3, 4),
                 "previous_actions": actions, "raw_actions": raw}
        failed = torch.tensor([False, False, True])
        terms = reward_components(state, actions, failed, 1 / 30, cfg)
        reward = locomotion_reward(state, actions, failed, 1 / 30, cfg)
        torch.testing.assert_close(reward, torch.tensor([4 / 30, 0., -50.]))
        torch.testing.assert_close(sum(terms.values()), reward)
        self.assertGreater(terms["positive_floor"][1].item(), 0.)
        cfg.only_positive_rewards = False
        self.assertLess(locomotion_reward(state, actions, failed, 1 / 30, cfg)[1].item(), 0.)

    def test_raw_action_constraint_survives_positive_floor(self):
        cfg = RewardCfg(alive=1., roll_weight=1., pitch_weight=1., height_weight=1.,
                        target_height=.3, action_rate_weight=1000., action_limit_weight=10.,
                        action_limit_outside_floor=True, only_positive_rewards=True, failure_penalty=50.,
                        undesired_contact_weight=0., horizontal_velocity_weight=0., yaw_velocity_weight=0.,
                        foot_support_weight=0.)
        raw = torch.tensor([1., 2., 3., 3.]).unsqueeze(-1).repeat(1, 12)
        actions = raw.clamp(-1, 1)
        state = {"roll": torch.zeros(4), "pitch": torch.zeros(4), "height": torch.full((4,), .3),
                 "angular_velocity": torch.zeros(4, 3),
                 "joint_offset": torch.zeros_like(raw), "hip_joint_pos": torch.zeros(4, 4),
                 "previous_actions": torch.zeros_like(raw), "raw_actions": raw}
        failed = torch.tensor([False, False, False, True])
        terms = reward_components(state, actions, failed, 1 / 30, cfg)
        reward = locomotion_reward(state, actions, failed, 1 / 30, cfg)
        # 常规正则已触发下限修正；原始动作越超界，独立扣分仍越多，失败仍单独扣 50。
        self.assertTrue(torch.all(terms["positive_floor"] > 0))
        torch.testing.assert_close(reward, torch.tensor([0., -4., -16., -66.]))
        torch.testing.assert_close(sum(terms.values()), reward)

    def test_body_roll_pitch_velocity_penalties_ignore_yaw_and_direction(self):
        cfg = RewardCfg(roll_velocity_weight=2., pitch_velocity_weight=3.)
        state = {"angular_velocity": torch.tensor([[1., 2., 100.], [-1., -2., -100.], [0., 0., 999.]])}
        torch.testing.assert_close(roll_velocity_reward(state, cfg), torch.tensor([-2., -2., 0.]))
        torch.testing.assert_close(pitch_velocity_reward(state, cfg), torch.tensor([-12., -12., 0.]))

    def test_velocity_components_are_scaled_by_dt_and_reduce_actual_reward(self):
        cfg = RewardCfg(alive=1., roll_weight=2., pitch_weight=3., height_weight=4., target_height=.3,
                        roll_velocity_weight=2., pitch_velocity_weight=3., action_limit_weight=0.,
                        undesired_contact_weight=0., horizontal_velocity_weight=0., yaw_velocity_weight=0.,
                        foot_support_weight=0.)
        actions = torch.zeros(2, 12)
        state = {"roll": torch.zeros(2), "pitch": torch.zeros(2), "height": torch.full((2,), .3),
                 "angular_velocity": torch.tensor([[0., 0., 100.], [1., -1., -100.]]),
                 "joint_offset": actions, "hip_joint_pos": torch.zeros(2, 4), "previous_actions": actions}
        failed = torch.zeros(2, dtype=torch.bool)
        dt = 1 / 30
        terms = reward_components(state, actions, failed, dt, cfg)
        reward = locomotion_reward(state, actions, failed, dt, cfg)
        torch.testing.assert_close(terms["roll_velocity"], torch.tensor([0., -2 * dt]))
        torch.testing.assert_close(terms["pitch_velocity"], torch.tensor([0., -3 * dt]))
        torch.testing.assert_close(reward, torch.tensor([10 * dt, 5 * dt]))
        torch.testing.assert_close(sum(terms.values()), reward)

    def test_contact_penalty_survives_floor_and_preserves_failure_accounting(self):
        cfg = RewardCfg(alive=1., roll_weight=1., pitch_weight=1., height_weight=1.,
                        target_height=.3, action_rate_weight=1000., action_limit_weight=0.,
                        undesired_contact_weight=10., failure_penalty=50.,
                        horizontal_velocity_weight=0., yaw_velocity_weight=0., foot_support_weight=0.)
        actions = torch.ones(4, 12)
        state = {"roll": torch.zeros(4), "pitch": torch.zeros(4), "height": torch.full((4,), .3),
                 "angular_velocity": torch.zeros(4, 3), "joint_offset": torch.zeros_like(actions),
                 "hip_joint_pos": torch.zeros(4, 4), "previous_actions": torch.zeros_like(actions),
                 "undesired_contact_count": torch.tensor([0., 1., 4., 4.])}
        failed = torch.tensor([False, False, False, True])
        for dt in (1 / 30, 1 / 60):
            terms = reward_components(state, actions, failed, dt, cfg)
            expected = -10 * state["undesired_contact_count"] * dt - 50 * failed.float()
            self.assertTrue(torch.all(terms["positive_floor"] > 0))
            torch.testing.assert_close(locomotion_reward(state, actions, failed, dt, cfg), expected)
            torch.testing.assert_close(sum(terms.values()), expected)
        cfg.undesired_contact_weight = 0.
        del state["undesired_contact_count"]
        torch.testing.assert_close(locomotion_reward(state, actions, failed, 1 / 30, cfg), -50 * failed.float())

    def test_contact_configuration_rejects_negative_or_nonfinite_values(self):
        for name in ("undesired_contact_weight", "contact_force_threshold"):
            for value in (-1., float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    RewardCfg(**{name: value})

    def test_stationary_rewards_use_world_horizontal_and_vertical_rotation_axes(self):
        cfg = RewardCfg(horizontal_velocity_weight=20., yaw_velocity_weight=10.,
                        horizontal_velocity_sigma=.5, yaw_velocity_sigma=1.)
        state = {
            "linear_velocity_w": torch.tensor([[0., 0., 99.], [.3, .4, 0.], [-.3, -.4, -99.], [1., 0., 0.]]),
            "angular_velocity_w": torch.tensor([[99., -99., 0.], [0., 0., 1.], [0., 0., -1.], [0., 0., 2.]]),
            # 机身坐标数据故意与世界数据不同，检查奖励没有取错坐标系。
            "linear_velocity": torch.full((4, 3), 999.),
            "angular_velocity": torch.full((4, 3), 999.),
        }
        expected = torch.exp(-torch.tensor([0., 1., 1., 4.]))
        torch.testing.assert_close(horizontal_velocity_reward(state, cfg), 20 * expected)
        torch.testing.assert_close(yaw_velocity_reward(state, cfg), 10 * expected)
        velocity = torch.tensor([[.2, -.3, 0.]], requires_grad=True)
        gradient = torch.autograd.grad(horizontal_velocity_reward({"linear_velocity_w": velocity}, cfg).sum(), velocity)[0]
        self.assertLess(gradient[0, 0], 0.)  # 正速度应减小。
        self.assertGreater(gradient[0, 1], 0.)  # 负速度应增大，向零靠拢。
        self.assertEqual(gradient[0, 2], 0.)

    def test_stationary_components_reduce_return_for_drift_and_scale_with_dt(self):
        cfg = RewardCfg(alive=1., roll_weight=0., pitch_weight=0., height_weight=0.,
                        roll_velocity_weight=0., pitch_velocity_weight=0.,
                        action_rate_weight=0., action_limit_weight=0., undesired_contact_weight=0.,
                        horizontal_velocity_weight=20., yaw_velocity_weight=10.,
                        horizontal_velocity_sigma=.5, yaw_velocity_sigma=1., foot_support_weight=0.)
        actions = torch.zeros(3, 12)
        state = {"roll": torch.zeros(3), "pitch": torch.zeros(3), "height": torch.full((3,), .3),
                 "joint_offset": actions, "hip_joint_pos": torch.zeros(3, 4), "previous_actions": actions,
                 "linear_velocity_w": torch.tensor([[0., 0., 0.], [.5, 0., 0.], [0., 0., 0.]]),
                 "angular_velocity_w": torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]])}
        failed = torch.zeros(3, dtype=torch.bool)
        for dt in (1 / 30, 1 / 60):
            terms = reward_components(state, actions, failed, dt, cfg)
            expected = (torch.tensor([31., 11., 21.]) + torch.tensor([0., 20., 10.]) * torch.exp(torch.tensor(-1.))) * dt
            torch.testing.assert_close(sum(terms.values()), expected)
            torch.testing.assert_close(locomotion_reward(state, actions, failed, dt, cfg), expected)
            self.assertGreater(expected[0], expected[1])
            self.assertGreater(expected[0], expected[2])

    def test_stationary_configuration_validation(self):
        for name in ("horizontal_velocity_weight", "yaw_velocity_weight"):
            for value in (-1., float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    RewardCfg(**{name: value})
        for name in ("horizontal_velocity_sigma", "yaw_velocity_sigma"):
            for value in (0., -1., float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    RewardCfg(**{name: value})

    def test_foot_support_is_gated_by_commands_not_actual_velocity(self):
        cfg = RewardCfg(foot_support_weight=20., standing_linear_threshold=.05, standing_yaw_threshold=.1)
        state = {
            "commands": torch.tensor([[0., 0., 0.], [.2, 0., 0.], [0., -.2, 0.],
                                       [0., 0., .2], [0., 0., -.2], [.05, 0., .1], [0., 0., 0.]]),
            "foot_support_fraction": torch.tensor([.75, 1., 1., 1., 1., 1., 1.]),
            "linear_velocity_w": torch.full((7, 3), 100.),
        }
        torch.testing.assert_close(foot_support_reward(state, cfg), torch.tensor([15., 0., 0., 0., 0., 20., 20.]))

    def test_foot_support_actual_reward_scales_with_dt_and_can_be_disabled(self):
        cfg = RewardCfg(alive=1., roll_weight=0., pitch_weight=0., height_weight=0.,
                        roll_velocity_weight=0., pitch_velocity_weight=0., horizontal_velocity_weight=0.,
                        yaw_velocity_weight=0., action_limit_weight=0., action_rate_weight=0.,
                        undesired_contact_weight=0., foot_support_weight=20.)
        actions = torch.zeros(3, 12)
        state = {"roll": torch.zeros(3), "pitch": torch.zeros(3), "height": torch.full((3,), .3),
                 "joint_offset": actions, "hip_joint_pos": torch.zeros(3, 4), "previous_actions": actions,
                 "commands": torch.zeros(3, 3), "foot_support_fraction": torch.tensor([0., .75, 1.])}
        failed = torch.zeros(3, dtype=torch.bool)
        for dt in (1 / 30, 1 / 60):
            terms = reward_components(state, actions, failed, dt, cfg)
            torch.testing.assert_close(terms["foot_support"], torch.tensor([0., 15., 20.]) * dt)
            torch.testing.assert_close(locomotion_reward(state, actions, failed, dt, cfg), torch.tensor([1., 16., 21.]) * dt)
            torch.testing.assert_close(sum(terms.values()), torch.tensor([1., 16., 21.]) * dt)
        cfg.foot_support_weight = 0.
        del state["commands"], state["foot_support_fraction"]
        torch.testing.assert_close(locomotion_reward(state, actions, failed, 1 / 30, cfg), torch.ones(3) / 30)

    def test_foot_support_config_rejects_negative_and_nonfinite_values(self):
        for name in ("foot_support_weight", "standing_linear_threshold", "standing_yaw_threshold"):
            for value in (-1., float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    RewardCfg(**{name: value})


if __name__ == "__main__":
    unittest.main()
