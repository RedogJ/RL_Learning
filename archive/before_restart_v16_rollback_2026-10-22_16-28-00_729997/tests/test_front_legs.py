"""前腿目标：机身坐标、实际足部名称、奖励与旧模型配置兼容。"""

import json
import math
import unittest

import torch

from go2_learning.simulation.kinematics import points_in_body_frame
from go2_learning.tasks import get_task
from go2_learning.tasks.upright.reward_cfg import RewardCfg
from go2_learning.tests.test_reset_settling import load_env_class, Scene, Sim


class FrontLegTests(unittest.TestCase):
    def test_world_translation_yaw_and_upright_pitch_leave_body_targets_unchanged(self):
        # 三台狗分别平放、世界偏航 90°、机身 +X 朝上；含不同环境原点偏移。
        s = math.sqrt(.5)
        q = torch.tensor([[0., 0., 0., 1.], [0., 0., s, s], [0., -s, 0., s]])
        origins = torch.tensor([[2., 4., .4], [-3., 9., .5], [20., -8., .6]])
        local = torch.tensor([[.16, .142, -.15], [.16, -.142, -.15]])
        yaw = torch.stack((-local[:, 1], local[:, 0], local[:, 2]), dim=-1)
        upright = torch.stack((-local[:, 2], local[:, 1], local[:, 0]), dim=-1)
        world = torch.stack((local, yaw, upright)) + origins[:, None, :]
        actual = points_in_body_frame(world, origins, q)
        torch.testing.assert_close(actual, local.expand(3, -1, -1), atol=2e-6, rtol=0.)

    def test_penalty_counts_both_feet_scales_with_dt_and_can_be_disabled(self):
        task = get_task("upright")
        task.cfg.rewards.front_foot_position_weight = 1000.
        scene = Scene()
        template = load_env_class()(Sim(scene), scene, task=task)._read_state()
        state = {key: value[:1].repeat(4, *([1] * (value.ndim - 1))) for key, value in template.items()}
        state["front_foot_error_b"] = torch.tensor([
            [[0., 0., 0.], [0., 0., 0.]],
            [[.1, 0., 0.], [-.1, 0., 0.]],
            [[.1, .2, 0.], [0., 0., .3]],
            [[0., 0., 0.], [.2, 0., 0.]],
        ])
        actions, failed = torch.zeros(4, 12), torch.zeros(4, dtype=torch.bool)
        terms = task.reward_components(state, actions, failed, .1)
        torch.testing.assert_close(terms["front_foot_position"], torch.tensor([0., -1., -7., -2.]))
        doubled = task.reward_components(state, actions, failed, .2)
        torch.testing.assert_close(doubled["front_foot_position"], 2 * terms["front_foot_position"])
        task.cfg.rewards.front_foot_position_weight = 0.
        del state["front_foot_error_b"]
        self.assertEqual(task.reward_components(state, actions, failed, .1)["front_foot_position"].sum(), 0.)
        self.assertNotIn("Task/front_foot_position_error_m", task.metrics(state))

    def test_environment_uses_named_front_feet_and_logs_physical_position_error(self):
        scene = Scene()
        scene.robot.body_names = list(reversed(scene.robot.body_names))
        task = get_task("upright")
        env = load_env_class()(Sim(scene), scene, task=task)
        env.finish_initial_settling()
        targets = torch.tensor([task.cfg.rewards.front_foot_target_left,
                                task.cfg.rewards.front_foot_target_right])
        root = scene.robot.data.root_link_pos_w.torch
        for i, leg in enumerate(("FL", "FR")):
            body_id = scene.robot.body_names.index(f"{leg}_foot")
            scene.robot.data.body_link_pos_w.torch[:, body_id] = root + targets[i]
        state = env._read_state()
        torch.testing.assert_close(state["front_foot_error_b"], torch.zeros(3, 2, 3), atol=1e-6, rtol=0.)
        # 右前脚多偏 0.1 m：两足平均距离 0.05 m，距离平方平均 0.005 m²。
        scene.robot.data.body_link_pos_w.torch[:, env.front_foot_body_ids[1], 0] += .1
        obs, reward, _, extras = env.step(env.standing_actions)
        self.assertEqual(obs["policy"].shape, (3, 52))
        log = extras["log"]
        torch.testing.assert_close(log["Task/front_foot_position_error_m"], torch.tensor(.05), atol=1e-6, rtol=0.)
        torch.testing.assert_close(log["Task/front_foot_FL_error_m"], torch.tensor(0.), atol=1e-6, rtol=0.)
        torch.testing.assert_close(log["Task/front_foot_FR_error_m"], torch.tensor(.1), atol=1e-6, rtol=0.)
        torch.testing.assert_close(log["Reward/front_foot_position"],
                                   torch.tensor(-task.cfg.rewards.front_foot_position_weight * .005 * env.dt),
                                   atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(reward, sum(env._reward_components.values()))

    def test_target_configuration_json_round_trip_and_legacy_playback(self):
        task = get_task("upright")
        task.cfg.rewards.front_foot_target_left = (.15, .16, -.17)
        saved = task.run_config(False)
        restored = get_task("upright")
        restored.load_playback_config(json.loads(json.dumps(saved)))
        self.assertEqual(json.loads(json.dumps(restored.run_config(False))), json.loads(json.dumps(saved)))
        for name in ("front_foot_position_weight", "front_foot_target_left", "front_foot_target_right"):
            saved["reward_cfg"].pop(name)
        restored.load_playback_config(saved)
        self.assertEqual(restored.cfg.rewards.front_foot_position_weight, 0.)
        scene = Scene()
        env = load_env_class()(Sim(scene), scene, task=restored)
        self.assertIsNone(env.front_foot_body_ids)
        self.assertNotIn("front_foot_error_b", env._read_state())

    def test_target_validation_accepts_signed_xyz_and_rejects_invalid_values(self):
        cfg = RewardCfg(front_foot_target_left=[-.1, .15, -.2])
        self.assertEqual(cfg.front_foot_target_left, (-.1, .15, -.2))
        for point in ((0., 0.), (0., float("nan"), 0.), (0., 0., float("inf")), "abc", (0., "x", 0.)):
            with self.subTest(point=point), self.assertRaises(ValueError):
                RewardCfg(front_foot_target_left=point)
        with self.assertRaises(ValueError):
            RewardCfg(front_foot_position_weight=-1.)


if __name__ == "__main__":
    unittest.main()
